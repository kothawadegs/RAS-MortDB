"""
RAS-MortDB Paper Agent — MCP server

Turns the paper:
  Ranjan, Kothawade, Sharrer, Tsukuda & Good (2026).
  "Does YOLO26 Truly Offer Advantages over Its Predecessors for Edge
  Deployment? A Benchmark Study in Aquaculture." AI, 7(9), 354.
  https://doi.org/10.3390/ai7090354

into a queryable, tool-using agent, following the Paper2Agent pattern
(Miao et al., Nature 2026, https://doi.org/10.1038/s41586-026-11044-y):
the paper's own public codebase (https://github.com/PA-RRanjan/RAS-MortDB)
is wrapped as executable MCP tools, backed by MCP resources holding the
paper's reported benchmark numbers verbatim from its README/tables, so an
LLM agent can run real mortality detection and give evidence-based
deployment answers rather than paraphrasing the abstract.

Run:
    pip install -r requirements.txt
    python mcp_server.py                      # stdio, for Claude Desktop / Claude Code
    python mcp_server.py --http --port 8000    # HTTP, e.g. for Hugging Face Spaces
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Literal

from fastmcp import FastMCP

# --------------------------------------------------------------------------
# Paths — point these at a full clone of https://github.com/PA-RRanjan/RAS-MortDB
# for all 12 models / the full test set. This package ships a slim subset
# (YOLO26n + YOLOv8n weights, 5 test images) so the tools are runnable out
# of the box.
# --------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent
WEIGHTS_PT = REPO_ROOT / "weights" / "pytorch"
WEIGHTS_ONNX = REPO_ROOT / "weights" / "onnx"

# --------------------------------------------------------------------------
# Reference numbers, copied verbatim from the paper's README / results
# tables (https://github.com/PA-RRanjan/RAS-MortDB#model-performance).
# These are NOT recomputed at query time — they are the paper's authoritative
# reported results, exposed as a structured resource so the agent can answer
# "what did the paper find" questions without re-deriving them.
# --------------------------------------------------------------------------
MODEL_TABLE: dict[str, dict] = {
    "yolo26n": {"arch": "YOLO26", "tier": "nano", "params_m": None,
                "mAP50": 94.44, "mAP50_sd": 0.38, "mAP50_95": 71.16,
                "precision": 90.94, "recall": 88.02,
                "rpi5_avg_ms": 127.5, "rpi5_fps": 7.84, "rpi5_fps_sd": 0.13,
                "onnx_mb": 9.8, "rel_fps_vs_yolov8_same_tier": "+21.7%"},
    "yolo11n": {"arch": "YOLO11", "tier": "nano",
                "mAP50": 94.79, "mAP50_sd": 0.32, "mAP50_95": 70.22,
                "precision": 89.13, "recall": 90.10,
                "rpi5_avg_ms": 149.3, "rpi5_fps": 6.70, "rpi5_fps_sd": 0.11,
                "onnx_mb": 10.6, "rel_fps_vs_yolov8_same_tier": "+4.0%"},
    "yolov8n": {"arch": "YOLOv8", "tier": "nano",
                "mAP50": 94.15, "mAP50_sd": 0.39, "mAP50_95": 69.59,
                "precision": 90.02, "recall": 88.15,
                "rpi5_avg_ms": 155.2, "rpi5_fps": 6.44, "rpi5_fps_sd": 0.08,
                "onnx_mb": 12.3, "rel_fps_vs_yolov8_same_tier": "baseline"},
    "yolov5nu": {"arch": "YOLOv5u", "tier": "nano",
                 "mAP50": 93.56, "mAP50_sd": 0.30, "mAP50_95": 67.60,
                 "precision": 89.46, "recall": 88.28,
                 "rpi5_avg_ms": 139.9, "rpi5_fps": 7.15, "rpi5_fps_sd": 0.06,
                 "onnx_mb": 10.3, "rel_fps_vs_yolov8_same_tier": "+11.0%"},
    "yolo26s": {"arch": "YOLO26", "tier": "small",
                "mAP50": 94.29, "mAP50_sd": 0.26, "mAP50_95": 71.35,
                "precision": 91.17, "recall": 88.50,
                "rpi5_avg_ms": 347.4, "rpi5_fps": 2.88, "rpi5_fps_sd": 0.03,
                "onnx_mb": 38.2, "rel_fps_vs_yolov8_same_tier": "+16.6%"},
    "yolo11s": {"arch": "YOLO11", "tier": "small",
                "mAP50": 94.54, "mAP50_sd": 0.40, "mAP50_95": 70.77,
                "precision": 90.67, "recall": 88.58,
                "rpi5_avg_ms": 361.4, "rpi5_fps": 2.77, "rpi5_fps_sd": 0.04,
                "onnx_mb": 37.9, "rel_fps_vs_yolov8_same_tier": "+12.1%"},
    "yolov8s": {"arch": "YOLOv8", "tier": "small",
                "mAP50": 94.34, "mAP50_sd": 0.36, "mAP50_95": 70.03,
                "precision": 90.85, "recall": 88.06,
                "rpi5_avg_ms": 405.1, "rpi5_fps": 2.47, "rpi5_fps_sd": 0.07,
                "onnx_mb": 44.7, "rel_fps_vs_yolov8_same_tier": "baseline"},
    "yolov5su": {"arch": "YOLOv5u", "tier": "small",
                 "mAP50": 94.09, "mAP50_sd": 0.22, "mAP50_95": 69.20,
                 "precision": 88.66, "recall": 89.94,
                 "rpi5_avg_ms": 347.7, "rpi5_fps": 2.88, "rpi5_fps_sd": 0.03,
                 "onnx_mb": 36.7, "rel_fps_vs_yolov8_same_tier": "+16.6%"},
    "yolo26m": {"arch": "YOLO26", "tier": "medium",
                "mAP50": 94.44, "mAP50_sd": 0.41, "mAP50_95": 71.96,
                "precision": 89.49, "recall": 89.22,
                "rpi5_avg_ms": 955.4, "rpi5_fps": 1.05, "rpi5_fps_sd": 0.01,
                "onnx_mb": 81.7, "rel_fps_vs_yolov8_same_tier": "0.0%"},
    "yolo11m": {"arch": "YOLO11", "tier": "medium",
                "mAP50": 94.81, "mAP50_sd": 0.10, "mAP50_95": 71.86,
                "precision": 90.19, "recall": 89.27,
                "rpi5_avg_ms": 963.7, "rpi5_fps": 1.04, "rpi5_fps_sd": 0.01,
                "onnx_mb": 80.4, "rel_fps_vs_yolov8_same_tier": "-1.0%"},
    "yolov8m": {"arch": "YOLOv8", "tier": "medium",
                "mAP50": 94.63, "mAP50_sd": 0.13, "mAP50_95": 71.34,
                "precision": 90.25, "recall": 88.59,
                "rpi5_avg_ms": 949.5, "rpi5_fps": 1.05, "rpi5_fps_sd": 0.01,
                "onnx_mb": 103.6, "rel_fps_vs_yolov8_same_tier": "baseline"},
    "yolov5mu": {"arch": "YOLOv5u", "tier": "medium",
                 "mAP50": 94.52, "mAP50_sd": 0.37, "mAP50_95": 70.88,
                 "precision": 90.54, "recall": 88.32,
                 "rpi5_avg_ms": 792.5, "rpi5_fps": 1.26, "rpi5_fps_sd": 0.00,
                 "onnx_mb": 100.5, "rel_fps_vs_yolov8_same_tier": "+20.0%"},
}

# Min. training images (of the 7 learning-curve sizes tested) each model
# needed to reach >=90% mAP50 — from Fig. 5 / Section 3.4 of the paper.
MIN_IMAGES_FOR_90_MAP50 = {
    "yolov8n": 400, "yolov8s": 400, "yolov8m": 400,
    "yolov5nu": 700, "yolov5su": 400, "yolov5mu": 700,
    "yolo11n": 700, "yolo11s": 700, "yolo11m": 700,
    "yolo26n": 1000, "yolo26s": 1000, "yolo26m": 700,
}

PAPER_CITATION = (
    "Ranjan, R.; Kothawade, G.S.; Sharrer, K.; Tsukuda, S.; Good, C. "
    "Does YOLO26 Truly Offer Advantages over Its Predecessors for Edge "
    "Deployment? A Benchmark Study in Aquaculture. AI 2026, 7(9), 354. "
    "https://doi.org/10.3390/ai7090354"
)

mcp = FastMCP(
    name="RAS-MortDB Paper Agent",
    instructions=(
        "You are the interactive agent for the paper "
        f'"{PAPER_CITATION}". '
        "You can run the paper's actual trained YOLO models on fish images "
        "to detect live/dead Atlantic salmon, and you can answer deployment "
        "questions (which model/tier to pick) using the paper's reported "
        "benchmark numbers rather than guessing. Always ground accuracy, "
        "speed, and data-efficiency claims in the get_model_performance / "
        "list_models resources, not in general YOLO knowledge."
    ),
)


def _resolve_weight(model: str, fmt: Literal["pytorch", "onnx"]) -> Path:
    model = model.lower()
    if model not in MODEL_TABLE:
        raise ValueError(f"Unknown model '{model}'. Options: {sorted(MODEL_TABLE)}")
    folder = WEIGHTS_PT if fmt == "pytorch" else WEIGHTS_ONNX
    suffix = "_best.pt" if fmt == "pytorch" else ".onnx"
    path = folder / f"{model}{suffix}"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not bundled in this package. Point WEIGHTS_PT/WEIGHTS_ONNX "
            "at a full clone of github.com/PA-RRanjan/RAS-MortDB to use all 12 models."
        )
    return path


@mcp.tool
def list_models() -> dict:
    """List all 12 YOLO model variants benchmarked in the paper (architecture,
    size tier, and headline mAP50/edge-FPS), so an agent or user can pick a
    model before calling detect_mortality or get_model_performance."""
    return {
        "models": [
            {"model": k, "architecture": v["arch"], "tier": v["tier"],
             "mAP50_pct": v["mAP50"], "raspberry_pi5_fps": v["rpi5_fps"]}
            for k, v in MODEL_TABLE.items()
        ],
        "note": "Full numbers per model via get_model_performance(model).",
    }


@mcp.tool
def get_model_performance(model: str) -> dict:
    """Return the paper's reported benchmark results for one model: full-dataset
    detection accuracy (mAP50, mAP50-95, precision, recall; mean +/- SD over 3
    training runs), Raspberry Pi 5 CPU edge-inference speed, ONNX export size,
    and the minimum number of training images needed to reach >=90% mAP50
    (data efficiency). These are the paper's own reported numbers, not
    re-derived. model: one of yolo26n/s/m, yolo11n/s/m, yolov8n/s/m,
    yolov5nu/su/mu."""
    model = model.lower()
    if model not in MODEL_TABLE:
        raise ValueError(f"Unknown model '{model}'. Options: {sorted(MODEL_TABLE)}")
    row = dict(MODEL_TABLE[model])
    row["min_training_images_for_90pct_mAP50"] = MIN_IMAGES_FOR_90_MAP50[model]
    row["source"] = PAPER_CITATION
    return row


@mcp.tool
def detect_mortality(
    image_path: str,
    model: str = "yolo26n",
    weight_format: Literal["pytorch", "onnx"] = "pytorch",
    conf: float = 0.25,
) -> dict:
    """Run the paper's trained fish-mortality detector on one image and count
    live vs. dead Atlantic salmon. Mirrors RAS-MortDB/inference/run_inference.py
    exactly (conf/iou defaults, Dead/Live class names from the trained
    weights). Use weight_format='onnx' to reproduce the paper's CPU/edge-style
    inference path; 'pytorch' for the GPU-trained weights directly.
    Returns per-detection boxes/confidences and wall-clock inference time."""
    from ultralytics import YOLO  # deferred import: only needed for this tool

    img = Path(image_path)
    if not img.exists():
        raise FileNotFoundError(image_path)
    weight_path = _resolve_weight(model, weight_format)

    net = YOLO(str(weight_path))
    t0 = time.perf_counter()
    results = net.predict(source=str(img), conf=conf, iou=0.45, verbose=False)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    r = results[0]
    detections = [
        {"class": r.names[int(c)], "confidence": round(float(conf_), 3)}
        for c, conf_ in zip(r.boxes.cls, r.boxes.conf)
    ]
    dead = sum(1 for d in detections if d["class"] == "Dead")
    live = sum(1 for d in detections if d["class"] == "Live")
    return {
        "image": str(img),
        "model": model,
        "weight_format": weight_format,
        "dead_count": dead,
        "live_count": live,
        "detections": detections,
        "inference_ms": round(elapsed_ms, 1),
    }


@mcp.tool
def compare_models_on_image(
    image_path: str,
    models: list[str] | None = None,
    weight_format: Literal["pytorch", "onnx"] = "pytorch",
    conf: float = 0.25,
) -> dict:
    """Run several trained models on the same image and compare their live/dead
    counts and confidences side by side. Only bundles models whose weight
    files are actually present locally (see list_models); defaults to every
    available model if `models` is omitted. Useful for sanity-checking whether
    architecture choice changes the detected mortality count on a given frame
    (the paper found <=1.25 pp mAP50 spread across all 12 models)."""
    candidates = models or list(MODEL_TABLE)
    out = {}
    for m in candidates:
        try:
            out[m] = detect_mortality(image_path, model=m, weight_format=weight_format, conf=conf)
        except FileNotFoundError as e:
            out[m] = {"skipped": str(e)}
    return {"image": image_path, "results": out}


@mcp.tool
def recommend_deployment(
    target_hardware: Literal["gpu_server", "raspberry_pi_or_cpu_edge"],
    annotated_training_images_available: int,
    priority: Literal["max_accuracy", "fastest_edge_inference", "smallest_model_size"] = "fastest_edge_inference",
) -> dict:
    """Recommend a model/tier following the paper's own deployment-oriented
    conclusions (Section 4): detection accuracy barely varies across
    architectures (<=1.25 pp mAP50), so model choice should be driven by data
    availability, target hardware, and inference-speed needs rather than
    architectural novelty. Returns a recommended model plus the paper-derived
    reasoning, not a generic recommendation."""
    reasons = []
    if target_hardware == "gpu_server":
        pick, reasons = "yolo11m", [
            "On GPU, tier/architecture accuracy differences are within noise "
            "(mAP50 93.56-94.81% across all 12 models, SD<=0.41%).",
            "YOLO11m had the single highest reported mAP50 (94.81%) of the 12 models.",
        ]
    else:
        if priority == "smallest_model_size":
            pick, reasons = "yolo26n", [
                "YOLO26n has the smallest ONNX export of all 12 models (9.8 MB), "
                "useful for storage/RAM-constrained on-farm edge devices.",
            ]
        elif priority == "fastest_edge_inference":
            pick, reasons = "yolo26n", [
                "YOLO26n had the highest Raspberry Pi 5 inference speed in the "
                "nano tier (7.84 +/- 0.13 FPS), 21.7% faster than same-tier YOLOv8n, "
                "due to its NMS-free/DFL-free design.",
            ]
        else:  # max_accuracy on edge hardware
            pick, reasons = "yolo11n", [
                "Within the nano tier (best edge FPS/size trade-off), YOLO11n had "
                "the highest mAP50 (94.79%) while still running at 6.70 FPS on a "
                "Raspberry Pi 5.",
            ]

    if annotated_training_images_available < MIN_IMAGES_FOR_90_MAP50[pick]:
        alt = min(MIN_IMAGES_FOR_90_MAP50, key=MIN_IMAGES_FOR_90_MAP50.get)
        reasons.append(
            f"Caveat: {pick} needed {MIN_IMAGES_FOR_90_MAP50[pick]} training images "
            f"to reach >=90% mAP50 in the paper's learning-curve analysis, but only "
            f"{annotated_training_images_available} are available here. YOLOv8 variants "
            f"were the most data-efficient (as few as {MIN_IMAGES_FOR_90_MAP50[alt]} images "
            "for the same threshold) and may be a safer choice until more annotated "
            "images are collected."
        )

    return {"recommended_model": pick, "reasoning": reasons, "source": PAPER_CITATION}


@mcp.resource("paper://abstract")
def paper_abstract() -> str:
    """The paper's citation, dataset/code availability, and headline findings."""
    return (
        f"{PAPER_CITATION}\n\n"
        "Dataset & weights (CC BY 4.0 data / AGPL-3.0 weights): "
        "https://github.com/PA-RRanjan/RAS-MortDB "
        "(archived: https://doi.org/10.5281/zenodo.20631780)\n\n"
        "Headline finding: across 12 YOLO model variants (YOLOv5u/v8/11/26 x "
        "nano/small/medium) trained on 2,800 images of live/dead Atlantic "
        "salmon in a RAS grow-out tank, full-dataset mAP50 varied by only "
        "1.25 percentage points (93.56-94.81%), so architecture generation "
        "alone does not meaningfully improve mortality-detection accuracy. "
        "YOLO26n was the fastest nano-tier model on a Raspberry Pi 5 CPU "
        "(7.84 FPS, +21.7% vs YOLOv8n) but the least data-efficient, needing "
        "1000 training images (vs 400 for YOLOv8) to reach 90% mAP50."
    )


@mcp.prompt
def benchmark_new_deployment() -> str:
    """Guides the agent through picking and validating a model for a new
    RAS mortality-monitoring deployment, using this paper's tools in order."""
    return (
        "1. Call list_models() and get_model_performance() to review the "
        "paper's reported accuracy/speed/data-efficiency trade-offs for all "
        "12 models.\n"
        "2. Ask the user: target hardware (GPU server vs Raspberry Pi/CPU "
        "edge), how many annotated training images they have, and their "
        "priority (accuracy vs. edge speed vs. model size).\n"
        "3. Call recommend_deployment() with those inputs.\n"
        "4. If the user has sample images, call detect_mortality() (or "
        "compare_models_on_image() for a couple of candidate models) on "
        "them with the recommended model to sanity-check real detections "
        "before committing to that choice.\n"
        "5. Report the recommendation with the paper-derived reasoning, "
        "not a generic opinion."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--http", action="store_true", help="Serve over HTTP instead of stdio")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.http:
        mcp.run(transport="http", port=args.port)
    else:
        mcp.run()
