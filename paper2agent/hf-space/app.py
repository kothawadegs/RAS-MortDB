"""RAS-MortDB Paper Agent: web demo for Hugging Face Spaces.

Runs the Paper2Agent-verified MCP tool ``ras_mortdb_detect_fish_mortality`` (tools/run_inference.py), which calls
the paper's own ``inference/run_inference.py``. The upstream script and the released weights are downloaded on
demand from https://github.com/PA-RRanjan/RAS-MortDB at the pinned commit and checked against SHA-256 hashes in
upstream_manifest.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import urllib.request
from pathlib import Path

try:  # ZeroGPU hardware: `spaces` must be imported before torch/CUDA; elsewhere this is a no-op
    import spaces
    _gpu = spaces.GPU(duration=60)
except ImportError:
    def _gpu(fn):
        return fn

import gradio as gr
import pandas as pd

HERE = Path(__file__).resolve().parent
MANIFEST = json.loads((HERE / "upstream_manifest.json").read_text())
COMMIT = MANIFEST["commit"]
RAW = f"https://raw.githubusercontent.com/PA-RRanjan/RAS-MortDB/{COMMIT}/"
CHECKOUT = Path(os.environ.setdefault("RAS_MORTDB_ROOT", str(Path(tempfile.gettempdir()) / f"ras-mortdb-{COMMIT[:7]}")))
OUTPUTS = Path(tempfile.gettempdir()) / "ras-mortdb-outputs"

try:
    import fastmcp  # noqa: F401
except ImportError:
    # Hugging Face Spaces always installs gradio[mcp] (mcp<2), which cannot coexist with FastMCP 4 (mcp>=2).
    # The web app only calls the tool function, so give the unchanged tool module a minimal stand-in whose
    # FastMCP(...).tool() returns the function as is, matching what FastMCP 4's decorator returns.
    import sys
    import types

    class _FastMCPStandIn:
        def __init__(self, *args, **kwargs):
            pass

        def tool(self, *args, **kwargs):
            return lambda fn: fn

    sys.modules["fastmcp"] = types.SimpleNamespace(FastMCP=_FastMCPStandIn)

from tools.run_inference import ras_mortdb_detect_fish_mortality  # noqa: E402  (reads RAS_MORTDB_ROOT per call)

MODELS = ["yolo26n", "yolo26s", "yolo26m", "yolo11n", "yolo11s", "yolo11m",
          "yolov8n", "yolov8s", "yolov8m", "yolov5nu", "yolov5su", "yolov5mu"]
_FETCH_LOCK = threading.Lock()


def _run_tool(image_path: str, model: str, weight_format: str, conf: float) -> dict:
    """Run the verified MCP tool on the CPU (fast enough for these models; uses no ZeroGPU quota)."""
    return ras_mortdb_detect_fish_mortality(image_path=image_path, model=model, weight_format=weight_format,
                                            conf=conf, output_dir=str(OUTPUTS))


# Opt-in GPU path: on ZeroGPU hardware this attaches a GPU and counts against the visitor's daily quota.
_run_tool_gpu = _gpu(_run_tool)


def _fetch(rel: str) -> Path:
    """Download one pinned upstream file into the local checkout and verify its SHA-256."""
    entry = MANIFEST["files"][rel]
    dest = CHECKOUT / rel
    with _FETCH_LOCK:
        if dest.is_file() and hashlib.sha256(dest.read_bytes()).hexdigest() == entry["sha256"]:
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        for _attempt in range(3):  # large weight downloads are occasionally cut short; retry before giving up
            try:
                urllib.request.urlretrieve(RAW + rel, tmp)
            except OSError:
                continue
            if hashlib.sha256(tmp.read_bytes()).hexdigest() == entry["sha256"]:
                tmp.replace(dest)
                return dest
        tmp.unlink(missing_ok=True)
        raise gr.Error(f"Could not download {rel} with the pinned SHA-256 after 3 attempts; please try again.")


def detect_fish_mortality(image_path: str, model: str = "yolo26n", weight_format: str = "onnx",
                          conf: float = 0.25, use_gpu: bool = False):
    """Count dead and live fish in a recirculating aquaculture system (RAS) tank image.

    Runs the RAS-MortDB paper's own inference script (Ranjan et al., AI 2026, 7(9), 354) with one of the 12
    released YOLO models.

    Args:
        image_path: Tank image to analyse.
        model: Released model id: yolo26n/s/m, yolo11n/s/m, yolov8n/s/m or yolov5nu/su/mu.
        weight_format: "onnx" (CPU/edge path used for the paper's Raspberry Pi 5 benchmark) or "pytorch".
        conf: Detection confidence threshold in (0, 1]; the paper's script defaults to 0.25.
        use_gpu: Run on a ZeroGPU GPU instead of the CPU (uses the visitor's daily GPU quota).

    Returns:
        Annotated image, a summary of dead/live counts, and a table of detections.
    """
    if not image_path:
        raise gr.Error("Upload an image or pick an example first.")
    _fetch("inference/run_inference.py")
    _fetch(f"weights/pytorch/{model}_best.pt" if weight_format == "pytorch" else f"weights/onnx/{model}.onnx")
    try:
        r = (_run_tool_gpu if use_gpu else _run_tool)(image_path, model, weight_format, float(conf))
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        raise gr.Error(str(exc)) from exc
    level = "Zero" if r["dead_count"] == 0 else "Low (<3 dead)" if r["dead_count"] < 3 else "High (≥3 dead)"
    summary = (f"### Dead: {r['dead_count']} · Live: {r['live_count']}\n"
               f"Mortality level: **{level}** · {r['total_detections']} detections · "
               f"`{model}` ({weight_format}), conf {conf}")
    table = pd.DataFrame([{"class": d["class_name"], "confidence": round(d["confidence"], 3),
                           "x1": round(d["box_xyxy"][0], 1), "y1": round(d["box_xyxy"][1], 1),
                           "x2": round(d["box_xyxy"][2], 1), "y2": round(d["box_xyxy"][3], 1)}
                          for d in r["detections"]])
    annotated = r["artifacts"][0]["path"] if r["artifacts"] else None
    return annotated, summary, table


CITATION = ("Ranjan, R.; Kothawade, G.S.; Sharrer, K.; Tsukuda, S.; Good, C. Does YOLO26 Truly Offer Advantages "
            "over Its Predecessors for Edge Deployment? A Benchmark Study in Aquaculture. *AI* **2026**, 7(9), 354. "
            "[doi:10.3390/ai7090354](https://doi.org/10.3390/ai7090354)")

with gr.Blocks(title="RAS-MortDB Paper Agent") as demo:
    gr.Markdown(
        "# RAS-MortDB Paper Agent: fish mortality detection\n"
        f"{CITATION}\n\n"
        "Built with [Paper2Agent](https://github.com/jmiao24/Paper2Agent). Detections run the paper's own "
        f"[`run_inference.py`](https://github.com/PA-RRanjan/RAS-MortDB/blob/{COMMIT}/inference/run_inference.py) "
        "with the released weights (dataset and code CC BY 4.0, weights AGPL-3.0).")
    with gr.Tab("Detect mortality"):
        gr.Markdown(
            "**Scope:** the models were trained on one dataset: adult Atlantic salmon (1.5–2.5 kg) in a single RAS "
            "grow-out tank, imaged by an underwater camera pointing down at the drain plate. Other species, camera "
            "positions (e.g. photos from above the water), life stages or lighting were not part of training, and "
            "detections on such images are less reliable. The paper notes that generalization to them \"remains to "
            "be established\".")
        with gr.Row():
            with gr.Column():
                img = gr.Image(type="filepath", label="Tank image")
                model = gr.Dropdown(MODELS, value="yolo26n", label="Model")
                fmt = gr.Radio(["onnx", "pytorch"], value="onnx", label="Weight format")
                conf = gr.Slider(0.05, 1.0, value=0.25, step=0.05, label="Confidence threshold")
                use_gpu = gr.Checkbox(False, label="Use GPU (only on ZeroGPU hardware; counts against your daily "
                                                   "GPU quota; CPU is fast enough for these models)")
                btn = gr.Button("Detect", variant="primary")
            with gr.Column():
                out_img = gr.Image(label="Detections")
                out_md = gr.Markdown()
                out_tab = gr.Dataframe(label="Boxes (pixels)")
        gr.Examples([[str(p), "yolo26n", "onnx", 0.25] for p in sorted((HERE / "examples").glob("*.jpg"))],
                    inputs=[img, model, fmt, conf], label="Test-set images from RAS-MortDB")
        gr.Markdown("The first run of each model downloads its weights (10–100 MB), so it takes a little longer.")
        btn.click(detect_fish_mortality, [img, model, fmt, conf, use_gpu], [out_img, out_md, out_tab],
                  api_name="detect")
    with gr.Tab("Paper results"):
        gr.Markdown("### Table 3: detection accuracy on the full dataset (mean ± SD over 3 seeds)")
        gr.Dataframe(pd.read_csv(HERE / "paper" / "table-3.csv", dtype=str), show_label=False)
        gr.Markdown("### Table 6: Raspberry Pi 5 ONNX inference benchmark (200 test images)")
        gr.Dataframe(pd.read_csv(HERE / "paper" / "table-6.csv", dtype=str), show_label=False)
        gr.Markdown("### Figure 5: learning curves (training images needed to reach 90% mAP50)")
        gr.Image(str(HERE / "paper" / "figure-5.jpg"), show_label=False)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()
