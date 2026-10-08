"""RAS-MortDB MCP server.

Exposes the RAS-MortDB benchmark results and model weights to MCP clients
(e.g. Claude Code) as tools:

    list_models           - all 12 models with accuracy + Raspberry Pi 5 metrics
    get_model_info        - details and available weight files for one model
    recommend_deployment  - pick a model for a target device / priority
    get_training_config   - the args.yaml used for a given training run
    detect_mortality      - run a model on an image and count Dead / Live fish

Run with:  python mcp_server.py   (stdio transport)
"""

from __future__ import annotations

import os
from pathlib import Path

from fastmcp import FastMCP

ROOT = Path(__file__).resolve().parent
WEIGHTS_PT = Path(os.environ.get("RAS_MORTDB_WEIGHTS_PT", ROOT / "weights" / "pytorch"))
WEIGHTS_ONNX = Path(os.environ.get("RAS_MORTDB_WEIGHTS_ONNX", ROOT / "weights" / "onnx"))
TRAINING_CONFIGS = ROOT / "training_configs"

# Metrics from README.md: mean across three seeds, full dataset (2,800 training
# images) and three Raspberry Pi 5 ONNX benchmark sessions.
MODELS: dict[str, dict] = {
    "yolo26n":  {"tier": "nano",   "map50": 94.44, "map50_95": 71.16, "precision": 90.94, "recall": 88.02, "pi5_ms": 127.5, "pi5_fps": 7.84, "ram_mb": 531, "onnx_mb": 9.8},
    "yolo11n":  {"tier": "nano",   "map50": 94.79, "map50_95": 70.22, "precision": 89.13, "recall": 90.10, "pi5_ms": 149.3, "pi5_fps": 6.70, "ram_mb": 531, "onnx_mb": 10.6},
    "yolov8n":  {"tier": "nano",   "map50": 94.15, "map50_95": 69.59, "precision": 90.02, "recall": 88.15, "pi5_ms": 155.2, "pi5_fps": 6.44, "ram_mb": 531, "onnx_mb": 12.3},
    "yolov5nu": {"tier": "nano",   "map50": 93.56, "map50_95": 67.60, "precision": 89.46, "recall": 88.28, "pi5_ms": 139.9, "pi5_fps": 7.15, "ram_mb": 531, "onnx_mb": 10.3},
    "yolo26s":  {"tier": "small",  "map50": 94.29, "map50_95": 71.35, "precision": 91.17, "recall": 88.50, "pi5_ms": 347.4, "pi5_fps": 2.88, "ram_mb": 626, "onnx_mb": 38.2},
    "yolo11s":  {"tier": "small",  "map50": 94.54, "map50_95": 70.77, "precision": 90.67, "recall": 88.58, "pi5_ms": 361.4, "pi5_fps": 2.77, "ram_mb": 626, "onnx_mb": 37.9},
    "yolov8s":  {"tier": "small",  "map50": 94.34, "map50_95": 70.03, "precision": 90.85, "recall": 88.06, "pi5_ms": 405.1, "pi5_fps": 2.47, "ram_mb": 626, "onnx_mb": 44.7},
    "yolov5su": {"tier": "small",  "map50": 94.09, "map50_95": 69.20, "precision": 88.66, "recall": 89.94, "pi5_ms": 347.7, "pi5_fps": 2.88, "ram_mb": 626, "onnx_mb": 36.7},
    "yolo26m":  {"tier": "medium", "map50": 94.44, "map50_95": 71.96, "precision": 89.49, "recall": 89.22, "pi5_ms": 955.4, "pi5_fps": 1.05, "ram_mb": 787, "onnx_mb": 81.7},
    "yolo11m":  {"tier": "medium", "map50": 94.81, "map50_95": 71.86, "precision": 90.19, "recall": 89.27, "pi5_ms": 963.7, "pi5_fps": 1.04, "ram_mb": 787, "onnx_mb": 80.4},
    "yolov8m":  {"tier": "medium", "map50": 94.63, "map50_95": 71.34, "precision": 90.25, "recall": 88.59, "pi5_ms": 949.5, "pi5_fps": 1.05, "ram_mb": 787, "onnx_mb": 103.6},
    "yolov5mu": {"tier": "medium", "map50": 94.52, "map50_95": 70.88, "precision": 90.54, "recall": 88.32, "pi5_ms": 792.5, "pi5_fps": 1.26, "ram_mb": 787, "onnx_mb": 100.5},
}

TRAINING_SUBSETS = [100, 200, 400, 700, 1000, 1400, 2800]
TIER_ORDER = ["nano", "small", "medium"]
EDGE_TARGETS = {"raspberry pi", "raspberry pi 5", "pi", "rpi", "edge", "cpu", "jetson"}

mcp = FastMCP("ras-mortdb")
_loaded: dict[str, object] = {}


def _normalize(name: str) -> str:
    key = name.lower().strip().removesuffix(".pt").removesuffix(".onnx").removesuffix("_best")
    if key not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from: {', '.join(MODELS)}")
    return key


def _weight_paths(key: str) -> dict[str, str | None]:
    pt = WEIGHTS_PT / f"{key}_best.pt"
    onnx = WEIGHTS_ONNX / f"{key}.onnx"
    return {"pytorch": str(pt) if pt.exists() else None, "onnx": str(onnx) if onnx.exists() else None}


def _summary(key: str) -> dict:
    return {"model": key, **MODELS[key], "weights": _weight_paths(key)}


@mcp.tool
def list_models(tier: str | None = None) -> list[dict]:
    """List RAS-MortDB models with accuracy (full dataset, mean of 3 seeds) and
    Raspberry Pi 5 ONNX inference metrics. Optionally filter by tier: nano, small, medium."""
    keys = [k for k, v in MODELS.items() if tier is None or v["tier"] == tier.lower()]
    return [_summary(k) for k in keys]


@mcp.tool
def get_model_info(model: str) -> dict:
    """Metrics, weight file locations and available training configs for one model (e.g. 'yolo26n')."""
    key = _normalize(model)
    configs = sorted(p.name for p in TRAINING_CONFIGS.glob(f"{key}_*imgs_args.yaml"))
    return {**_summary(key), "training_configs": configs}


@mcp.tool
def recommend_deployment(
    target: str = "raspberry pi 5",
    priority: str = "balanced",
    min_fps: float | None = None,
    training_images: int | None = None,
) -> dict:
    """Recommend a model for a deployment target.

    target: e.g. 'raspberry pi 5', 'edge', 'gpu', 'server'.
    priority: 'speed', 'accuracy' or 'balanced'.
    min_fps: optional minimum FPS on Raspberry Pi 5 (ONNX, CPU).
    training_images: size of the user's training set, used for caveats only.
    """
    priority = priority.lower()
    edge = target.lower().strip() in EDGE_TARGETS or "pi" in target.lower()
    candidates = [k for k in MODELS if min_fps is None or MODELS[k]["pi5_fps"] >= min_fps]
    if not candidates:
        return {"error": f"No model reaches {min_fps} FPS on Raspberry Pi 5 (max is {max(m['pi5_fps'] for m in MODELS.values())})."}
    if edge and min_fps is None:
        candidates = [k for k in candidates if MODELS[k]["tier"] == "nano"]

    def score(k: str) -> float:
        m = MODELS[k]
        if priority == "speed":
            return m["pi5_fps"]
        if priority == "accuracy":
            return m["map50_95"]
        # balanced: mAP50-95 plus a bonus for speed relative to the fastest candidate
        fastest = max(MODELS[c]["pi5_fps"] for c in candidates)
        return m["map50_95"] + 2.0 * m["pi5_fps"] / fastest

    ranked = sorted(candidates, key=score, reverse=True)
    best = ranked[0]
    weights = _weight_paths(best)
    notes = []
    if edge:
        notes.append("Use the ONNX export on CPU/edge devices; Pi 5 metrics were measured with ONNX.")
    if training_images is not None and training_images < 2800:
        nearest = min(TRAINING_SUBSETS, key=lambda n: abs(n - training_images))
        notes.append(
            f"Reported metrics are for the full 2,800-image training set; accuracy with "
            f"{training_images} images will be lower. The paper's learning-curve runs used subsets of "
            f"{', '.join(map(str, TRAINING_SUBSETS[:-1]))} images; the closest training config is "
            f"training_configs/{best}_{nearest}imgs_args.yaml. Fine-tuning from these RAS-MortDB weights "
            f"rather than COCO weights is advisable with small datasets."
        )
    if not weights["pytorch"]:
        notes.append("Only the ONNX weights are present in this repository for this model.")
    return {
        "recommended": _summary(best),
        "ranking": [{"model": k, "map50_95": MODELS[k]["map50_95"], "pi5_fps": MODELS[k]["pi5_fps"]} for k in ranked],
        "notes": notes,
    }


@mcp.tool
def get_training_config(model: str, training_images: int = 2800) -> str:
    """Return the Ultralytics args.yaml used to train `model` on `training_images`
    images (one of 100, 200, 400, 700, 1000, 1400, 2800)."""
    key = _normalize(model)
    path = TRAINING_CONFIGS / f"{key}_{training_images}imgs_args.yaml"
    if not path.exists():
        return f"No config for {key} with {training_images} images. Available subsets: {TRAINING_SUBSETS}"
    return path.read_text()


@mcp.tool
def detect_mortality(
    image_path: str,
    model: str = "yolo26n",
    conf: float = 0.25,
    weights_format: str = "auto",
) -> dict:
    """Run a RAS-MortDB model on an image (or directory of images) and count Dead and Live fish.

    weights_format: 'pytorch', 'onnx' or 'auto' (PyTorch if present, else ONNX).
    Mortality level follows the dataset definition: Zero, Low (<3 dead), High (>=3 dead).
    """
    from ultralytics import YOLO

    key = _normalize(model)
    paths = _weight_paths(key)
    fmt = weights_format.lower()
    if fmt == "auto":
        fmt = "pytorch" if paths["pytorch"] else "onnx"
    weight = paths.get(fmt)
    if not weight:
        raise FileNotFoundError(f"No {fmt} weights for {key} under {WEIGHTS_PT if fmt == 'pytorch' else WEIGHTS_ONNX}")
    source = Path(image_path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Image not found: {source}")

    if weight not in _loaded:
        _loaded[weight] = YOLO(weight, task="detect")
    results = _loaded[weight].predict(source=str(source), conf=conf, iou=0.45, verbose=False)

    images = []
    for r in results:
        detections = [
            {
                "class": r.names[int(c)],
                "confidence": round(float(s), 3),
                "box_xyxy": [round(float(v), 1) for v in box],
            }
            for c, s, box in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy)
        ]
        dead = sum(d["class"].lower() == "dead" for d in detections)
        live = sum(d["class"].lower() == "live" for d in detections)
        level = "Zero" if dead == 0 else "Low" if dead < 3 else "High"
        images.append({"image": r.path, "dead": dead, "live": live, "mortality_level": level, "detections": detections})

    return {"model": key, "weights": weight, "conf": conf, "images": images}


if __name__ == "__main__":
    mcp.run()
