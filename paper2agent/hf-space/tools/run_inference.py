"""Tools extracted from RAS-MortDB/inference/run_inference.py.

Wraps the upstream ``run_inference(image_path, model_path, conf=0.25)`` function, which applies one of the
released RAS-MortDB YOLO detectors (Ultralytics ``predict`` with ``iou=0.45, save=True, save_conf=True``) to a
recirculating-aquaculture-system tank image and counts Dead (class 0) and Live (class 1) fish.

The research checkout is located through the ``RAS_MORTDB_ROOT`` environment variable, defaulting to the pinned
checkout ``<project>/repo/RAS-MortDB`` next to this module's project.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import logging
import os
import sys
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Annotated, Literal

from fastmcp import FastMCP

REFERENCE = (
    "https://github.com/PA-RRanjan/RAS-MortDB/blob/6935cde29225de435cbaea1c73aa89fbb267bfa5/inference/run_inference.py"
)
DEFAULT_CHECKOUT = Path(__file__).resolve().parents[2] / "repo" / "RAS-MortDB"

# Ultralytics stores its settings under YOLO_CONFIG_DIR; use a writable location only when the user has not set one.
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(tempfile.gettempdir()) / "ras_mortdb_mcp" / "ultralytics_cfg"))

run_inference_mcp = FastMCP(name="run_inference")

# Upstream run_inference() calls predict(save=True), which writes runs/detect/predict* relative to the process CWD.
# Each call therefore runs with the CWD set to its fresh output directory; the lock serializes these CWD changes.
_CWD_LOCK = threading.Lock()
_UPSTREAM: dict[Path, object] = {}  # upstream run_inference per resolved checkout script


def _checkout_root() -> Path:
    return Path(os.environ.get("RAS_MORTDB_ROOT") or DEFAULT_CHECKOUT).expanduser().resolve()


def _load_upstream():
    """Import upstream run_inference() from the checkout without writing __pycache__ into it."""
    script = _checkout_root() / "inference" / "run_inference.py"
    if not script.is_file():
        raise FileNotFoundError(
            f"RAS-MortDB inference script not found at {script}. Set RAS_MORTDB_ROOT to a RAS-MortDB checkout "
            "(https://github.com/PA-RRanjan/RAS-MortDB, commit 6935cde29225de435cbaea1c73aa89fbb267bfa5)."
        )
    if script in _UPSTREAM:
        return _UPSTREAM[script]
    prev = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location("ras_mortdb_upstream_run_inference", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = prev
    # The Ultralytics logger is bound to stdout at import; move it to stderr so it never touches MCP protocol stdout.
    from ultralytics.utils import LOGGER

    for handler in LOGGER.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream in (sys.stdout, sys.__stdout__):
            handler.setStream(sys.stderr)
    _UPSTREAM[script] = module.run_inference
    return _UPSTREAM[script]


@run_inference_mcp.tool()
def ras_mortdb_detect_fish_mortality(
    image_path: Annotated[str, "Path to one RAS tank image file (jpg/png/...); directories and videos are rejected"],
    model: Annotated[
        Literal[
            "yolo26n", "yolo26s", "yolo26m",
            "yolo11n", "yolo11s", "yolo11m",
            "yolov8n", "yolov8s", "yolov8m",
            "yolov5nu", "yolov5su", "yolov5mu",
        ],
        "Released RAS-MortDB detector trained on the Dead/Live fish dataset",
    ] = "yolo26n",
    weight_format: Annotated[
        Literal["pytorch", "onnx"],
        "pytorch loads weights/pytorch/<model>_best.pt; onnx loads weights/onnx/<model>.onnx",
    ] = "pytorch",
    conf: Annotated[float, "Detection confidence threshold in (0, 1]; upstream default 0.25"] = 0.25,
    output_dir: Annotated[str | None, "Base output directory; a fresh subdirectory is created per call"] = None,
) -> dict:
    """Detect dead and live fish in one RAS tank image with a released RAS-MortDB YOLO model.
    Input is an image file; output is Dead/Live counts, per-box class/confidence/xyxy and the annotated image.
    """
    image = Path(image_path).expanduser().resolve()
    if not image.exists():
        raise FileNotFoundError(f"image_path does not exist: {image}")
    if not image.is_file():
        raise ValueError(f"image_path must be a single image file, not a directory: {image}")
    from ultralytics.data.utils import IMG_FORMATS

    if image.suffix.lower().lstrip(".") not in IMG_FORMATS:
        raise ValueError(f"Unsupported image format '{image.suffix}'; expected one of {sorted(IMG_FORMATS)}")
    if not 0.0 < conf <= 1.0:
        raise ValueError(f"conf must be in (0, 1], got {conf}")

    checkout = _checkout_root()
    if weight_format == "pytorch":
        weights = checkout / "weights" / "pytorch" / f"{model}_best.pt"
    else:
        weights = checkout / "weights" / "onnx" / f"{model}.onnx"
    run_inference = _load_upstream()
    if not weights.is_file():
        raise FileNotFoundError(f"Weight file for model '{model}' ({weight_format}) not found: {weights}")

    base = Path(output_dir).expanduser() if output_dir else Path(tempfile.gettempdir()) / "ras_mortdb_mcp_outputs"
    call_dir = (base / f"detect_{uuid.uuid4().hex}").resolve()
    call_dir.mkdir(parents=True)

    printed = io.StringIO()
    with _CWD_LOCK:
        prev_cwd = os.getcwd()
        os.chdir(call_dir)
        try:
            with contextlib.redirect_stdout(printed):
                results = run_inference(str(image), str(weights), conf)
        finally:
            os.chdir(prev_cwd)
    upstream_message = printed.getvalue().strip()
    if upstream_message:
        print(upstream_message, file=sys.stderr)

    # Upstream counts boxes on results[0] by class name; serialize the same Results object.
    result = results[0]
    names = {int(k): v for k, v in result.names.items()}
    class_ids = [int(c) for c in result.boxes.cls.tolist()]
    detections = [
        {"class_id": c, "class_name": names[c], "confidence": float(p), "box_xyxy": [float(v) for v in xyxy]}
        for c, p, xyxy in zip(class_ids, result.boxes.conf.tolist(), result.boxes.xyxy.tolist())
    ]
    dead = sum(1 for c in class_ids if names[c] == "Dead")
    live = sum(1 for c in class_ids if names[c] == "Live")

    annotated = (Path(result.save_dir) / Path(result.path).name).resolve()
    artifacts = []
    if annotated.is_file():
        artifacts.append({"description": "Ultralytics annotated prediction image (boxes with confidences)",
                          "path": str(annotated)})
    out = {
        "message": f"{model} ({weight_format}) at conf={conf}: Dead {dead}, Live {live} "
                   f"({len(detections)} detections) in {image.name}",
        "reference": REFERENCE,
        "artifacts": artifacts,
        "dead_count": dead,
        "live_count": live,
        "total_detections": len(detections),
        "detections": detections,
        "class_names": {str(k): v for k, v in names.items()},
        "image_shape_hw": [int(v) for v in result.orig_shape],
        "model": model,
        "weight_format": weight_format,
        "weights_path": str(weights),
        "conf": conf,
        "output_dir": str(call_dir),
        "upstream_message": upstream_message,
    }
    if not str(annotated).startswith(str(call_dir) + os.sep):
        out["warning"] = (f"Ultralytics saved outputs outside the per-call directory ({result.save_dir}); "
                          "check the Ultralytics 'runs_dir' setting.")
    return out
