"""Evaluate a released RAS-MortDB model on a labelled dataset split.

Reports precision, recall, mAP50 and mAP50-95 (overall and per class) with Ultralytics' validator, using
dataset/data.yaml. Outputs go to a temporary directory, and the tracked labels.cache files are restored
afterwards so the repository stays unchanged.

    python inference/evaluate.py --model weights/pytorch/yolo26n_best.pt --split test
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "dataset" / "data.yaml"


def evaluate(model_path, split="test", imgsz=640, batch=16, device=None):
    from ultralytics import YOLO

    caches = {p: p.read_bytes() for p in (REPO / "dataset").glob("*/labels.cache")}
    try:
        with tempfile.TemporaryDirectory() as out:
            metrics = YOLO(model_path, task="detect").val(
                data=str(DATA), split=split, imgsz=imgsz, batch=batch, device=device,
                project=out, name="val", plots=False, verbose=False)
    finally:
        for path, content in caches.items():
            path.write_bytes(content)
    box = metrics.box
    names = metrics.names
    return {
        "model": str(model_path), "split": split, "imgsz": imgsz,
        "precision": round(float(box.mp), 4), "recall": round(float(box.mr), 4),
        "mAP50": round(float(box.map50), 4), "mAP50-95": round(float(box.map), 4),
        "per_class": {names[int(c)]: {"precision": round(float(box.p[i]), 4), "recall": round(float(box.r[i]), 4),
                                      "mAP50": round(float(box.ap50[i]), 4), "mAP50-95": round(float(box.ap[i]), 4)}
                      for i, c in enumerate(box.ap_class_index)},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="weights/pytorch/yolo26n_best.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--batch", default=16, type=int)
    parser.add_argument("--device", default=None, help="e.g. cpu or 0; default lets Ultralytics choose")
    args = parser.parse_args()
    result_stream, sys.stdout = sys.stdout, sys.stderr  # Ultralytics logs/progress to stderr; JSON result to stdout
    result = evaluate(args.model, args.split, args.imgsz, args.batch, args.device)
    print(json.dumps(result, indent=2), file=result_stream)
