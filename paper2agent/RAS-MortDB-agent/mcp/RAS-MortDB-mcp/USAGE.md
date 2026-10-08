# RAS-MortDB MCP server

An MCP server generated with the [Paper2Agent](https://github.com/jmiao24/Paper2Agent) Paper2MCP workflow from the code and released weights of:

> Ranjan, R.; Kothawade, G.S.; Sharrer, K.; Tsukuda, S.; Good, C. Does YOLO26 Truly Offer Advantages over Its Predecessors for Edge Deployment? A Benchmark Study in Aquaculture. *AI* **2026**, *7*, 354. https://doi.org/10.3390/ai7090354

Source repository: https://github.com/PA-RRanjan/RAS-MortDB, pinned at commit `6935cde29225de435cbaea1c73aa89fbb267bfa5`.

## Scope

| Tool | Use it to |
|---|---|
| `ras_mortdb_detect_fish_mortality` | Detect and count dead and live fish in one recirculating aquaculture system (RAS) tank image with any of the paper's 12 released YOLO models, in PyTorch or ONNX format. |

The tool calls the repository's own `run_inference()` (`inference/run_inference.py`) unchanged: Ultralytics `predict` with `iou=0.45`, `save=True`, `save_conf=True`.

Not included, with the reason recorded during tool selection:

- **Reproducing the reported mAP/precision/recall.** The repository has no evaluation code, `dataset/data.yaml` is absent at the pinned commit, and the reported metrics are means over three seeds, while one weight file per model is released.
- **Training.** The `training_configs/` runs are 100-epoch A100 jobs whose `data:` paths point to the authors' machine, and the learning-curve subsets are not in the repository.
- **Raspberry Pi latency benchmarking, ONNX export, batch inference and model comparison.** There is no upstream code for these. To compare models, call the tool once per model.

## Requirements

- Python 3.12 (tested 3.12.3, Linux x86-64, CPU only; a GPU is optional and was not tested).
- The pinned runtime packages in `src/requirements.txt` (FastMCP 4.0.3, Ultralytics 8.4.21, torch 2.14.1, onnxruntime 1.30.0, and others). The Ultralytics version matches the version recorded in the released checkpoints.
- A checkout of the RAS-MortDB repository at the pinned commit. It provides `inference/run_inference.py` and the `weights/` folder (about 850 MB of model files). These are not bundled in this package.

## Installation

From the extracted `RAS-MortDB-mcp/` folder:

```bash
# 1. Python environment
python3.12 -m venv .venv
.venv/bin/python -m pip install -r src/requirements.txt
# (equivalent with uv: uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r src/requirements.txt)

# 2. Research checkout (code + released weights) at the pinned commit
git clone https://github.com/PA-RRanjan/RAS-MortDB.git repo/RAS-MortDB
git -C repo/RAS-MortDB checkout 6935cde29225de435cbaea1c73aa89fbb267bfa5
```

The server looks for the checkout at `repo/RAS-MortDB` inside this folder. To use a checkout elsewhere, set `RAS_MORTDB_ROOT=/path/to/RAS-MortDB`.

## Start the server

```bash
.venv/bin/python src/RAS-MortDB_mcp.py          # stdio transport
```

### Claude Code

```bash
claude mcp add ras-mortdb -- "$PWD/.venv/bin/python" "$PWD/src/RAS-MortDB_mcp.py"
# with a checkout elsewhere:
claude mcp add ras-mortdb -e RAS_MORTDB_ROOT=/path/to/RAS-MortDB -- "$PWD/.venv/bin/python" "$PWD/src/RAS-MortDB_mcp.py"
```

The FastMCP 4.0.3 CLI offers the same through `fastmcp install claude-code src/RAS-MortDB_mcp.py --name ras-mortdb --with-requirements src/requirements.txt --env RAS_MORTDB_ROOT=/path/to/RAS-MortDB`. That command creates its own environment with `uv`; the direct command above is the path that was tested.

Optional environment variables:

- `RAS_MORTDB_ROOT`: location of the research checkout.
- `YOLO_CONFIG_DIR`: Ultralytics settings directory. It defaults to a temporary folder.

## Tool reference

### `ras_mortdb_detect_fish_mortality`

| Parameter | Required | Default | Meaning |
|---|---|---|---|
| `image_path` | yes | n/a | One image file (any Ultralytics image format). Directories and videos are rejected, because upstream counts only the first result. |
| `model` | no | `yolo26n` | One of `yolo26n/s/m`, `yolo11n/s/m`, `yolov8n/s/m`, `yolov5nu/su/mu` |
| `weight_format` | no | `pytorch` | `pytorch` loads `weights/pytorch/<model>_best.pt`; `onnx` loads `weights/onnx/<model>.onnx` |
| `conf` | no | `0.25` | Confidence threshold in (0, 1] (the upstream default) |
| `output_dir` | no | a temporary folder | Base folder; every call writes to a fresh `detect_<uuid>` subfolder |

It returns:

- `dead_count` and `live_count`: counted from the class names on the result, as upstream does.
- `total_detections`.
- `detections`: one entry per box, with `class_id` (0 = Dead, 1 = Live), `class_name`, `confidence` and `box_xyxy` in pixels.
- `class_names`, `image_shape_hw`, the resolved `weights_path` and settings, and `upstream_message` (upstream's printed counts).
- `artifacts`: the absolute path of the annotated image that Ultralytics saved.

Calls are serialized within one server process, because upstream saves relative to the working directory.

## Validation performed

- **Reference runs:** upstream `run_inference()` and its command-line interface were run directly on repository test images, covering YOLO26n, YOLOv8n and YOLO11m, `.pt` and `.onnx`, and confidence 0.25 and 0.5. A fresh-process replay matched bit for bit.
- **Independent verification:** 34 pytest tests compare the tool with direct upstream calls.
  - Class ids and counts must match exactly. Confidences and boxes must match within 1e-4 (they were identical on the test CPU). Annotated-image hashes must match.
  - Changed inputs: a new image, YOLOv5su `.pt`, YOLO26s `.onnx`, and confidence 0.8 and 0.9.
  - Error cases, repeated-call output isolation, and a check that the input images are unchanged.
- **Real stdio acceptance:** 11 of 11 cases passed, both in the development environment and in a fresh environment installed from `src/requirements.txt`. The package was also extracted to a new location and checked there.

Limits:

- Only CPU on Linux x86-64 was tested.
- Detections reproduce the paper's released models. They do not re-establish the paper's accuracy figures.
- If the Ultralytics `runs_dir` setting is an absolute path, or Ultralytics believes tests are running, it may save the annotated image somewhere else. The tool still returns the real path, with a `warning`.

## Licenses

- RAS-MortDB code and dataset: CC BY 4.0 (The Conservation Fund). Model weights: AGPL-3.0, following Ultralytics.
- Ultralytics: AGPL-3.0.
- Cite the paper above when using this server.
