# RAS-MortDB: RAS Fish Mortality Detection Dataset and Model Weights

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md.svg)](https://huggingface.co/spaces/opticalResearcher/RAS-MortDB-agent)

## Overview

RAS-MortDB is a publicly available RAS fish mortality detection dataset and model repository:
> Ranjan, R.; Kothawade, G.S.; Sharrer, K.; Tsukuda, S.; Good, C. Does YOLO26 Truly Offer Advantages over Its Predecessors for Edge Deployment? A Benchmark Study in Aquaculture. *AI* **2026**, *7*(9), 354. https://doi.org/10.3390/ai7090354

The dataset comprises 2,000 annotated images of dead and live fish collected from a semi-commercial RAS over a 90-day deployment period under ambient and supplemental lighting conditions. Trained model weights for twelve Ultralytics YOLO architectures across three size tiers are provided in PyTorch (.pt) and ONNX (.onnx) formats.

## Repository Structure

```
RAS-MortDB/
├── dataset/               # 2,000 annotated images (YOLO format)
│   ├── train/             # 1,400 original + 1,400 augmented (2,800 total)
│   ├── valid/             # 400 images
│   ├── test/              # 200 images
│   └── data.yaml          # Dataset configuration
├── weights/
│   ├── pytorch/           # 12 x PyTorch model weights (.pt)
│   └── onnx/              # 12 x ONNX model weights (.onnx)
├── training_configs/      # args.yaml for all 84 training runs
├── inference/
│   ├── run_inference.py   # Example inference script
│   └── evaluate.py        # Precision/recall/mAP on a labelled split
├── docs/assets/           # Sample detection images
├── paper2agent/           # AI agent for the paper (Paper2Agent): paper skill, MCP server, web demo
├── CITATION.cff
└── LICENSE
```

## Dataset

- **Total images:** 2,000 (train: 2,800 augmented / valid: 400 / test: 200)
- **Classes:** Dead, Live
- **Split:** 70:20:10 (train:valid:test)
- **Annotation:** YOLO format (.txt)
- **Lighting:** Ambient and supplemental
- **Mortality levels:** Zero, Low (<3 fish), High (≥3 fish)
- **Collection:** 90-day commercial RAS deployment

## Model Performance

All metrics reported as average ± SD across three independent training runs (seeds 1, 2, 3).

### Full Dataset Results (2,800 training images)

| Model | Tier | mAP50 ± SD (%) | mAP50-95 ± SD (%) | Precision ± SD (%) | Recall ± SD (%) |
|-------|------|----------------|-------------------|-------------------|-----------------|
| YOLO26n | Nano | 94.44 ± 0.38 | 71.16 ± 0.70 | 90.94 ± 0.50 | 88.02 ± 0.98 |
| YOLO11n | Nano | 94.79 ± 0.32 | 70.22 ± 0.67 | 89.13 ± 0.36 | 90.10 ± 0.40 |
| YOLOv8n | Nano | 94.15 ± 0.39 | 69.59 ± 0.19 | 90.02 ± 1.05 | 88.15 ± 0.94 |
| YOLOv5nu | Nano | 93.56 ± 0.30 | 67.60 ± 0.38 | 89.46 ± 1.34 | 88.28 ± 1.54 |
| YOLO26s | Small | 94.29 ± 0.26 | 71.35 ± 0.13 | 91.17 ± 1.32 | 88.50 ± 1.46 |
| YOLO11s | Small | 94.54 ± 0.40 | 70.77 ± 0.67 | 90.67 ± 0.53 | 88.58 ± 1.42 |
| YOLOv8s | Small | 94.34 ± 0.36 | 70.03 ± 0.40 | 90.85 ± 0.58 | 88.06 ± 1.19 |
| YOLOv5su | Small | 94.09 ± 0.22 | 69.20 ± 0.25 | 88.66 ± 1.06 | 89.94 ± 1.13 |
| YOLO26m | Medium | 94.44 ± 0.41 | 71.96 ± 0.85 | 89.49 ± 1.56 | 89.22 ± 1.26 |
| YOLO11m | Medium | 94.81 ± 0.10 | 71.86 ± 0.24 | 90.19 ± 1.35 | 89.27 ± 1.30 |
| YOLOv8m | Medium | 94.63 ± 0.13 | 71.34 ± 0.46 | 90.25 ± 0.68 | 88.59 ± 0.69 |
| YOLOv5mu | Medium | 94.52 ± 0.37 | 70.88 ± 0.43 | 90.54 ± 0.60 | 88.32 ± 1.10 |

### Raspberry Pi 5 Edge Inference Results

Mean ± SD across three independent benchmark sessions with randomised model evaluation order.

| Model | Tier | Avg ± SD (ms) | FPS ± SD | CPU (%) | RAM (MB) | ONNX (MB) | Rel FPS % |
|-------|------|---------------|----------|---------|----------|-----------|-----------|
| YOLO26n | Nano | 127.5 ± 2.2 | 7.84 ± 0.13 | 49.9 | 531 | 9.8 | +21.7% |
| YOLO11n | Nano | 149.3 ± 2.5 | 6.70 ± 0.11 | 48.7 | 531 | 10.6 | +4.0% |
| YOLOv8n | Nano | 155.2 ± 2.0 | 6.44 ± 0.08 | 48.8 | 531 | 12.3 | Baseline |
| YOLOv5nu | Nano | 139.9 ± 1.2 | 7.15 ± 0.06 | 48.7 | 531 | 10.3 | +11.0% |
| YOLO26s | Small | 347.4 ± 4.2 | 2.88 ± 0.03 | 50.1 | 626 | 38.2 | +16.6% |
| YOLO11s | Small | 361.4 ± 4.5 | 2.77 ± 0.04 | 49.6 | 626 | 37.9 | +12.1% |
| YOLOv8s | Small | 405.1 ± 11.3 | 2.47 ± 0.07 | 49.6 | 626 | 44.7 | Baseline |
| YOLOv5su | Small | 347.7 ± 3.5 | 2.88 ± 0.03 | 49.5 | 626 | 36.7 | +16.6% |
| YOLO26m | Medium | 955.4 ± 9.6 | 1.05 ± 0.01 | 50.2 | 787 | 81.7 | 0.0% |
| YOLO11m | Medium | 963.7 ± 10.8 | 1.04 ± 0.01 | 49.9 | 787 | 80.4 | -1.0% |
| YOLOv8m | Medium | 949.5 ± 11.6 | 1.05 ± 0.01 | 49.9 | 787 | 103.6 | Baseline |
| YOLOv5mu | Medium | 792.5 ± 0.9 | 1.26 ± 0.00 | 50.0 | 787 | 100.5 | +20.0% |

## Training Configuration

All models trained with identical hyperparameters:

| Parameter | Value |
|-----------|-------|
| Epochs | 100 |
| Batch size | 32 |
| Image size | 640 × 640 |
| Optimizer | auto (MuSGD for YOLO26; SGD for others) |
| Learning rate (lr0/lrf) | 0.01 / 0.01 |
| Momentum | 0.937 |
| Weight decay | 0.0005 |
| Patience | 50 |
| IoU threshold | 0.7 |
| Workers | 4 |
| Hardware | NVIDIA A100-SXM4-80GB (USDA SCINet Atlas) |

Training configuration files (args.yaml) for all 84 runs are in `training_configs/`.

## Quick Start

### PyTorch Inference

```python
from ultralytics import YOLO

model = YOLO("weights/pytorch/yolo26n_best.pt")
results = model.predict("your_image.jpg", conf=0.25)
results[0].show()
```

### ONNX Inference (Raspberry Pi / CPU)

```python
from ultralytics import YOLO

model = YOLO("weights/onnx/yolo26n.onnx")
results = model.predict("your_image.jpg", conf=0.25)
results[0].show()
```

### Evaluation on a Labelled Split

`inference/evaluate.py` runs the Ultralytics validator with `dataset/data.yaml` and prints precision, recall, mAP50 and mAP50-95 (overall and per class) as JSON:

```bash
python inference/evaluate.py --model weights/pytorch/yolo26n_best.pt --split test   # or --split val
```

A single released checkpoint gives one run. The tables above report the mean ± SD over three training seeds, so expect values within about that spread. For example, `yolo26n_best.pt` gives 94.0% mAP50 on `val` and 95.1% on `test`.

## Paper Agent (Paper2Agent)

[`paper2agent/`](paper2agent/README.md) holds an AI agent for the paper, generated with the [Paper2Agent](https://github.com/jmiao24/Paper2Agent) framework:

- **Paper skill** (`paper2agent/RAS-MortDB-agent/skill/`): the reviewed paper text, figures and tables, packaged so Claude can answer questions about methods and results.
- **MCP server** (`paper2agent/RAS-MortDB-agent/mcp/`): the tool `ras_mortdb_detect_fish_mortality`, which runs `inference/run_inference.py` with any of the 12 released models. It was independently verified against direct runs of the script.
- **Web demo**: [huggingface.co/spaces/opticalResearcher/RAS-MortDB-agent](https://huggingface.co/spaces/opticalResearcher/RAS-MortDB-agent). Upload a tank image in the browser, no account needed. The source is in `paper2agent/hf-space/`.
- **GitHub demo**: **Actions → Agent demo → Run workflow**. It starts the MCP server, runs it on test images and posts a report on the run's summary page.

To use the agent in Claude Code from the repository root, install the server once and start Claude Code. The bundled `.mcp.json` registers the server.

```bash
pip install -r paper2agent/RAS-MortDB-agent/mcp/RAS-MortDB-mcp/src/requirements.txt
cp -R paper2agent/RAS-MortDB-agent/skill/ras-mortdb-yolo26-paper ~/.claude/skills/   # paper skill
claude
```

## Sample Detection Results

### YOLO26n Detections

![](docs/assets/yolo26n_image20.jpg) ![](docs/assets/yolo26n_image82.jpg)

### YOLOv8n Detections

![](docs/assets/yolov8n_image20.jpg) ![](docs/assets/yolov8n_image82.jpg)

## Citation

If you use RAS-MortDB, please cite:

**Paper:**
```
Ranjan, R.; Kothawade, G.S.; Sharrer, K.; Tsukuda, S.; Good, C. Does YOLO26
Truly Offer Advantages over Its Predecessors for Edge Deployment? A Benchmark
Study in Aquaculture. AI 2026, 7(9), 354. https://doi.org/10.3390/ai7090354
```

**Dataset:**
```
Ranjan, R. (2026). RAS-MortDB: Fish Mortality Detection Dataset and Model
Weights for Recirculating Aquaculture Systems (v1.1.0) [Data set].
Zenodo. https://doi.org/10.5281/zenodo.20631780
```

## License

- Dataset: Creative Commons Attribution 4.0 (CC BY 4.0)
- Model weights: AGPL-3.0 (consistent with Ultralytics YOLO license)

## Contact

Rakesh Ranjan  
The Conservation Fund Freshwater Institute  
Email: rranjan@conservationfund.org