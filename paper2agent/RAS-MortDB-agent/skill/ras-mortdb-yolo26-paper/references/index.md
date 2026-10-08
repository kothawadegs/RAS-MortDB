# Paper navigation

Use exact headings to locate current line numbers. Read relevant passages, not both complete documents. This index locates evidence; it does not summarize findings.

## Main paper — [paper.md](paper.md)

| Exact heading | Look here for |
| --- | --- |
| Abstract | Summary of objectives, models compared and headline results |
| 1. Introduction | Background on aquaculture mortality monitoring, YOLO generations and research gaps |
| 2.1. YOLO Model Architecture | Architectural differences between YOLOv5u, YOLOv8, YOLO11 and YOLO26 |
| 2.2. Dataset | RAS-MortDB image acquisition, annotation, splits and augmentation |
| 2.3. Model Training | Training hardware, hyperparameters and learning-curve subset design |
| 2.4. Model Performance on GPU | GPU evaluation metrics and protocol |
| 2.5. Edge Deployment Evaluation | Raspberry Pi 5 ONNX benchmark protocol |
| 3.1. Mortality Detection | Accuracy results overall, by tier and per class (Tables 3, Figures 3-4) |
| 3.2. GPU Inference Speed | GPU latency comparison |
| 3.3. Computational Efficiency | Parameters, GFLOPs and model size comparison (Table 4) |
| 3.4. Learning Curve Analysis | Data efficiency: training images needed to reach 90% mAP50 (Figure 5) |
| 3.5. Edge Deployment Performance | Raspberry Pi 5 latency, FPS, CPU, RAM and ONNX size (Tables 5-6) |
| 4. Conclusions | Main conclusions and deployment guidance |
| Data Availability Statement | Where the dataset, weights and code are published |
| References | Numbered bibliography 1-40 |

For long sections, search a narrower subsection or prompt. Headings inside fenced quotations are source content, not document section boundaries.

## Supplementary information — [supplement.md](supplement.md)

| Exact heading | Look here for |
| --- | --- |
| Document beginning | Source text or supplied-material notes |

For long sections, search a narrower subsection or prompt. Headings inside fenced quotations are source content, not document section boundaries.

## Assets

Figure and table captions in the documents link to the files below. Open only the needed image; for a table, read its header and relevant rows first.

- `assets/figure/`: main figures (JPEG).
- `assets/supp_figs/`: supplementary and extended-data figures (JPEG).
- `assets/table/`: main tables (CSV, or JPEG when transcription is unreliable).
- `assets/supp_table/`: supplementary tables (CSV, or JPEG fallback).

Asset paths are relative to the skill root. CSVs retain internal blank rows; captions and merged headers may also occupy rows. Consult the document notes before treating every row as data.
