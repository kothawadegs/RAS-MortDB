# RAS-MortDB Paper Agent (Paper2Agent)

This agent was generated with the [Paper2Agent](https://github.com/jmiao24/Paper2Agent) framework (Miao et al., *Nature* 2026, https://doi.org/10.1038/s41586-026-11044-y; skill revision `8c2d059`). Its source paper is:

> Ranjan, R.; Kothawade, G.S.; Sharrer, K.; Tsukuda, S.; Good, C. **Does YOLO26 Truly Offer Advantages over Its Predecessors for Edge Deployment? A Benchmark Study in Aquaculture.** *AI* 2026, 7(9), 354. https://doi.org/10.3390/ai7090354

The code side was built from https://github.com/PA-RRanjan/RAS-MortDB, pinned at commit `6935cde29225de435cbaea1c73aa89fbb267bfa5`.

Paper2Agent produces two parts that together make the agent:

```text
paper2agent/
├── RAS-MortDB-agent/
│   ├── skill/ras-mortdb-yolo26-paper/   Paper2Skill: the reviewed paper as an agent-readable skill
│   └── mcp/RAS-MortDB-mcp/              Paper2MCP: tested MCP server bound to the repository's code
└── RAS-MortDB-mcp.zip                   the verified MCP delivery archive (same files as mcp/)
```

## 1. Paper skill (`skill/ras-mortdb-yolo26-paper`)

This part answers questions about the paper's methods, results, figures and tables from the paper itself.

- **Contents:**
  - `references/paper.md`: the full article text, with front matter, all sections, back matter and references 1–40.
  - `assets/figure/figure-1…5.jpg`: the paper's five figures.
  - `assets/table/table-1…6.csv`: Tables 1–6. Table 4 spans two PDF pages, so it is split across `table-4.csv` and `table-4-continued.csv`.
  - `references/index.md`: a navigation index.
- **How it was built:** the 24-page PDF was split across three parallel reviewer agents. A fresh verifier agent then checked every table cell, every figure crop and the prose word for word against the PDF. The `paper_bundle.py verify --strict` check returned status **reviewed**.
- **Known limits:**
  - Bold best-in-tier cells are not marked in the CSVs.
  - Supplementary Table S1 is published separately by MDPI and is not included.
  - Figure 5 marks YOLO26n/s as ">1000" images, while §3.4 says they needed 1000. Both are kept as printed.
- **Install for Claude Code:** `cp -R RAS-MortDB-agent/skill/ras-mortdb-yolo26-paper ~/.claude/skills/`.

## 2. MCP server (`mcp/RAS-MortDB-mcp`)

| Tool | What it does |
|---|---|
| `ras_mortdb_detect_fish_mortality` | Runs the repository's own `inference/run_inference.py` on one tank image with any of the 12 released models, as `.pt` or `.onnx`. It returns Dead/Live counts, per-box class, confidence and coordinates, and the annotated image. |

Installation, startup, Claude Code registration and parameters are in [`mcp/RAS-MortDB-mcp/USAGE.md`](RAS-MortDB-agent/mcp/RAS-MortDB-mcp/USAGE.md). The server needs a checkout of this repository for the code and weights. It finds it through `RAS_MORTDB_ROOT`.

### Which tools were built and why

Paper2MCP only exposes operations that exist in the repository's code. A scanner agent reviewed the repository:

| Candidate | Decision | Reason |
|---|---|---|
| Mortality detection on new images | **exposed** | `inference/run_inference.py::run_inference` |
| mAP/precision/recall reproduction (`YOLO.val`) | deferred | No evaluation code in the repo; `dataset/data.yaml` is missing; reported metrics are 3-seed means, but one weight per model is released |
| Training with `training_configs/` | deferred | 100-epoch A100 runs; `data:` paths point to the authors' machine; learning-curve subsets are not in the repo |
| Raspberry Pi benchmark, ONNX export, model comparison, deployment recommendation | omitted | No upstream code. Comparison means calling the detection tool once per model; recommendations come from the paper skill |

Adding `dataset/data.yaml` and an evaluation script upstream would let a future run expose a metrics-reproduction tool.

### Validation (Paper2MCP stages 1–6)

1. **Environment:** Python 3.12.3, ultralytics 8.4.21 (the version recorded in the released checkpoints), torch 2.14.1 on CPU, fastmcp 4.0.3.
2. **Reference execution:** upstream `run_inference()` and its CLI were run directly on 9 cases (models, formats, thresholds, images). A fresh-process replay matched bit for bit.
3. **Implementation, then independent verification:** a separate verifier agent wrote 34 tests against direct upstream calls, and all pass. Class ids and counts match exactly, confidences and boxes match within 1e-4 (they were identical), and the annotated-image hashes match. The tests also cover changed inputs, error cases and repeated-call isolation. The verifier made one repair: the upstream module is now cached per checkout path.
4. **Real stdio acceptance in the project environment:** 11/11 cases passed.
5. **Fresh environment from `src/requirements.txt`:** 11/11 cases passed.
6. **ZIP delivery:** an independent agent extracted the archive to a new path containing a space and installed it by following USAGE.md. It passed 11/11 cases with the default checkout location and 11/11 with `RAS_MORTDB_ROOT`, and 15 content comparisons against upstream references had zero difference.

The workflow checker (`verify_workflow.py --through complete`) passes. Development evidence (agent run records, tests, logs, the review directory) is kept outside this folder, as the framework requires.

Scope: the detections reproduce the paper's released models. They do not re-establish the paper's accuracy figures, which come from the paper skill. Only CPU on Linux x86-64 was tested.

## Relation to `ras_mortdb_agent/` and the root `mcp_server.py`

Those are earlier hand-written agents. Their tools (recommendations, performance tables) are written out by hand from the README and paper rather than bound to repository code. Under the Paper2Agent split, that knowledge belongs in the paper skill and the computation in the MCP server, which is what this folder provides.
