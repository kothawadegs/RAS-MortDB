# RAS-MortDB Paper Agent

A Paper2Agent-style MCP server built from:

> Ranjan, R.; Kothawade, G.S.; Sharrer, K.; Tsukuda, S.; Good, C. **Does YOLO26
> Truly Offer Advantages over Its Predecessors for Edge Deployment? A
> Benchmark Study in Aquaculture.** *AI* 2026, 7(9), 354.
> https://doi.org/10.3390/ai7090354

Codebase agentified: https://github.com/PA-RRanjan/RAS-MortDB (CC BY 4.0
data / AGPL-3.0 weights, archived at
https://doi.org/10.5281/zenodo.20631780)

This follows the approach described in Miao et al., *"Reimagining research
papers as interactive and reliable AI agents"* (Paper2Agent), Nature (2026),
https://doi.org/10.1038/s41586-026-11044-y: wrap a paper's own public code
and trained artifacts as MCP tools, backed by the paper's reported
benchmark numbers as an MCP resource, and connect it to a chat agent
(Claude Code, Claude Desktop, or any MCP-compatible client).

## What's built here (manually, not via the automated Paper2Agent.sh pipeline)

The full Paper2Agent framework (github.com/jmiao24/Paper2Agent) is a
Claude-Code-orchestrated multi-agent pipeline that auto-discovers tutorials,
executes them, extracts tools, and validates them — normally 30 min–3 hrs
and ~$13–15 in API cost per paper. That orchestrator wasn't available in
this environment, so the same *result* was built by hand, following the
same shape: real code from the paper's repo → executable MCP tools →
validated against real trained weights and real test images before
delivery (see "Validation" below).

### Tools

| Tool | What it does |
|---|---|
| `list_models()` | All 12 model variants (YOLOv5u/v8/11/26 × nano/small/medium) with headline mAP50 / edge FPS |
| `get_model_performance(model)` | Full reported accuracy, Raspberry Pi 5 speed, ONNX size, and data-efficiency numbers for one model, straight from the paper's tables |
| `detect_mortality(image_path, model, weight_format, conf)` | Runs the actual trained weights (PyTorch or ONNX) on an image; counts Live/Dead fish, exactly matching `RAS-MortDB/inference/run_inference.py` |
| `compare_models_on_image(image_path, models)` | Runs several models on the same image side by side |
| `recommend_deployment(target_hardware, annotated_images_available, priority)` | Model/tier recommendation using the paper's own deployment-oriented conclusions (accuracy is nearly architecture-invariant; choose by hardware, data budget, and speed need) |

### Resources & prompts

- `paper://abstract` — citation, data/code availability, headline finding
- `benchmark_new_deployment` — an MCP prompt that walks an agent through
  picking + sanity-checking a model for a new deployment using the tools
  above, in order

## Setup

```bash
pip install -r requirements.txt
python mcp_server.py                     # stdio — for Claude Desktop / Claude Code
python mcp_server.py --http --port 8000  # HTTP — e.g. for Hugging Face Spaces
```

This package bundles only **2 of 12 models** (`yolo26n`, `yolov8n`) and
**5 sample test images**, to keep it lightweight. Run `fetch_full_repo.sh`
to pull the full repo (all 24 weight files, full 2,000-image dataset, all
84 training configs) and point `WEIGHTS_PT` / `WEIGHTS_ONNX` at the top of
`mcp_server.py` at it to unlock all 12 models.

### Connect to Claude Code / Claude Desktop

```json
{
  "mcpServers": {
    "ras-mortdb": {
      "command": "python",
      "args": ["/absolute/path/to/mcp_server.py"]
    }
  }
}
```

## Validation

Before delivery, every tool was run against the real repository, not just
inspected:
- `detect_mortality` was run with both PyTorch and ONNX weights on real
  RAS-MortDB test-set images, reproducing the same Dead/Live counting logic
  as the paper's own `run_inference.py` (verified class mapping
  `{0: 'Dead', 1: 'Live'}` matches the trained weights' embedded names).
- All 5 tools, 1 resource, and 1 prompt were exercised through the actual
  MCP protocol layer via `fastmcp`'s in-memory `Client`, not just called as
  plain Python functions.
- `get_model_performance` / `list_models` / `recommend_deployment` numbers
  are copied verbatim from the paper's README results tables — they are
  reference lookups, not re-derived at query time, so they can't drift from
  what's actually published.

Not validated here (would need the full repo + more disk/compute than this
sandbox had available): reproducing the paper's full 3-seed training runs,
or the Raspberry Pi 5 edge-latency benchmarks themselves (those numbers are
taken from the paper as reported, not re-measured).
