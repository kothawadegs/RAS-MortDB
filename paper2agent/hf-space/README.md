---
title: RAS-MortDB Paper Agent
emoji: 🐟
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.29.1
python_version: "3.12"
app_file: app.py
pinned: false
license: agpl-3.0
short_description: Fish mortality detection from the RAS-MortDB paper
---

# RAS-MortDB Paper Agent

Upload an image of a recirculating aquaculture system (RAS) tank and count dead and live fish with any of the 12 YOLO models released with:

> Ranjan, R.; Kothawade, G.S.; Sharrer, K.; Tsukuda, S.; Good, C. Does YOLO26 Truly Offer Advantages over Its Predecessors for Edge Deployment? A Benchmark Study in Aquaculture. *AI* 2026, 7(9), 354. https://doi.org/10.3390/ai7090354

This Space runs the MCP tool `ras_mortdb_detect_fish_mortality` (`tools/run_inference.py`) generated and independently verified with [Paper2Agent](https://github.com/jmiao24/Paper2Agent). That tool calls the paper's own [`inference/run_inference.py`](https://github.com/PA-RRanjan/RAS-MortDB/blob/6935cde29225de435cbaea1c73aa89fbb267bfa5/inference/run_inference.py). The script and the released weights are downloaded on demand from GitHub at commit `6935cde` and checked against the SHA-256 hashes in `upstream_manifest.json`.

The **Paper results** tab shows the paper's Table 3 (accuracy), Table 6 (Raspberry Pi 5 benchmark) and Figure 5 (learning curves).

Licenses: RAS-MortDB dataset and code are CC BY 4.0 (The Conservation Fund). The model weights and Ultralytics are AGPL-3.0. The example images come from the RAS-MortDB test split.
