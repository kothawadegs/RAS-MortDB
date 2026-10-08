"""Live demo of the Paper2Agent output for RAS-MortDB.

Starts the Paper2Agent MCP server (paper2agent/RAS-MortDB-agent/mcp/RAS-MortDB-mcp) over stdio, calls its tool the way
an MCP client such as Claude Code would, and writes a Markdown report. Reported benchmark numbers are read from the
Paper2Agent paper skill. In GitHub Actions the report goes to the job summary page; locally it is written to
paper2agent/demo/output/demo_report.md together with the annotated detection images.

    python paper2agent/demo/demo_report.py [--model yolo26n] [--format onnx] [--conf 0.25]
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

REPO = Path(__file__).resolve().parents[2]
SERVER = REPO / "paper2agent/RAS-MortDB-agent/mcp/RAS-MortDB-mcp/src/RAS-MortDB_mcp.py"
SKILL = REPO / "paper2agent/RAS-MortDB-agent/skill/ras-mortdb-yolo26-paper"
IMAGES = REPO / "dataset/test/images"
LABELS = REPO / "dataset/test/labels"
OUT = Path(__file__).resolve().parent / "output"
TEST_IMAGES = [
    "tl_0031_0153_20221022_001226_jpg.rf.5a6ee26f647296aab6267d96b17ced6f.jpg",
    "tl_0031_0160_20221022_015733_jpg.rf.8cd32362708799edca7a3e4c79809e3e.jpg",
    "tl_0031_0162_20221022_022736_jpg.rf.2c9ea99a86db78ccd36d6934f860eb74.jpg",
    "tl_0031_0163_20221022_024237_jpg.rf.3b4856fac2e25f359fdf55af74f6dde8.jpg",
    "tl_0031_0166_20221022_032740_jpg.rf.d4d43d3dbcc654647f4aa5de5672ceb0.jpg",
]
COMPARE_MODELS = ["yolo26n", "yolo11n", "yolov8n", "yolov5nu"]
CLASS_NAMES = {"0": "Dead", "1": "Live"}


def ground_truth(image: str) -> Counter:
    label = LABELS / f"{Path(image).stem}.txt"
    return Counter(CLASS_NAMES[ln.split()[0]] for ln in label.read_text().splitlines() if ln.strip())


def table6_rows(models: list[str]) -> list[dict]:
    with open(SKILL / "assets/table/table-6.csv", newline="") as fh:
        rows = {r["Model"].lower(): r for r in csv.DictReader(fh)}
    return [rows[m] for m in models]


async def main(model: str, fmt: str, conf: float) -> str:
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    transport = StdioTransport(sys.executable, [str(SERVER)], env={"RAS_MORTDB_ROOT": str(REPO)})
    md = ["# RAS-MortDB Paper Agent (Paper2Agent): live demo", ""]

    async with Client(transport) as client:
        tools = [t.name for t in await client.list_tools()]
        md += ["The Paper2Agent MCP server was started over stdio and called like an MCP client would call it.", "",
               f"**Exposed tools:** {', '.join(f'`{t}`' for t in tools)}", ""]

        md += [f"## Detection on held-out test images (`{model}`, {fmt}, conf {conf})", "",
               "| Image | Agent: dead / live | Ground truth: dead / live |", "|---|---|---|"]
        for image in TEST_IMAGES:
            r = (await client.call_tool("ras_mortdb_detect_fish_mortality", {
                "image_path": str(IMAGES / image), "model": model, "weight_format": fmt, "conf": conf,
                "output_dir": str(OUT / "runs")})).data
            gt = ground_truth(image)
            md.append(f"| `{image[:22]}…` | {r['dead_count']} / {r['live_count']} | {gt['Dead']} / {gt['Live']} |")
            if r["artifacts"]:
                shutil.copy(r["artifacts"][0]["path"], OUT / f"annotated_{image[:22]}.jpg")
        md += ["", "_Annotated images are attached to the run as the `demo-output` artifact._", ""]

        md += [f"## Same image, nano-tier models (one tool call per model, {fmt})", "",
               "| Model | Dead | Live | Paper: Raspberry Pi 5 FPS (Table 6) |", "|---|---|---|---|"]
        for m, paper in zip(COMPARE_MODELS, table6_rows(COMPARE_MODELS)):
            r = (await client.call_tool("ras_mortdb_detect_fish_mortality", {
                "image_path": str(IMAGES / TEST_IMAGES[0]), "model": m, "weight_format": fmt, "conf": conf,
                "output_dir": str(OUT / "runs")})).data
            rel = paper["Relative FPS % (vs. YOLOv8)"]
            rel = "baseline" if rel == "Baseline" else f"{rel} vs. YOLOv8n"
            md.append(f"| {m} | {r['dead_count']} | {r['live_count']} | {paper['CPU FPS (Mean ± SD)']} ({rel}) |")
        md += ["", "_FPS figures come from the paper skill (`assets/table/table-6.csv`), not from this run._", ""]

    shutil.rmtree(OUT / "runs", ignore_errors=True)
    report = "\n".join(md)
    (OUT / "demo_report.md").write_text(report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolo26n")
    p.add_argument("--format", default="onnx", choices=["onnx", "pytorch"])
    p.add_argument("--conf", type=float, default=0.25)
    a = p.parse_args()
    report = asyncio.run(main(a.model, a.format, a.conf))
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a") as fh:
            fh.write(report + "\n")
    print(report)
