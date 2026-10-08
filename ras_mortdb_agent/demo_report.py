"""End-to-end demo of the RAS-MortDB Paper Agent.

Starts mcp_server.py as a real MCP server over stdio, calls its tools the way
an MCP client (Claude Code, Claude Desktop, ...) would, and writes a Markdown
report. In GitHub Actions the report goes to the job summary page; locally it
is written to demo_output/demo_report.md. Annotated detection images are saved
to demo_output/ as well.

    python demo_report.py [--model yolo26n] [--format onnx] [--conf 0.25]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import Counter
from pathlib import Path

from fastmcp import Client

HERE = Path(__file__).resolve().parent
TEST_IMAGES = HERE / "test_images"
GT_LABELS = HERE.parent / "dataset" / "test" / "labels"
OUT = HERE / "demo_output"
CLASS_NAMES = {"0": "Dead", "1": "Live"}

SCENARIOS = [
    ("Raspberry Pi, 300 labelled images, speed first",
     {"target_hardware": "raspberry_pi_or_cpu_edge", "annotated_training_images_available": 300,
      "priority": "fastest_edge_inference"}),
    ("Raspberry Pi, 2,000 labelled images, accuracy first",
     {"target_hardware": "raspberry_pi_or_cpu_edge", "annotated_training_images_available": 2000,
      "priority": "max_accuracy"}),
    ("GPU server, 2,800 labelled images",
     {"target_hardware": "gpu_server", "annotated_training_images_available": 2800}),
]


def ground_truth(image: Path) -> Counter | None:
    label = GT_LABELS / f"{image.stem}.txt"
    if not label.exists():
        return None
    return Counter(CLASS_NAMES[line.split()[0]] for line in label.read_text().splitlines() if line.strip())


def save_annotated(image: Path, weight: Path, conf: float) -> Path:
    """Draw boxes with the same weights the agent used, for the report artifact."""
    from ultralytics import YOLO

    out = OUT / f"annotated_{image.stem[:22]}.jpg"
    YOLO(str(weight), task="detect").predict(str(image), conf=conf, iou=0.45, verbose=False)[0].save(str(out))
    return out


async def main(model: str, fmt: str, conf: float) -> str:
    OUT.mkdir(exist_ok=True)
    md: list[str] = ["# RAS-MortDB Paper Agent: live demo", ""]

    async with Client(str(HERE / "mcp_server.py")) as client:
        tools = await client.list_tools()
        resources = await client.list_resources()
        prompts = await client.list_prompts()
        md += [
            "MCP server started over stdio and answered the calls below.",
            "",
            f"**Tools ({len(tools)}):** " + ", ".join(f"`{t.name}`" for t in tools),
            f"**Resources:** " + ", ".join(f"`{r.uri}`" for r in resources),
            f"**Prompts:** " + ", ".join(f"`{p.name}`" for p in prompts),
            "",
        ]

        abstract = await client.read_resource("paper://abstract")
        md += ["## Paper (resource `paper://abstract`)", "", "> " + abstract[0].text.split("\n\n")[0], ""]

        md += ["## Deployment recommendations (`recommend_deployment`)", "",
               "| Scenario | Recommended | Reasoning |", "|---|---|---|"]
        for title, args in SCENARIOS:
            rec = (await client.call_tool("recommend_deployment", args)).data
            reasoning = "<br>".join(rec["reasoning"]).replace("|", "\\|")
            md.append(f"| {title} | **{rec['recommended_model']}** | {reasoning} |")
        md.append("")

        perf = (await client.call_tool("get_model_performance", {"model": model})).data
        md += [f"## Reported performance of {model} (`get_model_performance`)", "",
               f"mAP50 {perf['mAP50']}% ± {perf['mAP50_sd']} · mAP50-95 {perf['mAP50_95']}% · "
               f"Raspberry Pi 5 {perf['rpi5_fps']} FPS ({perf['rpi5_avg_ms']} ms) · ONNX {perf['onnx_mb']} MB · "
               f"needs {perf['min_training_images_for_90pct_mAP50']} training images for ≥90% mAP50", ""]

        md += [f"## Live detection on held-out test images (`detect_mortality`, {model}, {fmt})", "",
               "| Image | Agent: dead / live | Ground truth: dead / live | Inference |",
               "|---|---|---|---|"]
        images = sorted(TEST_IMAGES.glob("*.jpg"))
        weight = HERE / "weights" / ("pytorch" if fmt == "pytorch" else "onnx") / (
            f"{model}_best.pt" if fmt == "pytorch" else f"{model}.onnx")
        for image in images:
            det = (await client.call_tool("detect_mortality", {
                "image_path": str(image), "model": model, "weight_format": fmt, "conf": conf})).data
            gt = ground_truth(image)
            gt_text = f"{gt['Dead']} / {gt['Live']}" if gt is not None else "n/a"
            md.append(f"| `{image.name[:22]}…` | {det['dead_count']} / {det['live_count']} | {gt_text} | "
                      f"{det['inference_ms']} ms |")
            save_annotated(image, weight, conf)
        md += ["", "_First call includes model loading. Annotated images are in the `demo-output` artifact._", ""]

        cmp = (await client.call_tool("compare_models_on_image", {
            "image_path": str(images[0]), "weight_format": fmt, "conf": conf})).data
        md += ["## Same image, every bundled model (`compare_models_on_image`)", "",
               "| Model | Dead | Live |", "|---|---|---|"]
        for name, res in cmp["results"].items():
            if "skipped" not in res:
                md.append(f"| {name} | {res['dead_count']} | {res['live_count']} |")
        md.append("")

    report = "\n".join(md)
    (OUT / "demo_report.md").write_text(report)
    (OUT / "comparison.json").write_text(json.dumps(cmp, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo26n")
    parser.add_argument("--format", default="onnx", choices=["onnx", "pytorch"])
    parser.add_argument("--conf", type=float, default=0.25)
    a = parser.parse_args()
    report = asyncio.run(main(a.model, a.format, a.conf))
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a") as fh:
            fh.write(report + "\n")
    print(report)
