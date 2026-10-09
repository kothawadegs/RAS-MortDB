# Deploying the RAS-MortDB web demo to Hugging Face Spaces

This folder is a complete Hugging Face Space. Once deployed, anyone can open the link in a browser, with no account needed, upload a tank image and get dead/live fish counts from the paper's own models.

It runs on the free **CPU basic** hardware or on **ZeroGPU**. Detection runs on the CPU by default, which takes about a second per image and uses no ZeroGPU quota; the first use of each model also downloads its weights (10–100 MB). A **Use GPU** checkbox runs the same detection in a `@spaces.GPU` function, which counts against the visitor's daily ZeroGPU quota. Torch is pinned to 2.13.0, a version ZeroGPU supports. On a ZeroGPU Space, switching to **Settings → Space hardware → CPU basic** removes the quota entirely.

## 1. Create a Hugging Face account and token

1. Sign up at https://huggingface.co/join (free).
2. Go to **Settings → Access Tokens → Create new token**, choose type **Write**, and copy the token.

## 2. Create the Space

1. Open https://huggingface.co/new-space.
2. **Owner**: you, or your lab's organization. **Space name**: e.g. `RAS-MortDB-agent`.
3. **License**: `agpl-3.0`.
4. **Space SDK**: **Gradio** (blank template).
5. **Hardware**: **CPU basic** (free) if offered, otherwise **ZeroGPU**. **Visibility**: **Public**.
6. Click **Create Space**.

## 3. Upload this folder

Use **one** of the two options.

**Option A: command line (recommended)**, from the root of your RAS-MortDB clone:

```bash
pip install -U huggingface_hub
hf auth login                       # paste the Write token
hf upload <your-username>/RAS-MortDB-agent paper2agent/hf-space . --repo-type=space
```

**Option B: browser.** On the Space page, go to **Files → Add file → Upload files**. Drag in **everything inside** `paper2agent/hf-space/` (`app.py`, `README.md`, `requirements.txt`, `packages.txt`, `upstream_manifest.json`, and the folders `tools/`, `examples/`, `paper/`), keeping the folder structure, then click **Commit**.

The upload replaces the template's `README.md`. That is intended, because this one carries the Space settings (Gradio version, Python 3.12).

## 4. Wait for the build

The Space shows **Building**: it installs torch, Ultralytics and the pinned packages, which takes about 5–10 minutes the first time. When it shows **Running**, open the **App** tab:

1. Click an example image (or upload your own), then **Detect**.
2. For example `tank_0162.jpg` with `yolo26n` should show **Dead: 2 · Live: 6**, the same result as the paper's `run_inference.py`.

If the build fails, open **Logs → Build** and send me the last ~30 lines.

## 5. Share it

- The public link is `https://huggingface.co/spaces/<your-username>/RAS-MortDB-agent`.
- To add a badge to the GitHub README:
  ```markdown
  [![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md.svg)](https://huggingface.co/spaces/<your-username>/RAS-MortDB-agent)
  ```
- Free Spaces go to sleep after about 48 hours without visitors. The next visitor wakes the Space automatically, which takes about a minute.

## What runs, and how it was checked

- **The app.** `app.py` is a thin Gradio interface around `tools/run_inference.py`. That file is the Paper2Agent MCP tool, copied unchanged (sha256 `f56e5721…`) from the verified package in `paper2agent/RAS-MortDB-agent/mcp/`. It calls the paper's own `inference/run_inference.py`.
- **Downloads.** The script and the weights are downloaded from GitHub at the pinned commit `6935cde`. They are used only if their SHA-256 matches `upstream_manifest.json`.
- **Local test before publishing:**
  - The app was started from an empty cache; downloads and hash checks passed.
  - It was called through the Gradio API: `tank_0153` + yolo26n ONNX gave 2/3, `tank_0162` + yolo26n PyTorch gave 2/6 at conf 0.25 and 2/5 at conf 0.5, and `tank_0153` + yolo11m ONNX gave 2/3. All four are identical to direct runs of the paper's script.
  - It was also driven in a browser.
- **No FastMCP on Spaces.** Hugging Face always installs `gradio[mcp]`, which needs `mcp<2`, while FastMCP 4 needs `mcp>=2`. The Space therefore does not install FastMCP. `app.py` gives the unchanged tool module a minimal stand-in whose `FastMCP(...).tool()` returns the function as is, which is what FastMCP 4's decorator returns. The detection code path is identical.
  - This was re-tested by reproducing Hugging Face's two install steps from a build log: the packages resolve, and the four reference detections match upstream.
  - For MCP use, run the package in `paper2agent/RAS-MortDB-agent/mcp/` locally.
