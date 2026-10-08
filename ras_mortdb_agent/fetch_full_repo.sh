#!/usr/bin/env bash
# This package ships only 2 of the 12 trained models (yolo26n, yolov8n) and
# 5 sample test images, to keep it small. Run this once to pull the full
# RAS-MortDB repo (all 12 PyTorch + 12 ONNX weights, full 2000-image
# dataset, all 84 training configs) and point the MCP server at it.
set -e
git clone --depth 1 https://github.com/PA-RRanjan/RAS-MortDB.git "$(dirname "$0")/RAS-MortDB_full"
echo
echo "Full repo cloned to ./RAS-MortDB_full"
echo "Either:"
echo "  a) copy its weights/ subfolders over this package's weights/, or"
echo "  b) edit WEIGHTS_PT / WEIGHTS_ONNX at the top of mcp_server.py to point"
echo "     at RAS-MortDB_full/weights/pytorch and RAS-MortDB_full/weights/onnx"
