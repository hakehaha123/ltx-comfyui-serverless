#!/usr/bin/env bash
set -euo pipefail

echo "worker-comfyui-ltx: ensuring LTX models are on disk..."
python3 /download_ltx_models.py

echo "worker-comfyui-ltx: handing off to stock start.sh"
exec /start.sh
