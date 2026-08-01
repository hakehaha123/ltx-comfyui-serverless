#!/usr/bin/env bash
set -euo pipefail

echo "ltx-comfyui-serverless: ensuring LTX models are on disk..."
python3 /download_ltx_models.py

echo "ltx-comfyui-serverless: handing off to stock start.sh"
exec /start.sh
