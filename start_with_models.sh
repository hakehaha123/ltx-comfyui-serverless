#!/usr/bin/env bash
set -euo pipefail

COMFY_ROOT="${COMFYUI_PATH:-/comfyui}"
MODELS_ROOT="${COMFY_ROOT}/models"

echo "worker-comfyui-ltx: COMFYUI_PATH=${COMFY_ROOT}"
echo "worker-comfyui-ltx: pre-creating model directories under ${MODELS_ROOT}"

# Belt-and-suspenders: create dirs in shell BEFORE Python (covers old/broken images).
mkdir -p \
  "${MODELS_ROOT}/checkpoints" \
  "${MODELS_ROOT}/text_encoders" \
  "${MODELS_ROOT}/loras/ltxv/ltx2" \
  "${MODELS_ROOT}/clip" \
  "${MODELS_ROOT}/vae" \
  "${MODELS_ROOT}/diffusion_models" \
  /tmp/ltx-download

for d in \
  "${MODELS_ROOT}/checkpoints" \
  "${MODELS_ROOT}/text_encoders" \
  "${MODELS_ROOT}/loras/ltxv/ltx2" \
  /tmp/ltx-download
do
  if [[ ! -d "$d" ]]; then
    echo "worker-comfyui-ltx: FATAL — directory missing after mkdir -p: $d" >&2
    ls -la "$(dirname "$d")" >&2 || true
    exit 1
  fi
  echo "worker-comfyui-ltx: dir ok: $d"
done

echo "worker-comfyui-ltx: ensuring LTX models are on disk..."
python3 /download_ltx_models.py

echo "worker-comfyui-ltx: handing off to stock start.sh"
exec /start.sh
