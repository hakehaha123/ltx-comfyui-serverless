# LTX video worker for RunPod Serverless (MVP: nodes in image, models on cold start)
# Docs: https://github.com/runpod-workers/worker-comfyui/blob/main/docs/customization.md
#
# RunPod Endpoint:
#   - Container Disk: 200GB
#   - Env: HF_TOKEN=<your token>   (do NOT bake the token into this image)
#   - Active Workers: 0, no Network Volume
FROM runpod/worker-comfyui:5.8.6-base

# Avoid ComfyUI-LTXVideo import failure on kornia 0.8.x (pad removed from pyramid)
RUN pip install "kornia==0.7.3" "huggingface_hub[hf_transfer]" \
    && pip install "hf_transfer" || true

# Custom nodes required by the LTX + ClownSampler workflow
RUN comfy-node-install \
    https://github.com/Lightricks/ComfyUI-LTXVideo \
    https://github.com/ClownsharkBatwing/RES4LYF

# comfy-node-install can succeed while RES4LYF beta samplers fail to import later.
# Fail the image build if ClownSampler_Beta is not actually present on disk + install deps.
RUN set -e; \
    RES_DIR="$(find /comfyui/custom_nodes -maxdepth 1 -type d \( -iname 'RES4LYF' -o -iname 'res4lyf' \) | head -n1)"; \
    echo "RES4LYF dir=${RES_DIR}"; \
    test -n "${RES_DIR}"; \
    test -f "${RES_DIR}/beta/__init__.py"; \
    grep -q 'ClownSampler_Beta' "${RES_DIR}/beta/__init__.py"; \
    if [ -f "${RES_DIR}/requirements.txt" ]; then pip install -r "${RES_DIR}/requirements.txt"; fi; \
    ls "${RES_DIR}/beta/samplers.py"; \
    echo "RES4LYF ClownSampler_Beta present"

COPY download_ltx_models.py /download_ltx_models.py
COPY start_with_models.sh /start_with_models.sh
RUN chmod +x /start_with_models.sh /download_ltx_models.py \
    && mkdir -p \
      /comfyui/models/checkpoints \
      /comfyui/models/text_encoders \
      /comfyui/models/loras/ltxv/ltx2 \
      /comfyui/models/clip \
      /comfyui/models/vae \
      /comfyui/models/diffusion_models \
      /comfyui/models/.ltx-download-tmp \
    && test -d /comfyui/models/checkpoints \
    && test -d /comfyui/models/text_encoders \
    && test -d /comfyui/models/loras/ltxv/ltx2

# Download missing weights before starting ComfyUI + handler
CMD ["/start_with_models.sh"]
