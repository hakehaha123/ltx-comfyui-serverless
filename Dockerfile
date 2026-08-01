# LTX video worker for RunPod Serverless (MVP: nodes in image, models on cold start)
# Docs: https://github.com/runpod-workers/worker-comfyui/blob/main/docs/customization.md
#
# RunPod Endpoint:
#   - Container Disk: 100GB
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

COPY download_ltx_models.py /download_ltx_models.py
COPY start_with_models.sh /start_with_models.sh
RUN chmod +x /start_with_models.sh /download_ltx_models.py

# Download missing weights before starting ComfyUI + handler
CMD ["/start_with_models.sh"]
