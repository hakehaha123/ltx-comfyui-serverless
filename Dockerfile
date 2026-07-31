# 1. 基础镜像：使用 RunPod 官方成熟的 ComfyUI Worker (已内置 RunPod API Handler)
FROM runpod/worker-comfy:v2.7.0-cuda12.1.0

# 2. 安装系统依赖 (FFmpeg 是处理音频、音效合成的关键)
RUN apt-get update && apt-get install -y ffmpeg git-lfs && rm -rf /var/lib/apt/lists/*

# 3. 安装 Lightricks 官方 LTX-Video 自定义节点与音频/口型扩展
WORKDIR /workspace/ComfyUI/custom_nodes
RUN git clone https://github.com/Lightricks/ComfyUI-LTXVideo.git \
    && pip install --no-cache-dir -r ComfyUI-LTXVideo/requirements.txt

# 4. 安装加速下载工具
RUN pip install --no-cache-dir huggingface_hub hf_transfer

# 设置环境变量，提升 HF 下载速度
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# 5. 精简下载模型 (只下载必须文件，体积控制在 ~25GB)
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# 5.1 下载 FP8/精简版的 LTX 主模型 (示例使用精简打包路径)
RUN python3 -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='Lightricks/LTX-Video', filename='ltx-video-2b-v0.9.1.safetensors', local_dir='/workspace/ComfyUI/models/checkpoints')"

# 5.2 下载文本编码器 (T5xxl FP8 精简版)
RUN python3 -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='comfyanonymous/butterworth_filters', filename='t5xxl_fp8_e4m3fn.safetensors', local_dir='/workspace/ComfyUI/models/clip')"

# 5.3 下载 LipDub / 口型与音效 LoRA (核心音频驱动模块)
RUN python3 -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='Lightricks/LTX-Video-IC-LoRA-LipDub', filename='ltx-video-2b-ic-lora-lipdub.safetensors', local_dir='/workspace/ComfyUI/models/loras')"

# 6. 清理缓存以缩小镜像体积
RUN rm -rf /root/.cache/pip /tmp/*

WORKDIR /workspace
CMD ["python3", "-u", "handler.py"]
