# 1. 基础镜像：官方 PyTorch GPU 镜像
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# 设置全局环境变量，防止 apt-get 弹出交互菜单卡死
ENV DEBIAN_FRONTEND=noninteractive

# 2. 安装系统依赖（FFmpeg 为音频合成与音视频对齐核心依赖）
RUN apt-get update && \
    apt-get install -y ffmpeg git git-lfs tzdata curl && \
    rm -rf /var/lib/apt/lists/*

# 3. 安装 RunPod SDK、工具库（复用镜像内置 GPU 版 Torch）
RUN pip install --no-cache-dir runpod requests huggingface_hub hf_transfer

# 4. 下载 ComfyUI 主程序
WORKDIR /workspace
RUN git clone https://github.com/comfyanonymous/ComfyUI.git \
    && cd ComfyUI \
    && pip install --no-cache-dir -r requirements.txt

# 5. 安装 Lightricks 官方 LTX-Video 最新自定义节点与依赖
WORKDIR /workspace/ComfyUI/custom_nodes
RUN git clone https://github.com/Lightricks/ComfyUI-LTXVideo.git \
    && pip install --no-cache-dir -r ComfyUI-LTXVideo/requirements.txt

# 6. 设置环境变量与创建 2.3 专属的目录结构
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

RUN mkdir -p /workspace/ComfyUI/models/checkpoints \
    /workspace/ComfyUI/models/clip \
    /workspace/ComfyUI/models/vae \
    /workspace/ComfyUI/models/loras/ltxv/ltx2

# ==============================================================================
# 7. 模型预下载区域 (适配 LTX-2.3 完整生态)
# ==============================================================================

# 7.1 LTX-2.3 22B FP8 量化主模型 (24G 显存 Serverless 推荐)
RUN python3 -c "import os; from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='Lightricks/LTX-Video', filename='ltx-2.3-22b-dev-fp8.safetensors', local_dir='/workspace/ComfyUI/models/checkpoints', token=os.getenv('HF_TOKEN'))" || \
    python3 -c "import os; from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='Lightricks/LTX-Video', filename='ltx-2.3-22b-dev.safetensors', local_dir='/workspace/ComfyUI/models/checkpoints', token=os.getenv('HF_TOKEN'))"

# 7.2 Gemma 3 文本编码器 (2.3 专属，优先下载 FP8/Q4 优化版)
RUN python3 -c "import os; from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='google/gemma-3-12b-it-qat-q4_0-unquantized', filename='model-00001-of-00005.safetensors', local_dir='/workspace/ComfyUI/models/clip', token=os.getenv('HF_TOKEN'))" || \
    python3 -c "import os; from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='google/gemma-3-12b-it', filename='comfy_gemma_3_12B_it.safetensors', local_dir='/workspace/ComfyUI/models/clip', token=os.getenv('HF_TOKEN'))"

# 7.3 LTX-2.3 官方蒸馏单阶段加速 LoRA
RUN python3 -c "import os; from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='Lightricks/LTX-Video', filename='ltx-2.3-22b-distilled-lora-384-1.1.safetensors', local_dir='/workspace/ComfyUI/models/loras/ltxv/ltx2', token=os.getenv('HF_TOKEN'))"

# 7.4 LTX-2.3 Lipdub (音频驱动与口型对齐 LoRA)
RUN python3 -c "import os; from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='Lightricks/LTX-Video', filename='ltx-2.3-22b-ic-lora-lipdub.safetensors', local_dir='/workspace/ComfyUI/models/loras/ltxv/ltx2', token=os.getenv('HF_TOKEN'))"

# 7.5 LTX-2.3 Union (多功能姿态/边缘控制 LoRA)
RUN python3 -c "import os; from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='Lightricks/LTX-Video', filename='ltx-2.3-22b-ic-lora-union.safetensors', local_dir='/workspace/ComfyUI/models/loras/ltxv/ltx2', token=os.getenv('HF_TOKEN'))" || true

# 7.6 LTX-2.3 Motion Track (动态运动追踪驱动 LoRA)
RUN python3 -c "import os; from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='Lightricks/LTX-Video', filename='ltx-2.3-22b-ic-lora-motion-track.safetensors', local_dir='/workspace/ComfyUI/models/loras/ltxv/ltx2', token=os.getenv('HF_TOKEN'))" || true

# ==============================================================================

# 8. 清理缓存以控制镜像体积
RUN rm -rf /root/.cache/pip /tmp/*

# 9. 启动入口
WORKDIR /workspace
CMD ["python3", "-u", "handler.py"]
