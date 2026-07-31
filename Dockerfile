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

# ==============================================================================
# 5. 安装 LTX-2.3 必须的自定义节点 (Custom Nodes)
# ==============================================================================
WORKDIR /workspace/ComfyUI/custom_nodes

# 5.1 提前升级 LTX-2.3 核心依赖库（防止与 ComfyUI 自带旧库冲突导致 IMPORT FAILED）
RUN pip install --no-cache-dir --upgrade transformers diffusers accelerate sentencepiece peft kornia kornia-rs

# 5.1 安装 ComfyUI-Manager (节点管理器，方便管理与辅助解析)
RUN git clone https://github.com/ltdrdata/ComfyUI-Manager.git

# 5.2 安装 Lightricks 官方最新 2.3 自定义节点 (包含 MultimodalGuider, ClownSampler 等 8 个节点)
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
# 7. 模型预下载区域 (修复 404 文件名问题)
# ==============================================================================

# 7.1 LTX-2.3 22B FP8 量化版主 Checkpoint (~22GB)
RUN python3 -c "import os; from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='Lightricks/LTX-2.3-fp8', filename='ltx-2.3-22b-dev-fp8.safetensors', local_dir='/workspace/ComfyUI/models/checkpoints', token=os.getenv('HF_TOKEN'))"

# 7.2 Gemma 3 12B 文本编码器
RUN python3 -c "import os; from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='Comfy-Org/ltx-2', filename='split_files/text_encoders/gemma_3_12B_it.safetensors', local_dir='/workspace/ComfyUI/models/clip', token=os.getenv('HF_TOKEN'))"

# 7.3 LTX-2.3 蒸馏单阶段加速 LoRA (v1.1)
RUN python3 -c "import os; from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='Lightricks/LTX-2.3', filename='ltx-2.3-22b-distilled-lora-384-1.1.safetensors', local_dir='/workspace/ComfyUI/models/loras/ltxv/ltx2', token=os.getenv('HF_TOKEN'))"

# 7.4 LTX-2.3 Lipdub 音频口型驱动 LoRA (DubIt 专用)
RUN python3 -c "import os; from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='Lightricks/LTX-2.3-22b-IC-LoRA-LipDub', filename='ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors', local_dir='/workspace/ComfyUI/models/loras/ltxv/ltx2', token=os.getenv('HF_TOKEN'))"

# ==============================================================================

# 8. 清理缓存以控制镜像体积
RUN rm -rf /root/.cache/pip /tmp/*

# 9. 启动入口
WORKDIR /workspace
CMD ["python3", "-u", "handler.py"]
