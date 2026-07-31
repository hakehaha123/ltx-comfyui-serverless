# 1. 基础镜像：官方 PyTorch GPU 镜像（解决权限与基础环境）
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# 设置全局环境变量，防止 apt-get 弹出交互菜单卡死
ENV DEBIAN_FRONTEND=noninteractive

# 2. 安装系统依赖（加上 DEBIAN_FRONTEND=noninteractive 避免时区选择卡死）
RUN apt-get update && \
    apt-get install -y ffmpeg git git-lfs tzdata && \
    rm -rf /var/lib/apt/lists/*

# 3. 安装 RunPod SDK、工具库（注意：不重新安装 torch，直接复用镜像内置的 GPU 版 Torch）
RUN pip install --no-cache-dir runpod requests huggingface_hub hf_transfer

# 4. 下载 ComfyUI 主程序（从官方镜像切过来后，必须手动拉取 ComfyUI 代码）
WORKDIR /workspace
RUN git clone https://github.com/comfyanonymous/ComfyUI.git \
    && cd ComfyUI \
    && pip install --no-cache-dir -r requirements.txt

# 5. 安装 Lightricks 官方 LTX-Video 自定义节点与依赖
WORKDIR /workspace/ComfyUI/custom_nodes
RUN git clone https://github.com/Lightricks/ComfyUI-LTXVideo.git \
    && pip install --no-cache-dir -r ComfyUI-LTXVideo/requirements.txt

# 6. 设置环境变量，开启 Hugging Face 极速下载
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# 7. 模型预下载（下载核心 Checkpoint、T5 编码器、LipDub LoRA）
# 7.1 LTX 主模型
RUN python3 -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='Lightricks/LTX-Video', filename='ltx-video-2b-v0.9.1.safetensors', local_dir='/workspace/ComfyUI/models/checkpoints')"

# 7.2 T5xxl 文本编码器（修改为正确的 repo_id: comfyanonymous/flux_text_encoders）
RUN python3 -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='comfyanonymous/flux_text_encoders', filename='t5xxl_fp8_e4m3fn.safetensors', local_dir='/workspace/ComfyUI/models/clip')"

# 7.3 LipDub / 音频驱动 LoRA (修正 repo_id 并带上 HF Token 鉴权)
RUN python3 -c "import os; from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='Lightricks/LTX-Video', filename='ltx-video-2b-ic-lora-lipdub.safetensors', local_dir='/workspace/ComfyUI/models/loras', token=os.getenv('HF_TOKEN'))"

# 8. 清理缓存以控制镜像体积
RUN rm -rf /root/.cache/pip /tmp/*

# 9. 启动入口（确认你的 handler.py 放在仓库根目录下）
WORKDIR /workspace
CMD ["python3", "-u", "handler.py"]
