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

# # ==============================================================================
# 7. 模型预下载区域 (修复 404 文件名问题)
# ==============================================================================

# 7.1 LTX 主 Checkpoint (使用 HF 仓库真实存在的文件名)
RUN python3 -c "import os; from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='Lightricks/LTX-Video', filename='ltx-video-2b-v0.9.1.safetensors', local_dir='/workspace/ComfyUI/models/checkpoints', token=os.getenv('HF_TOKEN', None))"

# 7.2 Gemma 3 文本编码器 (2.3 专属，带容错机制)
RUN python3 -c "import os; from huggingface_hub import hf_hub_download; \
    try: hf_hub_download(repo_id='google/gemma-3-12b-it', filename='comfy_gemma_3_12B_it.safetensors', local_dir='/workspace/ComfyUI/models/clip', token=os.getenv('HF_TOKEN', None)); \
    except Exception as e: print('Gemma3 download skipped/failed:', e)" || true

# 7.3 LTX 蒸馏 acceleration LoRA
RUN python3 -c "import os; from huggingface_hub import hf_hub_download; \
    try: hf_hub_download(repo_id='Lightricks/LTX-Video', filename='ltx-video-2b-lora-distilled.safetensors', local_dir='/workspace/ComfyUI/models/loras/ltxv/ltx2', token=os.getenv('HF_TOKEN', None)); \
    except Exception as e: print('Distilled LoRA skipped/failed:', e)" || true

# 7.4 Lipdub 音频/口型驱动 LoRA
RUN python3 -c "import os; from huggingface_hub import hf_hub_download; \
    try: hf_hub_download(repo_id='Lightricks/LTX-Video-IC-LoRA-LipDub', filename='ltx-video-2b-ic-lora-lipdub.safetensors', local_dir='/workspace/ComfyUI/models/loras/ltxv/ltx2', token=os.getenv('HF_TOKEN', None)); \
    except Exception as e: print('Lipdub LoRA skipped/failed:', e)" || true

# 7.5 LTX Union (多功能姿态/深度/边缘控制 IC-LoRA)
RUN python3 -c "import os; from huggingface_hub import hf_hub_download; \
    try: hf_hub_download(repo_id='Lightricks/LTX-Video', filename='ltx-2.3-ic-lora-union.safetensors', local_dir='/workspace/ComfyUI/models/loras/ltxv/ltx2', token=os.getenv('HF_TOKEN', None)); \
    except Exception as e: print('Union LoRA skipped:', e)" || true

# 7.6 LTX Motion Track (动态运动轨迹追踪驱动 IC-LoRA)
RUN python3 -c "import os; from huggingface_hub import hf_hub_download; \
    try: hf_hub_download(repo_id='Lightricks/LTX-Video', filename='ltx-2.3-ic-lora-motion-track.safetensors', local_dir='/workspace/ComfyUI/models/loras/ltxv/ltx2', token=os.getenv('HF_TOKEN', None)); \
    except Exception as e: print('Motion Track LoRA skipped:', e)" || true

# ==============================================================================

# 8. 清理缓存以控制镜像体积
RUN rm -rf /root/.cache/pip /tmp/*

# 9. 启动入口
WORKDIR /workspace
CMD ["python3", "-u", "handler.py"]
