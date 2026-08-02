#!/usr/bin/env python3
"""Download LTX-2.3 weights for the Single-Stage Distilled Full workflow.

Aligned to workflow file references:
  - checkpoints/ltx-2.3-22b-dev.safetensors  (FP8 weights saved under this name)
  - text_encoders/comfy_gemma_3_12B_it.safetensors
  - loras/ltxv/ltx2/ltx-2.3-22b-distilled-lora-384-1.1.safetensors

HF_TOKEN must come from the environment (RunPod Endpoint env) — never hardcode it.
Set LTX_SKIP_MODEL_DOWNLOAD=1 to skip.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

COMFY_ROOT = Path(os.environ.get("COMFYUI_PATH", "/comfyui"))
MODELS_ROOT = COMFY_ROOT / "models"

# (repo_id, remote_filename, dest_relative_to_models)
MODEL_SPECS: list[tuple[str, str, str]] = [
    (
        "Lightricks/LTX-2.3-fp8",
        "ltx-2.3-22b-dev-fp8.safetensors",
        "checkpoints/ltx-2.3-22b-dev.safetensors",
    ),
    (
        "Comfy-Org/ltx-2",
        "split_files/text_encoders/gemma_3_12B_it.safetensors",
        "text_encoders/comfy_gemma_3_12B_it.safetensors",
    ),
    (
        "Lightricks/LTX-2.3",
        "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
        "loras/ltxv/ltx2/ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
    ),
]


def ensure_model_dirs() -> None:
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    for sub in (
        "checkpoints",
        "text_encoders",
        "loras/ltxv/ltx2",
        "clip",
        "vae",
        "diffusion_models",
    ):
        (MODELS_ROOT / sub).mkdir(parents=True, exist_ok=True)
    for _, _, dest_rel in MODEL_SPECS:
        (MODELS_ROOT / dest_rel).parent.mkdir(parents=True, exist_ok=True)
    print(f"ltx-download: comfy root exists={COMFY_ROOT.is_dir()} models={MODELS_ROOT}")


def download_one(repo_id: str, remote_name: str, dest_rel: str, token: str | None) -> None:
    dest = MODELS_ROOT / dest_rel
    if dest.is_file() and dest.stat().st_size > 0:
        print(f"ltx-download: skip (exists) {dest} ({dest.stat().st_size / (1024**3):.2f} GB)")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download

    # Keep scratch outside the destination tree; copy (don't move) while cache still exists.
    scratch_root = Path(os.environ.get("LTX_DOWNLOAD_TMP", "/tmp"))
    scratch_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="ltx-hf-", dir=str(scratch_root)) as tmp:
        tmp_path = Path(tmp)
        os.environ["HF_HOME"] = str(tmp_path)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(tmp_path / "hub")
        print(f"ltx-download: fetching {repo_id}/{remote_name} -> {dest}")
        cached = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=remote_name,
                token=token,
            )
        )
        if not cached.is_file():
            raise FileNotFoundError(f"HF download path missing: {cached}")

        # Stage beside dest then replace — avoids cross-device move edge cases
        staging = dest.parent / f".{dest.name}.partial"
        if staging.exists() or staging.is_symlink():
            staging.unlink()
        shutil.copy2(cached, staging)
        staging.replace(dest)

    if not dest.is_file():
        raise FileNotFoundError(f"dest missing after copy: {dest}")
    print(f"ltx-download: ready {dest} ({dest.stat().st_size / (1024**3):.2f} GB)")


def main() -> int:
    if os.environ.get("LTX_SKIP_MODEL_DOWNLOAD", "").strip() in {"1", "true", "TRUE", "yes"}:
        print("ltx-download: skipped (LTX_SKIP_MODEL_DOWNLOAD)")
        return 0

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print(
            "ltx-download: WARNING — HF_TOKEN not set; downloads may be slow or rate-limited.",
            file=sys.stderr,
        )

    try:
        ensure_model_dirs()
    except Exception as e:
        print(f"ltx-download: FAILED creating model dirs under {MODELS_ROOT}: {e}", file=sys.stderr)
        return 1

    print(f"ltx-download: models root = {MODELS_ROOT}")
    for repo_id, remote_name, dest_rel in MODEL_SPECS:
        try:
            download_one(repo_id, remote_name, dest_rel, token)
        except Exception as e:
            print(f"ltx-download: FAILED {repo_id}/{remote_name}: {e}", file=sys.stderr)
            return 1
    print("ltx-download: all models present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
