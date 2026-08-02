#!/usr/bin/env python3
"""Download LTX-2.3 weights for the Single-Stage Distilled Full workflow.

Aligned to workflow file references:
  - checkpoints/ltx-2.3-22b-dev.safetensors  (FP8 weights saved under this name)
  - text_encoders/comfy_gemma_3_12B_it.safetensors
  - loras/ltxv/ltx2/ltx-2.3-22b-distilled-lora-384-1.1.safetensors

Strategy (avoid Errno 2 / path races):
  1) Create every destination directory up front and verify with is_dir().
  2) Download into /tmp only (never let HF write under /comfyui).
  3) Copy verified file into place with a .partial + atomic replace.

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
# Always-writable scratch; must NOT be under MODELS_ROOT
SCRATCH_ROOT = Path(os.environ.get("LTX_DOWNLOAD_TMP", "/tmp/ltx-download"))

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

REQUIRED_SUBDIRS = (
    "checkpoints",
    "text_encoders",
    "loras/ltxv/ltx2",
    "clip",
    "vae",
    "diffusion_models",
)


def _mkdir_verified(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise FileNotFoundError(
            f"failed to create directory (exists but not a dir?): {path} "
            f"(parent={path.parent} parent_is_dir={path.parent.is_dir()})"
        )


def ensure_model_dirs() -> list[Path]:
    """Create + verify every directory we will write into. Returns dest parents."""
    if COMFY_ROOT.exists() and not COMFY_ROOT.is_dir():
        raise NotADirectoryError(f"COMFYUI_PATH is not a directory: {COMFY_ROOT}")

    _mkdir_verified(COMFY_ROOT)
    _mkdir_verified(MODELS_ROOT)
    _mkdir_verified(SCRATCH_ROOT)

    for sub in REQUIRED_SUBDIRS:
        _mkdir_verified(MODELS_ROOT / sub)

    dest_parents: list[Path] = []
    for _, _, dest_rel in MODEL_SPECS:
        parent = (MODELS_ROOT / dest_rel).parent
        _mkdir_verified(parent)
        dest_parents.append(parent)

    print(f"ltx-download: COMFYUI_PATH={COMFY_ROOT} exists={COMFY_ROOT.is_dir()}")
    print(f"ltx-download: MODELS_ROOT={MODELS_ROOT} exists={MODELS_ROOT.is_dir()}")
    print(f"ltx-download: SCRATCH_ROOT={SCRATCH_ROOT} exists={SCRATCH_ROOT.is_dir()}")
    for p in dest_parents:
        print(f"ltx-download: dest dir ok: {p}")
    return dest_parents


def download_one(repo_id: str, remote_name: str, dest_rel: str, token: str | None) -> None:
    dest = MODELS_ROOT / dest_rel
    _mkdir_verified(dest.parent)

    if dest.is_file() and dest.stat().st_size > 0:
        print(f"ltx-download: skip (exists) {dest} ({dest.stat().st_size / (1024**3):.2f} GB)")
        return

    # Remove broken/empty leftovers
    if dest.exists() or dest.is_symlink():
        dest.unlink()

    from huggingface_hub import hf_hub_download

    # Entire HF cache lives under /tmp; destination under /comfyui is write-only via copy.
    with tempfile.TemporaryDirectory(prefix="ltx-hf-", dir=str(SCRATCH_ROOT)) as tmp:
        tmp_path = Path(tmp)
        hf_home = tmp_path / "hf"
        hub_cache = hf_home / "hub"
        _mkdir_verified(hf_home)
        _mkdir_verified(hub_cache)

        os.environ["HF_HOME"] = str(hf_home)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
        # Prefer stability over speed on cold start; hf_transfer can fail oddly if misinstalled.
        os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)

        print(f"ltx-download: fetching {repo_id}/{remote_name}")
        print(f"ltx-download: HF cache = {hub_cache}")
        print(f"ltx-download: final dest = {dest} (parent_is_dir={dest.parent.is_dir()})")

        cached = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=remote_name,
                token=token,
            )
        )
        if not cached.is_file():
            raise FileNotFoundError(f"HF download path missing after download: {cached}")

        size = cached.stat().st_size
        if size <= 0:
            raise OSError(f"HF download empty: {cached}")

        # Stage next to dest, then atomic replace — parent already verified.
        staging = dest.parent / f".{dest.name}.partial"
        if staging.exists() or staging.is_symlink():
            staging.unlink()

        print(f"ltx-download: copying {size / (1024**3):.2f} GB -> {staging}")
        shutil.copy2(cached, staging)
        if not staging.is_file() or staging.stat().st_size != size:
            raise OSError(
                f"staging copy incomplete: {staging} "
                f"got={staging.stat().st_size if staging.exists() else 'missing'} expected={size}"
            )
        staging.replace(dest)

    if not dest.is_file() or dest.stat().st_size <= 0:
        raise FileNotFoundError(f"dest missing/empty after install: {dest}")
    print(f"ltx-download: ready {dest} ({dest.stat().st_size / (1024**3):.2f} GB)")


def main() -> int:
    if os.environ.get("LTX_SKIP_MODEL_DOWNLOAD", "").strip() in {"1", "true", "TRUE", "yes"}:
        print("ltx-download: skipped (LTX_SKIP_MODEL_DOWNLOAD)")
        return 0

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print(
            "ltx-download: WARNING — HF_TOKEN not set; large downloads may fail or rate-limit.",
            file=sys.stderr,
        )
    else:
        print("ltx-download: HF_TOKEN is set")

    try:
        ensure_model_dirs()
    except Exception as e:
        print(f"ltx-download: FAILED creating model dirs under {MODELS_ROOT}: {e}", file=sys.stderr)
        # Extra diagnostics without downloading anything
        for p in (Path("/"), Path("/comfyui"), Path("/comfyui/models"), COMFY_ROOT, MODELS_ROOT):
            print(
                f"ltx-download: probe {p}: exists={p.exists()} is_dir={p.is_dir()} "
                f"is_symlink={p.is_symlink()}",
                file=sys.stderr,
            )
        return 1

    for repo_id, remote_name, dest_rel in MODEL_SPECS:
        try:
            download_one(repo_id, remote_name, dest_rel, token)
        except Exception as e:
            dest = MODELS_ROOT / dest_rel
            print(f"ltx-download: FAILED {repo_id}/{remote_name}: {e}", file=sys.stderr)
            print(
                f"ltx-download: at failure dest.parent={dest.parent} "
                f"is_dir={dest.parent.is_dir()} exists={dest.parent.exists()}",
                file=sys.stderr,
            )
            return 1

    print("ltx-download: all models present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
