#!/usr/bin/env python3
"""Download LTX-2.3 weights for drama workflows (T2V/I2V/FFLF + IC-LoRA + upscale).

Core (~61GB):
  - checkpoints/ltx-2.3-22b-dev.safetensors  (FP8 weights saved under this name)
  - text_encoders/comfy_gemma_3_12B_it.safetensors
  - loras/ltxv/ltx2/ltx-2.3-22b-distilled-lora-384-1.1.safetensors

IC-LoRA + latent upscalers (~6GB; Pixel x4 intentionally omitted):
  - LipDub / Ingredients / Union / Pixel Spatial Upscaler x2
  - latent spatial x2 1.1 + temporal x2 (Two-Stage / upscale)

One RunPod endpoint image can host all of these; the app selects the workflow JSON
via workflow_kind — no per-workflow endpoint required.

Disk strategy (models ~67GB total; use Container Disk >=200GB):
  1) Scratch lives UNDER /comfyui/models so move() is same-filesystem rename (no 2x copy).
  2) After each file, wipe scratch immediately.
  3) Log free space before/after; fail early if clearly insufficient.

HF_TOKEN must come from the environment (RunPod Endpoint env) — never hardcode it.
Set LTX_SKIP_MODEL_DOWNLOAD=1 to skip.
Set LTX_SKIP_IC_LORA=1 to download core weights only.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

COMFY_ROOT = Path(os.environ.get("COMFYUI_PATH", "/comfyui"))
MODELS_ROOT = COMFY_ROOT / "models"
# Same filesystem as final models → shutil.move is rename, not copy+delete.
SCRATCH_ROOT = Path(
    os.environ.get("LTX_DOWNLOAD_TMP", str(MODELS_ROOT / ".ltx-download-tmp"))
)

# (repo_id, remote_filename, dest_relative_to_models, approx_bytes)
CORE_SPECS: list[tuple[str, str, str, int]] = [
    (
        "Lightricks/LTX-2.3-fp8",
        "ltx-2.3-22b-dev-fp8.safetensors",
        "checkpoints/ltx-2.3-22b-dev.safetensors",
        29_145_431_166,
    ),
    (
        "Comfy-Org/ltx-2",
        "split_files/text_encoders/gemma_3_12B_it.safetensors",
        "text_encoders/comfy_gemma_3_12B_it.safetensors",
        24_379_468_890,
    ),
    (
        "Lightricks/LTX-2.3",
        "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
        "loras/ltxv/ltx2/ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
        7_605_507_256,
    ),
]

# Drama Phase-2 adapters (Pixel x4 omitted — ~0.61GB saved; x2 covers enhance path).
IC_LORA_SPECS: list[tuple[str, str, str, int]] = [
    (
        "Lightricks/LTX-2.3-22b-IC-LoRA-LipDub",
        "ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors",
        "loras/ltxv/ltx2/ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors",
        2_465_000_000,  # ~2.3GB; exact size verified on first download
    ),
    (
        "Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients",
        "ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors",
        "loras/ltxv/ltx2/ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors",
        1_310_000_000,  # ~1.22GB
    ),
    (
        "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control",
        "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
        "loras/ltxv/ltx2/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
        654_465_352,
    ),
    (
        "Lightricks/LTX-2.3-22b-IC-LoRA-Pixel-Spatial-Upscaler",
        "ltx-2.3-22b-ic-lora-pixel-spatial-upscaler-x2-0.9.safetensors",
        "loras/ltxv/ltx2/ltx-2.3-22b-ic-lora-pixel-spatial-upscaler-x2-0.9.safetensors",
        654_465_286,
    ),
    (
        "Lightricks/LTX-2.3",
        "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        "latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        995_743_560,
    ),
    (
        "Lightricks/LTX-2.3",
        "ltx-2.3-temporal-upscaler-x2-1.0.safetensors",
        "latent_upscale_models/ltx-2.3-temporal-upscaler-x2-1.0.safetensors",
        261_944_000,
    ),
]

REQUIRED_SUBDIRS = (
    "checkpoints",
    "text_encoders",
    "loras/ltxv/ltx2",
    "latent_upscale_models",
    "clip",
    "vae",
    "diffusion_models",
)


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip() in {"1", "true", "TRUE", "yes"}


def active_specs() -> list[tuple[str, str, str, int]]:
    specs = list(CORE_SPECS)
    if not _truthy("LTX_SKIP_IC_LORA"):
        specs.extend(IC_LORA_SPECS)
    else:
        print("ltx-download: IC-LoRA/upscalers skipped (LTX_SKIP_IC_LORA)")
    return specs


def _gb(n: int | float) -> str:
    return f"{n / (1024**3):.2f} GB"


def _disk_free(path: Path) -> int:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free


def _log_disk(label: str, path: Path) -> int:
    usage = shutil.disk_usage(path)
    print(
        f"ltx-download: disk[{label}] path={path} "
        f"free={_gb(usage.free)} used={_gb(usage.used)} total={_gb(usage.total)}"
    )
    return usage.free


def _mkdir_verified(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise FileNotFoundError(
            f"failed to create directory (exists but not a dir?): {path} "
            f"(parent={path.parent} parent_is_dir={path.parent.is_dir()})"
        )


def _wipe_scratch() -> None:
    if SCRATCH_ROOT.exists():
        shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)
    _mkdir_verified(SCRATCH_ROOT)


def ensure_model_dirs() -> None:
    if COMFY_ROOT.exists() and not COMFY_ROOT.is_dir():
        raise NotADirectoryError(f"COMFYUI_PATH is not a directory: {COMFY_ROOT}")

    _mkdir_verified(COMFY_ROOT)
    _mkdir_verified(MODELS_ROOT)
    _wipe_scratch()

    for sub in REQUIRED_SUBDIRS:
        _mkdir_verified(MODELS_ROOT / sub)

    for _, _, dest_rel, _ in active_specs():
        _mkdir_verified((MODELS_ROOT / dest_rel).parent)

    print(f"ltx-download: COMFYUI_PATH={COMFY_ROOT} exists={COMFY_ROOT.is_dir()}")
    print(f"ltx-download: MODELS_ROOT={MODELS_ROOT} exists={MODELS_ROOT.is_dir()}")
    print(f"ltx-download: SCRATCH_ROOT={SCRATCH_ROOT} exists={SCRATCH_ROOT.is_dir()}")
    _log_disk("models", MODELS_ROOT)


def download_one(
    repo_id: str, remote_name: str, dest_rel: str, approx_bytes: int, token: str | None
) -> None:
    dest = MODELS_ROOT / dest_rel
    _mkdir_verified(dest.parent)

    if dest.is_file() and dest.stat().st_size > 0:
        print(f"ltx-download: skip (exists) {dest} ({_gb(dest.stat().st_size)})")
        return

    if dest.exists() or dest.is_symlink():
        dest.unlink()

    # Need room for the file in scratch (+ small margin). Same FS move → no second full copy.
    free = _log_disk(f"before {dest.name}", MODELS_ROOT)
    need = int(approx_bytes * 1.05) + (512 * 1024 * 1024)
    if free < need:
        raise OSError(
            f"not enough free disk for {dest.name}: free={_gb(free)} need~={_gb(need)}. "
            f"Raise RunPod Container Disk to >=200GB (models ~67GB + image + margin)."
        )

    from huggingface_hub import hf_hub_download

    _wipe_scratch()
    try:
        with tempfile.TemporaryDirectory(prefix="ltx-hf-", dir=str(SCRATCH_ROOT)) as tmp:
            tmp_path = Path(tmp)
            hf_home = tmp_path / "hf"
            hub_cache = hf_home / "hub"
            _mkdir_verified(hf_home)
            _mkdir_verified(hub_cache)

            os.environ["HF_HOME"] = str(hf_home)
            os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
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

            staging = dest.parent / f".{dest.name}.partial"
            if staging.exists() or staging.is_symlink():
                staging.unlink()

            # Prefer move (same FS → rename). Falls back to copy+delete across devices.
            print(f"ltx-download: installing {_gb(size)} -> {staging} (move)")
            shutil.move(str(cached), str(staging))
            if not staging.is_file() or staging.stat().st_size != size:
                raise OSError(
                    f"staging incomplete: {staging} "
                    f"got={staging.stat().st_size if staging.exists() else 'missing'} expected={size}"
                )
            staging.replace(dest)
    finally:
        _wipe_scratch()

    if not dest.is_file() or dest.stat().st_size <= 0:
        raise FileNotFoundError(f"dest missing/empty after install: {dest}")
    print(f"ltx-download: ready {dest} ({_gb(dest.stat().st_size)})")
    _log_disk(f"after {dest.name}", MODELS_ROOT)


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
        for p in (Path("/"), Path("/comfyui"), Path("/comfyui/models"), COMFY_ROOT, MODELS_ROOT):
            print(
                f"ltx-download: probe {p}: exists={p.exists()} is_dir={p.is_dir()} "
                f"is_symlink={p.is_symlink()}",
                file=sys.stderr,
            )
        return 1

    specs = active_specs()
    remaining = sum(
        approx
        for _, _, dest_rel, approx in specs
        if not ((MODELS_ROOT / dest_rel).is_file() and (MODELS_ROOT / dest_rel).stat().st_size > 0)
    )
    free = _disk_free(MODELS_ROOT)
    print(f"ltx-download: remaining downloads ~{_gb(remaining)}; free={_gb(free)}")
    if free < remaining + (2 * 1024**3):
        print(
            "ltx-download: WARNING — free space looks tight for remaining models. "
            "Set Container Disk to 200GB before retrying.",
            file=sys.stderr,
        )

    for repo_id, remote_name, dest_rel, approx in specs:
        try:
            download_one(repo_id, remote_name, dest_rel, approx, token)
        except Exception as e:
            dest = MODELS_ROOT / dest_rel
            print(f"ltx-download: FAILED {repo_id}/{remote_name}: {e}", file=sys.stderr)
            print(
                f"ltx-download: at failure dest.parent={dest.parent} "
                f"is_dir={dest.parent.is_dir()} exists={dest.parent.exists()}",
                file=sys.stderr,
            )
            _log_disk("failure", MODELS_ROOT)
            return 1

    print("ltx-download: all models present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
