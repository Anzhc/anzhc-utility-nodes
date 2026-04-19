from __future__ import annotations

from typing import Optional
import threading

import torch
import torch.nn.functional as F
import comfy.sample
import nodes


def normalize_spatially(tensor: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Normalize each channel of a BCHW tensor to zero mean and unit variance.
    """
    if tensor.ndim < 3:
        return tensor

    dims = tuple(range(2, tensor.ndim))
    mean = tensor.mean(dim=dims, keepdim=True)
    var = tensor.var(dim=dims, keepdim=True, unbiased=False)
    return (tensor - mean) / torch.sqrt(var + eps)


def apply_reso_block_noise(
    noise: torch.Tensor,
    latents: torch.Tensor,
    timesteps: Optional[torch.Tensor],
    target_sizes_hw: Optional[torch.Tensor] = None,
    *,
    enabled: bool = False,
    base_resolution: int = 512,
    num_train_timesteps: Optional[int] = None,
    scale_quant_step: float = 0.5,
    constant_alpha_strength: float = 0.7,
) -> torch.Tensor:
    """
    Apply block-structured noise to high-resolution samples so their low-frequency SNR better
    matches the 512 resolution bucket.
    """
    if not enabled or noise is None or not isinstance(noise, torch.Tensor) or noise.ndim != 4 or noise.shape[0] == 0:
        return noise

    device = noise.device
    work_dtype = torch.float32
    batch_size = noise.shape[0]

    if target_sizes_hw is not None:
        sizes_hw = target_sizes_hw.to(device=device, dtype=work_dtype)
    else:
        if isinstance(latents, torch.Tensor) and latents.ndim >= 4:
            lat_h = latents.shape[2] * 8
            lat_w = latents.shape[3] * 8
        else:
            lat_h = noise.shape[2] * 8
            lat_w = noise.shape[3] * 8
        sizes_hw = torch.tensor([lat_h, lat_w], device=device, dtype=work_dtype).repeat(batch_size, 1)

    heights = sizes_hw[:, 0]
    widths = sizes_hw[:, 1]
    effective_res = torch.sqrt(torch.clamp(heights * widths, min=1.0))

    step = max(float(scale_quant_step), 1e-6)
    raw_scale = effective_res / float(base_resolution)
    quantized_scale = torch.round(raw_scale / step) * step
    quantized_scale = torch.clamp(quantized_scale, min=step)
    extra_scale = torch.clamp(quantized_scale - 1.0, min=0.0)

    contributing = torch.nonzero(extra_scale > 1e-6, as_tuple=False).flatten()
    if contributing.numel() == 0:
        return noise

    alpha = extra_scale * float(constant_alpha_strength)
    alpha = torch.clamp(alpha, max=2.0)

    noise_seed = torch.randn_like(noise, dtype=work_dtype)
    block_noise = torch.zeros_like(noise, dtype=work_dtype)

    for idx in contributing.tolist():
        scale = float(quantized_scale[idx].item())
        if scale <= 1.0:
            continue
        sample = noise_seed[idx : idx + 1]
        _, _, h, w = sample.shape
        inv_scale = 1.0 / scale
        down_h = max(1, int(round(h * inv_scale)))
        down_w = max(1, int(round(w * inv_scale)))
        downsampled = F.interpolate(sample, size=(down_h, down_w), mode="area")
        block = F.interpolate(downsampled, size=(h, w), mode="nearest")
        block = normalize_spatially(block)
        block_noise[idx] = block[0]

    alpha = alpha.to(device=device, dtype=work_dtype).view(-1, 1, 1, 1)
    base = noise.to(work_dtype)
    denom = torch.sqrt(1.0 + alpha.pow(2))
    mixed = (base + alpha * block_noise) / denom

    return mixed.to(noise.dtype)


class AnzhcBlockNoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "enabled": ("BOOLEAN", {"default": True}),
                "base_resolution": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "scale_quant_step": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply"
    CATEGORY = "anzhc/utility"
    DESCRIPTION = "Attach block-noise sampling parameters to a latent. Injection happens during KSampler noise generation."

    def apply(self, latent, enabled: bool, base_resolution: int, scale_quant_step: float):
        if not isinstance(latent, dict) or "samples" not in latent:
            return (latent,)

        out = latent.copy()
        if enabled:
            out[_BLOCK_NOISE_CONFIG_KEY] = {
                "enabled": True,
                "base_resolution": int(base_resolution),
                "scale_quant_step": float(scale_quant_step),
                "constant_alpha_strength": 0.7,
            }
        else:
            out.pop(_BLOCK_NOISE_CONFIG_KEY, None)
        return (out,)


class AnzhcBlockNoiseDebugPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "base_resolution": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "scale_quant_step": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 2.0, "step": 0.05}),
                "latent_channels": ("INT", {"default": 4, "min": 1, "max": 128, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "preview"
    CATEGORY = "anzhc/utility"
    DESCRIPTION = "Debug preview: left half is normal noise, right half is block-noise, for the chosen resolution."

    @staticmethod
    def _noise_to_image(noise: torch.Tensor, width: int, height: int) -> torch.Tensor:
        # Visualize noise as grayscale: mean over channels, then normalize to [0, 1].
        gray = noise.to(torch.float32).mean(dim=1, keepdim=True)
        gray = F.interpolate(gray, size=(height, width), mode="bilinear", align_corners=False)

        minv = gray.amin(dim=(2, 3), keepdim=True)
        maxv = gray.amax(dim=(2, 3), keepdim=True)
        denom = torch.clamp(maxv - minv, min=1e-6)
        gray = (gray - minv) / denom

        return gray.movedim(1, -1).repeat(1, 1, 1, 3).clamp(0.0, 1.0)

    def preview(
        self,
        width: int,
        height: int,
        seed: int,
        base_resolution: int,
        scale_quant_step: float,
        latent_channels: int,
    ):
        latent_h = max(1, height // 8)
        latent_w = max(1, width // 8)
        latent = torch.zeros((1, latent_channels, latent_h, latent_w), dtype=torch.float32, device="cpu")
        noise = comfy.sample.prepare_noise(latent, seed)

        block_noise = apply_reso_block_noise(
            noise=noise,
            latents=latent,
            timesteps=None,
            target_sizes_hw=None,
            enabled=True,
            base_resolution=base_resolution,
            num_train_timesteps=None,
            scale_quant_step=scale_quant_step,
            constant_alpha_strength=0.7,
        )

        base_img = self._noise_to_image(noise, width, height)
        block_img = self._noise_to_image(block_noise, width, height)

        split_x = width // 2
        preview = base_img.clone()
        preview[:, :, split_x:, :] = block_img[:, :, split_x:, :]
        return (preview,)


_BLOCK_NOISE_CONFIG_KEY = "_anzhc_block_noise_config"
_PREPARE_NOISE_PATCH_LOCK = threading.RLock()


def _extract_block_noise_config(latent) -> Optional[dict]:
    if not isinstance(latent, dict):
        return None
    raw = latent.get(_BLOCK_NOISE_CONFIG_KEY)
    if not isinstance(raw, dict):
        return None
    if not bool(raw.get("enabled", False)):
        return None

    base_resolution = int(raw.get("base_resolution", 512))
    scale_quant_step = float(raw.get("scale_quant_step", 0.5))
    constant_alpha_strength = float(raw.get("constant_alpha_strength", 0.7))

    if base_resolution < 1:
        base_resolution = 512
    if scale_quant_step <= 0:
        scale_quant_step = 0.5

    return {
        "base_resolution": base_resolution,
        "scale_quant_step": scale_quant_step,
        "constant_alpha_strength": constant_alpha_strength,
    }


def _wrap_common_ksampler():
    target = nodes.common_ksampler
    if getattr(target, "_anzhc_block_noise_wrapped", False):
        return

    original = target

    def wrapped(
        model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent,
        denoise=1.0,
        disable_noise=False,
        start_step=None,
        last_step=None,
        force_full_denoise=False,
    ):
        block_cfg = _extract_block_noise_config(latent)
        if block_cfg is None or disable_noise:
            return original(
                model,
                seed,
                steps,
                cfg,
                sampler_name,
                scheduler,
                positive,
                negative,
                latent,
                denoise=denoise,
                disable_noise=disable_noise,
                start_step=start_step,
                last_step=last_step,
                force_full_denoise=force_full_denoise,
            )

        with _PREPARE_NOISE_PATCH_LOCK:
            original_prepare_noise = comfy.sample.prepare_noise

            def patched_prepare_noise(latent_image, noise_seed, noise_inds=None):
                noise = original_prepare_noise(latent_image, noise_seed, noise_inds)
                if not isinstance(noise, torch.Tensor):
                    return noise
                if noise.ndim != 4 or noise.shape[0] == 0:
                    return noise
                return apply_reso_block_noise(
                    noise=noise,
                    latents=latent_image if isinstance(latent_image, torch.Tensor) else noise,
                    timesteps=None,
                    target_sizes_hw=None,
                    enabled=True,
                    base_resolution=block_cfg["base_resolution"],
                    num_train_timesteps=None,
                    scale_quant_step=block_cfg["scale_quant_step"],
                    constant_alpha_strength=block_cfg["constant_alpha_strength"],
                )

            comfy.sample.prepare_noise = patched_prepare_noise
            try:
                return original(
                    model,
                    seed,
                    steps,
                    cfg,
                    sampler_name,
                    scheduler,
                    positive,
                    negative,
                    latent,
                    denoise=denoise,
                    disable_noise=disable_noise,
                    start_step=start_step,
                    last_step=last_step,
                    force_full_denoise=force_full_denoise,
                )
            finally:
                comfy.sample.prepare_noise = original_prepare_noise

    wrapped._anzhc_block_noise_wrapped = True
    nodes.common_ksampler = wrapped


_wrap_common_ksampler()


NODE_CLASS_MAPPINGS = {
    "Anzhc Block Noise": AnzhcBlockNoise,
    "Anzhc Block Noise Debug": AnzhcBlockNoiseDebugPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Anzhc Block Noise": "Block Noise (Anzhc)",
    "Anzhc Block Noise Debug": "Block Noise Debug (Anzhc)",
}
