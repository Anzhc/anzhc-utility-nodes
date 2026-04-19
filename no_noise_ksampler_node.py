from __future__ import annotations

import comfy.model_sampling
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview


def _validate_const_model_sampling(model) -> None:
    model_sampling = model.get_model_object("model_sampling")
    if isinstance(model_sampling, comfy.model_sampling.CONST):
        return

    raise RuntimeError(
        "No-Noise KSampler (Anzhc) requires a CONST-based model_sampling "
        f"(FLOW/FLUX-style). Got {type(model_sampling).__name__}."
    )


class AnzhcNoNoiseKSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "Kept for KSampler compatibility. No random noise is generated.",
                    },
                ),
                "steps": (
                    "INT",
                    {"default": 20, "min": 1, "max": 10000, "tooltip": "The number of denoising steps."},
                ),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                        "tooltip": "Classifier-Free Guidance scale.",
                    },
                ),
                "sampler_name": (
                    comfy.samplers.KSampler.SAMPLERS,
                    {"tooltip": "The sampler algorithm used during denoising."},
                ),
                "scheduler": (
                    comfy.samplers.KSampler.SCHEDULERS,
                    {"tooltip": "The scheduler controlling sigma progression."},
                ),
                "positive": ("CONDITIONING", {"tooltip": "Positive conditioning."}),
                "negative": ("CONDITIONING", {"tooltip": "Negative conditioning."}),
                "latent_image": (
                    "LATENT",
                    {
                        "tooltip": "Already-resized latent to denoise directly, without generating Gaussian noise."
                    },
                ),
                "denoise": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "The amount of denoising applied.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The denoised latent.",)
    FUNCTION = "sample"
    CATEGORY = "anzhc/utility"
    DESCRIPTION = "KSampler-style node that starts from the provided latent instead of generating random noise."

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0):
        if not isinstance(latent_image, dict) or "samples" not in latent_image:
            raise RuntimeError("No-Noise KSampler (Anzhc) expects a LATENT input containing 'samples'.")

        _validate_const_model_sampling(model)

        latent_samples = latent_image["samples"]
        latent_samples = comfy.sample.fix_empty_latent_channels(
            model, latent_samples, latent_image.get("downscale_ratio_spacial", None)
        )
        noise = latent_samples
        noise_mask = latent_image.get("noise_mask")

        callback = latent_preview.prepare_callback(model, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        samples = comfy.sample.sample(
            model,
            noise,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_samples,
            denoise=denoise,
            disable_noise=False,
            start_step=None,
            last_step=None,
            force_full_denoise=False,
            noise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=seed,
        )

        out = latent_image.copy()
        out.pop("downscale_ratio_spacial", None)
        out["samples"] = samples
        return (out,)


NODE_CLASS_MAPPINGS = {
    "Anzhc No-Noise KSampler": AnzhcNoNoiseKSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Anzhc No-Noise KSampler": "No-Noise KSampler (Anzhc)",
}
