import numpy as np
from PIL import Image

class HiresColorAdjustmentNode:
    """
    Applies shadow/middle/highlight color balance before hi-res fix.
    """
    CATEGORY = "Custom/Hires"
    NODE_HIDDEN = False
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                # Shadow sliders
                "shadow_r": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "shadow_g": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "shadow_b": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "shadow_brightness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "shadow_contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                # Middle sliders
                "middle_r": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "middle_g": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "middle_b": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "middle_brightness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "middle_contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                # Highlight sliders
                "highlight_r": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "highlight_g": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "highlight_b": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "highlight_brightness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "highlight_contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            }
        }
    RETURN_TYPES = ("LATENT","JSON")
    RETURN_NAMES = ("latent","metadata")
    FUNCTION = "process"

    def process(
        self,
        latent,
        shadow_r, shadow_g, shadow_b, shadow_brightness, shadow_contrast,
        middle_r, middle_g, middle_b, middle_brightness, middle_contrast,
        highlight_r, highlight_g, highlight_b, highlight_brightness, highlight_contrast
    ):
        # Convert IMAGE batch (torch.Tensor) to numpy uint8
        arr = latent.numpy() if hasattr(latent, 'numpy') else latent
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8) if arr.dtype in [np.float32, np.float64] else arr.astype(np.uint8)
        # Separate alpha
        alpha = None
        if arr.shape[3] == 4:
            alpha = arr[..., 3]
            arr = arr[..., :3]
        pil = Image.fromarray(arr[0])
        # Build adjustments
        adjust = {
            'shadow':    [shadow_r, shadow_g, shadow_b, shadow_brightness, shadow_contrast],
            'middle':    [middle_r, middle_g, middle_b, middle_brightness, middle_contrast],
            'highlight': [highlight_r, highlight_g, highlight_b, highlight_brightness, highlight_contrast],
        }
        # Color balance logic
        orig = np.array(pil, dtype=np.float32)
        lum = orig.mean(axis=2)
        ws = np.clip((128.0 - lum) / 128.0, 0.0, 1.0)
        wh = np.clip((lum - 128.0) / 128.0, 0.0, 1.0)
        wm = 1.0 - ws - wh
        res = np.zeros_like(orig)
        for w, region in zip((ws, wm, wh), ('shadow', 'middle', 'highlight')):
            r, g, b, bright, contrast = adjust[region]
            af = np.array([r, g, b], dtype=np.float32) / 100.0
            delta = (255.0 - orig) * np.maximum(af, 0.0) + orig * np.minimum(af, 0.0)
            rv = (orig + delta) * bright
            mean = rv.mean(axis=(0,1), keepdims=True)
            rv = (rv - mean) * contrast + mean
            res += np.clip(rv, 0, 255) * w[..., None]
        pil = Image.fromarray(np.clip(res, 0, 255).astype(np.uint8))
        if alpha is not None:
            pil.putalpha(Image.fromarray(alpha))
        # Convert back to tensor batch
        out = np.array(pil).astype(np.float32) / 255.0
        if alpha is not None:
            alpha_arr = alpha.astype(np.float32) / 255.0
            out = np.concatenate((out, alpha_arr[..., None]), axis=2)
        return (out[None], {'Hires Color Adjust': adjust})

# Register node
NODE_CLASS_MAPPINGS = {
    "HiresColorAdjustmentNode": HiresColorAdjustmentNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "HiresColorAdjustmentNode": "Hires Color Adjustment",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]