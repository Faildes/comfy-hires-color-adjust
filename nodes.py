import numpy as np
from PIL import Image
import torch

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
                "image": ("IMAGE",),
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
    RETURN_TYPES = ("IMAGE","JSON")
    RETURN_NAMES = ("image","metadata")
    FUNCTION = "process"

    def process(
        self,
        image,
        shadow_r, shadow_g, shadow_b, shadow_brightness, shadow_contrast,
        middle_r, middle_g, middle_b, middle_brightness, middle_contrast,
        highlight_r, highlight_g, highlight_b, highlight_brightness, highlight_contrast
    ):
        # Convert IMAGE batch (torch.Tensor) to numpy uint8
        samples = image.numpy() if hasattr(image, 'numpy') else image
        samples = np.clip(samples * 255.0, 0, 255).astype(np.uint8) if samples.dtype in [np.float32, np.float64] else samples.astype(np.uint8)
        
        # Build adjustments
        adjust = {
            'shadow':    [shadow_r, shadow_g, shadow_b, shadow_brightness, shadow_contrast],
            'middle':    [middle_r, middle_g, middle_b, middle_brightness, middle_contrast],
            'highlight': [highlight_r, highlight_g, highlight_b, highlight_brightness, highlight_contrast],
        }
        
        for sample_id in range(samples.shape[0]):
            rgb = samples[sample_id][..., :3]
            pil = Image.fromarray(rgb)
            
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
            
            # Put back the modified rgb values to the sample
            samples[sample_id][..., :3] = np.array(pil, dtype=np.uint8)

        # Convert back to torch.Tensor
        final = np.clip((samples.astype(np.float32) / 255.0), 0.0, 1.0)
        samples = torch.from_numpy(final).float() if not isinstance(samples, torch.Tensor) else final
        return (samples, {"Hires Color Adjust": adjust})

# Register node
NODE_CLASS_MAPPINGS = {
    "HiresColorAdjustmentNode": HiresColorAdjustmentNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "HiresColorAdjustmentNode": "Hires Color Adjustment",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
