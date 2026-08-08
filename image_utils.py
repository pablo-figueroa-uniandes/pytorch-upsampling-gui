"""PIL <-> torch tensor <-> Tkinter PhotoImage conversions, plus preview
scaling. No GUI event handling here -- just data conversion helpers.
"""

import torch
from PIL import Image, ImageTk
import torchvision.transforms.functional as TF


def load_image_from_path(path):
    """Load an image file as RGB. Raises on bad/corrupt files -- caller
    should catch (OSError, ValueError) and show an error dialog."""
    return Image.open(path).convert("RGB")


def guard_image_size(image, max_dim=1600):
    """Downscale in place (returns a possibly-new image) if either
    dimension exceeds max_dim. Returns (image, was_resized)."""
    if max(image.size) > max_dim:
        image = image.copy()
        image.thumbnail((max_dim, max_dim), Image.LANCZOS)
        return image, True
    return image, False


def pil_to_tensor(image):
    """RGB PIL Image -> (1,3,H,W) float32 tensor in [0,1]."""
    return TF.to_tensor(image).unsqueeze(0)


def tensor_to_pil(tensor):
    """(1,3,H,W) float tensor -> RGB PIL Image.

    Clamps to [0,1] before converting: bicubic overshoot/ringing and SR
    model outputs can exceed that range, which would otherwise wrap
    around when cast to uint8.
    """
    tensor = tensor.squeeze(0).clamp(0, 1)
    array = tensor.mul(255).round().to("cpu", dtype=torch.uint8)
    array = array.permute(1, 2, 0).numpy()
    return Image.fromarray(array, "RGB")


def make_preview(image, max_size=(480, 480)):
    """Display-only downscaled copy. Actual upsampling always runs on the
    full-resolution source image, independent of this."""
    preview = image.copy()
    preview.thumbnail(max_size, Image.LANCZOS)
    return preview


def to_photoimage(image):
    """Wrap a PIL Image for Tkinter display. Caller must keep a strong
    reference (e.g. self._photo = ...) or Tkinter will blank the canvas
    once the PhotoImage is garbage-collected."""
    return ImageTk.PhotoImage(image)
