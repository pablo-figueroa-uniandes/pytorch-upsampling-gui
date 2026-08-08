"""PIL <-> torch tensor <-> Tkinter PhotoImage conversions, plus preview
scaling. No GUI event handling here -- just data conversion helpers.
"""

import torch
from PIL import Image, ImageDraw, ImageTk
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


def magnify(image, center_x, center_y, radius, target_size):
    """Crop a (2*radius+1)-square pixel patch centered on (center_x,
    center_y) and scale it up to target_size with nearest-neighbor
    resampling, so individual source pixels are visible as flat blocks.
    A thin grid is overlaid on top of the pixel boundaries when each
    source pixel maps to a large enough block to make it useful.

    The center is clamped so the returned patch is always the full
    requested size (never distorted by clipping at the image border).
    """
    radius = max(1, int(round(radius)))
    box = radius * 2 + 1
    width, height = image.size

    if width <= box:
        left, right = 0, width
    else:
        cx = max(radius, min(center_x, width - radius - 1))
        left = int(round(cx - radius))
        right = left + box

    if height <= box:
        top, bottom = 0, height
    else:
        cy = max(radius, min(center_y, height - radius - 1))
        top = int(round(cy - radius))
        bottom = top + box

    patch = image.crop((left, top, right, bottom))
    patch = patch.resize((target_size, target_size), Image.NEAREST)

    cols, rows = right - left, bottom - top
    cell_w, cell_h = target_size / cols, target_size / rows
    if cell_w >= 6 and cell_h >= 6:
        draw = ImageDraw.Draw(patch)
        for i in range(1, cols):
            x = round(i * cell_w)
            draw.line([(x, 0), (x, target_size)], fill=(120, 120, 120))
        for j in range(1, rows):
            y = round(j * cell_h)
            draw.line([(0, y), (target_size, y)], fill=(120, 120, 120))

    return patch


def to_photoimage(image):
    """Wrap a PIL Image for Tkinter display. Caller must keep a strong
    reference (e.g. self._photo = ...) or Tkinter will blank the canvas
    once the PhotoImage is garbage-collected."""
    return ImageTk.PhotoImage(image)
