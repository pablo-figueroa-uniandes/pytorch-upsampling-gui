"""Upsampling implementations: classic torch interpolation and pretrained
super-resolution. No GUI imports here -- kept independent of Tkinter so it
can be tested/reused on its own.
"""

import torch
import torch.nn.functional as F

CLASSIC_MODES = ["nearest", "bilinear", "bicubic", "area"]

SR_VARIANTS = {
    "EDSR-base (fast)": "eugenesiow/edsr-base",
    "EDSR (quality)": "eugenesiow/edsr",
}
SR_SCALES = [2, 3, 4]

_sr_model_cache = {}  # (repo_id, scale) -> loaded model


def classic_upsample(tensor, mode, scale_factor, align_corners, antialias):
    """Upsample a (1,3,H,W) float tensor in [0,1] using F.interpolate."""
    kwargs = {}
    if mode in ("bilinear", "bicubic"):
        kwargs["align_corners"] = align_corners
        kwargs["antialias"] = antialias
    with torch.no_grad():
        return F.interpolate(tensor, scale_factor=scale_factor, mode=mode, **kwargs)


def get_sr_model(repo_id, scale):
    from super_image import EdsrModel  # deferred: avoid importing opencv/hf on startup

    key = (repo_id, scale)
    if key not in _sr_model_cache:
        model = EdsrModel.from_pretrained(repo_id, scale=scale)
        model.eval()
        _sr_model_cache[key] = model
    return _sr_model_cache[key]


def sr_upsample(tensor, repo_id, scale):
    """Run a pretrained super-resolution model on a (1,3,H,W) tensor in [0,1]."""
    model = get_sr_model(repo_id, scale)
    with torch.no_grad():
        return model(tensor)
