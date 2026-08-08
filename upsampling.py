"""Upsampling implementations: classic torch interpolation and pretrained
super-resolution. No GUI imports here -- kept independent of Tkinter so it
can be tested/reused on its own.
"""

import torch
import torch.nn.functional as F

CLASSIC_MODES = ["nearest", "bilinear", "bicubic", "area"]

# Each variant maps to (architecture, repo_id). The architecture selects
# which super_image model class knows how to load that repo's weights.
SR_VARIANTS = {
    "EDSR-base (fast)": ("edsr", "eugenesiow/edsr-base"),
    "EDSR (quality)": ("edsr", "eugenesiow/edsr"),
    "MSRN-BAM (fast)": ("msrn", "eugenesiow/msrn-bam"),
    "MSRN (quality)": ("msrn", "eugenesiow/msrn"),
    "DRLN-BAM (fast)": ("drln", "eugenesiow/drln-bam"),
    "DRLN (quality)": ("drln", "eugenesiow/drln"),
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


def _model_class(architecture):
    # Deferred: avoid importing opencv/huggingface_hub on startup.
    from super_image import DrlnModel, EdsrModel, MsrnModel

    return {"edsr": EdsrModel, "msrn": MsrnModel, "drln": DrlnModel}[architecture]


def get_sr_model(architecture, repo_id, scale):
    key = (repo_id, scale)
    if key not in _sr_model_cache:
        model = _model_class(architecture).from_pretrained(repo_id, scale=scale)
        model.eval()
        _sr_model_cache[key] = model
    return _sr_model_cache[key]


def sr_upsample(tensor, architecture, repo_id, scale):
    """Run a pretrained super-resolution model on a (1,3,H,W) tensor in [0,1]."""
    model = get_sr_model(architecture, repo_id, scale)
    with torch.no_grad():
        return model(tensor)
