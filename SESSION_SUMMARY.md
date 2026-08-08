# Session Summary

Built from scratch: a Tkinter + PyTorch GUI for comparing image
upsampling methods.

## What was built

- **Classic interpolation** (`upsampling.py`): nearest, bilinear,
  bicubic, area via `torch.nn.functional.interpolate`, with
  scale-factor, `align_corners`, and `antialias` controls.
- **Pretrained super-resolution**: EDSR models (`super-image` package,
  weights from Hugging Face Hub) as a genuinely learned alternative to
  the classic methods, selectable model variant and scale (2x/3x/4x).
- **GUI** (`gui.py`): Original/Result canvases, method dropdown with
  per-method parameter panels, Apply button that runs processing on a
  background thread (via `queue.Queue` + `after()` polling) so the UI
  never freezes, including during first-run model downloads.
- **Pixel magnifier**: hovering/clicking/dragging either image shows a
  zoomed, pixel-grid view of the same physical area in both panes at
  once, with a marker box on each image.
- **Sample images** (`sample_images.py`): two images generated in
  memory with PIL (test chart + soft blobs) so the app is usable with
  no external files.
- **Image conversions** (`image_utils.py`): PIL ↔ tensor ↔ PhotoImage,
  with output clamped to `[0,1]` before display (bicubic overshoot and
  SR outputs can exceed that range).

## Key decisions

- Explicit "Apply" button rather than live-updating on every parameter
  change, so classic and SR methods share one code path.
- `super-image`'s EDSR chosen over training a model from scratch —
  real pretrained weights, small dependency footprint.
- Magnifier logic verified by driving the app's event handlers directly
  in-process, since OS-level synthetic mouse clicks proved unreliable
  for automated testing in this environment.

## Repo

MIT licensed. Pushed to
[pytorch-upsampling-gui](https://github.com/pablo-figueroa-uniandes/pytorch-upsampling-gui).
See `README.md` for setup/run instructions and the theory behind each
method.
