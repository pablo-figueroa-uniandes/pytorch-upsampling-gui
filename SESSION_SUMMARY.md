# Session Summary

Built from scratch: a Tkinter + PyTorch GUI for comparing image
upsampling methods.

## What was built

- **Classic interpolation** (`upsampling.py`): nearest, bilinear,
  bicubic, area via `torch.nn.functional.interpolate`, with
  scale-factor, `align_corners`, and `antialias` controls.
- **Pretrained super-resolution**: EDSR, MSRN, and DRLN models
  (`super-image` package, weights from Hugging Face Hub) as a genuinely
  learned alternative to the classic methods, each with a fast "-BAM"
  variant and a higher-quality full variant, selectable model and scale
  (2x/3x/4x).
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

- **PDF write-up** (`docs/Upsampling_Theory_and_Code.pdf`): a 24-page
  document covering the theory behind each method (including a
  generated intensity-profile plot measuring bicubic's actual
  overshoot, and a five-way visual comparison rendered with the app's
  own code), an application walkthrough with screenshots, and a full
  annotated code listing. Built as HTML and rendered via headless
  Chrome.

## Key decisions

- Explicit "Apply" button rather than live-updating on every parameter
  change, so classic and SR methods share one code path.
- `super-image`'s pretrained models chosen over training from scratch —
  real pretrained weights, small dependency footprint. `SR_VARIANTS`
  maps each display name to an `(architecture, repo_id)` pair so
  `upsampling.py` can pick the right model class per variant.
- Magnifier logic verified by driving the app's event handlers directly
  in-process, since OS-level synthetic mouse clicks proved unreliable
  for automated testing in this environment. The same technique (plus
  Tkinter's own geometry info rather than OS window lookups) was reused
  to capture the app screenshots for the PDF deterministically.

## Repo

MIT licensed. Pushed to
[pytorch-upsampling-gui](https://github.com/pablo-figueroa-uniandes/pytorch-upsampling-gui).
See `README.md` for setup/run instructions, or
`docs/Upsampling_Theory_and_Code.pdf` for the full theory and code
write-up.
