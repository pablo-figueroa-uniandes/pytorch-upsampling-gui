# PyTorch Upsampling Comparison

A small desktop GUI (Tkinter + PyTorch) for visually comparing different
ways of upsampling an image: classic interpolation methods and a real
pretrained deep-learning super-resolution model.

<p align="center">
  <em>Original vs. Result panes, a method dropdown, per-method parameters,
  and an Apply button.</em>
</p>

For a full write-up of the theory behind each method and a guided tour of
the code, see
[`docs/Upsampling_Theory_and_Code.pdf`](docs/Upsampling_Theory_and_Code.pdf).

## Running it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then either:

```bash
./run.sh
```

or manually:

```bash
source .venv/bin/activate
python main.py
```

The app opens with a synthetic sample image already loaded, so it's usable
immediately — no need to supply your own image. You can also click
`Browse...` to load any local image, or `Load Sample 2` for a second
synthetic test image.

The first time you select **Pretrained SR (EDSR)** as the method, the
model weights (a few MB) are downloaded once from the Hugging Face Hub and
cached locally; every run after that is instant.

### Pixel magnifier

Hover (or click/drag) over either the Original or Result image to open a
zoomed, pixel-level view of that area under both panes simultaneously — a
yellow box marks the sampled region on each image. Because moving over
either pane updates both magnifiers at the matching physical location,
you can directly compare, pixel by pixel, what a method actually did to
a specific area rather than judging the whole image at a glance.

## Project layout

| File | Responsibility |
|---|---|
| `main.py` | Entry point — creates the Tk root window and the `App`. |
| `gui.py` | All Tkinter widgets, layout, event handling, and the background-thread glue that keeps the UI responsive while an image is processed. |
| `upsampling.py` | The actual upsampling implementations. No Tkinter imports — pure functions over PyTorch tensors. |
| `image_utils.py` | Conversions between PIL images, PyTorch tensors, and Tkinter-displayable images, plus preview scaling. |
| `sample_images.py` | Generates two synthetic images in memory (no bundled image files, no licensing concerns). |

The GUI layer never touches image data directly — it hands a PIL image to
`image_utils`, gets a tensor, passes it to `upsampling`, and converts the
result tensor back to something it can draw. This keeps the
"how do you upsample an image" logic testable and reusable independent of
the GUI.

### Why a background thread?

Tkinter's event loop runs on the main thread, and if `Apply` ran the
upsampling inline, the window would freeze while it computed — barely
noticeable for classic interpolation, but very noticeable for the SR model
(especially on the very first run, which also has to download weights
over the network). `gui.py` always runs the actual computation in a
background thread and polls a `queue.Queue` from the main thread via
`after(...)`, so the window stays responsive and shows a progress bar
regardless of which method is selected.

## The methods, and the theory behind them

An image of size `H x W` needs to become `sH x sW` for some scale factor
`s > 1`. Every method here answers the same question — *what color should
each new pixel be?* — differently.

### Nearest neighbor

For every pixel in the output, look up the single closest pixel in the
source image and copy its value directly. It's the cheapest possible
approach: no math beyond a coordinate lookup, and no new colors are ever
introduced — every output pixel's color already existed in the input.

The tradeoff is visible immediately: straight edges become staircases and
diagonal/curved edges look blocky, because many output pixels end up
copying the exact same source pixel. In the app, this is the most
"pixelated" looking result, especially at higher scale factors.

### Bilinear interpolation

Instead of snapping to the nearest source pixel, bilinear interpolation
maps each output pixel back to a (generally fractional) coordinate in the
source image, then takes a weighted average of the 4 nearest source
pixels — weighted linearly by how close the fractional coordinate is to
each of them in the x and y directions ("bi-linear" = linear in both
axes).

This produces smooth gradients and removes the staircasing of nearest
neighbor, at the cost of somewhat softening (blurring) sharp edges, since
edge pixels get blended with their neighbors.

### Bicubic interpolation

Bicubic extends the same idea to a 4x4 neighborhood (16 source pixels)
and fits a cubic polynomial through them rather than a straight line.
Cubic interpolation can curve, which lets it better approximate how
natural image intensity actually changes near an edge, typically
producing sharper results than bilinear with fewer visible blurring
artifacts.

The tradeoff is **overshoot/ringing**: because a cubic curve can dip below
or rise above the values it's interpolating between, bicubic output can
contain values slightly outside the original `[0, 1]` intensity range,
visible as faint halos near high-contrast edges. This is exactly why
`image_utils.tensor_to_pil` clamps values to `[0, 1]` before converting
back to a displayable image — without the clamp, those overshoot values
wrap around when cast to 8-bit color and produce visible artifacts.

### Area interpolation

`area` mode (PyTorch's implementation of what OpenCV calls
`INTER_AREA`) is designed for *downsampling* — each output pixel is
computed as the average of the source pixels that fall within its
corresponding area, which is the mathematically correct way to shrink an
image without aliasing. When used for *upsampling*, the "area" each
output pixel covers in the source is smaller than a single source pixel,
so it degenerates toward nearest-neighbor-like behavior. It's included in
this app deliberately as a pedagogical negative example: a method that's
well-suited to a different problem (shrinking) doesn't upsample well.

`align_corners` and `antialias` (exposed as checkboxes for
bilinear/bicubic only, since PyTorch's `F.interpolate` only supports them
for those modes) are two additional knobs worth understanding:

- **`align_corners`** changes how source and destination pixel grids are
  aligned when computing the fractional source coordinate for each output
  pixel. With it off (the default, and generally recommended), the grids
  are aligned by pixel *area* (corners of the outer pixels line up);
  with it on, the *corner pixel centers* are forced to line up exactly.
  The difference is subtle but shifts results slightly, especially near
  the image borders.
- **`antialias`** applies a low-pass filter before resampling, primarily
  intended to reduce aliasing when *downsampling*; for upsampling its
  effect is a mild extra smoothing.

### Pretrained super-resolution (EDSR)

All four methods above are purely mathematical resampling — they only
ever combine information that's already present in the source image. A
learned super-resolution model instead uses a convolutional neural
network trained on many pairs of (low-resolution, high-resolution)
images to learn what plausible fine detail *usually* looks like, and
synthesizes new detail accordingly.

This app uses **EDSR** (Enhanced Deep residual networks for Super-
Resolution) via the [`super-image`](https://pypi.org/project/super-image/)
package, which loads pretrained weights from the Hugging Face Hub
(`eugenesiow/edsr-base` and `eugenesiow/edsr`). EDSR is a stack of
residual blocks (convolution → ReLU → convolution, with a skip connection
added back to the block's input) operating at the low-resolution size,
followed by an upsampling stage at the end of the network. Learning at
the original resolution and only upsampling at the very end is more
efficient and lets the residual blocks focus purely on refining detail
rather than also having to reason at a larger spatial size throughout.

Because the network was trained to reconstruct realistic high-resolution
images, its output can look noticeably sharper and more detailed than any
of the classic methods at the same scale factor — but it can also
hallucinate detail that wasn't really there, since it's making a learned
guess rather than a mathematical guarantee. Like bicubic, its raw output
isn't guaranteed to stay within `[0, 1]`, which is why the same clamping
step in `image_utils.tensor_to_pil` is used for every method's output.

`Model` (EDSR-base vs. EDSR) and `Scale` (2x/3x/4x, the only scales with
published pretrained weights) are exposed as separate controls from the
classic methods' continuous scale slider, since the SR model only works
at the specific integer scales it was trained for.

## Error handling

- Loading a corrupt/unsupported file shows an error dialog instead of
  crashing.
- Very large images (>1600px on a side) are automatically downscaled
  before processing, with a status message noting it happened.
- If the SR model's weights can't be downloaded (e.g. no internet on
  first use), a friendly error dialog is shown instead of a raw
  traceback or a frozen window.
