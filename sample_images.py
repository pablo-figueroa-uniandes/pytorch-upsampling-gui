"""Synthetic sample images generated in-memory with PIL -- no bundled
files, no licensing concerns. Deliberately small (~160px) so upsampled
differences between methods are visually obvious and SR inference stays
fast on CPU.
"""

from PIL import Image, ImageDraw, ImageFilter


def make_test_chart(size=160):
    """Checkerboard, concentric circles, a gradient band, and text --
    good for showing aliasing, ringing, and edge-sharpness differences
    between upsampling methods."""
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    half = size // 2

    # Top-left: checkerboard (aliasing / moire demo)
    cell = max(size // 20, 2)
    for y in range(0, half, cell):
        for x in range(0, half, cell):
            if ((x // cell) + (y // cell)) % 2 == 0:
                draw.rectangle([x, y, x + cell, y + cell], fill=(30, 30, 30))

    # Top-right: concentric circles (ringing / staircasing demo)
    cx, cy = half + half // 2, half // 2
    for r in range(half // 2, 0, -8):
        color = (200, 30, 30) if (r // 8) % 2 == 0 else (255, 220, 220)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)

    # Bottom-left: smooth horizontal gradient
    for x in range(half):
        shade = int(255 * x / half)
        draw.line([(x, half), (x, size)], fill=(shade, shade, 255))

    # Bottom-right: text (edge-sharpness demo)
    draw.rectangle([half, half, size, size], fill=(20, 90, 20))
    draw.text((half + 8, half + half // 2 - 8), "TEST", fill="white")

    return image


def make_soft_blobs(size=160):
    """Overlapping colored ellipses with Gaussian blur -- mimics soft
    photographic gradients, stressing methods differently than the sharp
    test chart."""
    image = Image.new("RGB", (size, size), (250, 245, 235))
    draw = ImageDraw.Draw(image)

    blobs = [
        (0.25, 0.30, 0.28, (220, 90, 90)),
        (0.65, 0.35, 0.32, (90, 140, 220)),
        (0.45, 0.70, 0.30, (110, 200, 130)),
        (0.75, 0.75, 0.20, (230, 200, 90)),
    ]
    for cx, cy, r, color in blobs:
        x, y, rad = cx * size, cy * size, r * size
        draw.ellipse([x - rad, y - rad, x + rad, y + rad], fill=color)

    return image.filter(ImageFilter.GaussianBlur(radius=2))
