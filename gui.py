"""Tkinter GUI: layout, widgets, event handlers, and threading glue.
Delegates all actual image processing to upsampling.py / image_utils.py.
"""

import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import image_utils
import sample_images
import upsampling

PANE_SIZE = (480, 480)

MAGNIFIER_DISPLAY = 220  # size (px) of the zoomed-pixel canvases
MAGNIFIER_RADIUS = 6     # half-width, in *original*-image pixels, of the sampled area


class App(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.source_image = None  # full-resolution PIL image
        self.result_image = None  # full-resolution upsampled PIL image (after Apply)
        self._original_photo = None
        self._result_photo = None
        self._original_zoom_photo = None
        self._result_zoom_photo = None
        self._original_view = None  # {"image", "preview_size", "scale"} for coord mapping
        self._result_view = None
        self._original_box_id = None
        self._result_box_id = None
        self._queue = queue.Queue()

        self._build_widgets()
        self.load_sample(1)

    # ---- layout -----------------------------------------------------

    def _build_widgets(self):
        top_bar = ttk.Frame(self)
        top_bar.pack(side="top", fill="x", padx=8, pady=6)
        ttk.Button(top_bar, text="Browse...", command=self._on_browse).pack(side="left")
        ttk.Button(top_bar, text="Load Sample 1",
                   command=lambda: self.load_sample(1)).pack(side="left", padx=(6, 0))
        ttk.Button(top_bar, text="Load Sample 2",
                   command=lambda: self.load_sample(2)).pack(side="left", padx=(6, 0))

        panes = ttk.Frame(self)
        panes.pack(side="top", fill="both", expand=True, padx=8)

        original_col = ttk.Frame(panes)
        original_col.pack(side="left", padx=4)
        ttk.Label(original_col, text="Original").pack()
        self.original_canvas = tk.Canvas(original_col, width=PANE_SIZE[0],
                                          height=PANE_SIZE[1], background="#222")
        self.original_canvas.pack()
        ttk.Label(original_col, text="Zoomed pixels (hover or click either image)").pack(pady=(6, 0))
        self.original_zoom_canvas = tk.Canvas(original_col, width=MAGNIFIER_DISPLAY,
                                               height=MAGNIFIER_DISPLAY, background="#333")
        self.original_zoom_canvas.pack()

        result_col = ttk.Frame(panes)
        result_col.pack(side="left", padx=4)
        ttk.Label(result_col, text="Result").pack()
        self.result_canvas = tk.Canvas(result_col, width=PANE_SIZE[0],
                                        height=PANE_SIZE[1], background="#222")
        self.result_canvas.pack()
        ttk.Label(result_col, text="Zoomed pixels (hover or click either image)").pack(pady=(6, 0))
        self.result_zoom_canvas = tk.Canvas(result_col, width=MAGNIFIER_DISPLAY,
                                             height=MAGNIFIER_DISPLAY, background="#333")
        self.result_zoom_canvas.pack()

        for event in ("<Motion>", "<Button-1>", "<B1-Motion>"):
            self.original_canvas.bind(event, lambda e: self._on_hover("original", e))
            self.result_canvas.bind(event, lambda e: self._on_hover("result", e))
        self.original_canvas.bind("<Leave>", lambda e: self._on_leave())
        self.result_canvas.bind("<Leave>", lambda e: self._on_leave())
        self._clear_zoom_canvas(self.original_zoom_canvas, "Hover an image")
        self._clear_zoom_canvas(self.result_zoom_canvas, "Hover an image")

        controls = ttk.Frame(self)
        controls.pack(side="top", fill="x", padx=8, pady=8)

        method_row = ttk.Frame(controls)
        method_row.pack(side="top", fill="x")
        ttk.Label(method_row, text="Method:").pack(side="left")
        self.method_names = upsampling.CLASSIC_MODES + ["Pretrained SR"]
        self.method_var = tk.StringVar(value="bicubic")
        method_box = ttk.Combobox(method_row, textvariable=self.method_var,
                                   values=self.method_names, state="readonly")
        method_box.pack(side="left", padx=(6, 0))
        method_box.bind("<<ComboboxSelected>>", lambda e: self._on_method_change())

        self.param_frame = ttk.Frame(controls)
        self.param_frame.pack(side="top", fill="x", pady=(8, 0))

        self._build_classic_params()
        self._build_sr_params()
        self._on_method_change()

        action_row = ttk.Frame(controls)
        action_row.pack(side="top", fill="x", pady=(8, 0))
        self.apply_btn = ttk.Button(action_row, text="Apply Upsampling",
                                     command=self._on_apply)
        self.apply_btn.pack(side="left")
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(action_row, textvariable=self.status_var).pack(side="left", padx=(10, 0))
        self.progress = ttk.Progressbar(action_row, mode="indeterminate", length=150)

    def _build_classic_params(self):
        frame = ttk.Frame(self.param_frame)
        self.classic_frame = frame

        scale_row = ttk.Frame(frame)
        scale_row.pack(side="top", fill="x")
        ttk.Label(scale_row, text="Scale factor:").pack(side="left")
        self.scale_var = tk.DoubleVar(value=2.0)
        self.scale_label_var = tk.StringVar(value="2.0x")
        ttk.Scale(scale_row, from_=1.0, to=6.0, variable=self.scale_var,
                  orient="horizontal", length=200,
                  command=self._on_scale_drag).pack(side="left", padx=(6, 6))
        ttk.Label(scale_row, textvariable=self.scale_label_var).pack(side="left")

        opts_row = ttk.Frame(frame)
        opts_row.pack(side="top", fill="x", pady=(4, 0))
        self.align_corners_var = tk.BooleanVar(value=False)
        self.antialias_var = tk.BooleanVar(value=False)
        self.align_corners_check = ttk.Checkbutton(
            opts_row, text="align_corners", variable=self.align_corners_var)
        self.antialias_check = ttk.Checkbutton(
            opts_row, text="antialias", variable=self.antialias_var)
        self.align_corners_check.pack(side="left")
        self.antialias_check.pack(side="left", padx=(10, 0))

    def _build_sr_params(self):
        frame = ttk.Frame(self.param_frame)
        self.sr_frame = frame

        variant_row = ttk.Frame(frame)
        variant_row.pack(side="top", fill="x")
        ttk.Label(variant_row, text="Model:").pack(side="left")
        self.sr_variant_var = tk.StringVar(value="EDSR-base (fast)")
        ttk.Combobox(variant_row, textvariable=self.sr_variant_var,
                     values=list(upsampling.SR_VARIANTS.keys()),
                     state="readonly").pack(side="left", padx=(6, 0))

        scale_row = ttk.Frame(frame)
        scale_row.pack(side="top", fill="x", pady=(4, 0))
        ttk.Label(scale_row, text="Scale:").pack(side="left")
        self.sr_scale_var = tk.IntVar(value=2)
        ttk.Combobox(scale_row, textvariable=self.sr_scale_var,
                     values=upsampling.SR_SCALES, state="readonly",
                     width=5).pack(side="left", padx=(6, 0))

    # ---- event handlers ----------------------------------------------

    def _on_scale_drag(self, value):
        self.scale_label_var.set(f"{float(value):.1f}x")

    def _on_method_change(self):
        is_sr = self.method_var.get() == "Pretrained SR"
        self.classic_frame.pack_forget()
        self.sr_frame.pack_forget()
        if is_sr:
            self.sr_frame.pack(side="top", fill="x")
        else:
            self.classic_frame.pack(side="top", fill="x")
            is_area_or_nearest = self.method_var.get() in ("nearest", "area")
            state = "disabled" if is_area_or_nearest else "normal"
            self.align_corners_check.config(state=state)
            self.antialias_check.config(state=state)

    def _on_browse(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif *.webp"),
                       ("All files", "*.*")])
        if not path:
            return
        try:
            image = image_utils.load_image_from_path(path)
        except (OSError, ValueError):
            messagebox.showerror("Load failed", f"Could not open image:\n{path}")
            return
        self._set_source_image(image)

    def load_sample(self, which):
        image = sample_images.make_test_chart() if which == 1 else sample_images.make_soft_blobs()
        self._set_source_image(image)

    def _set_source_image(self, image):
        image, was_resized = image_utils.guard_image_size(image)
        self.source_image = image
        if was_resized:
            self.status_var.set("Image was large; downscaled to fit.")
        preview = image_utils.make_preview(image, PANE_SIZE)
        self._original_photo = image_utils.to_photoimage(preview)
        self.original_canvas.delete("all")
        self.original_canvas.create_image(PANE_SIZE[0] // 2, PANE_SIZE[1] // 2,
                                           image=self._original_photo, anchor="center")
        self._original_view = self._make_view(image, preview)
        self._original_box_id = None

        # Previous result no longer corresponds to the new source image.
        self.result_image = None
        self._result_view = None
        self._result_box_id = None
        self.result_canvas.delete("all")
        self._on_leave()

    def _collect_params(self):
        method = self.method_var.get()
        if method == "Pretrained SR":
            architecture, repo_id = upsampling.SR_VARIANTS[self.sr_variant_var.get()]
            return {
                "method": "sr",
                "architecture": architecture,
                "repo_id": repo_id,
                "scale": self.sr_scale_var.get(),
            }
        return {
            "method": "classic",
            "mode": method,
            "scale_factor": self.scale_var.get(),
            "align_corners": self.align_corners_var.get(),
            "antialias": self.antialias_var.get(),
        }

    def _on_apply(self):
        if self.source_image is None:
            return
        params = self._collect_params()
        self.apply_btn.config(state="disabled")
        self.progress.pack(side="left", padx=(10, 0))
        self.progress.start(10)
        if params["method"] == "sr":
            self.status_var.set("Running super-resolution (first use downloads the model)...")
        else:
            self.status_var.set("Processing...")

        source_image = self.source_image
        threading.Thread(target=self._worker, args=(source_image, params), daemon=True).start()
        self.after(100, self._poll_result)

    def _worker(self, source_image, params):
        try:
            tensor = image_utils.pil_to_tensor(source_image)
            if params["method"] == "sr":
                out = upsampling.sr_upsample(
                    tensor, params["architecture"], params["repo_id"], params["scale"])
            else:
                out = upsampling.classic_upsample(
                    tensor, params["mode"], params["scale_factor"],
                    params["align_corners"], params["antialias"])
            result_image = image_utils.tensor_to_pil(out)
            self._queue.put(("ok", result_image))
        except Exception as exc:
            self._queue.put(("error", str(exc)))

    def _poll_result(self):
        try:
            status, payload = self._queue.get_nowait()
        except queue.Empty:
            self.after(100, self._poll_result)
            return

        self.progress.stop()
        self.progress.pack_forget()
        self.apply_btn.config(state="normal")

        if status == "ok":
            self.status_var.set(f"Done ({payload.width}x{payload.height})")
            self._display_result(payload)
        else:
            self.status_var.set("Error")
            messagebox.showerror(
                "Upsampling failed",
                f"{payload}\n\nIf you were using the pretrained SR model, this may mean "
                "the model weights could not be downloaded (check your internet connection).")

    def _display_result(self, image):
        self.result_image = image
        preview = image_utils.make_preview(image, PANE_SIZE)
        self._result_photo = image_utils.to_photoimage(preview)
        self.result_canvas.delete("all")
        self.result_canvas.create_image(PANE_SIZE[0] // 2, PANE_SIZE[1] // 2,
                                         image=self._result_photo, anchor="center")
        self._result_view = self._make_view(image, preview)
        self._result_box_id = None

    # ---- magnifying glass ---------------------------------------------

    @staticmethod
    def _make_view(full_image, preview):
        return {
            "image": full_image,
            "preview_size": preview.size,
            "scale": preview.width / full_image.width,
        }

    def _clear_zoom_canvas(self, canvas, message):
        canvas.delete("all")
        canvas.create_text(MAGNIFIER_DISPLAY // 2, MAGNIFIER_DISPLAY // 2,
                            text=message, fill="#999")

    def _on_leave(self):
        self._set_box(self.original_canvas, "_original_box_id", None)
        self._set_box(self.result_canvas, "_result_box_id", None)
        self._clear_zoom_canvas(self.original_zoom_canvas, "Hover an image")
        self._clear_zoom_canvas(self.result_zoom_canvas, "Hover an image")

    def _on_hover(self, pane, event):
        view = self._original_view if pane == "original" else self._result_view
        if view is None:
            return

        pw, ph = view["preview_size"]
        x0 = PANE_SIZE[0] / 2 - pw / 2
        y0 = PANE_SIZE[1] / 2 - ph / 2
        px, py = event.x - x0, event.y - y0
        if not (0 <= px < pw and 0 <= py < ph):
            self._on_leave()
            return

        fx, fy = px / view["scale"], py / view["scale"]
        if pane == "original":
            ox, oy = fx, fy
        else:
            rx = self.source_image.width / self.result_image.width
            ry = self.source_image.height / self.result_image.height
            ox, oy = fx * rx, fy * ry

        self._update_magnifiers(ox, oy)

    def _update_magnifiers(self, ox, oy):
        orig_patch = image_utils.magnify(self.source_image, ox, oy,
                                          MAGNIFIER_RADIUS, MAGNIFIER_DISPLAY)
        self._original_zoom_photo = image_utils.to_photoimage(orig_patch)
        self.original_zoom_canvas.delete("all")
        self.original_zoom_canvas.create_image(0, 0, image=self._original_zoom_photo,
                                                anchor="nw")
        self._set_box(self.original_canvas, "_original_box_id",
                      self._preview_rect(self._original_view, ox, oy, MAGNIFIER_RADIUS))

        if self.result_image is None:
            self._clear_zoom_canvas(self.result_zoom_canvas, "Click Apply first")
            self._set_box(self.result_canvas, "_result_box_id", None)
            return

        rx = self.result_image.width / self.source_image.width
        ry = self.result_image.height / self.source_image.height
        result_patch = image_utils.magnify(self.result_image, ox * rx, oy * ry,
                                            MAGNIFIER_RADIUS * max(rx, ry), MAGNIFIER_DISPLAY)
        self._result_zoom_photo = image_utils.to_photoimage(result_patch)
        self.result_zoom_canvas.delete("all")
        self.result_zoom_canvas.create_image(0, 0, image=self._result_zoom_photo, anchor="nw")
        self._set_box(self.result_canvas, "_result_box_id",
                      self._preview_rect(self._result_view, ox * rx, oy * ry,
                                         MAGNIFIER_RADIUS * max(rx, ry)))

    @staticmethod
    def _preview_rect(view, cx, cy, radius):
        pw, ph = view["preview_size"]
        scale = view["scale"]
        x0 = PANE_SIZE[0] / 2 - pw / 2
        y0 = PANE_SIZE[1] / 2 - ph / 2
        return (x0 + (cx - radius) * scale, y0 + (cy - radius) * scale,
                x0 + (cx + radius + 1) * scale, y0 + (cy + radius + 1) * scale)

    def _set_box(self, canvas, attr, coords):
        box_id = getattr(self, attr)
        if coords is None:
            if box_id is not None:
                canvas.delete(box_id)
                setattr(self, attr, None)
            return
        if box_id is None:
            box_id = canvas.create_rectangle(*coords, outline="#ffcc00", width=2)
            setattr(self, attr, box_id)
        else:
            canvas.coords(box_id, *coords)
