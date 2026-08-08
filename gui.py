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


class App(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.source_image = None  # full-resolution PIL image
        self._original_photo = None
        self._result_photo = None
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

        result_col = ttk.Frame(panes)
        result_col.pack(side="left", padx=4)
        ttk.Label(result_col, text="Result").pack()
        self.result_canvas = tk.Canvas(result_col, width=PANE_SIZE[0],
                                        height=PANE_SIZE[1], background="#222")
        self.result_canvas.pack()

        controls = ttk.Frame(self)
        controls.pack(side="top", fill="x", padx=8, pady=8)

        method_row = ttk.Frame(controls)
        method_row.pack(side="top", fill="x")
        ttk.Label(method_row, text="Method:").pack(side="left")
        self.method_names = upsampling.CLASSIC_MODES + ["Pretrained SR (EDSR)"]
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
        is_sr = self.method_var.get() == "Pretrained SR (EDSR)"
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
        self.result_canvas.delete("all")

    def _collect_params(self):
        method = self.method_var.get()
        if method == "Pretrained SR (EDSR)":
            return {
                "method": "sr",
                "repo_id": upsampling.SR_VARIANTS[self.sr_variant_var.get()],
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
                out = upsampling.sr_upsample(tensor, params["repo_id"], params["scale"])
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
        preview = image_utils.make_preview(image, PANE_SIZE)
        self._result_photo = image_utils.to_photoimage(preview)
        self.result_canvas.delete("all")
        self.result_canvas.create_image(PANE_SIZE[0] // 2, PANE_SIZE[1] // 2,
                                         image=self._result_photo, anchor="center")
