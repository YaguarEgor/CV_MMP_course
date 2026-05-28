import tkinter as tk
from tkinter import filedialog, ttk

from PIL import Image, ImageTk

from debug_pipeline import TileSegmentationPipeline


PREVIEW_MAX = (720, 420)
TILE_PREVIEW_SIZE = (220, 220)


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Tantrix Debug UI")
        self.root.geometry("1800x1000")

        self.image_path = None
        self.tile_line_debug = []
        self.route_summary = None
        self.tk_refs = []

        self.pipeline = TileSegmentationPipeline()

        self._build_ui()

    def _build_ui(self):
        shell = ttk.Frame(self.root)
        shell.pack(fill=tk.BOTH, expand=True)

        self.page_canvas = tk.Canvas(shell, highlightthickness=0)
        self.page_scroll = ttk.Scrollbar(shell, orient=tk.VERTICAL, command=self.page_canvas.yview)
        self.page_inner = ttk.Frame(self.page_canvas, padding=8)

        self.page_inner.bind("<Configure>", self._on_page_configure)
        self.page_canvas.bind("<Configure>", self._on_canvas_configure)
        self.page_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        self.page_window = self.page_canvas.create_window((0, 0), window=self.page_inner, anchor="nw")
        self.page_canvas.configure(yscrollcommand=self.page_scroll.set)
        self.page_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.page_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        top = ttk.Frame(self.page_inner)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(top, text="Open Image", command=self.open_image).pack(side=tk.LEFT)
        ttk.Button(top, text="Refresh Debug View", command=self.process_current_image).pack(side=tk.LEFT, padx=8)

        self.status_var = tk.StringVar(value="Select an image")
        ttk.Label(top, textvariable=self.status_var).pack(side=tk.LEFT, padx=12)

        self.route_var = tk.StringVar(value="")
        ttk.Label(self.page_inner, textvariable=self.route_var, justify=tk.LEFT, wraplength=1600).pack(
            side=tk.TOP,
            fill=tk.X,
            pady=(6, 8),
        )

        main = ttk.Frame(self.page_inner)
        main.pack(fill=tk.BOTH, expand=True)

        previews = ttk.Panedwindow(main, orient=tk.HORIZONTAL)
        previews.pack(fill=tk.X, pady=(0, 8))

        original_frame, self.original_panel = self._make_image_panel(previews, "Original Image")
        mask_frame, self.tiles_mask_panel = self._make_image_panel(previews, "Tiles Mask")
        previews.add(original_frame, weight=1)
        previews.add(mask_frame, weight=1)

        tiles_box = ttk.LabelFrame(main, text="Normalized Tiles With Line Overlays", padding=8)
        tiles_box.pack(fill=tk.BOTH, expand=True)
        self.tiles_inner = ttk.Frame(tiles_box)
        self.tiles_inner.pack(fill=tk.BOTH, expand=True)

    def _make_image_panel(self, parent, title):
        frame = ttk.LabelFrame(parent, text=title, padding=8)
        label = ttk.Label(frame)
        label.pack(fill=tk.BOTH, expand=True)
        return frame, label

    def _on_page_configure(self, _event):
        self.page_canvas.configure(scrollregion=self.page_canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.page_canvas.itemconfigure(self.page_window, width=event.width)

    def _on_mousewheel(self, event):
        if self.page_canvas.winfo_exists():
            self.page_canvas.yview_scroll(int(-event.delta / 120), "units")

    def open_image(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Images", "*.bmp *.png *.jpg *.jpeg"), ("All files", "*.*")],
        )
        if not path:
            return

        self.image_path = path
        self.status_var.set(path)
        self.process_current_image()

    def process_current_image(self):
        if not self.image_path:
            return

        try:
            debug_result = self.pipeline.process_path(self.image_path)
        except Exception as exc:
            self.status_var.set(str(exc))
            return

        self.tile_line_debug = debug_result.tile_line_debug
        self.route_summary = debug_result.route_summary

        self._set_preview(self.original_panel, debug_result.original_rgb)
        self._set_preview(self.tiles_mask_panel, debug_result.tiles_mask_rgb)
        self._render_tiles()

        self._update_status()

    def _set_preview(self, panel: ttk.Label, image_rgb):
        pil = Image.fromarray(image_rgb)
        pil.thumbnail(PREVIEW_MAX, Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(pil)
        panel.configure(image=tk_img)
        panel.image = tk_img

    def _render_tiles(self):
        for child in self.tiles_inner.winfo_children():
            child.destroy()
        self.tk_refs.clear()

        if not self.tile_line_debug:
            ttk.Label(self.tiles_inner, text="Nothing to show yet").grid(row=0, column=0, padx=6, pady=6)
            return

        cols = 3
        for idx, line_debug in enumerate(self.tile_line_debug):
            title = f"Tile {idx + 1}"
            if line_debug.tile_number is not None:
                title += f" -> #{line_debug.tile_number}"
            frame = ttk.LabelFrame(self.tiles_inner, text=title, padding=6)
            row, col = divmod(idx, cols)
            frame.grid(row=row, column=col, padx=6, pady=6, sticky="nsew")

            pil = Image.fromarray(line_debug.overlay_rgb)
            pil.thumbnail(TILE_PREVIEW_SIZE, Image.Resampling.LANCZOS)
            tk_img = ImageTk.PhotoImage(pil)
            self.tk_refs.append(tk_img)

            label = ttk.Label(frame, image=tk_img)
            label.pack()

            for summary in line_debug.classification_lines:
                ttk.Label(
                    frame,
                    text=summary,
                    anchor="w",
                    justify=tk.LEFT,
                    wraplength=TILE_PREVIEW_SIZE[0],
                ).pack(fill=tk.X, pady=(4, 0))

    def _update_status(self):
        if not self.tile_line_debug:
            self.status_var.set("No tiles found")
            self.route_var.set("")
            return
        self.status_var.set(f"Tiles found: {len(self.tile_line_debug)}")
        self.route_var.set(self.route_summary or "")


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
