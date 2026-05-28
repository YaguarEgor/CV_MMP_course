import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional

import cv2
from PIL import Image, ImageTk

from Lab1_2 import process_for_gui


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FastSAM GUI (Tkinter)")
        self.geometry("1280x820")
        self.minsize(1000, 700)

        self.original_pil: Optional[Image.Image] = None
        self.current_pil: Optional[Image.Image] = None
        self.current_tk: Optional[ImageTk.PhotoImage] = None
        self.image_path: Optional[str] = None

        self.auto_update_job = None

        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self, padding=10)
        top.pack(fill="x")

        self.btn_load = ttk.Button(top, text="Загрузить изображение…", command=self.on_load)
        self.btn_load.pack(side="left")

        self.btn_segment = ttk.Button(top, text="Детекция", command=self.on_segment, state="disabled")
        self.btn_segment.pack(side="left", padx=(10, 0))

        self.btn_update = ttk.Button(top, text="Обновить", command=self.on_segment, state="disabled")
        self.btn_update.pack(side="left", padx=(10, 0))

        self.btn_reset = ttk.Button(top, text="Сброс", command=self.on_reset, state="disabled")
        self.btn_reset.pack(side="left", padx=(10, 0))

        ttk.Label(top, text="Режим:").pack(side="left", padx=(20, 6))

        self.view_mode = tk.StringVar(value="mask")
        self.cmb_view = ttk.Combobox(
            top,
            textvariable=self.view_mode,
            state="readonly",
            values=["mask", "panel"],
            width=12
        )
        self.cmb_view.pack(side="left")
        self.cmb_view.bind("<<ComboboxSelected>>", lambda e: self.on_segment() if self.image_path else None)

        main = ttk.Frame(self, padding=(10, 0, 10, 10))
        main.pack(fill="both", expand=True)

        main.columnconfigure(0, weight=4)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        left = ttk.Frame(main)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        self.canvas = tk.Canvas(left, bg="#111111", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        bottom = ttk.Labelframe(self, text="Результаты", padding=10)
        bottom.pack(fill="x", padx=10, pady=(0, 10))

        self.var_counts = tk.StringVar(value="Загрузи изображение.")
        self.lbl_counts = ttk.Label(bottom, textvariable=self.var_counts, justify="left")
        self.lbl_counts.pack(anchor="w")

        self.bind("<Configure>", self._on_resize)

    def on_load(self):
        path = filedialog.askopenfilename(
            title="Выбери изображение",
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        self.image_path = path

        try:
            pil = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось открыть файл:\n{e}")
            return

        self.original_pil = pil
        self.current_pil = pil

        self.var_counts.set("Изображение загружено. Нажми «Детекция».")
        self.btn_segment.config(state="normal")
        self.btn_update.config(state="normal")
        self.btn_reset.config(state="normal")

        self._render_current_image()

    def on_segment(self):
        if not self.image_path:
            return

        try:
            vis_bgr, result = process_for_gui(
                self.image_path,
                view_mode=self.view_mode.get(),
                params=None
            )

            vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
            self.current_pil = Image.fromarray(vis_rgb)
            self._render_current_image()

            eggs = result.get("eggs", 0)
            tomato_yellow = result.get("tomato_yellow", 0)
            tomato_red = result.get("tomato_red", 0)
            total = result.get("total", eggs + tomato_yellow + tomato_red)

            self.var_counts.set(
                f"Всего объектов: {total} | "
                f"Яиц: {eggs} | "
                f"Жёлтых томатов: {tomato_yellow} | "
                f"Красных томатов: {tomato_red}"
            )

        except Exception as e:
            messagebox.showerror("Ошибка детекции", str(e))

    def on_reset(self):
        if self.original_pil is None:
            return
        self.current_pil = self.original_pil
        self._render_current_image()
        self.var_counts.set("Сброшено к оригиналу. Нажми «Детекция» для повторной обработки.")

    def _on_resize(self, _event):
        if self.current_pil is not None:
            self._render_current_image()

    def _render_current_image(self):
        if self.current_pil is None:
            return

        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())

        if cw < 10 or ch < 10:
            self.after(50, self._render_current_image)
            return

        img = self.current_pil
        iw, ih = img.size

        scale = min(cw / iw, ch / ih)
        new_w = max(1, int(iw * scale))
        new_h = max(1, int(ih * scale))

        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.current_tk = ImageTk.PhotoImage(resized)

        self.canvas.delete("all")
        x = (cw - new_w) // 2
        y = (ch - new_h) // 2
        self.canvas.create_image(x, y, anchor="nw", image=self.current_tk)


if __name__ == "__main__":
    App().mainloop()