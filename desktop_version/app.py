# Created: 2026-06-02
"""
Handwritten Digit Recognizer — Desktop Version
Native tkinter GUI. No browser required.
"""

import os
import pickle
import threading
import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(PARENT_DIR, "digit_model.pkl")
NPZ_PATH   = os.path.join(PARENT_DIR, "mnist.npz")

# ── Design tokens ──────────────────────────────────────────────────────────────
BG      = "#0f0f1a"
CARD    = "#1a1a2e"
BORDER  = "#2a2a4a"
ACCENT  = "#7c6af7"
ACCENT2 = "#a78bfa"
TEXT    = "#e0e0f0"
MUTED   = "#666680"
BTN_CLR = "#2a2a3a"
RED     = "#ff8877"
WHITE   = "#ffffff"

CANVAS_PX = 280
BRUSH_DEF = 20
BRUSH_MIN = 8
BRUSH_MAX = 40


# ── Model helpers ──────────────────────────────────────────────────────────────

def _download_mnist():
    import urllib.request
    if not os.path.exists(NPZ_PATH):
        url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
        print(f"Downloading MNIST → {NPZ_PATH}")
        urllib.request.urlretrieve(url, NPZ_PATH)
    data    = np.load(NPZ_PATH)
    X_train = data["x_train"].reshape(-1, 784).astype(np.float32) / 255.0
    y_train = data["y_train"].astype(int)
    X_test  = data["x_test"].reshape(-1, 784).astype(np.float32) / 255.0
    y_test  = data["y_test"].astype(int)
    return X_train, y_train, X_test, y_test


def _train_and_save():
    X_train, y_train, X_test, y_test = _download_mnist()
    print(f"Training MLP on {len(X_train):,} samples …")
    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128), activation="relu",
        max_iter=30, random_state=42, verbose=True,
        early_stopping=True, n_iter_no_change=5,
    )
    clf.fit(X_train, y_train)
    print(f"Test accuracy: {accuracy_score(y_test, clf.predict(X_test)) * 100:.2f}%")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    print(f"Model saved → {MODEL_PATH}")
    return clf


def load_or_train():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return _train_and_save()


def preprocess(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("L")
    if np.array(img).mean() > 127:
        img = ImageOps.invert(img)
    bbox = img.getbbox()
    if bbox is None:
        return np.zeros((1, 784), dtype=np.float32)
    img  = img.crop(bbox)
    w, h = img.size
    pad  = max(int(max(w, h) * 0.2), 4)
    side = max(w, h) + pad * 2
    out  = Image.new("L", (side, side), 0)
    out.paste(img, (pad + (side - pad * 2 - w) // 2,
                    pad + (side - pad * 2 - h) // 2))
    arr = np.array(out.resize((28, 28), Image.LANCZOS), dtype=np.float32)
    return (arr.flatten() / 255.0).reshape(1, -1)


# ── Splash screen ──────────────────────────────────────────────────────────────

class SplashScreen:
    """Shown while the model loads in a background thread."""

    def __init__(self, root: tk.Tk):
        self.root  = root
        self._dots = 0
        self._base = "Loading model"

        root.title("Digit Recognizer")
        root.configure(bg=BG)
        root.resizable(False, False)

        frame = tk.Frame(root, bg=BG, padx=60, pady=40)
        frame.pack()

        tk.Label(frame, text="Handwritten Digit Recognizer",
                 bg=BG, fg=ACCENT2,
                 font=("Helvetica", 18, "bold")).pack()
        tk.Label(frame, text="Desktop Version",
                 bg=BG, fg=MUTED,
                 font=("Helvetica", 10)).pack(pady=(2, 24))

        self._status_var = tk.StringVar(value=self._base)
        tk.Label(frame, textvariable=self._status_var,
                 bg=BG, fg=TEXT, font=("Helvetica", 12)).pack()

        self._after_id = None
        self._animate()

    def _animate(self):
        self._dots = (self._dots + 1) % 4
        self._status_var.set(self._base + "." * self._dots)
        self._after_id = self.root.after(350, self._animate)

    def cancel(self):
        if self._after_id:
            self.root.after_cancel(self._after_id)
            self._after_id = None


# ── Main application ───────────────────────────────────────────────────────────

class DigitApp:
    def __init__(self, root: tk.Tk, model):
        self.root  = root
        self.model = model
        self._busy = False

        self.pil_img  = Image.new("RGB", (CANVAS_PX, CANVAS_PX), "white")
        self.pil_draw = ImageDraw.Draw(self.pil_img)
        self.last_xy  = None

        self._build_ui()
        self._bind_keys()

    # ── UI construction ─────────────────────────────────────────────────────────

    def _build_ui(self):
        self.root.title("Handwritten Digit Recognizer · Desktop")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)

        for w in self.root.winfo_children():
            w.destroy()

        # Header
        hdr = tk.Frame(self.root, bg=BG)
        hdr.pack(pady=(20, 6))
        tk.Label(hdr, text="Handwritten Digit Recognizer",
                 bg=BG, fg=ACCENT2,
                 font=("Helvetica", 18, "bold")).pack()
        tk.Label(hdr, text="Desktop Version",
                 bg=BG, fg=MUTED,
                 font=("Helvetica", 10)).pack(pady=(2, 0))

        # Drawing canvas
        cf = tk.Frame(self.root, bg=BORDER, padx=2, pady=2)
        cf.pack()
        self.canvas = tk.Canvas(cf, width=CANVAS_PX, height=CANVAS_PX,
                                bg="white", cursor="crosshair",
                                highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>",   self._press)
        self.canvas.bind("<B1-Motion>",       self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._release)

        # Brush slider
        sf = tk.Frame(self.root, bg=BG)
        sf.pack(pady=(10, 2))
        tk.Label(sf, text="Brush:", bg=BG, fg=MUTED,
                 font=("Helvetica", 10)).pack(side=tk.LEFT, padx=(0, 6))
        self.brush_var = tk.IntVar(value=BRUSH_DEF)
        tk.Scale(sf, from_=BRUSH_MIN, to=BRUSH_MAX, orient=tk.HORIZONTAL,
                 variable=self.brush_var, command=self._on_brush,
                 bg=BG, fg=TEXT, troughcolor=CARD,
                 highlightthickness=0, bd=0, sliderrelief=tk.FLAT,
                 activebackground=ACCENT2, length=160).pack(side=tk.LEFT)
        self.brush_lbl = tk.Label(sf, text=f"{BRUSH_DEF}px",
                                  bg=BG, fg=MUTED, font=("Helvetica", 10), width=4)
        self.brush_lbl.pack(side=tk.LEFT, padx=(4, 0))

        # Action buttons
        bf = tk.Frame(self.root, bg=BG)
        bf.pack(pady=(8, 0))
        tk.Button(bf, text="Clear  ⌫",
                  bg=BTN_CLR, fg=MUTED, activebackground="#333350",
                  relief=tk.FLAT, bd=0, padx=20, pady=8,
                  font=("Helvetica", 11, "bold"),
                  cursor="hand2", command=self.clear).pack(side=tk.LEFT, padx=(0, 10))
        self.pred_btn = tk.Button(bf, text="Predict  ▶",
                                  bg=ACCENT, fg=WHITE, activebackground=ACCENT2,
                                  relief=tk.FLAT, bd=0, padx=20, pady=8,
                                  font=("Helvetica", 11, "bold"),
                                  cursor="hand2", command=self.predict)
        self.pred_btn.pack(side=tk.LEFT)

        # Result card
        self.card = tk.Frame(self.root, bg=WHITE,
                             highlightbackground=BORDER, highlightthickness=1)
        self.card.pack(fill=tk.X, padx=16, pady=(12, 20))
        self._show_idle()

    # ── Result card states ──────────────────────────────────────────────────────

    def _clear_card(self):
        for w in self.card.winfo_children():
            w.destroy()

    def _show_idle(self):
        self._clear_card()
        tk.Label(self.card, text="Draw a digit (0–9) and click Predict",
                 bg=WHITE, fg=MUTED, font=("Helvetica", 11)).pack(pady=20)

    def _show_analyzing(self):
        self._clear_card()
        tk.Label(self.card, text="Analyzing…",
                 bg=WHITE, fg=ACCENT, font=("Helvetica", 12)).pack(pady=20)

    def _show_result(self, top3):
        self._clear_card()
        digit, conf = top3[0]
        pct = conf * 100

        tk.Label(self.card, text=str(digit),
                 bg=WHITE, fg=ACCENT,
                 font=("Helvetica", 72, "bold")).pack(pady=(10, 0))
        tk.Label(self.card, text=f"Confidence: {pct:.1f}%",
                 bg=WHITE, fg=MUTED, font=("Helvetica", 10)).pack()

        # Confidence bar
        track = tk.Frame(self.card, bg=BTN_CLR, height=6, width=260)
        track.pack(pady=(8, 4))
        track.pack_propagate(False)
        fill = tk.Frame(track, bg=ACCENT2, height=6)
        fill.place(x=0, y=0, height=6, width=int(260 * conf))

        # Top-3 chips
        chips_frame = tk.Frame(self.card, bg=WHITE)
        chips_frame.pack(pady=(4, 14))
        for d, p in top3:
            chip = tk.Frame(chips_frame, bg=BTN_CLR)
            chip.pack(side=tk.LEFT, padx=3)
            tk.Label(chip, text=str(d),
                     bg=BTN_CLR, fg=ACCENT2,
                     font=("Helvetica", 11, "bold")).pack(side=tk.LEFT, padx=(8, 2), pady=5)
            tk.Label(chip, text=f"{p * 100:.1f}%",
                     bg=BTN_CLR, fg=MUTED,
                     font=("Helvetica", 9)).pack(side=tk.LEFT, padx=(0, 8), pady=5)

    def _show_error(self, msg):
        self._clear_card()
        tk.Label(self.card, text=f"Error: {msg}",
                 bg=WHITE, fg=RED, font=("Helvetica", 10),
                 wraplength=260).pack(pady=20)

    # ── Drawing ─────────────────────────────────────────────────────────────────

    def _press(self, event):
        self.last_xy = (event.x, event.y)
        r = self.brush_var.get() // 2
        x, y = event.x, event.y
        self.canvas.create_oval(x - r, y - r, x + r, y + r,
                                fill="black", outline="")
        self.pil_draw.ellipse([x - r, y - r, x + r, y + r], fill="black")
        self.canvas.update_idletasks()

    def _drag(self, event):
        if self.last_xy is None:
            return
        x0, y0 = self.last_xy
        x1, y1 = event.x, event.y
        w = self.brush_var.get()
        self.canvas.create_line(x0, y0, x1, y1,
                                fill="black", width=w,
                                capstyle=tk.ROUND, smooth=True)
        self.pil_draw.line([x0, y0, x1, y1], fill="black", width=w)
        self.last_xy = (x1, y1)
        self.canvas.update_idletasks()

    def _release(self, _event):
        self.last_xy = None

    def _on_brush(self, val):
        self.brush_lbl.config(text=f"{val}px")

    # ── Actions ─────────────────────────────────────────────────────────────────

    def _bind_keys(self):
        self.root.bind("<Return>",   lambda _: self.predict())
        self.root.bind("<KP_Enter>", lambda _: self.predict())
        self.root.bind("<Escape>",   lambda _: self.clear())
        self.root.bind("<BackSpace>", lambda _: self.clear())

    def clear(self):
        self.canvas.delete("all")
        self.pil_img  = Image.new("RGB", (CANVAS_PX, CANVAS_PX), "white")
        self.pil_draw = ImageDraw.Draw(self.pil_img)
        self._show_idle()

    def predict(self):
        if self._busy:
            return
        self._busy = True
        self.pred_btn.config(state=tk.DISABLED)
        self._show_analyzing()
        snapshot = self.pil_img.copy()
        threading.Thread(target=self._run_inference,
                         args=(snapshot,), daemon=True).start()

    def _run_inference(self, img):
        try:
            x     = preprocess(img)
            probs = self.model.predict_proba(x)[0]
            idx   = np.argsort(probs)[::-1][:3]
            top3  = [(int(i), float(probs[i])) for i in idx]
            self.root.after(0, lambda: self._on_done(top3, None))
        except Exception as exc:
            self.root.after(0, lambda: self._on_done(None, str(exc)))

    def _on_done(self, top3, error):
        self._busy = False
        self.pred_btn.config(state=tk.NORMAL)
        if error:
            self._show_error(error)
        else:
            self._show_result(top3)


# ── Entry point ────────────────────────────────────────────────────────────────

def _center(root: tk.Tk):
    root.update_idletasks()
    w = root.winfo_reqwidth()
    h = root.winfo_reqheight()
    x = (root.winfo_screenwidth()  - w) // 2
    y = (root.winfo_screenheight() - h) // 2
    root.geometry(f"+{x}+{y}")


def main():
    root   = tk.Tk()
    splash = SplashScreen(root)
    _center(root)

    def _load():
        try:
            model = load_or_train()
            root.after(0, lambda: _ready(model))
        except Exception as exc:
            root.after(0, lambda: _failed(str(exc)))

    def _ready(model):
        splash.cancel()
        DigitApp(root, model)
        _center(root)

    def _failed(msg):
        splash.cancel()
        for w in root.winfo_children():
            w.destroy()
        root.configure(bg=BG)
        tk.Label(root, text="Failed to load model",
                 bg=BG, fg=RED,
                 font=("Helvetica", 14, "bold")).pack(pady=(30, 8))
        tk.Label(root, text=msg, bg=BG, fg=TEXT,
                 font=("Helvetica", 10), wraplength=320).pack(padx=24, pady=(0, 30))

    threading.Thread(target=_load, daemon=True).start()
    root.mainloop()


if __name__ == "__main__":
    main()
