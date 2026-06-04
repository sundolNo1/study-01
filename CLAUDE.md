# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the App

**Double-click launcher (Finder):**
```
손글씨 숫자 인식기.command   ← double-click in Finder
```

**From terminal:**
```bash
python3 digit_recognizer.py
# then open http://127.0.0.1:5001
```

**Stop a running server:**
```bash
kill $(lsof -ti tcp:5001)
```

**Force retrain the model** (delete the cached pickle first):
```bash
rm digit_model.pkl && python3 digit_recognizer.py
```

## Architecture

Everything lives in a single file (`digit_recognizer.py`) with three logical layers:

**1. Model layer** (`download_mnist` → `train_and_save` → `load_or_train`)
- On first run: downloads `mnist.npz` (~11 MB) from Google CDN, trains an `MLPClassifier(256, 128)` on 60k samples, pickles the result to `digit_model.pkl`.
- On subsequent runs: loads `digit_model.pkl` directly (fast).
- Both `mnist.npz` and `digit_model.pkl` are cached on disk beside the script.

**2. Preprocessing pipeline** (`preprocess_canvas`)
- Accepts a base64 PNG string from the browser canvas (280×280, white-on-black).
- Crops to the digit's bounding box → adds 20% padding → resizes to 28×28 (LANCZOS) → flattens to a 784-dim float32 vector matching MNIST format.
- Handles white-background canvases by auto-inverting when mean pixel > 127.

**3. Flask web layer** (`/` and `/predict`)
- The entire HTML/CSS/JS UI is embedded as the `HTML` string constant — no templates directory.
- `POST /predict` receives `{ image: "data:image/png;base64,..." }` and returns `{ digit, confidence, top3 }`.
- The global `model` variable is loaded at startup before `app.run()`.

## Dependencies

```
flask, numpy, pillow, scikit-learn
```

Install: `pip3 install flask numpy pillow scikit-learn`

The `.command` launcher auto-installs any missing packages on startup.

## Key Design Decisions

- **Single-file app**: HTML template is inlined as a Python string to keep the project self-contained.
- **Pickle for model persistence**: `digit_model.pkl` is not committed to git (large binary). Delete it to force a retrain.
- **Port 5001**: hardcoded in both `digit_recognizer.py` and the `.command` launcher — change both if you need a different port.
- **No test suite**: this is a study/demo project; accuracy is validated against the MNIST test set (10k samples) printed at training time.
