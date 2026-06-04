# CLAUDE.md — Desktop Version

## Running

```bash
python3 app.py          # from this directory
# or double-click run.command in Finder
```

## Architecture

Single-file tkinter app (`app.py`).

### Startup flow
1. `main()` creates the Tk root and shows `SplashScreen` immediately
2. A daemon thread calls `load_or_train()` (fast pickle load, or full MNIST train)
3. On completion → `root.after(0, ...)` switches the UI to `DigitApp` on the main thread
4. Window is re-centered after the full UI is built

### Inference flow
- `predict()` disables the button and starts a daemon thread
- `_run_inference()` calls `preprocess()` + `model.predict_proba()` off the main thread
- Result is posted back via `root.after(0, ...)` — tkinter is never called from a worker thread

### Drawing surfaces
Two surfaces kept in sync on every stroke:
- **tk.Canvas** — for display (anti-aliased lines via `capstyle=ROUND`)
- **PIL Image** — for inference input (same pixel data, used by `preprocess()`)

### Preprocessing (`preprocess`)
Same pipeline as the web version: invert if white background → crop to bounding box → 20% padding → resize to 28×28 (LANCZOS) → 784-dim float32 vector.

## Keyboard shortcuts
| Key | Action |
|-----|--------|
| Enter / KP_Enter | Predict |
| Escape / Backspace | Clear |

## Dependencies
```
numpy  pillow  scikit-learn
```
`tkinter` is bundled with macOS Python. If missing: `brew install python-tk`.

## Key files
| File | Purpose |
|------|---------|
| `app.py` | Entire application |
| `run.command` | Finder launcher — pauses on error so you can read the message |
| `../digit_model.pkl` | Shared trained MLP (created on first run) |
| `../mnist.npz` | Shared MNIST cache (~11 MB) |
