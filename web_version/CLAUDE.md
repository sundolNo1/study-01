# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running

```bash
# From this directory
python3 app.py
# then open http://127.0.0.1:5001

# Or double-click in Finder
run.command
```

**Stop the server:**
```bash
kill $(lsof -ti tcp:5001)
```

## Architecture

Single-file Flask app (`app.py`). All HTML/CSS/JS is embedded as the `HTML` string constant — no `templates/` directory.

**Request flow:**
1. Browser loads `GET /` → receives the full SPA (canvas + JS)
2. User draws on the HTML5 Canvas (280×280, white-on-black)
3. On "Predict": JS sends `POST /predict` with `{ image: "data:image/png;base64,…" }`
4. `preprocess()` crops → pads → resizes to 28×28 → returns a 784-dim float32 vector
5. `model.predict_proba()` returns per-class probabilities → top-3 sent back as JSON
6. JS renders the result card with an animated confidence bar

**Model:** loaded from `../digit_model.pkl` (shared with the parent project). If the file is missing, `load_or_train()` downloads `mnist.npz` and trains from scratch (~2–5 min).

## Dependencies

```
flask  numpy  pillow  scikit-learn
```

## Key Files

| File | Purpose |
|------|---------|
| `app.py` | Entire application (Flask routes + HTML template + preprocessing) |
| `run.command` | Finder double-click launcher; auto-installs deps, opens browser |
| `../digit_model.pkl` | Shared trained MLP model (created by first run) |
| `../mnist.npz` | Shared MNIST dataset cache (~11 MB) |
