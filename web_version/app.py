# Created: 2026-06-02 15:27
"""
Handwritten Digit Recognizer — Web Version
Flask server with an HTML5 canvas drawing interface.
Shares the trained model from the parent directory.
"""

import os
import io
import base64
import pickle
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, request, jsonify, render_template_string

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(PARENT_DIR, "digit_model.pkl")
NPZ_PATH   = os.path.join(PARENT_DIR, "mnist.npz")
PORT       = 5001

# ── HTML template ─────────────────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Digit Recognizer · Web</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --accent: #7c6af7;
      --accent2: #a78bfa;
      --bg: #0f0f1a;
      --card: #1a1a2e;
      --border: #2a2a4a;
      --text: #e0e0f0;
      --muted: #888;
    }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 20px;
      padding: 24px;
    }
    header { text-align: center; }
    h1 {
      font-size: 1.8rem;
      font-weight: 800;
      background: linear-gradient(135deg, var(--accent), var(--accent2));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .badge {
      display: inline-block;
      margin-top: 6px;
      padding: 2px 10px;
      border-radius: 99px;
      font-size: 0.7rem;
      font-weight: 600;
      letter-spacing: 1px;
      text-transform: uppercase;
      background: rgba(124,106,247,0.15);
      color: var(--accent2);
      border: 1px solid rgba(124,106,247,0.3);
    }
    .canvas-wrapper {
      border-radius: 16px;
      overflow: hidden;
      box-shadow: 0 0 0 2px #2a2a3a, 0 12px 48px rgba(0,0,0,0.6);
    }
    canvas {
      display: block;
      cursor: crosshair;
      background: #000;
      touch-action: none;
    }
    .toolbar {
      display: flex;
      align-items: center;
      gap: 16px;
      flex-wrap: wrap;
      justify-content: center;
    }
    .brush-row {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 0.85rem;
      color: var(--muted);
    }
    input[type=range] { accent-color: var(--accent); cursor: pointer; width: 100px; }
    .controls { display: flex; gap: 10px; }
    button {
      padding: 10px 26px;
      border: none;
      border-radius: 10px;
      font-size: 0.95rem;
      font-weight: 600;
      cursor: pointer;
      transition: transform 0.1s, box-shadow 0.2s;
    }
    button:active { transform: scale(0.96); }
    #predictBtn {
      background: linear-gradient(135deg, var(--accent), var(--accent2));
      color: #fff;
      box-shadow: 0 4px 16px rgba(124,106,247,0.4);
    }
    #predictBtn:hover { box-shadow: 0 6px 24px rgba(124,106,247,0.6); }
    #clearBtn { background: #2a2a3a; color: var(--muted); }
    #clearBtn:hover { background: #333350; color: var(--text); }

    .result-card {
      width: 300px;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 22px;
      text-align: center;
      min-height: 110px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 10px;
      transition: border-color 0.3s;
    }
    .result-card.active { border-color: var(--accent); }
    .result-label { font-size: 0.7rem; color: #555; text-transform: uppercase; letter-spacing: 1.5px; }
    .result-digit { font-size: 5.5rem; font-weight: 900; color: var(--accent2); line-height: 1; }
    .result-conf { font-size: 0.82rem; color: var(--muted); }
    .conf-track { width: 100%; height: 5px; background: #2a2a3a; border-radius: 3px; overflow: hidden; }
    .conf-fill {
      height: 100%;
      background: linear-gradient(90deg, var(--accent), var(--accent2));
      border-radius: 3px;
      transition: width 0.5s cubic-bezier(.4,0,.2,1);
      width: 0%;
    }
    .top-preds { display: flex; gap: 6px; flex-wrap: wrap; justify-content: center; margin-top: 4px; }
    .chip {
      background: #2a2a3a;
      border-radius: 8px;
      padding: 4px 10px;
      font-size: 0.75rem;
      color: var(--muted);
    }
    .chip b { color: var(--accent2); }
    .hint { font-size: 0.78rem; color: #444; }
  </style>
</head>
<body>
  <header>
    <h1>Handwritten Digit Recognizer</h1>
    <span class="badge">Web Version</span>
  </header>

  <div class="canvas-wrapper">
    <canvas id="canvas" width="280" height="280"></canvas>
  </div>

  <div class="toolbar">
    <div class="brush-row">
      Brush
      <input type="range" id="brush" min="8" max="40" value="20"/>
      <span id="brushLbl">20px</span>
    </div>
    <div class="controls">
      <button id="clearBtn">Clear</button>
      <button id="predictBtn">Predict ▶</button>
    </div>
  </div>

  <div class="result-card" id="card">
    <span class="hint">Draw a digit (0 – 9) and click Predict</span>
  </div>

  <script>
    const canvas = document.getElementById('canvas');
    const ctx    = canvas.getContext('2d');
    let drawing = false, brushSize = 20;

    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, 280, 280);

    document.getElementById('brush').addEventListener('input', e => {
      brushSize = +e.target.value;
      document.getElementById('brushLbl').textContent = brushSize + 'px';
    });

    const pos = e => {
      const r = canvas.getBoundingClientRect();
      const sx = 280 / r.width, sy = 280 / r.height;
      const src = e.touches ? e.touches[0] : e;
      return { x: (src.clientX - r.left) * sx, y: (src.clientY - r.top) * sy };
    };

    const start = e => { e.preventDefault(); drawing = true; const p = pos(e); ctx.beginPath(); ctx.moveTo(p.x, p.y); };
    const move  = e => {
      e.preventDefault(); if (!drawing) return;
      const p = pos(e);
      ctx.lineWidth = brushSize; ctx.lineCap = 'round'; ctx.strokeStyle = '#fff';
      ctx.lineTo(p.x, p.y); ctx.stroke(); ctx.beginPath(); ctx.moveTo(p.x, p.y);
    };
    const stop  = () => { drawing = false; ctx.beginPath(); };

    canvas.addEventListener('mousedown',  start);
    canvas.addEventListener('mousemove',  move);
    canvas.addEventListener('mouseup',    stop);
    canvas.addEventListener('mouseleave', stop);
    canvas.addEventListener('touchstart', start, { passive: false });
    canvas.addEventListener('touchmove',  move,  { passive: false });
    canvas.addEventListener('touchend',   stop);

    document.getElementById('clearBtn').addEventListener('click', () => {
      ctx.fillStyle = '#000'; ctx.fillRect(0, 0, 280, 280);
      const card = document.getElementById('card');
      card.className = 'result-card';
      card.innerHTML = '<span class="hint">Draw a digit (0 – 9) and click Predict</span>';
    });

    document.getElementById('predictBtn').addEventListener('click', async () => {
      const card = document.getElementById('card');
      card.innerHTML = '<span style="color:#7c6af7;font-size:.9rem">Analyzing…</span>';
      try {
        const res  = await fetch('/predict', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: canvas.toDataURL('image/png') })
        });
        const data = await res.json();
        if (data.error) { card.innerHTML = `<span style="color:#f87">${data.error}</span>`; return; }
        const pct    = (data.confidence * 100).toFixed(1);
        const chips  = data.top3.map(([d,p]) => `<div class="chip"><b>${d}</b> ${(p*100).toFixed(1)}%</div>`).join('');
        card.className = 'result-card active';
        card.innerHTML = `
          <span class="result-label">Predicted Digit</span>
          <div class="result-digit">${data.digit}</div>
          <span class="result-conf">Confidence: ${pct}%</span>
          <div class="conf-track"><div class="conf-fill" id="fill"></div></div>
          <div class="top-preds">${chips}</div>`;
        requestAnimationFrame(() => document.getElementById('fill').style.width = pct + '%');
      } catch(e) { card.innerHTML = `<span style="color:#f87">${e.message}</span>`; }
    });
  </script>
</body>
</html>
"""

# ── Flask app ─────────────────────────────────────────────────────────────────
app   = Flask(__name__)
model = None


def download_mnist():
    import urllib.request
    if not os.path.exists(NPZ_PATH):
        url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
        print(f"Downloading MNIST → {NPZ_PATH}")
        urllib.request.urlretrieve(url, NPZ_PATH)
    data = np.load(NPZ_PATH)
    X_train = data["x_train"].reshape(-1, 784).astype(np.float32) / 255.0
    y_train = data["y_train"].astype(int)
    X_test  = data["x_test"].reshape(-1, 784).astype(np.float32) / 255.0
    y_test  = data["y_test"].astype(int)
    return X_train, y_train, X_test, y_test


def train_and_save():
    X_train, y_train, X_test, y_test = download_mnist()
    print(f"Training MLP on {len(X_train):,} samples…")
    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128), activation="relu",
        max_iter=30, random_state=42, verbose=True,
        early_stopping=True, n_iter_no_change=5,
    )
    clf.fit(X_train, y_train)
    print(f"Test accuracy: {accuracy_score(y_test, clf.predict(X_test))*100:.2f}%")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    print(f"Model saved → {MODEL_PATH}")
    return clf


def load_or_train():
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}")
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return train_and_save()


def preprocess(b64: str) -> np.ndarray:
    """Crop, center, and resize canvas PNG to a 784-dim MNIST-compatible vector."""
    _, data = b64.split(",", 1)
    img = Image.open(io.BytesIO(base64.b64decode(data))).convert("L")
    if np.array(img).mean() > 127:
        img = ImageOps.invert(img)
    bbox = img.getbbox()
    if bbox is None:
        return np.zeros((1, 784), dtype=np.float32)
    img = img.crop(bbox)
    w, h = img.size
    pad  = max(int(max(w, h) * 0.2), 4)
    side = max(w, h) + pad * 2
    out  = Image.new("L", (side, side), 0)
    out.paste(img, (pad + (side - pad*2 - w)//2, pad + (side - pad*2 - h)//2))
    return (np.array(out.resize((28, 28), Image.LANCZOS), dtype=np.float32).flatten() / 255.0).reshape(1, -1)


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        b64 = request.get_json(force=True).get("image", "")
        if not b64:
            return jsonify({"error": "No image data"}), 400
        probs    = model.predict_proba(preprocess(b64))[0]
        top3_idx = np.argsort(probs)[::-1][:3]
        top3     = [(int(i), float(probs[i])) for i in top3_idx]
        return jsonify({"digit": top3[0][0], "confidence": top3[0][1], "top3": top3})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    model = load_or_train()
    print(f"\nServer → http://127.0.0.1:{PORT}\n")
    app.run(host="127.0.0.1", port=PORT, debug=False)
