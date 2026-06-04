"""
Handwritten Digit Recognizer
Trains a neural network on MNIST, then serves a web interface
where users can draw digits and get real-time predictions.
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

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "digit_model.pkl")

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Handwritten Digit Recognizer</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: #0f0f1a;
      color: #e0e0f0;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 24px;
      padding: 24px;
    }
    h1 {
      font-size: 1.8rem;
      font-weight: 700;
      background: linear-gradient(135deg, #7c6af7, #a78bfa);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .subtitle { color: #888; font-size: 0.9rem; }
    .canvas-wrapper {
      border-radius: 16px;
      overflow: hidden;
      box-shadow: 0 0 0 2px #333, 0 8px 40px rgba(0,0,0,0.5);
    }
    canvas {
      display: block;
      cursor: crosshair;
      background: #000;
      touch-action: none;
    }
    .controls { display: flex; gap: 12px; }
    button {
      padding: 12px 28px;
      border: none;
      border-radius: 10px;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      transition: transform 0.1s;
    }
    button:active { transform: scale(0.97); }
    #predictBtn { background: linear-gradient(135deg, #7c6af7, #a78bfa); color: #fff; }
    #clearBtn { background: #2a2a3a; color: #aaa; }
    .result-box {
      width: 280px;
      background: #1a1a2e;
      border: 1px solid #2a2a4a;
      border-radius: 16px;
      padding: 20px 24px;
      text-align: center;
      min-height: 100px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 8px;
    }
    .result-label { font-size: 0.75rem; color: #666; text-transform: uppercase; letter-spacing: 1px; }
    .result-digit { font-size: 5rem; font-weight: 800; color: #a78bfa; line-height: 1; }
    .result-conf { font-size: 0.85rem; color: #888; }
    .conf-bar-bg {
      width: 100%; height: 6px; background: #2a2a3a;
      border-radius: 3px; overflow: hidden; margin-top: 4px;
    }
    .conf-bar {
      height: 100%;
      background: linear-gradient(90deg, #7c6af7, #a78bfa);
      border-radius: 3px;
      transition: width 0.4s ease;
      width: 0%;
    }
    .top-preds {
      width: 100%; margin-top: 8px;
      display: flex; gap: 6px; justify-content: center; flex-wrap: wrap;
    }
    .pred-chip {
      background: #2a2a3a; border-radius: 8px;
      padding: 4px 10px; font-size: 0.75rem; color: #aaa;
    }
    .pred-chip span { color: #a78bfa; font-weight: 700; }
    .brush-row {
      display: flex; align-items: center; gap: 10px;
      font-size: 0.85rem; color: #888;
    }
    input[type=range] { accent-color: #7c6af7; cursor: pointer; }
  </style>
</head>
<body>
  <h1>Handwritten Digit Recognizer</h1>
  <p class="subtitle">Draw a digit (0-9) on the canvas, then click Predict</p>

  <div class="canvas-wrapper">
    <canvas id="canvas" width="280" height="280"></canvas>
  </div>

  <div class="brush-row">
    Brush size:
    <input type="range" id="brushSize" min="8" max="40" value="20" />
    <span id="brushVal">20px</span>
  </div>

  <div class="controls">
    <button id="clearBtn">Clear</button>
    <button id="predictBtn">Predict</button>
  </div>

  <div class="result-box" id="resultBox">
    <span class="result-label">Draw a digit and click Predict</span>
  </div>

  <script>
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    let drawing = false;
    let brushSize = 20;

    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    document.getElementById('brushSize').addEventListener('input', e => {
      brushSize = parseInt(e.target.value);
      document.getElementById('brushVal').textContent = brushSize + 'px';
    });

    function getPos(e) {
      const rect = canvas.getBoundingClientRect();
      const sx = canvas.width / rect.width, sy = canvas.height / rect.height;
      if (e.touches) return { x: (e.touches[0].clientX - rect.left) * sx, y: (e.touches[0].clientY - rect.top) * sy };
      return { x: (e.clientX - rect.left) * sx, y: (e.clientY - rect.top) * sy };
    }

    canvas.addEventListener('mousedown', e => { drawing = true; const p = getPos(e); ctx.beginPath(); ctx.moveTo(p.x, p.y); });
    canvas.addEventListener('mousemove', e => {
      if (!drawing) return;
      const p = getPos(e);
      ctx.lineWidth = brushSize; ctx.lineCap = 'round'; ctx.strokeStyle = '#fff';
      ctx.lineTo(p.x, p.y); ctx.stroke(); ctx.beginPath(); ctx.moveTo(p.x, p.y);
    });
    canvas.addEventListener('mouseup', () => { drawing = false; ctx.beginPath(); });
    canvas.addEventListener('mouseleave', () => { drawing = false; ctx.beginPath(); });
    canvas.addEventListener('touchstart', e => { e.preventDefault(); drawing = true; const p = getPos(e); ctx.beginPath(); ctx.moveTo(p.x, p.y); }, { passive: false });
    canvas.addEventListener('touchmove', e => {
      e.preventDefault(); if (!drawing) return;
      const p = getPos(e);
      ctx.lineWidth = brushSize; ctx.lineCap = 'round'; ctx.strokeStyle = '#fff';
      ctx.lineTo(p.x, p.y); ctx.stroke(); ctx.beginPath(); ctx.moveTo(p.x, p.y);
    }, { passive: false });
    canvas.addEventListener('touchend', () => { drawing = false; ctx.beginPath(); });

    document.getElementById('clearBtn').addEventListener('click', () => {
      ctx.fillStyle = '#000'; ctx.fillRect(0, 0, canvas.width, canvas.height);
      document.getElementById('resultBox').innerHTML = '<span class="result-label">Draw a digit and click Predict</span>';
    });

    document.getElementById('predictBtn').addEventListener('click', async () => {
      const box = document.getElementById('resultBox');
      box.innerHTML = '<span style="color:#7c6af7">Analyzing...</span>';
      try {
        const resp = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: canvas.toDataURL('image/png') })
        });
        const data = await resp.json();
        if (data.error) { box.innerHTML = `<span style="color:#f87">Error: ${data.error}</span>`; return; }
        const conf = (data.confidence * 100).toFixed(1);
        const topHtml = data.top3.map(([d, p]) =>
          `<div class="pred-chip"><span>${d}</span> ${(p*100).toFixed(1)}%</div>`).join('');
        box.innerHTML = `
          <span class="result-label">Predicted Digit</span>
          <div class="result-digit">${data.digit}</div>
          <div class="result-conf">Confidence: ${conf}%</div>
          <div class="conf-bar-bg"><div class="conf-bar" id="cbar"></div></div>
          <div class="top-preds">${topHtml}</div>`;
        requestAnimationFrame(() => { document.getElementById('cbar').style.width = conf + '%'; });
      } catch(err) { box.innerHTML = `<span style="color:#f87">${err.message}</span>`; }
    });
  </script>
</body>
</html>
"""

app = Flask(__name__)
model = None


def download_mnist():
    """Download MNIST from Keras/Google CDN as a single .npz file."""
    import urllib.request
    npz_path = os.path.join(os.path.dirname(MODEL_PATH), "mnist.npz")
    if not os.path.exists(npz_path):
        url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
        print(f"Downloading MNIST from {url} ...")
        urllib.request.urlretrieve(url, npz_path)
        print("Download complete.")
    data = np.load(npz_path)
    X_train = data["x_train"].reshape(-1, 784).astype(np.float32) / 255.0
    y_train = data["y_train"].astype(int)
    X_test  = data["x_test"].reshape(-1, 784).astype(np.float32) / 255.0
    y_test  = data["y_test"].astype(int)
    return X_train, y_train, X_test, y_test


def train_and_save():
    X_train, y_train, X_test, y_test = download_mnist()
    print(f"Training MLP on {len(X_train):,} samples ...")
    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        max_iter=30,
        random_state=42,
        verbose=True,
        early_stopping=True,
        n_iter_no_change=5,
    )
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"Test accuracy: {acc * 100:.2f}%")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    print(f"Model saved to {MODEL_PATH}")
    return clf


def load_or_train():
    if os.path.exists(MODEL_PATH):
        print(f"Loading saved model from {MODEL_PATH} ...")
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return train_and_save()


def preprocess_canvas(b64_image: str) -> np.ndarray:
    """Convert base64 canvas PNG to a 784-dim MNIST-compatible float32 vector."""
    _, data = b64_image.split(",", 1)
    img = Image.open(io.BytesIO(base64.b64decode(data))).convert("L")

    # MNIST uses white digit on black background; invert if needed
    if np.array(img).mean() > 127:
        img = ImageOps.invert(img)

    bbox = img.getbbox()
    if bbox is None:
        return np.zeros((1, 784), dtype=np.float32)

    img = img.crop(bbox)
    w, h = img.size
    pad = max(int(max(w, h) * 0.2), 4)
    side = max(w, h) + pad * 2
    canvas_img = Image.new("L", (side, side), 0)
    canvas_img.paste(img, (pad + (side - pad * 2 - w) // 2,
                           pad + (side - pad * 2 - h) // 2))

    resized = canvas_img.resize((28, 28), Image.LANCZOS)
    return (np.array(resized, dtype=np.float32).flatten() / 255.0).reshape(1, -1)


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        b64 = request.get_json(force=True).get("image", "")
        if not b64:
            return jsonify({"error": "No image data"}), 400
        x = preprocess_canvas(b64)
        probs = model.predict_proba(x)[0]
        top3_idx = np.argsort(probs)[::-1][:3]
        top3 = [(int(i), float(probs[i])) for i in top3_idx]
        return jsonify({"digit": top3[0][0], "confidence": top3[0][1], "top3": top3})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    model = load_or_train()
    print("\nServer starting at http://127.0.0.1:5001")
    print("Open the URL in your browser, draw a digit, and click Predict.\n")
    app.run(host="127.0.0.1", port=5001, debug=False)
