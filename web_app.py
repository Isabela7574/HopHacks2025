# web_app.py
import io, traceback
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from joblib import load

SR = 16000
N_MELS = 40

def featurize_bytes(wav_bytes: bytes):
    # Lazy import (safer on macOS)
    import librosa
    y, sr = librosa.load(io.BytesIO(wav_bytes), sr=SR, mono=True,
                         dtype=np.float32, res_type="kaiser_fast")
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    logS = librosa.power_to_db(S, ref=np.max)
    d1 = librosa.feature.delta(logS)
    d2 = librosa.feature.delta(logS, order=2)
    def ms(a): return np.hstack([a.mean(axis=1), a.std(axis=1)])
    feat = np.hstack([ms(logS), ms(d1), ms(d2)])
    return feat.reshape(1, -1)

bundle = load("baseline_emotion.joblib")
MODEL = bundle["model"]

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

@app.after_request
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp

@app.route("/")
def root():
    return send_from_directory("static", "index.html")

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return ("", 204)
    try:
        if "file" not in request.files:
            return jsonify(ok=False, error="No 'file' field."), 400
        f = request.files["file"]
        if not f or f.filename == "":
            return jsonify(ok=False, error="Empty filename."), 400

        data = f.read()
        if not data:
            return jsonify(ok=False, error="Empty file body."), 400

        X = featurize_bytes(data)
        pred = MODEL.predict(X)[0]

        top = None
        if hasattr(MODEL, "predict_proba"):
            p = MODEL.predict_proba(X)[0]
            classes = getattr(MODEL, "classes_", [])
            top = sorted(zip(classes, p), key=lambda t: t[1], reverse=True)[:3]
            top = [(c, float(round(v, 4))) for c, v in top]

        print(f"[PREDICT] {pred}")
        return jsonify(ok=True, prediction=pred, top=top)
    except Exception as e:
        traceback.print_exc()
        return jsonify(ok=False, error=str(e)), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
