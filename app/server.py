"""OpenStethoscope — inference server."""
import os, io, pickle, warnings, tempfile, subprocess
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_XLA_FLAGS"]         = "--tf_xla_auto_jit=0"

# Paths (env-var configurable)
BASE       = Path(__file__).parent
HEAR_PATH  = os.environ.get("HEAR_PATH", None)   # if "" or None → download from HF
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/app/models"))
PORT       = int(os.environ.get("PORT", 8080))

# Fallback for local dev: use models next to this file
if not MODELS_DIR.exists():
    MODELS_DIR = BASE.parent / "models"

app = Flask(__name__)

# HeAR model loading
_hear_infer = None

def get_hear():
    global _hear_infer, HEAR_PATH
    if _hear_infer is not None:
        return _hear_infer
    import tensorflow as tf
    hp = HEAR_PATH
    if not hp:
        # Check known path-hint locations (Docker base image or local download)
        for _pf in [Path("/app/hear_path.txt"), BASE / "hear_path.txt", BASE.parent / "hear_path.txt"]:
            if _pf.exists():
                hp = _pf.read_text().strip()
                print(f"HeAR path from {_pf}: {hp}")
                break
        if not hp:
            print("HEAR_PATH not set — downloading from Hugging Face Hub…")
            from huggingface_hub import snapshot_download
            hp = snapshot_download("google/hear")
        HEAR_PATH = hp
    print(f"Loading HeAR from {hp}")
    m = tf.saved_model.load(hp)
    _hear_infer = m.signatures["serving_default"]
    print("HeAR loaded.")
    return _hear_infer

# Classifier loading
_models: dict = {}
MODEL_NAMES = [
    "copd_vs_all",
    "copd_vs_healthy",
    "pneumonia_vs_healthy",
    "asthma_vs_healthy",
    "heart_failure_vs_healthy",
]

def get_models() -> dict:
    global _models
    if _models:
        return _models
    for name in MODEL_NAMES:
        path = MODELS_DIR / f"{name}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                _models[name] = pickle.load(f)
            print(f"  Loaded {name}: AUROC={_models[name]['cv_auroc_mean']:.3f}")
        else:
            print(f"  WARNING: model not found at {path}")
    return _models

# DSP pipeline
# Training: raw WAVs, no software filter. Inference: LP2000 to match stethoscope hardware band-limiting.

# Inference
def run_inference(audio_bytes: bytes, content_type: str = "audio/webm") -> dict:
    import numpy as np, tensorflow as tf

    SR, WIN = 16000, 32000

    # Map content-type → file extension
    ext_map = {
        "audio/webm": ".webm", "audio/mp4": ".mp4", "audio/ogg": ".ogg",
        "audio/wav":  ".wav",  "audio/mpeg": ".mp3", "audio/x-m4a": ".m4a",
        "audio/aac":  ".aac",
    }
    ext = next((v for k, v in ext_map.items() if k in content_type.lower()), ".webm")

    # Write to temp file — needed for non-WAV formats
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        # Convert to WAV via ffmpeg, then load with soundfile
        wav_path = tmp_path + ".wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_path, "-ar", str(SR), "-ac", "1", wav_path],
            capture_output=True, check=True,
        )
        import soundfile as sf
        y, _ = sf.read(wav_path)
        y = y.astype(np.float32)
        os.unlink(wav_path)
    except Exception:
        # Fallback: try loading directly
        try:
            import soundfile as sf
            y, loaded_sr = sf.read(tmp_path)
            y = y.astype(np.float32)
            if loaded_sr != SR:
                from scipy.signal import resample
                y = resample(y, int(len(y) * SR / loaded_sr)).astype(np.float32)
        except Exception as e2:
            raise RuntimeError(f"Could not decode audio: {e2}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    duration = len(y) / SR

    # LP2000: compensates for DJI Mic Mini broadband response vs stethoscope hardware band-limiting
    from scipy.signal import butter, sosfilt
    sos = butter(5, 2000.0 / (SR / 2), btype="low", output="sos")
    y   = sosfilt(sos, y).astype(np.float32)

    if len(y) < WIN:
        y = np.pad(y, (0, WIN - len(y)))
    wins = [y[s:s + WIN] for s in range(0, len(y) - WIN + 1, SR)] or [y[:WIN]]

    # HeAR embedding
    infer = get_hear()
    batch = np.array(wins, dtype=np.float32)
    out   = infer(tf.constant(batch))
    key   = next((k for k in out if "embedding" in k.lower()), list(out.keys())[0])
    emb   = out[key].numpy().mean(axis=0).reshape(1, -1)

    # Score with each classifier
    models = get_models()
    scores = {}
    for task_name, m in models.items():
        prob = float(m["model"].predict_proba(emb)[0, 1])
        scores[task_name] = {
            "prob":       round(prob, 4),
            "prob_pct":   round(prob * 100, 1),
            "auroc":      round(m["cv_auroc_mean"], 3),
            "n_positive": m["n_positive"],
            "n_negative": m["n_negative"],
        }

    return {
        "duration": round(duration, 1),
        "windows":  len(wins),
        "scores":   scores,
    }

# Routes
@app.route("/")
def index():
    return send_from_directory(BASE, "index.html")

@app.route("/health")
def health():
    models = get_models()
    return jsonify({"ok": True, "models_loaded": list(models.keys())})

@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        audio_bytes = request.data
        if not audio_bytes:
            return jsonify({"ok": False, "error": "No audio data"}), 400
        content_type = request.content_type or "audio/webm"
        result = run_inference(audio_bytes, content_type)
        return jsonify({"ok": True, **result})
    except Exception as e:
        import traceback
        return jsonify({
            "ok":    False,
            "error": str(e),
            "trace": traceback.format_exc(),
        }), 500

# Entry point
if __name__ == "__main__":
    print(f"OpenStethoscope starting on port {PORT}")
    print(f"Models dir: {MODELS_DIR}")
    print("Pre-loading classifiers…")
    get_models()
    print("Starting Flask server…")
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
