# app.py
import os
import json
import logging
import numpy as np
import torch
import torch.nn.functional as F
import librosa
from io import BytesIO
from flask import Flask, request, jsonify, send_from_directory
from transformers import Wav2Vec2Processor, Wav2Vec2Model
# from model_pipe import FidemixModel

BASE_DIR    = os.path.dirname(__file__)
TRACK_LIST  = os.path.join(BASE_DIR, "synthetics/processed_tracks.json")
AUDIO_DIR   = os.path.join(BASE_DIR, "synthetics/spectrogram")
MODEL_PATH  = os.path.join(BASE_DIR, "fidemix_triplet_model.pth")
SAMPLE_RATE = 16_000
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__, static_folder=BASE_DIR, static_url_path="")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec    = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(DEVICE).eval()

with open(TRACK_LIST, "r") as f:
    track_ids = [str(t) for t in json.load(f)]
logging.info(f"Found {len(track_ids)} tracks in database")

db_embs = []
for tid in track_ids:
    arr = np.load(os.path.join(AUDIO_DIR, tid, "original.npy"))
    if arr.ndim > 1:
        arr = arr.mean(axis=0)
    wav = librosa.resample(arr, orig_sr=SAMPLE_RATE, target_sr=SAMPLE_RATE)
    inputs = processor(wav, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        h = wav2vec(**inputs).last_hidden_state.mean(dim=1)
        emb = h.squeeze(0)
    db_embs.append(emb)
db_embs = torch.stack(db_embs, dim=0)  # (N_tracks, 768)
logging.info("Precomputed database embeddings")

# ─── ROUTES ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")

@app.route("/similarity", methods=["POST"])
def similarity():
    if "file" not in request.files:
        return jsonify(error="no file uploaded"), 400

    uploaded = request.files["file"].read()
    wav, _ = librosa.load(BytesIO(uploaded), sr=SAMPLE_RATE, mono=True)
    inputs = processor(wav, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True).to(DEVICE)

    with torch.no_grad():
        hq = wav2vec(**inputs).last_hidden_state.mean(dim=1).squeeze(0)  # (768,)
        q_emb = hq

    sims = F.cosine_similarity(q_emb.unsqueeze(0), db_embs, dim=1)  # (N_tracks,)

    probs = F.softmax(sims, dim=0)

    top5 = torch.topk(probs, k=5)
    results = []
    for score, idx in zip(top5.values.tolist(), top5.indices.tolist()):
        results.append({
            "track": track_ids[idx],
            "score": f"{100*score:.1f}%"
        })

    return jsonify(results=results)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
