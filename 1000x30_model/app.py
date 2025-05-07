import os, json
import numpy as np
import torch, torch.nn.functional as F
import librosa
from io import BytesIO
from flask import Flask, request, jsonify, send_from_directory
from train import mel_stats, Embed, SR
import random

from mutagen import File as MutagenFile


def extract_meta(uploaded_file):
    """Return (title, artist) if found in ID3 tags, else (None,None)."""
    try:
        audio = MutagenFile(BytesIO(uploaded_file.read()), easy=True)
        title = audio.get("title", [None])[0]
        artist = audio.get("artist", [None])[0]
        uploaded_file.seek(0)
        return title, artist
    except:
        uploaded_file.seek(0)
        return None, None


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir))

TRACK_JSON = os.path.join(PROJECT_ROOT, "synthetics", "processed_tracks.json")
SPEC_DIR   = os.path.join(PROJECT_ROOT, "synthetics", "spectrogram")
CKPT       = os.path.join(BASE_DIR, "mel_triplet_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__, static_folder='.', static_url_path='')

embed = Embed().to(DEVICE)
embed.load_state_dict(torch.load(CKPT, map_location="cpu"))
embed.eval()

with open(TRACK_JSON) as f:
    track_ids = [str(t) for t in json.load(f)]

with torch.no_grad():
    db_embs = torch.stack([
        embed(mel_stats(np.load(os.path.join(SPEC_DIR, tid, "original.npy"))).to(DEVICE))
        for tid in track_ids
    ])  # (N,256)


@app.route("/", defaults={"path": "index.html"})
@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(BASE_DIR, path)


@app.route("/similarity", methods=["POST"])
def similarity():
    f = request.files.get("file")
    if not f:
        return jsonify(error="no file"), 400

    title, artist = extract_meta(f)
    label = title or f.filename
    if artist:
        label = f"{label} — {artist}"

    wav, _ = librosa.load(BytesIO(f.read()), sr=SR, mono=True)
    q = embed(mel_stats(wav).to(DEVICE))  # (256,)

    sims = F.cosine_similarity(q.unsqueeze(0), db_embs, dim=1)  # (N,)
    top5 = torch.topk(sims, k=5)
    s_vals = top5.values.tolist()       # descending

    # Randomize endpoints within your desired bands
    p1 = random.uniform(88.0, 93.0)     # top-1 ∈ [88,93]%
    p5 = random.uniform(20.0, 30.0)     # top-5 ∈ [20,30]%

    # Normalize sims to [0,1] over the top-5 window
    s_max, s_min = s_vals[0], s_vals[-1]
    span = s_max - s_min if s_max != s_min else 1e-6
    norms = [(s - s_min) / span for s in s_vals]

    # Interpolate each norm into [p5,p1]
    perc = [p5 + n * (p1 - p5) for n in norms]

    results = []
    for pct, idx in zip(perc, top5.indices.tolist()):
        results.append({
            "name":        track_ids[idx],
            "probability": f"{pct:.1f}%"
        })

    return jsonify(file_label=label, results=results)


if __name__ == "__main__":
    app.run("127.0.0.1", 5000, debug=True)
