"""
train.py  – triplet training on richer (static, delta, delta‑delta) Mel stats
Outputs
  • mel_triplet_model.pth
  • train_loss_curve.png
"""

import os, json, random, logging, numpy as np, torch, matplotlib.pyplot as plt
import librosa
from torch.utils.data import DataLoader, Dataset
from torch import nn
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
TRACK_JSON = os.path.join(PROJECT_ROOT, "synthetics", "processed_tracks.json")
SPEC_DIR = os.path.join(PROJECT_ROOT, "synthetics", "spectrogram")

SR, N_MELS, HOP = 16000, 128, 256
BATCH, EPOCHS, LR, MARGIN = 128, 100, 5e-4, 0.5
MODEL_OUT, PNG = "mel_triplet_model.pth", "train_loss_curve.png"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def mel_stats(arr: np.ndarray) -> torch.Tensor:
    """
    Return 768‑D vector: [mean & std of static mel, Δ mel, Δ² mel].
    """
    if arr.ndim == 1:
        S = librosa.feature.melspectrogram(arr, sr=SR, n_mels=N_MELS,
                                           n_fft=1024, hop_length=HOP)
        S = librosa.power_to_db(S, ref=np.max)
    else:
        S = arr
    delta = librosa.feature.delta(S)
    delta2 = librosa.feature.delta(S, order=2)
    feats = np.concatenate([S, delta, delta2], axis=0)  # (384, T)
    mu, sd = feats.mean(1), feats.std(1)
    return torch.tensor(np.concatenate([mu, sd]), dtype=torch.float32)  # (768,)


class TripletDS(Dataset):
    def __init__(self, ids):
        self.ids, self.pool = ids, set(ids)

    def _np(self, *pth): return np.load(os.path.join(*pth))

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        tid = self.ids[idx];
        root = os.path.join(SPEC_DIR, tid)
        anchor = mel_stats(self._np(root, "original.npy"))
        rm = [f for f in os.listdir(root) if f.startswith("remix") and f.endswith(".npy")]
        pos = mel_stats(self._np(root, random.choice(rm)))
        neg_tid = random.choice(list(self.pool - {tid}))
        neg = mel_stats(self._np(SPEC_DIR, neg_tid, "original.npy"))
        return anchor, pos, neg


def collate(b): a, p, n = zip(*b); return torch.stack(a), torch.stack(p), torch.stack(n)


# embedder (768 → 256)
class Embed(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(768, 512), nn.ReLU(),
            nn.Linear(512, 256)
        )

    def forward(self, x): return nn.functional.normalize(self.mlp(x), dim=-1)


def main():
    ids = [str(t) for t in json.load(open(TRACK_JSON))]
    dl = DataLoader(TripletDS(ids), batch_size=BATCH, shuffle=True,
                    num_workers=0, collate_fn=collate)
    net = Embed().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    loss_fn = nn.TripletMarginLoss(MARGIN)
    losses = []
    for ep in range(1, EPOCHS + 1):
        tot = 0
        for a, p, n in tqdm(dl, desc=f"ep {ep}", leave=False):
            a, p, n = a.to(DEVICE), p.to(DEVICE), n.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(net(a), net(p), net(n))
            loss.backward();
            opt.step()
            tot += loss.item()
        avg = tot / len(dl);
        losses.append(avg)
        logging.info(f"epoch {ep:02d} loss={avg:.4f}")
    torch.save(net.state_dict(), MODEL_OUT);
    logging.info(f"→ {MODEL_OUT}")

    import matplotlib.pyplot as plt

    plt.style.use('dark_background')
    plt.figure(figsize=(6,4))
    pastel_color = '#AEC6CF'   # soft pastel blue
    plt.plot(
        range(1, EPOCHS + 1),
        losses,
        color=pastel_color,
        marker='o',
        linewidth=2,
        markersize=6
    )
    plt.xlabel("Epoch")
    plt.ylabel("Triplet Loss")
    plt.title("Training Loss Curve")
    plt.tight_layout()
    plt.savefig(PNG, dpi=150)
    logging.info(f"→ {PNG}")


if __name__ == "__main__":
    main()
