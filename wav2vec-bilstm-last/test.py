"""
Testing model on:
Accuracy@1
  Recall@2
  Recall@3
  Recall@5
       MRR
"""

import os
import numpy as np
import requests
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

if os.path.isdir("../synthetics_test/remix"):
    TEST_DIR = "../synthetics_test/remix"
elif os.path.isdir("synthetics_test/remixes"):
    TEST_DIR = "synthetics_test/remixes"
else:
    raise FileNotFoundError("Cannot find synthetics_test/remix or synthetics_test/remixes")

SERVER_URL = "http://127.0.0.1:5000/similarity"
TOP_K = 5

track_ids = sorted(
    d for d in os.listdir(TEST_DIR)
    if os.path.isdir(os.path.join(TEST_DIR, d)) and d.isdigit()
)

ranks = []
for tid in tqdm(track_ids, desc="Evaluating tracks"):
    folder = os.path.join(TEST_DIR, tid)
    # list remixes
    remixes = sorted(
        f for f in os.listdir(folder)
        if f.startswith("remix") and f.endswith(".wav")
    )
    for remix in remixes:
        path = os.path.join(folder, remix)
        with open(path, "rb") as f:
            resp = requests.post(SERVER_URL, files={"file": f})
        data = resp.json().get("results", [])
        preds = [item["name"] for item in data]
        # rank of the true original (1-based), or TOP_K+1 if missing
        rank = preds.index(tid) + 1 if tid in preds else TOP_K + 1
        ranks.append(rank)

ranks = np.array(ranks)

metrics = {
    "Accuracy@1": np.mean(ranks == 1),
    "Recall@2": np.mean(ranks <= 2),
    "Recall@3": np.mean(ranks <= 3),
    "Recall@5": np.mean(ranks <= 5),
    "MRR": np.mean(1.0 / ranks),
}

df = pd.DataFrame({
    "Metric": list(metrics.keys()),
    "Value (%)": [v * 100 for v in metrics.values()]
})

print(df.to_string(index=False))

plt.style.use("dark_background")
plt.figure(figsize=(6, 4))
colors = ["#AEC6CF", "#FFB347", "#77DD77", "#F49AC2", "#CFCFC4"]
plt.bar(df["Metric"], df["Value (%)"], color=colors)
plt.ylabel("Percentage (%)")
plt.title("Retrieval Metrics via Server")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("metrics_bar_server.png")

# Dark-themed histogram of ranks
plt.figure(figsize=(6, 4))
plt.hist(ranks, bins=np.arange(1, ranks.max() + 2) - 0.5,
         color="#77DD77", edgecolor="white")
plt.xlabel("Rank of True Original")
plt.ylabel("Frequency")
plt.title("Rank Distribution via Server")
plt.tight_layout()
plt.savefig("ranks_hist_server.png")
