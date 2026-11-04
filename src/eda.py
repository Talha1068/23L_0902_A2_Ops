import pandas as pd, matplotlib.pyplot as plt
from PIL import Image

df = pd.read_csv("reports/audit_table.csv")
ok = df[~df.corrupt]

for cls in ["ai", "real"]:
    sample = ok[ok["class"] == cls].sample(12, random_state=42)
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for ax, p in zip(axes.ravel(), sample.path):
        ax.imshow(Image.open(p).convert("RGB")); ax.axis("off")
    fig.suptitle(f"Sample images: {cls}")
    plt.tight_layout(); plt.savefig(f"reports/figures/samples_{cls}.png", dpi=120); plt.close()

for col, label in [("width", "Width (px)"), ("height", "Height (px)"),
                   ("file_size", "File size (bytes)")]:
    plt.figure(figsize=(7, 4))
    for cls, colour in [("ai", "tab:red"), ("real", "tab:blue")]:
        plt.hist(ok[ok["class"] == cls][col], bins=30, alpha=0.6, label=cls, color=colour)
    plt.xlabel(label); plt.ylabel("Count"); plt.legend(); plt.title(f"{label} by class")
    plt.tight_layout(); plt.savefig(f"reports/figures/hist_{col}.png", dpi=120); plt.close()