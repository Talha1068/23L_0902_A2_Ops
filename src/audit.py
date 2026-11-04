import hashlib
from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm

RAW = Path("data/raw")
CLASS_DIRS = {"ai": RAW / "AiArtData", "real": RAW / "RealArt"}

def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()

rows = []
for label, folder in CLASS_DIRS.items():
    files = [p for p in folder.rglob("*") if p.is_file()]
    for p in tqdm(files, desc=label):
        row = {"path": p.as_posix(), "class": label, "file_size": p.stat().st_size,
               "sha256": sha256(p), "width": None, "height": None,
               "mode": None, "format": None, "corrupt": False, "error": ""}
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                im.load()
                row.update(width=im.width, height=im.height, mode=im.mode, format=im.format)
        except Exception as e:
            row["corrupt"] = True
            row["error"] = str(e)[:80]
        rows.append(row)

df = pd.DataFrame(rows)
df["is_duplicate"] = df.duplicated("sha256", keep="first") & ~df["corrupt"]
cls_per_hash = df[~df.corrupt].groupby("sha256")["class"].nunique()
df["cross_class_dup"] = df["sha256"].map(cls_per_hash).fillna(1) > 1
df.to_csv("reports/audit_table.csv", index=False)

ok = df[~df.corrupt]
print("\n=== Counts per class ===")
print(df.groupby("class").agg(total=("path", "count"), corrupt=("corrupt", "sum"),
                              duplicates=("is_duplicate", "sum")))
print("\n=== Min / median / max (valid images) ===")
print(ok.groupby("class")[["width", "height", "file_size"]].agg(["min", "median", "max"]).T)
print("\nColour modes:\n", ok.groupby(["class", "mode"]).size())
print("\nFormats:\n", ok.groupby(["class", "format"]).size())
print("\nTotal corrupt files        :", int(df.corrupt.sum()))
print("Total exact duplicates     :", int(df.is_duplicate.sum()))
print("Cross-class duplicate rows :", int(df.cross_class_dup.sum()))