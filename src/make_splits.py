import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
df = pd.read_csv("reports/audit_table.csv")
clean = df[(~df.corrupt) & (~df.is_duplicate) & (~df.cross_class_dup)].copy()
print("Kept", len(clean), "of", len(df), "images")

clean["label"] = (clean["class"] == "ai").astype(int)   # ai = 1, real = 0
keep = ["path", "class", "label", "sha256"]

train, temp = train_test_split(clean, test_size=0.30, stratify=clean["label"], random_state=SEED)
val, test = train_test_split(temp, test_size=0.50, stratify=temp["label"], random_state=SEED)

for name, part in [("train", train), ("val", val), ("test", test)]:
    part[keep].to_csv(f"splits/{name}.csv", index=False)
    print(name, len(part), part["class"].value_counts().to_dict())