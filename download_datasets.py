import os
from datasets import load_dataset

os.makedirs("data/raw", exist_ok=True)

datasets_to_load = {
    "docvqa": ("lmms-lab/DocVQA", "DocVQA", "validation"),
    "chartqa": ("lmms-lab/ChartQA", None, "test"),
    "infovqa": ("lmms-lab/DocVQA", "InfographicVQA", "validation"),
    "textvqa": ("lmms-lab/textvqa", None, "validation"),
}

for name, (repo, subset, split) in datasets_to_load.items():
    try:
        print(f"Dang tai {name} ...")
        ds = subset and load_dataset(repo, subset, split=split) or load_dataset(repo, split=split)
        ds.save_to_disk(f"data/raw/{name}")
        print(f"  -> OK, {len(ds)} mau, da luu vao data/raw/{name}")
    except Exception as e:
        print(f"  -> LOI voi {name}: {e}")

print("Hoan tat. Kiem tra thu muc data/raw de xem ket qua.")
