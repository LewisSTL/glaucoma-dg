"""
Builds one unified label index across the datasets in data/archive/,
since every public glaucoma dataset ships its labels in a different format:

  - G1020:  G1020.csv          -> imageID, binaryLabels
  - ORIGA:  OrigaList.csv      -> Eye, Filename, ExpCDR, Set, Glaucoma
  - REFUGE: train/val/test/index.json -> {"0": {"ImgName":..., "Label":...}, ...}

Output: data/unified_labels.csv with columns:
  image_path  - path to the image, relative to the project root
  label       - 0 (no glaucoma) / 1 (glaucoma), as an int
  dataset     - "G1020" / "ORIGA" / "REFUGE"
  split       - dataset's own official split if it has one (REFUGE: train/val/test,
                ORIGA: A/B), else empty string if the dataset doesn't define one (G1020)

Run from anywhere:
    python src/data_prep/unify_labels.py
"""

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_DIR = PROJECT_ROOT / "data" / "archive"


def load_g1020() -> pd.DataFrame:
    csv_path = ARCHIVE_DIR / "G1020" / "G1020.csv"
    images_dir = ARCHIVE_DIR / "G1020" / "Images"
    df = pd.read_csv(csv_path)
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "image_path": str((images_dir / row["imageID"]).relative_to(PROJECT_ROOT)),
            "label": int(row["binaryLabels"]),
            "dataset": "G1020",
            "split": "",
        })
    return pd.DataFrame(rows)


def load_origa() -> pd.DataFrame:
    csv_path = ARCHIVE_DIR / "ORIGA" / "OrigaList.csv"
    images_dir = ARCHIVE_DIR / "ORIGA" / "Images"
    df = pd.read_csv(csv_path)
    rows = []
    for _, row in df.iterrows():
        # A couple of rows in this file are missing a label (seen while inspecting
        # the raw csv) -- skip anything that doesn't parse cleanly instead of crashing.
        if pd.isna(row.get("Glaucoma")) or pd.isna(row.get("Filename")):
            continue
        rows.append({
            "image_path": str((images_dir / row["Filename"]).relative_to(PROJECT_ROOT)),
            "label": int(row["Glaucoma"]),
            "dataset": "ORIGA",
            "split": row["Set"],  # "A" or "B"
        })
    return pd.DataFrame(rows)


def load_refuge() -> pd.DataFrame:
    # Note: REFUGE's "test" split has no ground-truth labels in this download (its
    # labels were never publicly released as part of the original grand challenge),
    # so we only use train + val, which both have real Label values.
    rows = []
    for split in ["train", "val"]:
        index_path = ARCHIVE_DIR / "REFUGE" / split / "index.json"
        images_dir = ARCHIVE_DIR / "REFUGE" / split / "Images"
        with open(index_path) as f:
            index = json.load(f)
        for entry in index.values():
            if "Label" not in entry:
                continue
            rows.append({
                "image_path": str((images_dir / entry["ImgName"]).relative_to(PROJECT_ROOT)),
                "label": int(entry["Label"]),
                "dataset": "REFUGE",
                "split": split,
            })
    return pd.DataFrame(rows)


def main():
    frames = [load_g1020(), load_origa(), load_refuge()]
    unified = pd.concat(frames, ignore_index=True)

    # Sanity check: drop any rows whose image file doesn't actually exist, and say so loudly
    # rather than silently training on broken paths.
    exists_mask = unified["image_path"].apply(lambda p: (PROJECT_ROOT / p).exists())
    missing = unified[~exists_mask]
    if len(missing) > 0:
        print(f"WARNING: {len(missing)} rows point at files that don't exist, dropping them:")
        print(missing.head(10))
    unified = unified[exists_mask].reset_index(drop=True)

    out_path = PROJECT_ROOT / "data" / "unified_labels.csv"
    unified.to_csv(out_path, index=False)

    print(f"\nWrote {len(unified)} rows to {out_path}")
    print("\nPer-dataset counts and class balance:")
    print(unified.groupby(["dataset", "label"]).size().unstack(fill_value=0))
    print("\nPer-dataset split breakdown:")
    print(unified.groupby(["dataset", "split"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
