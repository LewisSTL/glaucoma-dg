"""
PyTorch Dataset for the unified glaucoma label index built by
src/data_prep/unify_labels.py (data/unified_labels.csv).
"""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FundusDataset(Dataset):
    """Wraps a slice of the unified_labels.csv dataframe as a PyTorch Dataset.

    Pass in an already-filtered dataframe (e.g. only rows where dataset == "G1020"),
    not the whole unified file, so callers control exactly what's in train/val/test.
    """

    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = PROJECT_ROOT / row["image_path"]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return image, label


def load_unified_labels(csv_path: Path = None) -> pd.DataFrame:
    if csv_path is None:
        csv_path = PROJECT_ROOT / "data" / "unified_labels.csv"
    return pd.read_csv(csv_path)
