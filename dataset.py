import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class PairedDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        npy = np.load(record["npy"]).astype(np.float32)
        if npy.shape == (16, 16, 30):
            npy = np.transpose(npy, (2, 0, 1))
        elif npy.shape != (30, 16, 16):
            raise ValueError(f"Unexpected NPY shape {npy.shape}: {record['npy']}")
        minimum = npy.min()
        maximum = npy.max()
        if maximum > minimum:
            npy = (npy - minimum) / (maximum - minimum)
        png = np.asarray(
            Image.open(record["png"]).convert("RGB").resize((16, 16)),
            dtype=np.float32,
        ) / 255.0
        png = np.transpose(png, (2, 0, 1))
        return (
            torch.from_numpy(npy),
            torch.from_numpy(png),
            torch.tensor(record["label"], dtype=torch.long),
        )


def collect_records(category_dir):
    label_names = sorted(path.name for path in category_dir.iterdir() if path.is_dir())
    label_to_index = {name: index for index, name in enumerate(label_names)}
    records = []
    for label_name in label_names:
        label_dir = category_dir / label_name
        for npy_path in sorted(label_dir.glob("*.npy")):
            png_path = npy_path.with_suffix(".png")
            if not png_path.exists():
                raise FileNotFoundError(f"Missing PNG pair for {npy_path}")
            records.append({
                "npy": npy_path,
                "png": png_path,
                "label": label_to_index[label_name],
            })
    if not records:
        raise FileNotFoundError(f"No samples found in {category_dir}")
    return records, label_names


def split_records(records, label_names, seed=42, val_ratio=0.15, test_ratio=0.15):
    rng = random.Random(seed)
    train_records = []
    val_records = []
    test_records = []
    for label_index, _ in enumerate(label_names):
        label_records = [record for record in records if record["label"] == label_index]
        rng.shuffle(label_records)
        test_count = max(1, round(len(label_records) * test_ratio))
        val_count = max(1, round(len(label_records) * val_ratio))
        train_count = len(label_records) - val_count - test_count
        if train_count < 1:
            raise ValueError(f"Not enough samples to split label index {label_index}")
        train_records.extend(label_records[:train_count])
        val_records.extend(label_records[train_count:train_count + val_count])
        test_records.extend(label_records[train_count + val_count:])
    return train_records, val_records, test_records
