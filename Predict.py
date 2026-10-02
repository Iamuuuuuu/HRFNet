from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import torch
from torch.utils.data import Dataset, DataLoader

from model import DualInputModel

# ===== Editable test settings =====
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
WEIGHTS_DIR = BASE_DIR / "weights"
BATCH_SIZE = 10
# ==================================

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
        png = np.asarray(Image.open(record["png"]).convert("RGB").resize((16, 16)), dtype=np.float32) / 255.0
        png = np.transpose(png, (2, 0, 1))
        return torch.from_numpy(npy), torch.from_numpy(png), torch.tensor(record["label"], dtype=torch.long)


def collect_records(category_dir, label_names):
    label_to_index = {name: index for index, name in enumerate(label_names)}
    records = []
    for label_name in label_names:
        label_dir = category_dir / label_name
        for npy_path in sorted(label_dir.glob("*.npy")):
            png_path = npy_path.with_suffix(".png")
            if not png_path.exists():
                raise FileNotFoundError(f"Missing PNG pair for {npy_path}")
            records.append({"npy": npy_path, "png": png_path, "label": label_to_index[label_name]})
    return records


def predict_category(category_name, device):
    weight_category_name = category_name.replace("_test", "_best")
    checkpoint_path = WEIGHTS_DIR / f"{weight_category_name}_dual_input_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    label_names = checkpoint["label_names"]
    records = collect_records(DATA_DIR / category_name, label_names)
    dataset = PairedDataset(records)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model = DualInputModel(num_classes=checkpoint["num_classes"], npy_in_channels=30).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    targets = []
    predictions = []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for npy_batch, png_batch, target_batch in loader:
            npy_batch = npy_batch.to(device)
            png_batch = png_batch.to(device)
            target_batch = target_batch.to(device)
            logits = model(npy_batch, png_batch)
            total_loss += criterion(logits, target_batch).item() * target_batch.size(0)
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(target_batch.cpu().tolist())

    labels = list(range(len(label_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, labels=labels, zero_division=0
    )
    matrix = confusion_matrix(targets, predictions, labels=labels)
    total = matrix.sum()
    rows = []
    metric_name = {
        "color_30_test": "color",
        "maturity_30_test": "maturity",
        "structure_30_test": "structure",
        "identity_30_test": "identity",
        "oil_content_30_test": "oil_content",
        "chroma_30_test": "chroma",
    }[category_name]
    specificities = []
    for index, label_name in enumerate(label_names):
        true_positive = matrix[index, index]
        false_positive = matrix[:, index].sum() - true_positive
        false_negative = matrix[index, :].sum() - true_positive
        true_negative = total - true_positive - false_positive - false_negative
        specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive else 0.0
        specificities.append(specificity)
        class_accuracy = (true_positive + true_negative) / total if total else 0.0
        rows.append({
            "metric": metric_name,
            "class": label_name,
            "accuracy": round(class_accuracy * 100, 2),
            "precision": round(precision[index] * 100, 2),
            "sensitivity": round(recall[index] * 100, 2),
            "specificity": round(specificity * 100, 2),
            "f1_score": round(f1[index] * 100, 2),
            "sample_count": int(support[index]),
        })
    overall = {
        "metric": metric_name,
        "class": "overall",
        "accuracy": round(float(np.mean(np.asarray(predictions) == np.asarray(targets))) * 100, 2),
        "precision": round(float(np.mean(precision)) * 100, 2),
        "sensitivity": round(float(np.mean(recall)) * 100, 2),
        "specificity": round(float(np.mean(specificities)) * 100, 2),
        "f1_score": round(float(np.mean(f1)) * 100, 2),
        "sample_count": len(records),
        "loss": round(total_loss / len(records), 6),
    }
    return rows, overall


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    category_names = sorted(path.name for path in DATA_DIR.iterdir() if path.is_dir())
    overall_rows = []
    for category_name in category_names:
        _, overall = predict_category(category_name, device)
        overall_rows.append(overall)
    overall_results = pd.DataFrame(overall_rows)
    output_path = WEIGHTS_DIR / "predict_overall_results.csv"
    overall_results.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"device: {device}")
    print("\nOverall results:")
    print(overall_results.to_string(index=False))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
