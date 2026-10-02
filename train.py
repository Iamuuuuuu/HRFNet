from pathlib import Path
import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import DataLoader

from dataset import PairedDataset, collect_records, split_records
from model import DualInputModel

# ===== training settings =====
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / r"E:\烟草项目二期\烟草项目二期\PCA_Reduced_3D"
WEIGHTS_DIR = BASE_DIR / "weights"
EPOCHS = 300
BATCH_SIZE = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42
PRINT_EVERY = 10
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
# ============================


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    targets = []
    predictions = []
    with torch.no_grad():
        for npy_batch, png_batch, target_batch in loader:
            npy_batch = npy_batch.to(device, non_blocking=True)
            png_batch = png_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)
            logits = model(npy_batch, png_batch)
            loss_sum += criterion(logits, target_batch).item() * target_batch.size(0)
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(target_batch.cpu().tolist())
    accuracy = accuracy_score(targets, predictions)
    return loss_sum / len(loader.dataset), accuracy, targets, predictions


def train_category(category_name, device):
    category_dir = DATA_DIR / category_name
    records, label_names = collect_records(category_dir)
    train_records, val_records, test_records = split_records(
        records,
        label_names,
        seed=SEED,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
    )
    train_dataset = PairedDataset(train_records)
    val_dataset = PairedDataset(val_records)
    test_dataset = PairedDataset(test_records)
    loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model = DualInputModel(num_classes=len(label_names), npy_in_channels=30).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    history = []
    best_val_accuracy = -1.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for npy_batch, png_batch, target_batch in loader:
            npy_batch = npy_batch.to(device, non_blocking=True)
            png_batch = png_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(npy_batch, png_batch), target_batch)
            loss.backward()
            optimizer.step()
        scheduler.step()
        train_loss, train_accuracy, _, _ = evaluate(model, loader, criterion, device)
        val_loss, val_accuracy, _, _ = evaluate(model, val_loader, criterion, device)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        })
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % PRINT_EVERY == 0 or epoch == EPOCHS:
            print(
                f"{category_name}: epoch {epoch:03d}/{EPOCHS}, "
                f"train_accuracy={train_accuracy:.4f}, val_accuracy={val_accuracy:.4f}"
            )

    model.load_state_dict(best_state)
    test_loss, test_accuracy, _, _ = evaluate(model, test_loader, criterion, device)
    checkpoint_path = WEIGHTS_DIR / f"{category_name.replace('_test', '_best')}_dual_input_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "num_classes": len(label_names),
        "label_names": label_names,
        "category": category_name,
        "architecture": "DualInputModel_fixed_pseudocode",
        "input_shapes": {"npy": [30, 16, 16], "png": [3, 16, 16]},
        "history": history,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "val_accuracy": best_val_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "epochs_trained": len(history),
    }, checkpoint_path)
    return {
        "category": category_name,
        "samples": len(records),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "classes": len(label_names),
        "test_accuracy": test_accuracy,
        "test_loss": test_loss,
        "model_path": str(checkpoint_path),
    }


def main():
    set_seed(SEED)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    category_names = sorted(path.name for path in DATA_DIR.iterdir() if path.is_dir())
    if len(category_names) != 6:
        raise ValueError(f"Expected 6 category folders in {DATA_DIR}, found {category_names}")
    print(f"device: {device}")
    print(f"training data: {DATA_DIR}")
    print(f"weights output: {WEIGHTS_DIR}")
    results = [train_category(category_name, device) for category_name in category_names]
    results_df = pd.DataFrame(results)
    results_df.to_csv(WEIGHTS_DIR / "training_results.csv", index=False, encoding="utf-8-sig")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
