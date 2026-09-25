from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import getpass
import os
import random
import shutil
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def configure_juliacall_environment() -> None:
    home_dir = Path.home()
    os.environ.setdefault("PYTHON_JULIAPKG_PROJECT", str(home_dir / "Documents" / "dev" / "MAGENetRunner" / "python_utils"))
    os.environ.setdefault("PYTHON_JULIAPKG_OFFLINE", "yes")
    julia_exe = shutil.which("julia")
    if julia_exe is not None:
        os.environ.setdefault("PYTHON_JULIAPKG_EXE", julia_exe)


configure_juliacall_environment()

from juliacall import Main as jl
from python_utils import jld2_python
from python_utils.cnn.utils import calculate_statistics, save_checkpoint, save_metrics_to_csv

dloader_args = {
    "num_workers": 4,
    "pin_memory": False,
    "persistent_workers": False,
    "prefetch_factor": 2,
}


@dataclass
class DatasetConfig:
    dataset_name: str
    train_file: str
    val_file: str
    test_file: str
    num_classes: int
    batch_size: int
    val_batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    use_class_weights: bool
    seed: int
    resize_to: int
    output_dir: str
    experiment_name: str
    backbone_name: str
    nd: int = 1
    pretrained_weights: str = "default"


@dataclass
class ExperimentContext:
    config: DatasetConfig
    device: torch.device
    folder: Path
    metrics_file: Path
    checkpoint_dir: Path
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset
    train_dataloader: DataLoader
    val_dataloader: DataLoader
    test_dataloader: DataLoader
    criterion: nn.Module
    class_weights: torch.Tensor
    train_mean: list[float]
    train_std: list[float]


class DatasetJLD2(Dataset):
    def __init__(self, compressed_file: str, transform=None, nd: int = 1, repeat_grayscale_to_rgb: bool = True):
        self.compressed_file = compressed_file
        self.transform = transform
        self.nd = nd
        self.repeat_grayscale_to_rgb = repeat_grayscale_to_rgb
        tmp = jld2_python.read_jld2_dataset(compressed_file)
        self.images, self.labels = tmp[0], tmp[1]

        if nd == 3:
            self.images = np.asarray([jl.cat(x[0], x[1], x[2], dims=3) for x in self.images])
        elif nd == 1:
            self.images = np.asarray([jl.cat(x[0], dims=3) for x in self.images])
        else:
            raise RuntimeError(f"Unsupported nd={nd}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx]).type(torch.float32)
        image = torch.permute(image, (2, 0, 1))
        image = image / 255.0

        if self.repeat_grayscale_to_rgb and image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        label = int(self.labels[idx]) - 1

        if self.transform:
            image = self.transform(image)

        return image, label


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms(train_mean: list[float], train_std: list[float], resize_to: int, nd: int):
    normalize = transforms.Normalize(train_mean, train_std)
    train_steps = [
        transforms.RandomResizedCrop(resize_to, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    ]
    if nd == 1:
        train_steps.append(
            transforms.ColorJitter(brightness=0.3, contrast=0.3)
        )
    elif nd == 3:
        train_steps.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.1),
                transforms.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.1,
                    hue=0.02,
                ),
            ]
        )
    else:
        raise RuntimeError(f"Unsupported nd={nd}")
    train_steps.append(normalize)
    train_transform = transforms.Compose(train_steps)
    eval_transform = transforms.Compose(
        [
            transforms.Resize((resize_to, resize_to)),
            normalize,
        ]
    )
    return train_transform, eval_transform


def _compute_class_weights(train_dataset: DatasetJLD2) -> torch.Tensor:
    labels = np.asarray(train_dataset.labels, dtype=np.int64) - 1
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels,
    )
    return torch.tensor(class_weights, dtype=torch.float32)


def prepare_experiment(config: DatasetConfig) -> ExperimentContext:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    folder = ROOT / config.output_dir / config.experiment_name
    metrics_file = folder / "cnn_metrics_resnet.csv"
    checkpoint_dir = folder / "DLmodels_resnet" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if metrics_file.exists():
        metrics_file.unlink()

    for checkpoint_path in checkpoint_dir.glob("*.pth"):
        checkpoint_path.unlink()

    train_dataset = DatasetJLD2(config.train_file, None, config.nd)
    val_dataset = DatasetJLD2(config.val_file, None, config.nd)
    test_dataset = DatasetJLD2(config.test_file, None, config.nd)

    train_mean, train_std = calculate_statistics(train_dataset, batch_size=config.batch_size, num_workers=1)
    train_transform, eval_transform = build_transforms(
        train_mean,
        train_std,
        config.resize_to,
        config.nd,
    )

    train_dataset.transform = train_transform
    val_dataset.transform = eval_transform
    test_dataset.transform = eval_transform

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        **dloader_args,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.val_batch_size,
        shuffle=False,
        **dloader_args,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.val_batch_size,
        shuffle=False,
        **dloader_args,
    )

    class_weights = _compute_class_weights(train_dataset)
    if config.use_class_weights:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    print(f"TORCH DEVICE {device}")
    print(f"N CPUS {os.cpu_count()}")
    print(f"CWD : {os.getcwd()}")
    print(f"Metrics file: {metrics_file}")
    print(f"TRAIN DATASET SIZE : {len(train_dataset)}")
    print(f"VAL DATASET SIZE : {len(val_dataset)}")
    print(f"TEST DATASET SIZE : {len(test_dataset)}")
    print(f"normalization mean: {train_mean} | std: {train_std}")
    print(f"class weights: {class_weights.tolist()}")

    return ExperimentContext(
        config=config,
        device=device,
        folder=folder,
        metrics_file=metrics_file,
        checkpoint_dir=checkpoint_dir,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        criterion=criterion,
        class_weights=class_weights,
        train_mean=train_mean,
        train_std=train_std,
    )


def resolve_backbone(backbone_name: str, pretrained_weights: str):
    normalized_name = backbone_name.lower()
    normalized_weights = pretrained_weights.lower()
    use_default_weights = normalized_weights == "default"

    if normalized_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if use_default_weights else None
        return models.resnet18(weights=weights)
    if normalized_name == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if use_default_weights else None
        return models.resnet34(weights=weights)
    if normalized_name == "resnext50":
        weights = models.ResNeXt50_32X4D_Weights.DEFAULT if use_default_weights else None
        return models.resnext50_32x4d(weights=weights)
    raise ValueError(f"Unsupported backbone_name={backbone_name}")


def build_model(backbone_name: str, num_classes: int, pretrained_weights: str = "default") -> nn.Module:
    model = resolve_backbone(backbone_name, pretrained_weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def run_epoch(model, dataloader, criterion, device, optimizer=None):
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    total = 0
    predictions = []
    labels_list = []
    probabilities = []

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size
        predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        labels_list.extend(labels.cpu().numpy())
        probabilities.append(torch.softmax(outputs, dim=1).detach().cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = accuracy_score(labels_list, predictions)
    epoch_bal_acc = balanced_accuracy_score(labels_list, predictions)
    epoch_probs = np.concatenate(probabilities, axis=0)
    return epoch_loss, epoch_acc, epoch_bal_acc, np.asarray(labels_list), np.asarray(predictions), epoch_probs


def train_model(context: ExperimentContext, model: nn.Module, optimizer):
    train_loss, train_acc, train_bal_acc, _, _, _ = run_epoch(
        model,
        context.train_dataloader,
        context.criterion,
        context.device,
    )
    val_loss, val_acc, val_bal_acc, _, _, _ = run_epoch(
        model,
        context.val_dataloader,
        context.criterion,
        context.device,
    )

    save_metrics_to_csv(
        context.metrics_file,
        0,
        train_loss,
        val_loss,
        train_acc,
        val_acc,
        train_bal_acc,
        val_bal_acc,
    )

    print(
        f"Epoch [0/{context.config.epochs}], "
        f"Train Loss: {train_loss:.4f}, Train ACC: {train_acc:.4f}, Train BACC: {train_bal_acc:.4f}"
    )
    print(
        f"Epoch [0/{context.config.epochs}], "
        f"Val Loss: {val_loss:.4f}, Val ACC: {val_acc:.4f}, Val BACC: {val_bal_acc:.4f}"
    )

    for epoch in range(1, context.config.epochs + 1):
        train_loss, train_acc, train_bal_acc, _, _, _ = run_epoch(
            model,
            context.train_dataloader,
            context.criterion,
            context.device,
            optimizer=optimizer,
        )
        val_loss, val_acc, val_bal_acc, _, _, _ = run_epoch(
            model,
            context.val_dataloader,
            context.criterion,
            context.device,
        )

        save_checkpoint(
            model,
            optimizer,
            epoch,
            train_loss,
            val_loss,
            train_acc,
            val_acc,
            base_filename=context.checkpoint_dir,
        )
        save_metrics_to_csv(
            context.metrics_file,
            epoch,
            train_loss,
            val_loss,
            train_acc,
            val_acc,
            train_bal_acc,
            val_bal_acc,
        )

        print(
            f"Epoch [{epoch}/{context.config.epochs}], "
            f"Train Loss: {train_loss:.4f}, Train ACC: {train_acc:.4f}, Train BACC: {train_bal_acc:.4f}"
        )
        print(
            f"Epoch [{epoch}/{context.config.epochs}], "
            f"Val Loss: {val_loss:.4f}, Val ACC: {val_acc:.4f}, Val BACC: {val_bal_acc:.4f}"
        )

    print("Training complete.")
    return pd.read_csv(context.metrics_file)


def load_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    print(f"Loaded checkpoint {checkpoint_path}")


def select_best_checkpoint(context: ExperimentContext, metric_column: str):
    df = pd.read_csv(context.metrics_file)
    trained_df = df[df["epoch"] > 0].copy()
    best_epoch = int(trained_df.loc[trained_df[metric_column].idxmax(), "epoch"])
    checkpoint_path = context.checkpoint_dir / f"{best_epoch}.pth"
    print(f"Best {metric_column}: epoch {best_epoch}")
    print(f"Checkpoint to read: {checkpoint_path}")
    return checkpoint_path, df


def evaluate_loader(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_labels.append(labels.numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)

    try:
        if num_classes == 2:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")
    except ValueError:
        auc = float("nan")

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "balanced_accuracy": balanced_accuracy_score(all_labels, all_preds),
        "auc": auc,
    }
    return metrics, all_labels, all_preds, all_probs


def print_metrics(split_name: str, metrics: dict) -> None:
    print(f"{split_name} Accuracy: {metrics['accuracy']:.3f}")
    print(f"{split_name} Balanced Accuracy: {metrics['balanced_accuracy']:.3f}")
    print(f"{split_name} AUC: {metrics['auc']:.3f}")


def plot_training_curves(df: pd.DataFrame, title: str, output_path: Path):
    import matplotlib.pyplot as plt

    best_idx = df["val_balanced_accuracy"].idxmax()
    best_epoch = int(df.loc[best_idx, "epoch"])
    best_bacc = float(df.loc[best_idx, "val_balanced_accuracy"]) * 100.0

    epochs = df["epoch"]
    val_acc = df["val_accuracy"] * 100.0
    val_bacc = df["val_balanced_accuracy"] * 100.0
    train_acc = df["accuracy"] * 100.0
    train_bacc = df["balanced_accuracy"] * 100.0

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(epochs, val_bacc, color="#1f4e79", linewidth=2.6, label="Val BACC")
    ax.plot(epochs, val_acc, color="#4f81bd", linewidth=1.5, linestyle="--", alpha=0.95, label="Val ACC")
    ax.plot(epochs, train_bacc, color="#b45f06", linewidth=1.7, alpha=0.9, label="Train BACC")
    ax.plot(epochs, train_acc, color="#e69138", linewidth=1.4, linestyle="--", alpha=0.9, label="Train ACC")
    ax.axvline(best_epoch, color="#444444", linestyle=":", linewidth=1.0, alpha=0.85)
    ax.scatter([best_epoch], [best_bacc], color="#222222", s=28, zorder=5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score (%)")
    ax.set_title(title, pad=10)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    ax.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")
    ax.tick_params(colors="#444444")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.show()
    print(f"Saved plot to {output_path}")


def clean_checkpoints(checkpoint_dir: Path, keep_paths: set[Path]) -> None:
    keep_resolved = {Path(path).resolve() for path in keep_paths}
    for checkpoint_path in Path(checkpoint_dir).glob("*.pth"):
        if checkpoint_path.resolve() not in keep_resolved:
            print(f"Deleting: {checkpoint_path}")
            checkpoint_path.unlink()
