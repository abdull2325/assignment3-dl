import copy
import hashlib
import json
import math
import os
import random
import tarfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

def configure_environment(base_dir: Path) -> None:
    mpl_dir = base_dir / "artifacts" / "mplconfig"
    font_dir = base_dir / "artifacts" / "fontconfig"
    cache_dir = base_dir / "artifacts" / "cache"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    font_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    os.environ.setdefault("FONTCONFIG_PATH", "/opt/homebrew/etc/fonts")
    os.environ.setdefault("FONTCONFIG_FILE", "/opt/homebrew/etc/fonts/fonts.conf")
    os.environ.setdefault("FC_CACHEDIR", str(font_dir))


ROOT = Path(__file__).resolve().parents[1]
configure_environment(ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset, random_split
from torchvision import datasets, models, transforms
from torchvision.transforms import functional as TF


MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
STL10_CLASSES = [
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
]
STL10_REQUIRED_FILES = {
    "train_X.bin": "918c2871b30a85fa023e0c44e0bee87f",
    "train_y.bin": "5a34089d4802c674881badbb80307741",
    "test_X.bin": "7f263ba9f9e0b06b93213547f721ac82",
    "test_y.bin": "36f9794fa4beb8a2c72628de14fa638e",
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int = 600) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


class SmallCNN(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 24, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(24 * 7 * 7, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_stl10_train_test(root: Path) -> Path:
    base_dir = root / "stl10_binary"
    archive_path = root / "stl10_binary.tar.gz"
    base_dir.mkdir(parents=True, exist_ok=True)

    def missing_or_corrupt() -> List[str]:
        failures = []
        for filename, expected_md5 in STL10_REQUIRED_FILES.items():
            file_path = base_dir / filename
            if not file_path.exists() or file_md5(file_path) != expected_md5:
                failures.append(filename)
        return failures

    outstanding = missing_or_corrupt()
    if not outstanding:
        return base_dir

    if not archive_path.exists():
        raise RuntimeError(
            "STL-10 archive is missing. Expected data/stl10_binary.tar.gz before extracting train/test files."
        )

    wanted_members = {f"stl10_binary/{filename}" for filename in STL10_REQUIRED_FILES}
    try:
        with tarfile.open(archive_path, "r|gz") as tar:
            for member in tar:
                if member.name in wanted_members:
                    tar.extract(member, path=root)
                if not missing_or_corrupt():
                    break
    except tarfile.ReadError:
        # Partial archives fail once the stream reaches the unfinished unlabeled split.
        pass

    outstanding = missing_or_corrupt()
    if outstanding:
        raise RuntimeError(
            "STL-10 train/test files are not fully available yet. Missing or incomplete files: "
            + ", ".join(outstanding)
        )

    return base_dir


class STL10BinaryDataset(Dataset):
    def __init__(self, root: Path, split: str) -> None:
        if split not in {"train", "test"}:
            raise ValueError(f"Unsupported STL-10 split: {split}")
        self.base_dir = prepare_stl10_train_test(root)
        data_file = f"{split}_X.bin"
        labels_file = f"{split}_y.bin"
        self.data, self.labels = self._load_split(data_file, labels_file)
        self.classes = STL10_CLASSES

    def _load_split(self, data_file: str, labels_file: str) -> Tuple[np.ndarray, np.ndarray]:
        labels_path = self.base_dir / labels_file
        data_path = self.base_dir / data_file

        labels = np.fromfile(labels_path, dtype=np.uint8) - 1
        everything = np.fromfile(data_path, dtype=np.uint8)
        images = np.reshape(everything, (-1, 3, 96, 96))
        images = np.transpose(images, (0, 1, 3, 2))
        return images, labels

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __getitem__(self, index: int):
        image = Image.fromarray(np.transpose(self.data[index], (1, 2, 0)))
        label = int(self.labels[index])
        return image, label


class STL10ViewDataset(Dataset):
    def __init__(self, base_dataset: Dataset, indices: Sequence[int], transform) -> None:
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        image, label = self.base_dataset[self.indices[item]]
        return self.transform(image), label


def build_mnist_loaders(
    data_dir: Path,
    batch_size: int = 256,
    val_size: int = 5000,
) -> Dict[str, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ]
    )
    train_full = datasets.MNIST(root=data_dir, train=True, download=False, transform=transform)
    test_ds = datasets.MNIST(root=data_dir, train=False, download=False, transform=transform)
    train_size = len(train_full) - val_size
    train_ds, val_ds = random_split(
        train_full,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(600),
    )
    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0),
    }


def build_cmnist_loaders(
    train_path: Path,
    biased_test_path: Path,
    unbiased_test_path: Path,
    batch_size: int = 256,
    val_size: int = 5000,
) -> Dict[str, DataLoader]:
    train_images, train_labels = torch.load(train_path, map_location="cpu")
    biased_images, biased_labels = torch.load(biased_test_path, map_location="cpu")
    unbiased_images, unbiased_labels = torch.load(unbiased_test_path, map_location="cpu")

    train_full = TensorDataset(train_images, train_labels)
    biased_test = TensorDataset(biased_images, biased_labels)
    unbiased_test = TensorDataset(unbiased_images, unbiased_labels)

    train_size = len(train_full) - val_size
    train_ds, val_ds = random_split(
        train_full,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(600),
    )
    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0),
        "biased_test": DataLoader(biased_test, batch_size=batch_size, shuffle=False, num_workers=0),
        "unbiased_test": DataLoader(unbiased_test, batch_size=batch_size, shuffle=False, num_workers=0),
    }


def build_stl10_loaders(
    data_dir: Path,
    batch_size: int = 64,
    val_size: int = 500,
) -> Dict[str, DataLoader]:
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    base_train = STL10BinaryDataset(data_dir, split="train")
    base_test_raw = STL10BinaryDataset(data_dir, split="test")
    test_indices = list(range(len(base_test_raw)))
    base_test = STL10ViewDataset(base_test_raw, test_indices, eval_transform)

    indices = list(range(len(base_train)))
    generator = torch.Generator().manual_seed(600)
    shuffled = torch.randperm(len(indices), generator=generator).tolist()
    val_indices = shuffled[:val_size]
    train_indices = shuffled[val_size:]

    train_ds = STL10ViewDataset(base_train, train_indices, train_transform)
    val_ds = STL10ViewDataset(base_train, val_indices, eval_transform)

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0),
        "test": DataLoader(base_test, batch_size=batch_size, shuffle=False, num_workers=0),
    }


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def train_model(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> Tuple[Dict[str, List[float]], Dict[str, torch.Tensor]]:
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    best_state = copy.deepcopy(model.state_dict())
    best_val_acc = -math.inf

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0
        for inputs, labels in loaders["train"]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            running_correct += (logits.argmax(dim=1) == labels).sum().item()
            total += batch_size

        train_loss = running_loss / total
        train_acc = running_correct / total
        val_loss, val_acc = evaluate_model(model, loaders["val"], criterion, device)
        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch + 1:02d}/{epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}",
            flush=True,
        )

    model.load_state_dict(best_state)
    return history, best_state


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs)
        loss = criterion(logits, labels)
        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size
    return running_loss / total, running_correct / total


def plot_history(history: Dict[str, List[float]], out_path: Path, title: str) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, history["train_loss"], marker="o", label="Train")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="Validation")
    axes[0].set_title(f"{title}: Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], marker="o", label="Train")
    axes[1].plot(epochs, history["val_acc"], marker="o", label="Validation")
    axes[1].set_title(f"{title}: Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.01)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def denormalize_tensor(tensor: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    mean_tensor = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    std_tensor = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    return tensor * std_tensor + mean_tensor


def plot_first_layer_filters(model: SmallCNN, out_path: Path) -> None:
    weights = model.features[0].weight.detach().cpu().numpy()
    num_filters = weights.shape[0]
    cols = 4
    rows = math.ceil(num_filters / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.4))
    axes = np.array(axes).reshape(rows, cols)
    vmax = np.abs(weights).max()

    for idx, axis in enumerate(axes.flat):
        axis.axis("off")
        if idx >= num_filters:
            continue
        image = weights[idx, 0]
        axis.imshow(image, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        axis.set_title(f"Filter {idx + 1}")

    fig.suptitle("First Convolutional Filters", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_dataset_examples(
    dataset: Dataset,
    out_path: Path,
    title: str,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
    rows: int = 2,
    cols: int = 5,
) -> None:
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    indices = np.linspace(0, len(dataset) - 1, rows * cols, dtype=int)

    for axis, idx in zip(axes.flat, indices):
        image, label = dataset[idx]
        if isinstance(image, Image.Image):
            image = TF.to_tensor(image)
        if mean is not None and std is not None:
            image = denormalize_tensor(image, mean, std)
        image = image.detach().cpu().clamp(0, 1)
        if image.shape[0] == 1:
            axis.imshow(image.squeeze(0), cmap="gray")
        else:
            axis.imshow(np.transpose(image.numpy(), (1, 2, 0)))
        axis.set_title(f"y={label}")
        axis.axis("off")

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_bar(values: Dict[str, float], out_path: Path, title: str, ylabel: str) -> None:
    labels = list(values.keys())
    scores = [values[key] for key in labels]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, scores, color=["#24527a", "#c0392b", "#2d6a4f"][: len(labels)])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, max(scores) * 1.15 if scores else 1.0)
    ax.grid(axis="y", alpha=0.25)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, score + 0.01, f"{score:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output) -> None:
        self.activations = output.detach()
        if output.requires_grad:
            output.register_hook(self._gradient_hook)

    def _gradient_hook(self, grad: torch.Tensor) -> None:
        self.gradients = grad.detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> Tuple[np.ndarray, int]:
        input_tensor = input_tensor.detach().clone().requires_grad_(True)
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        score = logits[:, class_idx].sum()
        score.backward()

        assert self.activations is not None
        assert self.gradients is not None
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        cam = cam[0, 0]
        cam -= cam.min()
        cam /= cam.max().clamp(min=1e-8)
        return cam.detach().cpu().numpy(), class_idx

    def close(self) -> None:
        self.forward_handle.remove()


def heatmap_overlay(image: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    cmap = plt.get_cmap("jet")
    heatmap = cmap(cam)[..., :3]
    return np.clip((1 - alpha) * image + alpha * heatmap, 0, 1)


def collect_gradcam_examples(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    need_correct: int = 2,
    need_incorrect: int = 2,
) -> List[Dict]:
    model.eval()
    correct_examples: List[Dict] = []
    incorrect_examples: List[Dict] = []

    with torch.no_grad():
        for inputs, labels in loader:
            logits = model(inputs.to(device))
            preds = logits.argmax(dim=1).cpu()
            for image, label, pred in zip(inputs, labels, preds):
                entry = {
                    "image": image,
                    "label": int(label.item()),
                    "pred": int(pred.item()),
                }
                if pred.item() == label.item() and len(correct_examples) < need_correct:
                    correct_examples.append(entry)
                elif pred.item() != label.item() and len(incorrect_examples) < need_incorrect:
                    incorrect_examples.append(entry)
                if len(correct_examples) >= need_correct and len(incorrect_examples) >= need_incorrect:
                    return correct_examples + incorrect_examples
    return correct_examples + incorrect_examples


def plot_gradcam_examples(
    model: nn.Module,
    target_layer: nn.Module,
    examples: List[Dict],
    class_names: Sequence[str],
    out_path: Path,
    device: torch.device,
    mean: Sequence[float],
    std: Sequence[float],
) -> None:
    cam = GradCAM(model, target_layer)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    for axis, example in zip(axes.flat, examples):
        input_tensor = example["image"].unsqueeze(0).to(device)
        heatmap, pred = cam.generate(input_tensor, class_idx=example["pred"])
        image = denormalize_tensor(example["image"], mean, std).cpu().clamp(0, 1).numpy()
        image = np.transpose(image, (1, 2, 0))
        overlay = heatmap_overlay(image, heatmap)
        axis.imshow(overlay)
        verdict = "Correct" if example["pred"] == example["label"] else "Incorrect"
        axis.set_title(
            f"{verdict}: pred={class_names[pred]}\ntrue={class_names[example['label']]}"
        )
        axis.axis("off")

    fig.suptitle("GradCAM on STL-10 Test Samples", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    cam.close()


def build_resnet18_head(num_classes: int = 10) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def analyze_generalization(history: Dict[str, List[float]]) -> str:
    train_acc = history["train_acc"][-1]
    val_acc = history["val_acc"][-1]
    gap = train_acc - val_acc
    abs_gap = abs(gap)
    if gap > 0.05:
        return (
            f"The model shows mild overfitting: training accuracy ({train_acc:.3f}) "
            f"is noticeably above validation accuracy ({val_acc:.3f}), leaving a gap of {gap:.3f}."
        )
    if gap < -0.02:
        return (
            f"The model still generalizes well: validation accuracy ({val_acc:.3f}) is slightly above "
            f"training accuracy ({train_acc:.3f}), likely because dropout and data shuffling make the "
            f"training metric noisier than validation. The absolute gap is only {abs_gap:.3f}."
        )
    if val_acc < 0.90:
        return (
            f"The model is underfitting: validation accuracy ({val_acc:.3f}) remains low "
            f"and tracks the training curve closely."
        )
    return (
        f"The model generalizes well: training accuracy ({train_acc:.3f}) and validation accuracy "
        f"({val_acc:.3f}) stay close, with an absolute gap of {abs_gap:.3f}."
    )


def summarize_filters(weights: np.ndarray) -> str:
    center_weights = weights[:, 0]
    flat = center_weights.reshape(center_weights.shape[0], -1)
    polarity = np.sign(flat.mean(axis=1))
    edge_like = []
    for idx, kernel in enumerate(center_weights):
        horizontal = np.abs(kernel[0].sum() - kernel[-1].sum())
        vertical = np.abs(kernel[:, 0].sum() - kernel[:, -1].sum())
        if horizontal > vertical:
            direction = "horizontal edge / stroke transitions"
        else:
            direction = "vertical edge / stroke transitions"
        tone = "bright-center" if polarity[idx] > 0 else "dark-center / contrast"
        edge_like.append(f"Filter {idx + 1} emphasizes {direction} with a {tone} bias")
    return "; ".join(edge_like[:4]) + "."


def solve_toy_cnn() -> Dict:
    X = np.array([[1, 2, 0], [0, 1, 1], [1, 0, 2]], dtype=float)
    W = np.array([[1, 0], [-1, 1]], dtype=float)
    b = 0.0
    V = np.array([1, -1, 2, 0.5], dtype=float)
    c = 1.0
    target = 3.0

    Z = np.zeros((2, 2), dtype=float)
    for i in range(2):
        for j in range(2):
            patch = X[i : i + 2, j : j + 2]
            Z[i, j] = float((patch * W).sum() + b)
    A = np.maximum(Z, 0.0)
    flat = A.reshape(-1)
    Y = float(V @ flat + c)
    L = float(0.5 * (Y - target) ** 2)

    dL_dY = Y - target
    dL_dV = dL_dY * flat
    dL_dAflat = dL_dY * V
    dL_dA = dL_dAflat.reshape(2, 2)
    dL_dZ = dL_dA * (Z > 0)
    dL_dW = np.zeros_like(W)
    for i in range(2):
        for j in range(2):
            dL_dW += dL_dZ[i, j] * X[i : i + 2, j : j + 2]

    dL_dX = np.zeros_like(X)
    for i in range(2):
        for j in range(2):
            dL_dX[i : i + 2, j : j + 2] += dL_dZ[i, j] * W

    y_pool = float(A.max())
    pool_position = np.argwhere(A == y_pool)[0].tolist()
    pool_loss = float(0.5 * (y_pool - target) ** 2)

    mu = float(A.mean())
    variance = float(((A - mu) ** 2).mean())
    normalized = (A - mu) / math.sqrt(variance)

    x_crop = X[:2, :2]
    A_res = A + x_crop
    y_res = float(V @ A_res.reshape(-1) + c)
    loss_res = float(0.5 * (y_res - target) ** 2)
    dLres_dY = y_res - target
    dLres_dAres = (dLres_dY * V).reshape(2, 2)
    dLres_dZ = dLres_dAres * (Z > 0)
    dLres_dX_conv = np.zeros_like(X)
    for i in range(2):
        for j in range(2):
            dLres_dX_conv[i : i + 2, j : j + 2] += dLres_dZ[i, j] * W
    dLres_dX_skip = np.zeros_like(X)
    dLres_dX_skip[:2, :2] += dLres_dAres
    dLres_dX = dLres_dX_conv + dLres_dX_skip

    strongest_weight_index = np.unravel_index(np.abs(dL_dW).argmax(), dL_dW.shape)
    strongest_gradient = float(dL_dW[strongest_weight_index])

    return {
        "X": X.tolist(),
        "W": W.tolist(),
        "V": V.tolist(),
        "Z": Z.tolist(),
        "A": A.tolist(),
        "Y": Y,
        "loss": L,
        "dL_dY": dL_dY,
        "dL_dV": dL_dV.tolist(),
        "dL_dA": dL_dA.tolist(),
        "dL_dZ": dL_dZ.tolist(),
        "dL_dW": dL_dW.tolist(),
        "dL_dX": dL_dX.tolist(),
        "strongest_weight_index": [int(strongest_weight_index[0]), int(strongest_weight_index[1])],
        "strongest_weight_gradient": strongest_gradient,
        "q3_maxpool_output": y_pool,
        "q3_maxpool_loss": pool_loss,
        "q3_max_position": pool_position,
        "q3_layernorm_mean": mu,
        "q3_layernorm_variance": variance,
        "q3_layernorm_output": normalized.tolist(),
        "A_res": A_res.tolist(),
        "Y_res": y_res,
        "loss_res": loss_res,
        "dLres_dX": dLres_dX.tolist(),
    }


def latex_escape(text: str) -> str:
    mapping = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for key, value in mapping.items():
        text = text.replace(key, value)
    return text


def write_report_macros(results: Dict, out_path: Path) -> None:
    stl10_accuracy = results.get("stl10", {}).get("test_accuracy", 0.0)
    stl10_status = (
        "Completed"
        if not results.get("stl10", {}).get("skipped", False)
        else results.get("stl10", {}).get("reason", "Not run")
    )
    lines = [
        f"\\newcommand{{\\MNISTTestAcc}}{{{results['mnist']['test_accuracy'] * 100:.2f}\\%}}",
        f"\\newcommand{{\\CMNISTBiasedAcc}}{{{results['cmnist']['biased_test_accuracy'] * 100:.2f}\\%}}",
        f"\\newcommand{{\\CMNISTUnbiasedAcc}}{{{results['cmnist']['unbiased_test_accuracy'] * 100:.2f}\\%}}",
        f"\\newcommand{{\\STLTestAcc}}{{{stl10_accuracy * 100:.2f}\\%}}",
        f"\\newcommand{{\\STLStatus}}{{{latex_escape(stl10_status)}}}",
        f"\\newcommand{{\\MNISTParamCount}}{{{results['mnist']['trainable_parameters']}}}",
        f"\\newcommand{{\\CMNISTParamCount}}{{{results['cmnist']['trainable_parameters']}}}",
        f"\\newcommand{{\\ToyOutput}}{{{results['toy_cnn']['Y']:.3f}}}",
        f"\\newcommand{{\\ToyLoss}}{{{results['toy_cnn']['loss']:.3f}}}",
        f"\\newcommand{{\\ToyResidualOutput}}{{{results['toy_cnn']['Y_res']:.3f}}}",
        f"\\newcommand{{\\ToyResidualLoss}}{{{results['toy_cnn']['loss_res']:.3f}}}",
        f"\\newcommand{{\\MNISTGeneralization}}{{{latex_escape(results['mnist']['generalization_comment'])}}}",
        f"\\newcommand{{\\FilterInterpretation}}{{{latex_escape(results['mnist']['filter_comment'])}}}",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
