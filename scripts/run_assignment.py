import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.assignment3 import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    MNIST_MEAN,
    MNIST_STD,
    ROOT as PROJECT_ROOT,
    SmallCNN,
    analyze_generalization,
    build_cmnist_loaders,
    build_mnist_loaders,
    build_resnet18_head,
    build_stl10_loaders,
    collect_gradcam_examples,
    count_parameters,
    ensure_dir,
    plot_bar,
    plot_dataset_examples,
    plot_first_layer_filters,
    plot_gradcam_examples,
    plot_history,
    save_json,
    set_seed,
    solve_toy_cnn,
    summarize_filters,
    train_model,
    evaluate_model,
    write_report_macros,
    STL10_CLASSES,
)


def run_mnist(results: dict, epochs: int, device: torch.device) -> None:
    print("Running MNIST experiment", flush=True)
    loaders = build_mnist_loaders(PROJECT_ROOT / "data", batch_size=256, val_size=5000)
    plot_dataset_examples(
        loaders["train"].dataset.dataset,
        PROJECT_ROOT / "reports" / "figures" / "mnist_samples.png",
        "MNIST Samples",
        mean=MNIST_MEAN,
        std=MNIST_STD,
    )

    model = SmallCNN(in_channels=1).to(device)
    params = count_parameters(model)
    if params > 50000:
        raise ValueError(f"MNIST model exceeds parameter budget: {params}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history, best_state = train_model(model, loaders, optimizer, criterion, device, epochs)
    torch.save(best_state, PROJECT_ROOT / "models" / "mnist_cnn.pt")

    test_loss, test_acc = evaluate_model(model, loaders["test"], criterion, device)
    plot_history(history, PROJECT_ROOT / "reports" / "figures" / "mnist_history.png", "MNIST")
    plot_first_layer_filters(model, PROJECT_ROOT / "reports" / "figures" / "mnist_filters.png")

    filter_comment = summarize_filters(model.features[0].weight.detach().cpu().numpy())
    generalization_comment = analyze_generalization(history)

    results["mnist"] = {
        "trainable_parameters": params,
        "history": history,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "generalization_comment": generalization_comment,
        "filter_comment": filter_comment,
    }


def run_cmnist(results: dict, epochs: int, device: torch.device) -> None:
    print("Running Colored-MNIST experiment", flush=True)
    loaders = build_cmnist_loaders(
        PROJECT_ROOT / "train_biased.pt",
        PROJECT_ROOT / "test_biased.pt",
        PROJECT_ROOT / "test_unbiased.pt",
        batch_size=256,
        val_size=5000,
    )
    plot_dataset_examples(
        loaders["train"].dataset.dataset,
        PROJECT_ROOT / "reports" / "figures" / "cmnist_samples_train.png",
        "Colored-MNIST Biased Training Samples",
    )
    plot_dataset_examples(
        loaders["unbiased_test"].dataset,
        PROJECT_ROOT / "reports" / "figures" / "cmnist_samples_unbiased.png",
        "Colored-MNIST Unbiased Test Samples",
    )

    model = SmallCNN(in_channels=3).to(device)
    params = count_parameters(model)
    if params > 50000:
        raise ValueError(f"C-MNIST model exceeds parameter budget: {params}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history, best_state = train_model(model, loaders, optimizer, criterion, device, epochs)
    torch.save(best_state, PROJECT_ROOT / "models" / "cmnist_cnn.pt")

    biased_loss, biased_acc = evaluate_model(model, loaders["biased_test"], criterion, device)
    unbiased_loss, unbiased_acc = evaluate_model(model, loaders["unbiased_test"], criterion, device)
    plot_history(
        history,
        PROJECT_ROOT / "reports" / "figures" / "cmnist_history.png",
        "Colored-MNIST",
    )
    plot_bar(
        {
            "Biased test": biased_acc,
            "Unbiased test": unbiased_acc,
        },
        PROJECT_ROOT / "reports" / "figures" / "cmnist_bias_gap.png",
        "Shortcut Learning on Colored-MNIST",
        "Accuracy",
    )

    shortcut_gap = biased_acc - unbiased_acc
    explanation = (
        f"The network attains {biased_acc:.3f} accuracy on the biased test split but falls to "
        f"{unbiased_acc:.3f} on the unbiased split, a drop of {shortcut_gap:.3f}. "
        "Cross-entropy minimization rewards whichever feature most reliably reduces training loss. "
        "Because color is highly correlated with the label in the biased training data and is easier "
        "to separate than shape, the optimizer converges to color-sensitive filters first."
    )

    results["cmnist"] = {
        "trainable_parameters": params,
        "history": history,
        "biased_test_loss": biased_loss,
        "biased_test_accuracy": biased_acc,
        "unbiased_test_loss": unbiased_loss,
        "unbiased_test_accuracy": unbiased_acc,
        "shortcut_comment": explanation,
    }


def run_stl10(results: dict, epochs: int, device: torch.device) -> None:
    print("Running STL-10 transfer learning experiment", flush=True)
    loaders = build_stl10_loaders(PROJECT_ROOT / "data", batch_size=64, val_size=500)
    plot_dataset_examples(
        loaders["train"].dataset,
        PROJECT_ROOT / "reports" / "figures" / "stl10_samples.png",
        "STL-10 Samples",
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    )

    model = build_resnet18_head(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(epochs // 2, 1), gamma=0.3)

    history, best_state = train_model(model, loaders, optimizer, criterion, device, epochs, scheduler=scheduler)
    torch.save(best_state, PROJECT_ROOT / "models" / "stl10_resnet18_head.pt")

    test_loss, test_acc = evaluate_model(model, loaders["test"], criterion, device)
    plot_history(
        history,
        PROJECT_ROOT / "reports" / "figures" / "stl10_history.png",
        "STL-10 ResNet-18 Head Fine-Tuning",
    )

    examples = collect_gradcam_examples(model, loaders["test"], device)
    if len(examples) >= 4:
        plot_gradcam_examples(
            model,
            model.layer4[-1].conv2,
            examples[:4],
            STL10_CLASSES,
            PROJECT_ROOT / "reports" / "figures" / "stl10_gradcam.png",
            device,
            IMAGENET_MEAN,
            IMAGENET_STD,
        )

    results["stl10"] = {
        "history": history,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "frozen_backbone": True,
        "trainable_parameters": sum(param.numel() for param in model.fc.parameters()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AI600 Assignment 3 experiments.")
    parser.add_argument("--mnist-epochs", type=int, default=8)
    parser.add_argument("--cmnist-epochs", type=int, default=8)
    parser.add_argument("--stl10-epochs", type=int, default=12)
    parser.add_argument("--skip-mnist", action="store_true")
    parser.add_argument("--skip-cmnist", action="store_true")
    parser.add_argument("--skip-stl10", action="store_true")
    args = parser.parse_args()

    set_seed(600)
    ensure_dir(PROJECT_ROOT / "artifacts")
    ensure_dir(PROJECT_ROOT / "models")
    ensure_dir(PROJECT_ROOT / "reports" / "figures")

    device = torch.device("cpu")
    torch.set_num_threads(max(1, min(8, (os.cpu_count() or 4))))

    results_path = PROJECT_ROOT / "artifacts" / "results.json"
    if results_path.exists():
        results = json.loads(results_path.read_text())
    else:
        results = {}

    results["toy_cnn"] = solve_toy_cnn()
    save_json(results["toy_cnn"], PROJECT_ROOT / "artifacts" / "toy_cnn_results.json")

    if not args.skip_mnist:
        run_mnist(results, args.mnist_epochs, device)
        save_json(results, results_path)
    elif "mnist" not in results:
        raise RuntimeError("MNIST results were skipped but no prior results exist in artifacts/results.json.")

    if not args.skip_cmnist:
        run_cmnist(results, args.cmnist_epochs, device)
        save_json(results, results_path)
    elif "cmnist" not in results:
        raise RuntimeError("Colored-MNIST results were skipped but no prior results exist in artifacts/results.json.")

    if args.skip_stl10:
        results["stl10"] = {
            "skipped": True,
            "reason": "STL-10 dataset download was skipped.",
        }
    else:
        run_stl10(results, args.stl10_epochs, device)

    save_json(results, results_path)
    write_report_macros(results, PROJECT_ROOT / "reports" / "generated_results.tex")
    print("Finished. Results written to artifacts/results.json", flush=True)


if __name__ == "__main__":
    main()
