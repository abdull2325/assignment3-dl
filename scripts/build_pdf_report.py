import json
import math
import os
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
mpl_dir = ROOT / "artifacts" / "mplconfig"
font_dir = ROOT / "artifacts" / "fontconfig"
cache_dir = ROOT / "artifacts" / "cache"
mpl_dir.mkdir(parents=True, exist_ok=True)
font_dir.mkdir(parents=True, exist_ok=True)
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
os.environ.setdefault("FONTCONFIG_PATH", "/opt/homebrew/etc/fonts")
os.environ.setdefault("FONTCONFIG_FILE", "/opt/homebrew/etc/fonts/fonts.conf")
os.environ.setdefault("FC_CACHEDIR", str(font_dir))

import matplotlib
matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


REPORT_PATH = ROOT / "reports" / "assignment3_report.pdf"
RESULTS_PATH = ROOT / "artifacts" / "results.json"


def load_results() -> dict:
    return json.loads(RESULTS_PATH.read_text())


def add_text_page(pdf: PdfPages, title: str, sections) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    y = 0.96
    fig.text(0.08, y, title, fontsize=18, fontweight="bold", va="top")
    y -= 0.05

    for heading, body in sections:
        fig.text(0.08, y, heading, fontsize=12.5, fontweight="bold", va="top")
        y -= 0.024
        wrapped = textwrap.fill(body, width=95)
        fig.text(0.08, y, wrapped, fontsize=10.5, va="top", linespacing=1.45)
        line_count = wrapped.count("\n") + 1
        y -= 0.028 * line_count + 0.018
        if y < 0.08:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.patch.set_facecolor("white")
            y = 0.96

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_image_page(pdf: PdfPages, title: str, image_paths, captions, cols: int = 1) -> None:
    rows = math.ceil(len(image_paths) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(8.27, 11.69))
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)
    axes = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for axis, image_path, caption in zip(axes, image_paths, captions):
        axis.imshow(mpimg.imread(image_path))
        axis.set_title(caption, fontsize=11)
        axis.axis("off")

    for axis in axes[len(image_paths) :]:
        axis.axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.965))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    results = load_results()
    toy = results["toy_cnn"]

    with PdfPages(REPORT_PATH) as pdf:
        add_text_page(
            pdf,
            "AI600 Assignment 3 Report",
            [
                (
                    "Course Context",
                    "Lahore University of Management Sciences, School of Science and Engineering, "
                    "Department of Electrical Engineering. This report covers the analytical Toy-CNN "
                    "questions and the programming experiments for standard MNIST, Colored-MNIST, and STL-10. "
                    "Target GitHub repository: https://github.com/abdull2325/assignment3-dl.",
                ),
                (
                    "Execution Status",
                    "Standard MNIST, Colored-MNIST, and STL-10 were all executed successfully. For STL-10, "
                    "the required train/test binaries were extracted from the official archive before the "
                    "large unlabeled split completed downloading, which was sufficient for Task 2.",
                ),
                (
                    "Key Metrics",
                    f"MNIST test accuracy: {results['mnist']['test_accuracy']*100:.2f}%. "
                    f"Colored-MNIST biased test accuracy: {results['cmnist']['biased_test_accuracy']*100:.2f}%. "
                    f"Colored-MNIST unbiased test accuracy: {results['cmnist']['unbiased_test_accuracy']*100:.2f}%. "
                    f"STL-10 test accuracy: {results['stl10']['test_accuracy']*100:.2f}%. "
                    f"Custom CNN parameter count: {results['mnist']['trainable_parameters']}.",
                ),
            ],
        )

        add_text_page(
            pdf,
            "Analytical Solutions",
            [
                (
                    "Q1 Forward Pass",
                    f"The convolution pre-activation map is Z = [[2, 2], [-1, 3]]. After ReLU, "
                    f"A = [[2, 2], [0, 3]]. Flattening gives [2, 2, 0, 3]^T. The scalar prediction is "
                    f"Y = {toy['Y']:.3f} and the loss is L = {toy['loss']:.3f}.",
                ),
                (
                    "Q1 Backward Pass",
                    "The loss gradient with respect to the fully connected weights is "
                    "dL/dV = [-1, -1, 0, -1.5]. The convolution-filter gradient is "
                    "dL/dW = [[0.25, -1.25], [0.5, -0.5]]. The largest-magnitude entry is the top-right "
                    "weight, so gradient descent increases that weight the most.",
                ),
                (
                    "Q3 Pooling and Normalization",
                    "With global max pooling, Y = max(A) = 3, which matches the target, so the loss is zero and "
                    "all input gradients are zero in this numeric example. For layer normalization on A, the mean "
                    "is 1.75, the variance is 1.1875, and the normalized map is approximately "
                    "[[0.2294, 0.2294], [-1.6059, 1.1471]].",
                ),
                (
                    "Practice Questions",
                    "For completeness, the input gradient in Q2 is "
                    "dL/dX = [[-0.5, 0.5, 0], [0.5, -1.25, 0.5], [0, 0.25, -0.25]], so an adversary would "
                    "decrease the center pixel to increase the loss. In the residual variant, "
                    "A_res = [[3, 4], [0, 4]], Y_res = 2.0, and dL/dX_11 becomes -2.0, illustrating the "
                    "gradient-highway effect of skip connections.",
                ),
            ],
        )

        add_text_page(
            pdf,
            "Programming Analysis",
            [
                (
                    "Task 1A: Standard MNIST",
                    f"The custom CNN uses three convolutional layers and two linear layers while staying under "
                    f"the 50,000-parameter budget with {results['mnist']['trainable_parameters']} trainable "
                    f"parameters. Final test accuracy is {results['mnist']['test_accuracy']*100:.2f}%. "
                    f"{results['mnist']['generalization_comment']}",
                ),
                (
                    "Task 1B: Colored-MNIST",
                    f"The same architecture was adapted to RGB input. It reached "
                    f"{results['cmnist']['biased_test_accuracy']*100:.2f}% on the biased test set but only "
                    f"{results['cmnist']['unbiased_test_accuracy']*100:.2f}% on the unbiased test set, "
                    f"confirming shortcut learning through color. {results['cmnist']['shortcut_comment']}",
                ),
                (
                    "Strategies to Reduce Shortcut Learning",
                    "Useful interventions include grayscale conversion, heavy color jitter, randomized recoloring, "
                    "balanced color-label pairings, and invariant-learning objectives such as Group DRO or IRM.",
                ),
                (
                    "Task 2: Transfer Learning and GradCAM",
                    f"A frozen-backbone pretrained ResNet-18 was fine-tuned on STL-10 by training only the "
                    f"classification head. The final STL-10 test accuracy is "
                    f"{results['stl10']['test_accuracy']*100:.2f}%. The GradCAM overlays show that correct "
                    f"predictions usually center on the main object, while mistakes often over-emphasize a "
                    f"localized part such as a face, branch crossing, or ambiguous body contour.",
                ),
            ],
        )

        add_image_page(
            pdf,
            "MNIST Results",
            [
                ROOT / "reports" / "figures" / "mnist_history.png",
                ROOT / "reports" / "figures" / "mnist_filters.png",
            ],
            [
                "Training and validation curves",
                "First-layer convolutional filters",
            ],
            cols=1,
        )

        add_image_page(
            pdf,
            "Colored-MNIST Results",
            [
                ROOT / "reports" / "figures" / "cmnist_history.png",
                ROOT / "reports" / "figures" / "cmnist_bias_gap.png",
            ],
            [
                "Training and validation curves",
                "Accuracy gap between biased and unbiased tests",
            ],
            cols=1,
        )

        add_image_page(
            pdf,
            "Sample Visualizations",
            [
                ROOT / "reports" / "figures" / "mnist_samples.png",
                ROOT / "reports" / "figures" / "cmnist_samples_train.png",
                ROOT / "reports" / "figures" / "cmnist_samples_unbiased.png",
            ],
            [
                "MNIST sample grid",
                "Colored-MNIST biased training samples",
                "Colored-MNIST unbiased test samples",
            ],
            cols=1,
        )

        add_image_page(
            pdf,
            "STL-10 Results",
            [
                ROOT / "reports" / "figures" / "stl10_history.png",
                ROOT / "reports" / "figures" / "stl10_gradcam.png",
            ],
            [
                "Frozen-backbone ResNet-18 training curves",
                "GradCAM overlays on correct and incorrect test predictions",
            ],
            cols=1,
        )

        add_text_page(
            pdf,
            "AI Policy Disclosure",
            [
                (
                    "Prompt Used",
                    "you have to complete the following assessmetn make sure things are aligned and we are good "
                    "to go accordingly make sure you complete the assignment along with the visualisation of "
                    "everything and make sure its completed thoroughly and greatly",
                ),
                (
                    "Generated Output",
                    "The AI-assisted workflow generated analytical derivations, PyTorch training code, plotting "
                    "utilities, report scaffolding, and written interpretations of the observed results.",
                ),
                (
                    "Edits Applied",
                    "The generated content was edited to match the exact numerical outputs, enforce the parameter "
                    "budget, incorporate measured metrics from artifacts/results.json, patch the STL-10 loader "
                    "to use the extracted train/test binaries from the official archive, and add the actual "
                    "GradCAM-based interpretation from the generated heatmaps.",
                ),
            ],
        )

    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
