# assignment3-dl

This repository contains the analytical derivations, PyTorch code, figures, and report source for AI600 Assignment 3.

GitHub target: `https://github.com/abdull2325/assignment3-dl`

## Workspace Layout

- `src/assignment3.py`: reusable models, loaders, plotting utilities, GradCAM helper, and Toy-CNN derivations
- `scripts/run_assignment.py`: end-to-end experiment runner
- `reports/assignment3_report.tex`: LaTeX report source
- `reports/generated_results.tex`: auto-generated metrics/macros from the latest run
- `reports/figures/`: generated figures for the report
- `artifacts/results.json`: experiment metrics and analysis text

## Reproducing Results

1. Download MNIST and STL-10 into `data/` or let torchvision download them.
2. Run:

```bash
PYTHONPYCACHEPREFIX=artifacts/pycache python3 scripts/run_assignment.py
```

To skip STL-10 while the archive is still downloading:

```bash
PYTHONPYCACHEPREFIX=artifacts/pycache python3 scripts/run_assignment.py --skip-stl10
```

## Current Status

- Standard MNIST test accuracy: `98.76%`
- Colored-MNIST biased / unbiased test accuracy: `98.49%` / `75.52%`
- STL-10 frozen-backbone ResNet-18 test accuracy: `79.61%`
- GradCAM overlays are generated in `reports/figures/stl10_gradcam.png`
- The STL-10 pipeline works from the extracted train/test binaries even if the full unlabeled archive is still downloading
