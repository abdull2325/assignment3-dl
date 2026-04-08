# assignment3-dl

AI600 Assignment 3 submission repository for LUMS.

Student: `Muhammad Abdullah`  
Roll Number: `25280050`  
GitHub: `https://github.com/abdull2325/assignment3-dl`

## Overview

This repository contains the full submission package for Assignment 3:

- analytical Toy-CNN derivations
- PyTorch implementations for MNIST, Colored-MNIST, and STL-10
- training curves, learned-filter visualizations, and GradCAM outputs
- LaTeX report source and compiled PDF

## Final Results

| Task | Model | Result |
| --- | --- | --- |
| Standard MNIST | Custom CNN | `98.76%` test accuracy |
| Colored-MNIST biased split | Custom CNN (RGB) | `98.49%` accuracy |
| Colored-MNIST unbiased split | Custom CNN (RGB) | `75.52%` accuracy |
| STL-10 | Frozen ResNet-18 + linear head | `79.61%` test accuracy |

## Repository Layout

- `src/assignment3.py`  
  Core code for datasets, models, training loops, plotting, GradCAM, and analytical calculations.

- `scripts/run_assignment.py`  
  Main experiment runner for MNIST, Colored-MNIST, and STL-10.

- `scripts/build_pdf_report.py`  
  Fallback PDF builder. The main report is now compiled directly from LaTeX.

- `reports/assignment3_report.tex`  
  Final LaTeX report source.

- `reports/assignment3_report.pdf`  
  Final compiled submission PDF.

- `reports/generated_results.tex`  
  Auto-generated LaTeX macros populated from experiment outputs.

- `reports/figures/`  
  All figures used in the report, including:
  - MNIST training curves
  - first-layer filter visualizations
  - Colored-MNIST bias-gap plots
  - STL-10 training curves
  - GradCAM overlays

- `artifacts/results.json`  
  Consolidated experiment metrics and textual summaries.

- `artifacts/toy_cnn_results.json`  
  Exact Toy-CNN numerical outputs used in the analytical section.

## How To Reproduce

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full experiment pipeline:

```bash
PYTHONPYCACHEPREFIX=artifacts/pycache python3 scripts/run_assignment.py
```

Run only STL-10 using previously saved MNIST and Colored-MNIST results:

```bash
PYTHONPYCACHEPREFIX=artifacts/pycache python3 scripts/run_assignment.py --skip-mnist --skip-cmnist
```

If needed, skip STL-10:

```bash
PYTHONPYCACHEPREFIX=artifacts/pycache python3 scripts/run_assignment.py --skip-stl10
```

## Report Build

Compile the LaTeX report:

```bash
tectonic reports/assignment3_report.tex --outdir reports
```

This produces:

- `reports/assignment3_report.pdf`

## Key Deliverables

- Final report PDF: [`reports/assignment3_report.pdf`](reports/assignment3_report.pdf)
- Final report source: [`reports/assignment3_report.tex`](reports/assignment3_report.tex)
- Main experiment runner: [`scripts/run_assignment.py`](scripts/run_assignment.py)
- Core implementation: [`src/assignment3.py`](src/assignment3.py)
- Final metrics: [`artifacts/results.json`](artifacts/results.json)

## Notes

- The STL-10 pipeline uses the required train/test binary files extracted from the official archive.
- The ResNet-18 backbone is frozen, and only the final classification head is trained.
- GradCAM visualizations are generated from the final convolutional block and included in the report.
