# Probabilistic AI — ETH Zürich Assignments

![Python](https://img.shields.io/badge/Python-3.8-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red?logo=pytorch)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikit-learn)

Solutions to three assignments from the **Probabilistic Artificial Intelligence** course at ETH Zürich. Each task focuses on a different aspect of uncertainty quantification and principled decision-making under uncertainty.

---

## Table of Contents

- [Task 1 — Pollution Prediction with Gaussian Processes](#task-1--pollution-prediction-with-gaussian-processes)
- [Task 2 — Uncertainty-Aware Image Classification (SWAG)](#task-2--uncertainty-aware-image-classification-swag)
- [Task 3 — Safe Bayesian Optimization](#task-3--safe-bayesian-optimization)
- [How to Run](#how-to-run)

---

## Task 1 — Pollution Prediction with Gaussian Processes

**Folder:** `task1-gaussian-processes/`

Predict air pollution concentration across city districts from 2D spatial coordinates. The solution uses **Gaussian Mixture Models** (40 clusters) to partition the spatial domain, then fits a separate **Gaussian Process regressor** per cluster with a composite RBF + Matérn kernel. Predictions account for an asymmetric cost function: underpredictions in residential areas are penalized more heavily, so the model shifts its quantile estimate depending on area type.

**Key techniques:** GP regression, GMM clustering, kernel composition, asymmetric cost-aware predictions.

---

## Task 2 — Uncertainty-Aware Image Classification (SWAG)

**Folder:** `task2-swag-uncertainty/`

6-class image classification (60×60 RGB) with **calibrated uncertainty** using **Stochastic Weight Averaging-Gaussian (SWAG)**. Implements three inference modes:

- **MAP**: standard point-estimate prediction with pretrained weights.
- **SWAG-Diagonal**: tracks running mean and variance of weights during SGD to approximate a Gaussian posterior.
- **SWAG-Full**: extends SWAG-Diagonal with a low-rank deviation matrix for a richer posterior approximation.

At test time, multiple networks are sampled from the posterior and averaged (Bayesian model averaging). The model can output "don't know" (class −1) for samples where predictive confidence falls below a calibrated threshold.

**Key techniques:** Bayesian deep learning, SWAG posterior approximation, Expected Calibration Error (ECE), predictive uncertainty, batch-norm recalibration.

---

## Task 3 — Safe Bayesian Optimization

**Folder:** `task3-safe-bayesian-opt/`

Maximize a black-box objective function f(x) over x ∈ [0, 10] subject to a **hard safety constraint** v(x) ≤ 4. Two separate Gaussian Processes model the objective and the constraint. The algorithm tracks a *safe region* — the set of points where the upper confidence bound on v stays below the threshold — and only evaluates the acquisition function inside it. The acquisition function balances exploration and exploitation while penalizing proximity to the unsafe boundary.

**Key techniques:** Constrained Bayesian optimization, GP-based safety modeling, safe region tracking, upper/lower confidence bounds, L-BFGS-B acquisition optimization with multiple restarts.

---

## How to Run

Each task folder contains a self-contained environment:

```bash
# 1. Create and activate the conda environment
conda env create -f env.yaml
conda activate <env-name>   # see env.yaml for the name

# 2. Run the solution locally
bash runner.sh
```

> **Note:** Large data files (`.npz`, `.pt`) and the encrypted grading client are excluded from this repository. The CSV training data for task1 is included.
