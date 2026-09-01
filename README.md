# README

## Overview

This repository provides the Python code for a data-driven materials design workflow, including:

1. Pearson correlation analysis  
2. Model comparison and accuracy test  
3. Feature selection  
4. Multi-objective optimization  

The uploaded dataset is a **small demo dataset**. It is provided to demonstrate that the scripts can run correctly and that the workflow is complete. Because the demo dataset is smaller than the full research dataset, the model accuracy and optimization results may differ from those reported in the manuscript. However, the code structure, model-training procedure, feature-selection workflow and optimization functions are exactly the same.

---

## System Requirements

Recommended environment:

```text
Python >= 3.9
Windows 10/11, Linux or macOS
Standard CPU
RAM >= 8 GB
GPU is not required
```

Install required packages:

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn openpyxl xgboost lightgbm pymoo
```

Typical installation time is several minutes on a standard desktop computer.

---

## Input Data

The main input file is:

```text
data.xlsx
```

The default format is:

```text
Input features | GS | TR | UTS | EL | T1 | T2 | T3
```

The last seven columns are target-related variables:

```text
GS, TR, UTS, EL, T1, T2, T3
```

The columns before the last seven columns are used as input features.

In this workflow:

- `GS`, `TR`, `UTS` and `EL` are target features obtained from experiments or preprocessed data.
- `T1`, `T2` and `T3` are reconstructed optimization targets.
- The last seven target-related columns are excluded from the input feature matrix to avoid information leakage.

---

## 1. Pearson Correlation Analysis

Script:

```bash
python "1.Pearson correlation analysis.py"
```

Purpose:

- Calculate Pearson correlation coefficients among numerical variables.
- Generate a correlation matrix and heatmap.
- Help identify strongly correlated or redundant variables.

Typical outputs:

```text
pearson_correlation_results/
pearson_correlation_matrix.xlsx
pearson_correlation_heatmap.png
pearson_correlation_heatmap.pdf
```

---

## 2. Model Comparison and Accuracy Test

Script:

```bash
python "2.Model comparison and accuracy test.py"
```

Purpose:

- Predict each of the seven targets independently.
- Compare different regression algorithms.
- Select the best-performing model for each target.

Targets:

```text
GS, TR, UTS, EL, T1, T2, T3
```

Algorithms:

```text
GBDT, XGB, LGBM, RF, DT, SVR, LR, ANN, KNN
```

Evaluation metrics:

```text
R2
RMSE
```

The model comparison bar charts include error bars calculated from repeated train/test splits. The best model for each target is selected mainly according to test-set R2, with RMSE used as an additional reference.

Typical outputs:

```text
model_comparison_7_targets_results/
all_targets_model_comparison_summary.xlsx
*_model_comparison_bar.png
*_best_model_scatter.png
```

---

## 3. Feature Selection

Script:

```bash
python "3.Feature Selection.py"
```

Purpose:

- Rank input feature importance.
- Remove highly correlated redundant features.
- Select the final feature subset using recursive feature elimination with cross-validation.

Feature-selection procedure:

```text
1. Model-derived feature-importance ranking
2. Pearson-correlation-based redundancy filtering
3. Recursive feature elimination with cross-validation
```

Typical outputs:

```text
feature_selection_RFE_results/
*_feature_importance_correlation_filter.png
*_RFECV_curve.png
*_RFE_ranking.png
*_scatter_RFE_model.png
*_RFE_feature_selection_results.xlsx
all_targets_RFE_feature_selection_summary.xlsx
```

---

## 4. Multi-Objective Optimization

Script:

```bash
python "4.Multi-objective optimization.py"
```

Purpose:

- Train surrogate models for reconstructed targets.
- Perform multi-objective optimization using NSGA-III.
- Recommend candidate compositions or processing parameters.

Default optimization targets:

```text
T1, T2, T3
```

Default surrogate models:

```text
T1: XGB
T2: LGBM
T3: GBDT
```

Typical outputs:

```text
NSGAIII_recommended_candidates.xlsx
```

Typical sheets:

```text
Recommended
All_Pareto_Candidates
Model_Report
Search_Bounds
Feature_Info
```

---

## Suggested Running Order

```bash
python "1.Pearson correlation analysis.py"
python "2.Model comparison and accuracy test.py"
python "3.Feature Selection.py"
python "4.Multi-objective optimization.py"
```

---

## Notes on Demo Data and Reproducibility

The uploaded dataset is a **small dataset for code demonstration**. It is intended to verify the functionality of the scripts, including data loading, model training, plotting, feature selection and optimization.

Because the demo dataset contains fewer samples than the full dataset used in the study:

- The prediction accuracy may be lower or different.
- The selected best model may change.
- The selected feature subset may change.
- The final Pareto candidates may differ.

These differences are expected and do not indicate a code error. The computational workflow and functions are the same as those used for the full dataset.

For better reproducibility, record the exact package versions using:

```bash
pip freeze > requirements.txt
```

---

## Troubleshooting

### Matplotlib or Tkinter error

For batch execution, use a non-interactive backend:

```python
import matplotlib
matplotlib.use("Agg")
```

This avoids errors such as:

```text
RuntimeError: main thread is not in main loop
Tcl_AsyncDelete: async handler deleted by the wrong thread
```

### Missing package

Install the missing package:

```bash
pip install package_name
```

For example:

```bash
pip install xgboost lightgbm pymoo openpyxl
```

### Incorrect data format

Make sure that `data.xlsx` is placed in the same folder as the scripts and that the last seven columns are:

```text
GS, TR, UTS, EL, T1, T2, T3
```
