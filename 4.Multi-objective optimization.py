# -*- coding: utf-8 -*-

import os
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

# ============================================================
# 1. Basic settings
# ============================================================

DATA_PATH = "data-selection.xlsx"

SHEET_CONFIG = {
    "T1": 0,   # Sheet1: features + T1
    "T2": 1,   # Sheet2: features + T2
    "T3": 2,   # Sheet3: features + T3
}

TARGETS = ["T1", "T2", "T3"]

RANDOM_STATE = 42

N_RECOMMEND = 30

NSGA_POP_SIZE = 300
NSGA_N_GEN = 1000

DISCRETE_UNIQUE_THRESHOLD = 8

FORCE_DISCRETE_COLS = []

FORCE_CONTINUOUS_COLS = []

OUTPUT_EXCEL = "NSGAIII_recommended_candidates.xlsx"


# ============================================================
# 2. Read each sheet
# ============================================================

def read_target_sheet(data_path, sheet_index, target_name):
    df = pd.read_excel(data_path, sheet_name=sheet_index)
    df = df.dropna(axis=1, how="all")
    df.columns = [str(c).strip() for c in df.columns]

    if df.shape[1] < 2:
        raise ValueError(
            f"Sheet{sheet_index + 1} must contain at least one feature column and one target column."
        )

    target_col = df.columns[-1]
    feature_cols = list(df.columns[:-1])

    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    df = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)

    if len(df) < 3:
        raise ValueError(
            f"Sheet{sheet_index + 1} has too few valid samples after removing missing values."
        )

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    return df, X, y, feature_cols, target_col


data_dict = {}
X_dict = {}
y_dict = {}
feature_dict = {}
target_col_dict = {}

for target_name, sheet_index in SHEET_CONFIG.items():
    df, X, y, feature_cols, target_col = read_target_sheet(
        DATA_PATH,
        sheet_index,
        target_name
    )

    data_dict[target_name] = df
    X_dict[target_name] = X
    y_dict[target_name] = y
    feature_dict[target_name] = feature_cols
    target_col_dict[target_name] = target_col

    print("=" * 70)
    print(f"Target: {target_name}")
    print(f"Sheet: Sheet{sheet_index + 1}")
    print(f"Samples: {len(df)}")
    print(f"Input features: {feature_cols}")
    print(f"Target column: {target_col}")


# ============================================================
# 3. Automatically build design space
# ============================================================

def ordered_union(list_of_lists):
    result = []
    for cols in list_of_lists:
        for c in cols:
            if c not in result:
                result.append(c)
    return result


def is_nearly_integer(values, tol=1e-8):
    values = np.asarray(values, dtype=float)
    return np.all(np.abs(values - np.round(values)) < tol)


design_cols = ordered_union([feature_dict[t] for t in TARGETS])

bounds = {}
observed_values = {}
discrete_values = {}

for col in design_cols:
    values = []

    for target_name in TARGETS:
        if col in data_dict[target_name].columns:
            v = data_dict[target_name][col].dropna().values
            values.extend(v)

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        raise ValueError(f"No valid values were found for design variable: {col}")

    vmin = float(np.min(values))
    vmax = float(np.max(values))

    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    bounds[col] = (vmin, vmax)

    unique_vals = np.unique(np.round(values, 10))
    observed_values[col] = unique_vals

    use_discrete = False

    if col in FORCE_DISCRETE_COLS:
        use_discrete = True

    if col in FORCE_CONTINUOUS_COLS:
        use_discrete = False

    if col not in FORCE_CONTINUOUS_COLS:
        if len(unique_vals) <= DISCRETE_UNIQUE_THRESHOLD:
            use_discrete = True
        elif is_nearly_integer(unique_vals) and len(unique_vals) <= 20:
            use_discrete = True

    if use_discrete:
        discrete_values[col] = unique_vals


print("\n" + "=" * 70)
print("Automatically detected design variables:")
print(design_cols)

print("\nSearch bounds:")
for col in design_cols:
    print(f"{col}: {bounds[col]}")

print("\nDiscrete variables:")
if len(discrete_values) == 0:
    print("None")
else:
    for col, vals in discrete_values.items():
        print(f"{col}: {vals}")


# ============================================================
# 4. Fixed surrogate models
# ============================================================

def build_fixed_model(target_name):
    if target_name == "T1":
        model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        model_name = "XGBRegressor"

    elif target_name == "T2":
        model = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=-1,
            num_leaves=15,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1
        )
        model_name = "LGBMRegressor"

    elif target_name == "T3":
        model = GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.85,
            random_state=RANDOM_STATE
        )
        model_name = "GradientBoostingRegressor"

    else:
        raise ValueError(f"Unknown target name: {target_name}")

    return model, model_name


def evaluate_and_train_model(X, y, target_name):
    model, model_name = build_fixed_model(target_name)

    n_samples = len(X)
    n_splits = min(5, n_samples)

    cv = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    try:
        cv_r2_scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring="r2",
            n_jobs=-1
        )

        cv_rmse_scores = -cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1
        )

        cv_r2 = float(np.nanmean(cv_r2_scores))
        cv_rmse = float(np.nanmean(cv_rmse_scores))

    except Exception:
        cv_r2 = np.nan
        cv_rmse = np.nan

    model.fit(X, y)

    y_fit = model.predict(X)
    train_r2 = r2_score(y, y_fit)
    train_rmse = np.sqrt(mean_squared_error(y, y_fit))

    report = {
        "Target": target_name,
        "Model": model_name,
        "CV_R2": cv_r2,
        "CV_RMSE": cv_rmse,
        "Training_R2": train_r2,
        "Training_RMSE": train_rmse,
        "Samples": len(X),
        "Features": ", ".join(list(X.columns))
    }

    print("\n" + "=" * 70)
    print(f"Target: {target_name}")
    print(f"Model: {model_name}")
    print(f"CV R2: {cv_r2:.4f}")
    print(f"CV RMSE: {cv_rmse:.4f}")
    print(f"Training R2: {train_r2:.4f}")
    print(f"Training RMSE: {train_rmse:.4f}")

    return model, model_name, report


model_dict = {}
model_name_dict = {}
model_report_list = []

for target_name in TARGETS:
    model, model_name, report = evaluate_and_train_model(
        X_dict[target_name],
        y_dict[target_name],
        target_name
    )

    model_dict[target_name] = model
    model_name_dict[target_name] = model_name
    model_report_list.append(report)

model_report_df = pd.DataFrame(model_report_list)


# ============================================================
# 5. NSGA-III multi-objective optimization
# ============================================================

try:
    from pymoo.core.problem import Problem
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.util.ref_dirs import get_reference_directions
    from pymoo.optimize import minimize
except ImportError:
    raise ImportError(
        "pymoo is required for NSGA-III optimization. "
        "Please install it using: pip install pymoo"
    )


xl = np.array([bounds[c][0] for c in design_cols], dtype=float)
xu = np.array([bounds[c][1] for c in design_cols], dtype=float)


def repair_design_matrix(X_array):
    X_array = np.asarray(X_array, dtype=float).copy()

    for j, col in enumerate(design_cols):
        X_array[:, j] = np.clip(X_array[:, j], bounds[col][0], bounds[col][1])

        if col in discrete_values:
            vals = np.asarray(discrete_values[col], dtype=float)
            idx = np.abs(
                X_array[:, j].reshape(-1, 1) - vals.reshape(1, -1)
            ).argmin(axis=1)
            X_array[:, j] = vals[idx]

    return X_array


def predict_targets_from_array(X_array):
    X_array = repair_design_matrix(X_array)

    candidate_df = pd.DataFrame(X_array, columns=design_cols)

    pred_df = candidate_df.copy()

    for target_name in TARGETS:
        input_cols = feature_dict[target_name]
        pred_df[f"Pred_{target_name}"] = model_dict[target_name].predict(
            candidate_df[input_cols]
        )

    return pred_df


class ReconstructedTargetProblem(Problem):
    def __init__(self):
        super().__init__(
            n_var=len(design_cols),
            n_obj=3,
            n_constr=0,
            xl=xl,
            xu=xu
        )

    def _evaluate(self, X, out, *args, **kwargs):
        pred_df = predict_targets_from_array(X)

        T1 = pred_df["Pred_T1"].values
        T2 = pred_df["Pred_T2"].values
        T3 = pred_df["Pred_T3"].values

        # pymoo 默认是最小化问题。
        # 因为 T1、T2、T3 都是越大越好，所以这里取负号。
        out["F"] = np.column_stack([
            -T1,
            -T2,
            -T3
        ])


problem = ReconstructedTargetProblem()

ref_dirs = get_reference_directions(
    "das-dennis",
    3,
    n_partitions=12
)

algorithm = NSGA3(
    pop_size=NSGA_POP_SIZE,
    ref_dirs=ref_dirs,
    eliminate_duplicates=True
)

result = minimize(
    problem,
    algorithm,
    termination=("n_gen", NSGA_N_GEN),
    seed=RANDOM_STATE,
    verbose=True
)


# ============================================================
# 6. Process Pareto candidates
# ============================================================

if result.X is None:
    raise RuntimeError("NSGA-III failed to return valid solutions.")

pareto_X = repair_design_matrix(result.X)

pareto_df = predict_targets_from_array(pareto_X)

pareto_df = pareto_df.drop_duplicates(
    subset=design_cols,
    keep="first"
).reset_index(drop=True)


def minmax_normalize(series):
    values = np.asarray(series, dtype=float)
    vmin = np.min(values)
    vmax = np.max(values)

    if np.isclose(vmin, vmax):
        return np.ones_like(values)

    return (values - vmin) / (vmax - vmin)


pareto_df["Norm_T1"] = minmax_normalize(pareto_df["Pred_T1"])
pareto_df["Norm_T2"] = minmax_normalize(pareto_df["Pred_T2"])
pareto_df["Norm_T3"] = minmax_normalize(pareto_df["Pred_T3"])

# 综合推荐分数。
# 如果想更重视耐蚀性，可以提高 Norm_T3 的权重，例如 0.3 或 0.4。
pareto_df["Balanced_score"] = (
    0.4 * pareto_df["Norm_T1"]
    + 0.4 * pareto_df["Norm_T2"]
    + 0.2 * pareto_df["Norm_T3"]
)

# 用于避免只某一个目标特别高，而另两个目标较低的极端解。
pareto_df["Min_normalized_target"] = pareto_df[
    ["Norm_T1", "Norm_T2", "Norm_T3"]
].min(axis=1)

pareto_df = pareto_df.sort_values(
    by=[
        "Balanced_score",
        "Min_normalized_target",
        "Pred_T1",
        "Pred_T2",
        "Pred_T3"
    ],
    ascending=[False, False, False, False, False]
).reset_index(drop=True)

recommended_df = pareto_df.head(N_RECOMMEND).copy()
recommended_df.insert(0, "Rank", np.arange(1, len(recommended_df) + 1))


# ============================================================
# 7. Save results
# ============================================================

bounds_df = pd.DataFrame([
    {
        "Variable": col,
        "Lower_bound": bounds[col][0],
        "Upper_bound": bounds[col][1],
        "Is_discrete": col in discrete_values,
        "Discrete_values": ", ".join(map(str, discrete_values[col]))
        if col in discrete_values else ""
    }
    for col in design_cols
])

feature_info_df = pd.DataFrame([
    {
        "Target": target_name,
        "Sheet": f"Sheet{SHEET_CONFIG[target_name] + 1}",
        "Target_column_in_excel": target_col_dict[target_name],
        "Model": model_name_dict[target_name],
        "Input_features": ", ".join(feature_dict[target_name])
    }
    for target_name in TARGETS
])

with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
    recommended_df.to_excel(writer, sheet_name="Recommended", index=False)
    pareto_df.to_excel(writer, sheet_name="All_Pareto_Candidates", index=False)
    model_report_df.to_excel(writer, sheet_name="Model_Report", index=False)
    bounds_df.to_excel(writer, sheet_name="Search_Bounds", index=False)
    feature_info_df.to_excel(writer, sheet_name="Feature_Info", index=False)

print("\n" + "=" * 70)
print("Optimization finished.")
print(f"Recommended candidates saved to: {os.path.abspath(OUTPUT_EXCEL)}")

print("\nTop recommended candidates:")
display_cols = (
    ["Rank"]
    + design_cols
    + [
        "Pred_T1",
        "Pred_T2",
        "Pred_T3",
        "Balanced_score",
        "Min_normalized_target"
    ]
)

print(recommended_df[display_cols].to_string(index=False))