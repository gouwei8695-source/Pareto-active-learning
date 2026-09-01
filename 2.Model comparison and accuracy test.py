from pathlib import Path
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import r2_score, mean_squared_error

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


warnings.filterwarnings("ignore")


# =============================================================================
# Basic settings
# =============================================================================

FILE_PATH = "data.xlsx"
SHEET_NAME = 0

N_TARGET_COLUMNS = 7

RANDOM_STATE = 41
N_FOLDS = 5

SHOW_FIGURE = False

OUTPUT_DIR = Path("five_fold_model_comparison_results")
OUTPUT_DIR.mkdir(exist_ok=True)

colors = {
    "train": "#4477AA",
    "test": "#EE6677"
}

MODEL_ORDER = [
    "GBDT",
    "XGB",
    "LGBM",
    "RF",
    "DT",
    "SVR",
    "LR",
    "ANN",
    "KNN"
]


# =============================================================================
# Plot settings
# =============================================================================

def set_plot_style():
    plt.rcParams["font.family"] = ["Arial"]
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.it"] = "Arial:italic"
    plt.rcParams["mathtext.rm"] = "Arial"
    plt.rcParams["mathtext.tt"] = "Arial"
    plt.rcParams["mathtext.bf"] = "Arial:bold"
    plt.rcParams["font.size"] = 20
    plt.rcParams["axes.linewidth"] = 1.5
    plt.rcParams["lines.linewidth"] = 2.0
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


# =============================================================================
# Utility functions
# =============================================================================

def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def safe_sheet_name(name):
    invalid_chars = ["\\", "/", "*", "[", "]", ":", "?"]
    name = str(name)

    for ch in invalid_chars:
        name = name.replace(ch, "_")

    return name[:31]


def remove_possible_index_column(df):
    first_col = str(df.columns[0])

    if first_col.startswith("Unnamed") or first_col in ["序号", "样本序号", "No.", "No"]:
        df = df.drop(columns=df.columns[0])

    return df


def make_target_scaled_model(pipeline):
    return TransformedTargetRegressor(
        regressor=pipeline,
        transformer=StandardScaler()
    )


# =============================================================================
# Data loading
# =============================================================================

def load_data(file_path, sheet_name):
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Input file was not found: {file_path.resolve()}")

    df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)

    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")
    df.columns = [str(c).strip() for c in df.columns]
    df = remove_possible_index_column(df)

    if df.shape[1] <= N_TARGET_COLUMNS:
        raise ValueError(
            "The number of columns is too small. "
            "The dataset must contain input features and the last target columns."
        )

    feature_columns = list(df.columns[:-N_TARGET_COLUMNS])
    target_columns = list(df.columns[-N_TARGET_COLUMNS:])

    df = df.apply(pd.to_numeric, errors="coerce")

    print("=" * 80)
    print("Data loaded successfully.")
    print(f"Data shape: {df.shape}")
    print(f"Input feature columns: {feature_columns}")
    print(f"Target columns: {target_columns}")
    print("=" * 80)

    return df, feature_columns, target_columns


def prepare_xy(df, feature_columns, target_name):
    data_used = df[feature_columns + [target_name]].copy()
    data_used = data_used.dropna(subset=[target_name]).reset_index(drop=True)

    X = data_used[feature_columns].copy()
    y = data_used[target_name].copy()

    valid_feature_columns = []
    removed_feature_columns = []

    for col in feature_columns:
        if X[col].notna().sum() > 0:
            valid_feature_columns.append(col)
        else:
            removed_feature_columns.append(col)

    X = X[valid_feature_columns].copy()

    if X.shape[1] == 0:
        raise ValueError(f"No valid input features were found for target {target_name}.")

    if len(X) < N_FOLDS:
        raise ValueError(f"Too few samples for 5-fold cross-validation: {target_name}")

    return X, y, valid_feature_columns, removed_feature_columns


# =============================================================================
# Model candidates
# =============================================================================

def get_model_candidates(random_state):
    candidates = {
        "GBDT": [
            (
                "GBDT_default",
                make_target_scaled_model(
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("model", GradientBoostingRegressor(
                            n_estimators=300,
                            learning_rate=0.05,
                            max_depth=3,
                            random_state=random_state
                        ))
                    ])
                )
            )
        ],

        "XGB": [
            (
                "XGB_default",
                make_target_scaled_model(
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("model", XGBRegressor(
                            n_estimators=500,
                            learning_rate=0.03,
                            max_depth=3,
                            subsample=0.85,
                            colsample_bytree=0.85,
                            reg_lambda=1.0,
                            objective="reg:squarederror",
                            random_state=random_state,
                            n_jobs=1
                        ))
                    ])
                )
            )
        ],

        "LGBM": [
            (
                "LGBM_default",
                make_target_scaled_model(
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("model", LGBMRegressor(
                            n_estimators=500,
                            learning_rate=0.03,
                            num_leaves=15,
                            subsample=0.85,
                            colsample_bytree=0.85,
                            reg_lambda=1.0,
                            random_state=random_state,
                            n_jobs=1,
                            verbose=-1
                        ))
                    ])
                )
            )
        ],

        "RF": [
            (
                "RF_default",
                make_target_scaled_model(
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("model", RandomForestRegressor(
                            n_estimators=300,
                            random_state=random_state,
                            n_jobs=1
                        ))
                    ])
                )
            )
        ],

        "DT": [
            (
                "DT_default",
                make_target_scaled_model(
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("model", DecisionTreeRegressor(
                            random_state=random_state
                        ))
                    ])
                )
            )
        ],

        "SVR": [
            (
                "SVR_default",
                make_target_scaled_model(
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("model", SVR(
                            kernel="rbf",
                            C=100,
                            gamma=0.1,
                            epsilon=0.1
                        ))
                    ])
                )
            )
        ],

        "LR": [
            (
                "OLS",
                make_target_scaled_model(
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("model", LinearRegression())
                    ])
                )
            ),
            (
                "Ridge_alpha_0.1",
                make_target_scaled_model(
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("model", Ridge(alpha=0.1))
                    ])
                )
            ),
            (
                "Ridge_alpha_1",
                make_target_scaled_model(
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("model", Ridge(alpha=1.0))
                    ])
                )
            ),
            (
                "Ridge_alpha_10",
                make_target_scaled_model(
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("model", Ridge(alpha=10.0))
                    ])
                )
            ),
            (
                "Lasso_alpha_0.001",
                make_target_scaled_model(
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("model", Lasso(
                            alpha=0.001,
                            max_iter=10000
                        ))
                    ])
                )
            ),
            (
                "ElasticNet_alpha_0.001",
                make_target_scaled_model(
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("model", ElasticNet(
                            alpha=0.001,
                            l1_ratio=0.5,
                            max_iter=10000
                        ))
                    ])
                )
            )
        ],

        "ANN": [
            (
                "ANN_16_lbfgs_alpha_0.001",
                make_target_scaled_model(
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("model", MLPRegressor(
                            hidden_layer_sizes=(16,),
                            solver="lbfgs",
                            alpha=0.001,
                            max_iter=5000,
                            random_state=random_state
                        ))
                    ])
                )
            ),
            (
                "ANN_32_lbfgs_alpha_0.001",
                make_target_scaled_model(
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("model", MLPRegressor(
                            hidden_layer_sizes=(32,),
                            solver="lbfgs",
                            alpha=0.001,
                            max_iter=5000,
                            random_state=random_state
                        ))
                    ])
                )
            ),
            (
                "ANN_64_lbfgs_alpha_0.01",
                make_target_scaled_model(
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("model", MLPRegressor(
                            hidden_layer_sizes=(64,),
                            solver="lbfgs",
                            alpha=0.01,
                            max_iter=5000,
                            random_state=random_state
                        ))
                    ])
                )
            ),
            (
                "ANN_32_16_adam_alpha_0.01",
                make_target_scaled_model(
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("model", MLPRegressor(
                            hidden_layer_sizes=(32, 16),
                            solver="adam",
                            activation="relu",
                            alpha=0.01,
                            learning_rate_init=0.001,
                            early_stopping=True,
                            validation_fraction=0.15,
                            n_iter_no_change=50,
                            max_iter=5000,
                            random_state=random_state
                        ))
                    ])
                )
            )
        ],

        "KNN": [
            (
                "KNN_default",
                make_target_scaled_model(
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("model", KNeighborsRegressor(
                            n_neighbors=5,
                            weights="distance"
                        ))
                    ])
                )
            )
        ]
    }

    return candidates


# =============================================================================
# 5-fold cross-validation
# =============================================================================

def evaluate_one_candidate_cv(estimator, X, y, candidate_name):
    y_array = np.asarray(y, dtype=float)

    oof_pred = np.full(len(y_array), np.nan)

    train_true_all = []
    train_pred_all = []
    train_fold_all = []

    fold_rows = []

    kf = KFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    for fold_id, (train_index, test_index) in enumerate(kf.split(X), start=1):
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        model = clone(estimator)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        oof_pred[test_index] = y_test_pred

        train_true_all.extend(y_train.values)
        train_pred_all.extend(y_train_pred)
        train_fold_all.extend([fold_id] * len(y_train))

        fold_rows.append({
            "Candidate": candidate_name,
            "Fold": fold_id,
            "Train_R2": r2_score(y_train, y_train_pred),
            "Test_R2": r2_score(y_test, y_test_pred),
            "Train_RMSE": calculate_rmse(y_train, y_train_pred),
            "Test_RMSE": calculate_rmse(y_test, y_test_pred)
        })

    fold_df = pd.DataFrame(fold_rows)

    oof_r2 = r2_score(y_array, oof_pred)
    oof_rmse = calculate_rmse(y_array, oof_pred)

    summary = {
        "Candidate": candidate_name,
        "Train_R2": fold_df["Train_R2"].mean(),
        "Train_R2_std": fold_df["Train_R2"].std(ddof=1),
        "Test_R2": fold_df["Test_R2"].mean(),
        "Test_R2_std": fold_df["Test_R2"].std(ddof=1),
        "Train_RMSE": fold_df["Train_RMSE"].mean(),
        "Train_RMSE_std": fold_df["Train_RMSE"].std(ddof=1),
        "Test_RMSE": fold_df["Test_RMSE"].mean(),
        "Test_RMSE_std": fold_df["Test_RMSE"].std(ddof=1),
        "OOF_R2": oof_r2,
        "OOF_RMSE": oof_rmse
    }

    scatter_result = {
        "train": {
            "true": np.asarray(train_true_all),
            "pred": np.asarray(train_pred_all)
        },
        "test": {
            "true": y_array,
            "pred": oof_pred
        },
        "metrics": {
            "train_r2": fold_df["Train_R2"].mean(),
            "test_r2": oof_r2,
            "train_rmse": fold_df["Train_RMSE"].mean(),
            "test_rmse": oof_rmse
        }
    }

    train_prediction_df = pd.DataFrame({
        "Fold": train_fold_all,
        "Train_true": train_true_all,
        "Train_pred": train_pred_all
    })

    test_prediction_df = pd.DataFrame({
        "Sample_Index": np.arange(1, len(y_array) + 1),
        "Test_true": y_array,
        "Test_pred": oof_pred
    })

    return summary, fold_df, scatter_result, train_prediction_df, test_prediction_df


def evaluate_one_algorithm(algorithm_name, candidate_list, X, y):
    candidate_summary_rows = []
    all_fold_rows = []
    candidate_outputs = {}

    for candidate_name, estimator in candidate_list:
        print(f"  Testing {algorithm_name}: {candidate_name}")

        try:
            summary, fold_df, scatter_result, train_pred_df, test_pred_df = evaluate_one_candidate_cv(
                estimator=estimator,
                X=X,
                y=y,
                candidate_name=candidate_name
            )

            candidate_summary_rows.append(summary)
            fold_df["Algorithm"] = algorithm_name
            all_fold_rows.append(fold_df)

            candidate_outputs[candidate_name] = {
                "scatter_result": scatter_result,
                "train_prediction_df": train_pred_df,
                "test_prediction_df": test_pred_df
            }

        except Exception as e:
            candidate_summary_rows.append({
                "Candidate": candidate_name,
                "Train_R2": np.nan,
                "Train_R2_std": np.nan,
                "Test_R2": np.nan,
                "Test_R2_std": np.nan,
                "Train_RMSE": np.nan,
                "Train_RMSE_std": np.nan,
                "Test_RMSE": np.nan,
                "Test_RMSE_std": np.nan,
                "OOF_R2": np.nan,
                "OOF_RMSE": np.nan,
                "Error": str(e)
            })

    candidate_summary_df = pd.DataFrame(candidate_summary_rows)

    candidate_summary_df = candidate_summary_df.sort_values(
        by=["Test_R2", "Test_RMSE"],
        ascending=[False, True]
    ).reset_index(drop=True)

    best_candidate_name = candidate_summary_df.loc[0, "Candidate"]

    algorithm_summary = {
        "Model": algorithm_name,
        "Best_Params": best_candidate_name,
        "Train_R2": candidate_summary_df.loc[0, "Train_R2"],
        "Train_R2_std": candidate_summary_df.loc[0, "Train_R2_std"],
        "Test_R2": candidate_summary_df.loc[0, "Test_R2"],
        "Test_R2_std": candidate_summary_df.loc[0, "Test_R2_std"],
        "Train_RMSE": candidate_summary_df.loc[0, "Train_RMSE"],
        "Train_RMSE_std": candidate_summary_df.loc[0, "Train_RMSE_std"],
        "Test_RMSE": candidate_summary_df.loc[0, "Test_RMSE"],
        "Test_RMSE_std": candidate_summary_df.loc[0, "Test_RMSE_std"],
        "OOF_R2": candidate_summary_df.loc[0, "OOF_R2"],
        "OOF_RMSE": candidate_summary_df.loc[0, "OOF_RMSE"]
    }

    if len(all_fold_rows) > 0:
        all_fold_df = pd.concat(all_fold_rows, axis=0).reset_index(drop=True)
    else:
        all_fold_df = pd.DataFrame()

    best_output = candidate_outputs.get(best_candidate_name, None)

    return algorithm_summary, candidate_summary_df, all_fold_df, best_output


def evaluate_all_algorithms_cv(X, y):
    model_candidates = get_model_candidates(RANDOM_STATE)

    algorithm_summary_rows = []
    all_candidate_summary = []
    all_fold_details = []
    best_outputs = {}

    for algorithm_name in MODEL_ORDER:
        print(f"Evaluating algorithm: {algorithm_name}")

        algorithm_summary, candidate_summary_df, fold_df, best_output = evaluate_one_algorithm(
            algorithm_name=algorithm_name,
            candidate_list=model_candidates[algorithm_name],
            X=X,
            y=y
        )

        algorithm_summary_rows.append(algorithm_summary)

        candidate_summary_df["Algorithm"] = algorithm_name
        all_candidate_summary.append(candidate_summary_df)

        if len(fold_df) > 0:
            all_fold_details.append(fold_df)

        best_outputs[algorithm_name] = best_output

    results_df = pd.DataFrame(algorithm_summary_rows)
    results_df["Model"] = pd.Categorical(
        results_df["Model"],
        categories=MODEL_ORDER,
        ordered=True
    )
    results_df = results_df.sort_values("Model").reset_index(drop=True)
    results_df["Model"] = results_df["Model"].astype(str)

    ranked_results_df = results_df.sort_values(
        by=["Test_R2", "Test_RMSE"],
        ascending=[False, True]
    ).reset_index(drop=True)

    candidate_summary_df = pd.concat(all_candidate_summary, axis=0).reset_index(drop=True)

    if len(all_fold_details) > 0:
        fold_details_df = pd.concat(all_fold_details, axis=0).reset_index(drop=True)
    else:
        fold_details_df = pd.DataFrame()

    return results_df, ranked_results_df, candidate_summary_df, fold_details_df, best_outputs


# =============================================================================
# Scatter plot
# =============================================================================

def safe_kde_x(data, ax, color):
    data = np.asarray(data)

    if len(data) >= 3 and len(np.unique(data)) >= 2:
        try:
            sns.kdeplot(
                x=data,
                ax=ax,
                color=color,
                fill=True,
                alpha=0.3,
                linewidth=1.5
            )
        except Exception:
            pass


def safe_kde_y(data, ax, color):
    data = np.asarray(data)

    if len(data) >= 3 and len(np.unique(data)) >= 2:
        try:
            sns.kdeplot(
                y=data,
                ax=ax,
                color=color,
                fill=True,
                alpha=0.3,
                linewidth=1.5
            )
        except Exception:
            pass


def plot_scatter(result, model_name, target_name, output_dir):
    set_plot_style()

    fig = plt.figure(figsize=(10, 8))

    gs = gridspec.GridSpec(
        2,
        2,
        width_ratios=[6, 1],
        height_ratios=[1, 6],
        wspace=0.0,
        hspace=0.0
    )

    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    ax_top.axis("off")
    ax_right.axis("off")

    train_true = result["train"]["true"]
    train_pred = result["train"]["pred"]
    test_true = result["test"]["true"]
    test_pred = result["test"]["pred"]

    ax_main.scatter(
        train_true,
        train_pred,
        c=colors["train"],
        s=80,
        alpha=0.35,
        edgecolors="black",
        linewidths=0.5,
        zorder=2,
        label="Train"
    )

    ax_main.scatter(
        test_true,
        test_pred,
        c=colors["test"],
        s=90,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.8,
        zorder=3,
        label="Test"
    )

    all_true = np.concatenate([train_true, test_true])
    all_pred = np.concatenate([train_pred, test_pred])

    min_val = min(all_true.min(), all_pred.min())
    max_val = max(all_true.max(), all_pred.max())

    if max_val == min_val:
        margin = 1.0
    else:
        margin = (max_val - min_val) * 0.10

    limit_range = [min_val - margin, max_val + margin]

    ax_main.plot(
        limit_range,
        limit_range,
        "--",
        color="black",
        alpha=0.6,
        lw=1.5,
        zorder=1,
        label="y=x"
    )

    ax_main.set_xlim(limit_range)
    ax_main.set_ylim(limit_range)

    metrics = result["metrics"]

    text_str = (
        f"$R^2$ = {metrics['test_r2']:.2f}\n"
        f"RMSE = {metrics['test_rmse']:.2f}"
    )

    props = dict(
        boxstyle="round,pad=0.5",
        facecolor="white",
        alpha=0.9,
        edgecolor="gray"
    )

    ax_main.text(
        0.95,
        0.05,
        text_str,
        transform=ax_main.transAxes,
        fontsize=18,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=props,
        zorder=5
    )

    safe_kde_x(train_true, ax_top, colors["train"])
    safe_kde_x(test_true, ax_top, colors["test"])
    safe_kde_y(train_pred, ax_right, colors["train"])
    safe_kde_y(test_pred, ax_right, colors["test"])

    ax_main.set_xlabel(f"Experimental {target_name}", fontsize=22, fontname="Arial")
    ax_main.set_ylabel(f"Predicted {target_name}", fontsize=18, fontname="Arial")

    ax_main.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=14
    )

    ax_main.tick_params(axis="both", labelsize=16)

    for spine in ax_main.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()

    png_path = output_dir / f"{target_name}_{model_name}_5fold_scatter.png"

    fig.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white")

    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(fig)

    print(f"Scatter plot saved to: {png_path}")


# =============================================================================
# One-target workflow
# =============================================================================

def analyze_one_target(df, feature_columns, target_name):
    print("\n" + "=" * 80)
    print(f"Analyzing target: {target_name}")
    print("=" * 80)

    X, y, valid_feature_columns, removed_feature_columns = prepare_xy(
        df=df,
        feature_columns=feature_columns,
        target_name=target_name
    )

    results_df, ranked_results_df, candidate_summary_df, fold_details_df, best_outputs = evaluate_all_algorithms_cv(
        X=X,
        y=y
    )

    best_model_name = ranked_results_df.loc[0, "Model"]
    best_params = ranked_results_df.loc[0, "Best_Params"]
    best_output = best_outputs[best_model_name]

    print("\nModel comparison results:")
    print(results_df.to_string(index=False))

    print(f"\nBest model for {target_name}: {best_model_name}")
    print(f"Best params: {best_params}")
    print(f"Best mean Test R2: {ranked_results_df.loc[0, 'Test_R2']:.4f}")
    print(f"Best mean Test RMSE: {ranked_results_df.loc[0, 'Test_RMSE']:.4f}")
    print(f"Best OOF R2: {ranked_results_df.loc[0, 'OOF_R2']:.4f}")
    print(f"Best OOF RMSE: {ranked_results_df.loc[0, 'OOF_RMSE']:.4f}")

    plot_scatter(
        result=best_output["scatter_result"],
        model_name=best_model_name,
        target_name=target_name,
        output_dir=OUTPUT_DIR
    )

    summary = {
        "Target": target_name,
        "Best_model": best_model_name,
        "Best_params": best_params,
        "Best_mean_Train_R2": ranked_results_df.loc[0, "Train_R2"],
        "Best_std_Train_R2": ranked_results_df.loc[0, "Train_R2_std"],
        "Best_mean_Test_R2": ranked_results_df.loc[0, "Test_R2"],
        "Best_std_Test_R2": ranked_results_df.loc[0, "Test_R2_std"],
        "Best_mean_Train_RMSE": ranked_results_df.loc[0, "Train_RMSE"],
        "Best_std_Train_RMSE": ranked_results_df.loc[0, "Train_RMSE_std"],
        "Best_mean_Test_RMSE": ranked_results_df.loc[0, "Test_RMSE"],
        "Best_std_Test_RMSE": ranked_results_df.loc[0, "Test_RMSE_std"],
        "Best_OOF_R2": ranked_results_df.loc[0, "OOF_R2"],
        "Best_OOF_RMSE": ranked_results_df.loc[0, "OOF_RMSE"],
        "Input_feature_number": len(valid_feature_columns),
        "Input_features": ", ".join(valid_feature_columns),
        "Removed_empty_features": ", ".join(removed_feature_columns)
    }

    return {
        "target_name": target_name,
        "results_df": results_df,
        "ranked_results_df": ranked_results_df,
        "candidate_summary_df": candidate_summary_df,
        "fold_details_df": fold_details_df,
        "train_prediction_df": best_output["train_prediction_df"],
        "test_prediction_df": best_output["test_prediction_df"],
        "summary": summary
    }


# =============================================================================
# Main workflow
# =============================================================================

def main():
    df, feature_columns, target_columns = load_data(
        file_path=FILE_PATH,
        sheet_name=SHEET_NAME
    )

    all_summary = []
    all_target_results = {}

    for target_name in target_columns:
        target_result = analyze_one_target(
            df=df,
            feature_columns=feature_columns,
            target_name=target_name
        )

        all_summary.append(target_result["summary"])
        all_target_results[target_name] = target_result

    summary_df = pd.DataFrame(all_summary)

    output_excel = OUTPUT_DIR / "all_targets_5fold_model_comparison_summary.xlsx"

    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

        for target_name, target_result in all_target_results.items():
            sheet_prefix = safe_sheet_name(target_name)

            target_result["results_df"].to_excel(
                writer,
                sheet_name=safe_sheet_name(f"{sheet_prefix}_models"),
                index=False
            )

            target_result["ranked_results_df"].to_excel(
                writer,
                sheet_name=safe_sheet_name(f"{sheet_prefix}_ranking"),
                index=False
            )

            target_result["candidate_summary_df"].to_excel(
                writer,
                sheet_name=safe_sheet_name(f"{sheet_prefix}_params"),
                index=False
            )

            target_result["fold_details_df"].to_excel(
                writer,
                sheet_name=safe_sheet_name(f"{sheet_prefix}_folds"),
                index=False
            )

            target_result["train_prediction_df"].to_excel(
                writer,
                sheet_name=safe_sheet_name(f"{sheet_prefix}_train_pred"),
                index=False
            )

            target_result["test_prediction_df"].to_excel(
                writer,
                sheet_name=safe_sheet_name(f"{sheet_prefix}_test_pred"),
                index=False
            )

    print("\n" + "=" * 80)
    print("All 5-fold model comparison tasks completed.")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print(f"\nResults saved to: {output_excel.resolve()}")


if __name__ == "__main__":
    main()