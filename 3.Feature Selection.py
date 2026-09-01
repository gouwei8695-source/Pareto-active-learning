"""
Feature selection based on feature importance, Pearson correlation filtering,
and recursive feature elimination with cross-validation.
"""

from pathlib import Path
from copy import deepcopy
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_selection import RFECV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import clone

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


warnings.filterwarnings("ignore")

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

# =============================================================================
# Basic settings
# =============================================================================

DATA_PATH = "data-selection.xlsx"

SHEET_CONFIG = {
    "T1": 0,   # Sheet1: features + T1
    "T2": 1,   # Sheet2: features + T2
    "T3": 2,   # Sheet3: features + T3
}

TARGETS = ["T1", "T2", "T3"]

TEST_SIZE = 0.2
RANDOM_STATE = 42

CORR_THRESHOLD = 0.80
MIN_FEATURES_TO_SELECT = 1
CV_FOLDS = 5

OUTPUT_DIR = Path("feature_selection_RFE_results")
OUTPUT_DIR.mkdir(exist_ok=True)

COLOR_RETAINED = "#4477AA"
COLOR_REMOVED = "#EE6677"
COLOR_SELECTED = "#91CCC0"
COLOR_UNSELECTED = "#BDBDBD"

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 18
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


# =============================================================================
# Data loading
# =============================================================================

def read_target_sheet(data_path, sheet_index):
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Input file not found: {data_path}")

    df = pd.read_excel(data_path, sheet_name=sheet_index, header=0)
    df = df.dropna(axis=1, how="all")
    df.columns = [str(c).strip() for c in df.columns]

    first_col = str(df.columns[0])
    if first_col.startswith("Unnamed") or first_col in ["序号", "样本序号", "No.", "No"]:
        df = df.drop(columns=df.columns[0])

    if df.shape[1] < 2:
        raise ValueError(
            f"Sheet{sheet_index + 1} must contain at least one feature column and one target column."
        )

    feature_columns = list(df.columns[:-1])
    target_column = df.columns[-1]

    X = df[feature_columns].copy()
    y = df[target_column].copy()

    return df, X, y, feature_columns, target_column


# =============================================================================
# Model definition
# =============================================================================

def build_model(target_name):
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


# =============================================================================
# Feature importance
# =============================================================================

def calculate_feature_importance(model, X_train, y_train, feature_columns):
    fitted_model = clone(model)
    fitted_model.fit(X_train, y_train)

    if not hasattr(fitted_model, "feature_importances_"):
        raise AttributeError("The selected model does not provide feature_importances_.")

    importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": fitted_model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    ).reset_index(drop=True)

    importance_df["Importance_rank"] = np.arange(1, len(importance_df) + 1)

    return importance_df


# =============================================================================
# Pearson correlation filtering
# =============================================================================

def correlation_filter(X_df, importance_df, corr_threshold):
    ordered_features = importance_df["Feature"].tolist()
    corr_matrix = X_df[ordered_features].corr(method="pearson").abs()

    retained_features = []
    removed_features = []
    filter_records = []

    for feature in ordered_features:
        if len(retained_features) == 0:
            retained_features.append(feature)
            filter_records.append({
                "Feature": feature,
                "Status": "Retained",
                "Correlated_with": "",
                "Correlation": ""
            })
            continue

        corr_with_retained = corr_matrix.loc[feature, retained_features]
        max_corr = corr_with_retained.max()
        correlated_feature = corr_with_retained.idxmax()

        if max_corr > corr_threshold:
            removed_features.append(feature)
            filter_records.append({
                "Feature": feature,
                "Status": "Removed",
                "Correlated_with": correlated_feature,
                "Correlation": max_corr
            })
        else:
            retained_features.append(feature)
            filter_records.append({
                "Feature": feature,
                "Status": "Retained",
                "Correlated_with": "",
                "Correlation": ""
            })

    filter_df = pd.DataFrame(filter_records)

    return retained_features, removed_features, filter_df, corr_matrix


# =============================================================================
# Recursive feature elimination
# =============================================================================

def run_rfecv(model, X_train, y_train, retained_features):
    X_train_selected = X_train[retained_features].copy()

    n_samples = len(X_train_selected)
    n_splits = min(CV_FOLDS, n_samples)

    if n_splits < 2:
        raise ValueError("At least two training samples are required for RFECV.")

    if len(retained_features) == 1:
        rfe_result_df = pd.DataFrame({
            "Feature": retained_features,
            "Selected": [True],
            "RFE_ranking": [1]
        })

        cv_curve_df = pd.DataFrame({
            "n_features": [1],
            "Mean_CV_R2": [np.nan]
        })

        return retained_features, rfe_result_df, cv_curve_df

    cv = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    selector = RFECV(
        estimator=clone(model),
        step=1,
        min_features_to_select=MIN_FEATURES_TO_SELECT,
        cv=cv,
        scoring="r2",
        n_jobs=-1
    )

    selector.fit(X_train_selected, y_train)

    selected_features = list(X_train_selected.columns[selector.support_])

    rfe_result_df = pd.DataFrame({
        "Feature": retained_features,
        "Selected": selector.support_,
        "RFE_ranking": selector.ranking_
    })

    rfe_result_df = rfe_result_df.sort_values(
        by=["RFE_ranking", "Feature"],
        ascending=[True, True]
    ).reset_index(drop=True)

    cv_results = selector.cv_results_
    mean_scores = cv_results["mean_test_score"]

    if "n_features" in cv_results:
        n_features = cv_results["n_features"]
    else:
        n_features = np.arange(
            MIN_FEATURES_TO_SELECT,
            MIN_FEATURES_TO_SELECT + len(mean_scores)
        )

    cv_curve_df = pd.DataFrame({
        "n_features": n_features,
        "Mean_CV_R2": mean_scores
    })

    return selected_features, rfe_result_df, cv_curve_df


# =============================================================================
# Final model evaluation
# =============================================================================

def evaluate_final_model(model, X_train, X_test, y_train, y_test, selected_features):
    final_model = clone(model)

    final_model.fit(X_train[selected_features], y_train)

    y_train_pred = final_model.predict(X_train[selected_features])
    y_test_pred = final_model.predict(X_test[selected_features])

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    result = {
        "Train_R2": train_r2,
        "Test_R2": test_r2,
        "Train_RMSE": train_rmse,
        "Test_RMSE": test_rmse
    }

    return result


# =============================================================================
# Plotting
# =============================================================================

def plot_feature_importance(target_name, importance_df, filter_df, output_dir):
    plot_df = importance_df.merge(
        filter_df[["Feature", "Status"]],
        on="Feature",
        how="left"
    )

    colors = [
        COLOR_RETAINED if status == "Retained" else COLOR_REMOVED
        for status in plot_df["Status"]
    ]

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.bar(
        plot_df["Feature"],
        plot_df["Importance"],
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.85
    )

    ax.set_xlabel("Feature", fontsize=22)
    ax.set_ylabel("Feature importance", fontsize=22)
    ax.set_title(target_name, fontsize=24, pad=12)

    ax.tick_params(axis="x", labelrotation=45, labelsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()

    fig.savefig(
        output_dir / f"{target_name}_feature_importance_correlation_filter.png",
        dpi=600,
        bbox_inches="tight",
        facecolor="white"
    )

    fig.savefig(
        output_dir / f"{target_name}_feature_importance_correlation_filter.pdf",
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)


def plot_rfe_cv_curve(target_name, cv_curve_df, selected_feature_number, output_dir):
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        cv_curve_df["n_features"],
        cv_curve_df["Mean_CV_R2"],
        marker="o",
        markersize=8,
        linewidth=2.0,
        color="black",
        label="RFECV"
    )

    if not cv_curve_df["Mean_CV_R2"].isna().all():
        selected_score = cv_curve_df.loc[
            cv_curve_df["n_features"] == selected_feature_number,
            "Mean_CV_R2"
        ]

        if len(selected_score) > 0:
            ax.scatter(
                selected_feature_number,
                selected_score.iloc[0],
                s=180,
                color=COLOR_SELECTED,
                edgecolor="black",
                linewidth=1.2,
                zorder=5,
                label="Selected"
            )

    ax.set_xlabel("Number of selected features", fontsize=22)
    ax.set_ylabel("Mean CV $R^2$", fontsize=22)
    ax.set_title(target_name, fontsize=24, pad=12)

    ax.tick_params(axis="both", labelsize=16)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=14, frameon=True)

    plt.tight_layout()

    fig.savefig(
        output_dir / f"{target_name}_RFECV_curve.png",
        dpi=600,
        bbox_inches="tight",
        facecolor="white"
    )

    fig.savefig(
        output_dir / f"{target_name}_RFECV_curve.pdf",
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)


def plot_rfe_ranking(target_name, rfe_result_df, output_dir):
    plot_df = rfe_result_df.copy()
    plot_df = plot_df.sort_values(by="RFE_ranking", ascending=True)

    colors = [
        COLOR_SELECTED if selected else COLOR_UNSELECTED
        for selected in plot_df["Selected"]
    ]

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.bar(
        plot_df["Feature"],
        plot_df["RFE_ranking"],
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.85
    )

    ax.set_xlabel("Feature", fontsize=22)
    ax.set_ylabel("RFE ranking", fontsize=22)
    ax.set_title(target_name, fontsize=24, pad=12)

    ax.invert_yaxis()
    ax.tick_params(axis="x", labelrotation=45, labelsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()

    fig.savefig(
        output_dir / f"{target_name}_RFE_ranking.png",
        dpi=600,
        bbox_inches="tight",
        facecolor="white"
    )

    fig.savefig(
        output_dir / f"{target_name}_RFE_ranking.pdf",
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)


# =============================================================================
# One-target workflow
# =============================================================================

def analyze_one_target(target_name, sheet_index):
    print("\n" + "=" * 80)
    print(f"Analyzing target: {target_name}")
    print("=" * 80)

    df, X, y, feature_columns, target_column = read_target_sheet(
        data_path=DATA_PATH,
        sheet_index=sheet_index
    )

    model, model_name = build_model(target_name)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    importance_df = calculate_feature_importance(
        model=model,
        X_train=X_train,
        y_train=y_train,
        feature_columns=feature_columns
    )

    retained_features, removed_features, filter_df, corr_matrix = correlation_filter(
        X_df=X_train,
        importance_df=importance_df,
        corr_threshold=CORR_THRESHOLD
    )

    selected_features, rfe_result_df, cv_curve_df = run_rfecv(
        model=model,
        X_train=X_train,
        y_train=y_train,
        retained_features=retained_features
    )

    evaluation_result = evaluate_final_model(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        selected_features=selected_features
    )

    print(f"Model: {model_name}")
    print(f"Target column: {target_column}")
    print(f"Initial features: {feature_columns}")
    print(f"Retained after correlation filtering: {retained_features}")
    print(f"Removed by correlation filtering: {removed_features}")
    print(f"Selected by RFE: {selected_features}")
    print(f"Test R2: {evaluation_result['Test_R2']:.4f}")
    print(f"Test RMSE: {evaluation_result['Test_RMSE']:.4f}")

    plot_feature_importance(
        target_name=target_name,
        importance_df=importance_df,
        filter_df=filter_df,
        output_dir=OUTPUT_DIR
    )

    plot_rfe_cv_curve(
        target_name=target_name,
        cv_curve_df=cv_curve_df,
        selected_feature_number=len(selected_features),
        output_dir=OUTPUT_DIR
    )

    plot_rfe_ranking(
        target_name=target_name,
        rfe_result_df=rfe_result_df,
        output_dir=OUTPUT_DIR
    )

    evaluation_df = pd.DataFrame([evaluation_result])

    selected_feature_df = pd.DataFrame({
        "Selected_features": selected_features
    })

    excel_path = OUTPUT_DIR / f"{target_name}_RFE_feature_selection_results.xlsx"

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        importance_df.to_excel(writer, sheet_name="Feature_importance", index=False)
        filter_df.to_excel(writer, sheet_name="Correlation_filter", index=False)
        corr_matrix.to_excel(writer, sheet_name="Correlation_matrix")
        rfe_result_df.to_excel(writer, sheet_name="RFE_ranking", index=False)
        cv_curve_df.to_excel(writer, sheet_name="RFECV_curve", index=False)
        selected_feature_df.to_excel(writer, sheet_name="Selected_features", index=False)
        evaluation_df.to_excel(writer, sheet_name="Final_model_evaluation", index=False)

    summary = {
        "Target": target_name,
        "Sheet": f"Sheet{sheet_index + 1}",
        "Target_column": target_column,
        "Model": model_name,
        "Initial_n_features": len(feature_columns),
        "Retained_n_features": len(retained_features),
        "Selected_n_features": len(selected_features),
        "Selected_features": ", ".join(selected_features),
        "Removed_by_correlation": ", ".join(removed_features),
        "Train_R2": evaluation_result["Train_R2"],
        "Test_R2": evaluation_result["Test_R2"],
        "Train_RMSE": evaluation_result["Train_RMSE"],
        "Test_RMSE": evaluation_result["Test_RMSE"]
    }

    return summary


# =============================================================================
# Main workflow
# =============================================================================

def main():
    all_summary = []

    for target_name in TARGETS:
        summary = analyze_one_target(
            target_name=target_name,
            sheet_index=SHEET_CONFIG[target_name]
        )
        all_summary.append(summary)

    summary_df = pd.DataFrame(all_summary)

    summary_path = OUTPUT_DIR / "all_targets_RFE_feature_selection_summary.xlsx"
    summary_df.to_excel(summary_path, index=False)

    print("\n" + "=" * 80)
    print("RFE feature selection completed.")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print(f"\nSummary saved to: {summary_path.resolve()}")


if __name__ == "__main__":
    main()