"""
Pearson correlation analysis for feature screening.

This script calculates the Pearson correlation coefficients among numeric
variables and generates a lower-triangular correlation heatmap. The correlation
coefficient matrix is also saved as an Excel file.

"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# =============================================================================
# Basic settings
# =============================================================================

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.unicode_minus"] = False



import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
# =============================================================================
# Data loading
# =============================================================================

def load_data(file_path):
    """
    Load numeric data from an Excel file.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    df = pd.read_excel(file_path)

    # Pearson correlation is calculated only for numeric variables.
    numeric_df = df.select_dtypes(include=[np.number]).copy()

    if numeric_df.empty:
        raise ValueError("No numeric columns were found in the input file.")

    print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns.")
    print(f"Numeric columns used: {numeric_df.shape[1]}")

    return numeric_df


# =============================================================================
# Pearson correlation calculation
# =============================================================================

def calculate_pearson_correlation(df):
    """
    Calculate Pearson correlation coefficient matrix.
    """
    corr_df = df.corr(method="pearson")
    return corr_df


# =============================================================================
# Heatmap plotting
# =============================================================================

def plot_correlation_heatmap(corr_df, output_path):
    """
    Plot and save the lower-triangular Pearson correlation heatmap.
    """
    output_path = Path(output_path)

    fig, ax = plt.subplots(figsize=(12, 9))

    # Blue-white-red colormap.
    cmap = LinearSegmentedColormap.from_list(
        "pearson_red_blue",
        ["#4477AA", "white", "#D62728"],
        N=256
    )

    corr_values = corr_df.values
    n_cols = corr_df.shape[0]

    # Mask upper triangle.
    mask = np.triu(np.ones_like(corr_values, dtype=bool))
    masked_corr = np.ma.masked_where(mask, corr_values)

    im = ax.imshow(
        masked_corr,
        cmap=cmap,
        vmin=-1,
        vmax=1
    )

    # Axis labels.
    columns = corr_df.columns.tolist()

    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_cols))
    ax.set_xticklabels(columns, rotation=45, ha="right")
    ax.set_yticklabels(columns)

    ax.set_title(
        "Pearson Correlation Coefficient Matrix",
        fontsize=16,
        fontweight="bold",
        pad=20
    )

    # Grid lines.
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", which="major", length=0)

    # Color bar.
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(
        "Pearson Correlation Coefficient",
        fontsize=12,
        fontweight="bold"
    )

    plt.tight_layout()

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"Heatmap saved to: {output_path}")
    print(f"PDF version saved to: {output_path.with_suffix('.pdf')}")


# =============================================================================
# Save correlation matrix
# =============================================================================

def save_correlation_matrix(corr_df, output_path):
    """
    Save Pearson correlation coefficient matrix to Excel.
    """
    output_path = Path(output_path)
    corr_df.to_excel(output_path, sheet_name="Pearson_correlation")

    print(f"Correlation matrix saved to: {output_path}")


# =============================================================================
# Main function
# =============================================================================

def main():
    input_file = "data.xlsx"

    output_dir = Path("pearson_correlation_results")
    output_dir.mkdir(exist_ok=True)

    heatmap_path = output_dir / "pearson_correlation_heatmap.png"
    excel_path = output_dir / "pearson_correlation_matrix.xlsx"

    df = load_data(input_file)

    corr_df = calculate_pearson_correlation(df)

    plot_correlation_heatmap(corr_df, heatmap_path)

    save_correlation_matrix(corr_df, excel_path)

    print("Pearson correlation analysis completed.")


if __name__ == "__main__":
    main()