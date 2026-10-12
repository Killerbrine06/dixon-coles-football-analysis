"""
Multivariate Linear Regression via Conjugate Gradient Least Squares (CGLS).

Loads a window of 10 consecutive finished matches from SQLite, extracts predictive
contextual features (home advantage, rest days, 3-match rolling form), trains a
linear regression model via CGLS iterative optimization on matches 1-9, and forecasts
expected goals scored for the 10th match.

Usage:
    python general_models/multivar_lin_reg.py
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from data_loader import PLOTS_DIR, load_consecutive_matches
except ImportError:
    from general_models.data_loader import PLOTS_DIR, load_consecutive_matches

# ---------------------------------------------------------------------------
# Global Constants & Paths
# ---------------------------------------------------------------------------
ALGORITHM_NAME = "multivar_lin_reg"
WINDOW_SIZE = 10
FEATURE_COLS = ["Is_Home", "Rest_Days", "Form_Poss_3M", "Form_Sh_3M", "Form_SoT_3M"]
TARGET_COL = "GF"


# ---------------------------------------------------------------------------
# Mathematical Algorithm: Conjugate Gradient Least Squares (CGLS)
# ---------------------------------------------------------------------------
def CGLS(
    X: np.ndarray,
    Y: np.ndarray,
    tol: float = 1e-6,
    max_iter: Optional[int] = None,
) -> Tuple[np.ndarray, List[float]]:
    """
    Solves min_beta ||X * beta - Y||_2 via Conjugate Gradient Least Squares (CGLS).

    Solves the normal equations (X^T X) beta = X^T Y iteratively without
    explicitly computing the full Gram matrix X^T X.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    Y : np.ndarray
        Target vector of shape (n_samples,).
    tol : float, default 1e-6
        Tolerance threshold for the normal equation residual norm ||s||.
    max_iter : Optional[int], default None
        Maximum iterations (defaults to n_features + 10).

    Returns
    -------
    Tuple[np.ndarray, List[float]]
        Fitted beta solution vector and residual norm history per iteration.
    """
    n, p = X.shape
    if max_iter is None:
        max_iter = p + 10

    beta = np.zeros(p)
    r = Y - X @ beta
    s = X.T @ r
    p_dir = s.copy()

    error_history = [float(np.linalg.norm(s))]

    for k in range(max_iter):
        q = X @ p_dir
        s_norm_sq = float(np.dot(s, s))
        q_norm_sq = float(np.dot(q, q))

        if q_norm_sq == 0:
            break

        alpha = s_norm_sq / q_norm_sq
        beta = beta + alpha * p_dir
        r = r - alpha * q
        s_new = X.T @ r

        current_error = float(np.linalg.norm(s_new))
        error_history.append(current_error)

        if current_error < tol:
            break

        gamma = float(np.dot(s_new, s_new)) / s_norm_sq
        p_dir = s_new + gamma * p_dir
        s = s_new

    return beta, error_history


def run_cgls_regression(
    df: pd.DataFrame,
    feature_cols: List[str] = FEATURE_COLS,
    target_col: str = TARGET_COL,
) -> Dict[str, Any]:
    """
    Sets up design matrices, executes CGLS solver, and evaluates out-of-sample forecast.

    Parameters
    ----------
    df : pd.DataFrame
        Consecutive matches dataframe.
    feature_cols : List[str]
        List of predictor column names.
    target_col : str
        Target column name (e.g. 'GF').

    Returns
    -------
    Dict[str, Any]
        Dictionary with fitted weights, error history, predictions, and actuals.
    """
    X_raw = df[feature_cols].values
    Y_raw = df[target_col].values

    # Add a column of 1s at the front to serve as Intercept (Beta_0)
    X_all = np.c_[np.ones(X_raw.shape[0]), X_raw]

    # Strict Train/Test Split: Matches 1 to 9 (Indices 0 to 8) train, Match 10 (Index 9) test
    n_train = len(df) - 1
    X_train = X_all[:n_train]
    Y_train = Y_raw[:n_train]

    X_test = X_all[n_train : n_train + 1]
    Y_test_actual = float(Y_raw[n_train])

    # Compute beta solution using CGLS
    beta_solution, errors = CGLS(X_train, Y_train)

    # In-sample fitted goals on training set
    fitted_train = X_train @ beta_solution

    # Out-of-sample prediction for match 10: Y_hat = X * beta
    predicted_goals = float((X_test @ beta_solution)[0])
    prediction_error = abs(Y_test_actual - predicted_goals)

    return {
        "feature_cols": feature_cols,
        "beta_solution": beta_solution,
        "error_history": errors,
        "X_train": X_train,
        "Y_train": Y_train,
        "fitted_train": fitted_train,
        "X_test": X_test,
        "Y_test_actual": Y_test_actual,
        "predicted_goals": predicted_goals,
        "prediction_error": prediction_error,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_and_save(
    df: pd.DataFrame,
    results: Dict[str, Any],
    output_path: Path,
) -> None:
    """
    Renders and saves the CGLS optimization convergence and prediction plot.
    """
    team_name = df["Team"].iloc[0] if "Team" in df.columns else "Selected Window"
    errors = results["error_history"]
    Y_train = results["Y_train"]
    fitted_train = results["fitted_train"]
    Y_test_actual = results["Y_test_actual"]
    predicted_goals = results["predicted_goals"]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Subplot 1: Conjugate Gradient Residual Error Convergence
    ax1.plot(errors, marker="o", linestyle="-", color="indigo", linewidth=2)
    ax1.set_title("CGLS Optimization: Residual Matrix Error", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Iteration Step ($k$)", fontsize=10)
    ax1.set_ylabel(r"Normal Equation Residual Norm ($||s_k||$)", fontsize=10)
    ax1.set_yscale("log")
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)

    # Subplot 2: Goal Predictions (In-Sample Fitted + Out-of-Sample Test)
    x_train_idx = np.arange(1, len(Y_train) + 1)
    ax2.plot(x_train_idx, Y_train, "ko-", label="Actual Goals (Matches 1-9)", linewidth=1.5)
    ax2.plot(x_train_idx, fitted_train, "b--", label="CGLS Fitted Goals", linewidth=1.5)

    x_test_idx = len(df)
    ax2.scatter(
        [x_test_idx],
        [Y_test_actual],
        color="red",
        marker="*",
        s=200,
        label=f"Match 10 Actual ({Y_test_actual:.0f} Goals)",
        zorder=5,
    )
    ax2.scatter(
        [x_test_idx],
        [predicted_goals],
        color="green",
        marker="X",
        s=150,
        label=f"Match 10 Predicted ({predicted_goals:.2f} Goals)",
        zorder=5,
    )
    ax2.vlines(
        x_test_idx,
        min(Y_test_actual, predicted_goals),
        max(Y_test_actual, predicted_goals),
        colors="gray",
        linestyles="dashed",
        label=f"Prediction Error ({results['prediction_error']:.2f})",
    )

    ax2.set_title(f"Goal Forecast (Out-of-Sample) - {team_name}", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Match Index", fontsize=10)
    ax2.set_ylabel("Goals Scored (GF)", fontsize=10)
    ax2.set_xticks(np.arange(1, len(df) + 1))
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=9)

    plt.suptitle(
        f"CGLS Multivariate Linear Regression - {team_name} "
        f"({df['Date'].iloc[0]} to {df['Date'].iloc[-1]})",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()


# ---------------------------------------------------------------------------
# Main Execution Workflow
# ---------------------------------------------------------------------------
def main(args: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Multivariate Linear Regression via CGLS on consecutive match metrics."
    )
    parser.add_argument(
        "--db",
        type=str,
        required=True,
        help="Path to SQLite database.",
    )
    parser.add_argument(
        "--team",
        type=str,
        required=True,
        help="Team name to sample 10 consecutive matches for.",
    )
    parsed_args = parser.parse_args(args)

    output_path = PLOTS_DIR / f"{ALGORITHM_NAME}.png"

    # 1. Load consecutive matches from SQLite
    df = load_consecutive_matches(
        db_path=parsed_args.db,
        team=parsed_args.team,
        window_size=WINDOW_SIZE,
    )

    # 2. Run the mathematical algorithm
    results = run_cgls_regression(df)

    test_row = df.iloc[-1]
    venue = "Home" if test_row["Is_Home"] == 1 else "Away"
    beta_solution = results["beta_solution"]

    print("=" * 60)
    print("             CGLS PREDICTION REPORT             ")
    print("=" * 60)
    print(
        f"Match #10 Context   : {venue} vs {test_row['Opponent']} "
        f"(Date: {test_row['Date']})"
    )
    print(
        f"Pre-Match Form      : {test_row['Form_Sh_3M']:.1f} Shots/Game, "
        f"{test_row['Form_SoT_3M']:.1f} SoT/Game, "
        f"{test_row['Form_Poss_3M']:.1f}% Poss Est"
    )
    print("-" * 60)
    print(f"Calculated Intercept (Beta_0): {beta_solution[0]:.4f}")
    print(f"Feature Weights (Beta_1 to Beta_5):")
    for col, b in zip(results["feature_cols"], beta_solution[1:]):
        print(f"  -> {col:<14}: {b:.4f}")
    print("-" * 60)
    print(f"PREDICTED GOALS FOR MATCH #10 : {results['predicted_goals']:.2f}")
    print(f"ACTUAL GOALS SCORED           : {results['Y_test_actual']:.0f}")
    print(f"ABSOLUTE PREDICTION ERROR     : {results['prediction_error']:.2f}")
    print(f"CGLS Residual Iterations      : {len(results['error_history'])} steps")
    print("-" * 60)

    # 3. Plot and save results
    plot_and_save(df, results, output_path)
    print(f"Plot Successfully Saved -> {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()