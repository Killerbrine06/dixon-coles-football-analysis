"""
Least Squares Polynomial Fit for Match Metrics.

Loads a window of 10 consecutive finished matches from SQLite, fits a Least
Squares Polynomial of degree d (default d=3) across the first 9 matches, and
evaluates out-of-sample forecast performance against the 10th match.

Usage:
    python general_models/polynomial_fit.py
"""

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

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
ALGORITHM_NAME = "polynomial_fit"
WINDOW_SIZE = 10
DEFAULT_DEGREE = 3
COL_OF_INTEREST = "SoT%"


# ---------------------------------------------------------------------------
# Mathematical Algorithm: Least Squares Polynomial Fit
# ---------------------------------------------------------------------------
def run_polynomial_fit(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_real: float,
    y_real: float,
    degree: int = DEFAULT_DEGREE,
) -> Dict[str, Any]:
    """
    Fits a Least Squares Polynomial on training matches and evaluates on the target point.

    Parameters
    ----------
    x_train : np.ndarray
        X coordinates for training matches (indices 1 to 9).
    y_train : np.ndarray
        Y metric values for training matches.
    x_real : float
        X coordinate of the excluded evaluation match (index 10).
    y_real : float
        Actual Y metric value of the excluded evaluation match.
    degree : int, default 3
        Degree of the polynomial basis function.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing polynomial coefficients, poly1d object, prediction, and curve.
    """
    # 1. np.polyfit uses least squares optimization to solve the Vandermonde normal equations
    coefficients = np.polyfit(x_train, y_train, deg=degree)
    polynomial = np.poly1d(coefficients)

    # 2. Evaluate prediction at the out-of-sample point
    y_approx = float(polynomial(x_real))
    prediction_error = abs(y_real - y_approx)

    # 3. Dense curve for smooth plotting spanning from match 1 to beyond match 10
    x_curve = np.linspace(min(x_train) - 0.5, max(x_train) + 1.2, 200)
    y_curve = polynomial(x_curve)

    return {
        "degree": degree,
        "coefficients": coefficients,
        "polynomial": polynomial,
        "x_train": x_train,
        "y_train": y_train,
        "x_curve": x_curve,
        "y_curve": y_curve,
        "x_real": x_real,
        "y_real": y_real,
        "y_approx": y_approx,
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
    Renders and saves the Least Squares Polynomial fit plot.
    """
    team_name = df["Team"].iloc[0] if "Team" in df.columns else "Selected Window"
    degree = results["degree"]
    x_train = results["x_train"]
    y_train = results["y_train"]
    x_curve = results["x_curve"]
    y_curve = results["y_curve"]
    x_real = results["x_real"]
    y_real = results["y_real"]
    y_approx = results["y_approx"]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    # Plot original training points
    plt.scatter(
        x_train,
        y_train,
        color="blue",
        label="Data Points (Matches 1-9)",
        s=60,
        zorder=5,
    )

    # Plot polynomial approximation curve
    plt.plot(
        x_curve,
        y_curve,
        color="orange",
        linewidth=2,
        label=f"Least Squares Polynomial (Degree {degree})",
    )

    # Plot excluded real point
    plt.scatter(
        [x_real],
        [y_real],
        color="red",
        marker="*",
        s=200,
        label=f"Real Point (Match 10 Actual: {y_real:.1f}%)",
        zorder=5,
    )

    # Plot approximated point on polynomial curve
    plt.scatter(
        [x_real],
        [y_approx],
        color="green",
        marker="X",
        s=150,
        label=f"Approximated Point (Poly: {y_approx:.1f}%)",
        zorder=5,
    )

    # Draw vertical dashed line visualizing prediction error
    plt.vlines(
        x=x_real,
        ymin=min(y_real, y_approx),
        ymax=max(y_real, y_approx),
        colors="gray",
        linestyles="dashed",
        label=f"Prediction Error ({results['prediction_error']:.1f}%)",
    )

    plt.title(
        f"Least Squares Polynomial Fit (Degree {degree}) - {team_name}\n"
        f"10 Consecutive Matches ({df['Date'].iloc[0]} to {df['Date'].iloc[-1]})",
        fontsize=13,
        fontweight="bold",
    )
    plt.xlabel("Match Index", fontsize=11)
    plt.ylabel(f"{COL_OF_INTEREST}", fontsize=11)
    plt.xticks(np.arange(1, WINDOW_SIZE + 1))
    plt.legend(frameon=True, facecolor="white", framealpha=0.9)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()


# ---------------------------------------------------------------------------
# Main Execution Workflow
# ---------------------------------------------------------------------------
def main(args: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Least Squares Polynomial fit on consecutive match metrics."
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
    parser.add_argument(
        "--degree",
        type=int,
        default=DEFAULT_DEGREE,
        help="Degree of the polynomial fit (default: 3).",
    )
    parsed_args = parser.parse_args(args)

    output_path = PLOTS_DIR / f"{ALGORITHM_NAME}.png"

    print("=" * 60)
    print("      LEAST SQUARES POLYNOMIAL APPROXIMATION & FORECAST       ")
    print("=" * 60)

    # 1. Load consecutive matches from SQLite
    df = load_consecutive_matches(
        db_path=parsed_args.db,
        team=parsed_args.team,
        window_size=WINDOW_SIZE,
    )

    team_name = df["Team"].iloc[0] if "Team" in df.columns else "League"
    print(f"Sampled Team        : {team_name}")
    print(f"Match Date Range    : {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
    print(f"Window Size         : {len(df)} consecutive matches (9 Train, 1 Test)")
    print("-" * 60)
    print(f"{'Idx':<4} {'Date':<12} {'Opponent':<18} {'Venue':<6} {'GF':<4} {'SoT%':<8} {'Role'}")
    print("-" * 60)
    for idx, row in df.iterrows():
        venue = "Home" if row["Is_Home"] == 1 else "Away"
        role = "Train" if idx < len(df) - 1 else "Test (Out-of-Sample)"
        print(
            f"{idx + 1:<4} {row['Date']:<12} {row['Opponent']:<18} "
            f"{venue:<6} {row['GF']:<4.0f} {row['SoT%']:<8.2f}% {role}"
        )
    print("-" * 60)

    # 2. Run the mathematical algorithm
    n_train = len(df) - 1
    x_train = np.arange(1, n_train + 1)
    y_train = df[COL_OF_INTEREST].iloc[:n_train].values
    x_real = float(len(df))
    y_real = float(df[COL_OF_INTEREST].iloc[-1])

    results = run_polynomial_fit(
        x_train, y_train, x_real, y_real, degree=parsed_args.degree
    )

    print(f"Algorithm Executed  : Least Squares Polynomial (Degree {results['degree']})")
    print(f"Polynomial Formula  :\n{results['polynomial']}")
    print(f"Match #10 Opponent  : {df['Opponent'].iloc[-1]} (Date: {df['Date'].iloc[-1]})")
    print(f"ACTUAL SOT% FOR MATCH #10     : {results['y_real']:.2f}%")
    print(f"POLYNOMIAL MODEL PREDICTION   : {results['y_approx']:.2f}%")
    print(f"ABSOLUTE PREDICTION ERROR     : {results['prediction_error']:.2f}%")
    print("-" * 60)

    # 3. Plot and save results
    plot_and_save(df, results, output_path)
    print(f"Plot Successfully Saved -> {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()