"""
Natural Cubic Spline (C2) Interpolation for Match Metrics.

Loads a window of 10 consecutive finished matches from SQLite, fits a Natural
Cubic Spline (with boundary condition f''(x_0) = 0 and f''(x_n) = 0) across the first
9 matches, and evaluates out-of-sample extrapolation performance against the 10th match.

Usage:
    python general_models/splinec2.py
"""

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

try:
    from data_loader import PLOTS_DIR, load_consecutive_matches
except ImportError:
    from general_models.data_loader import PLOTS_DIR, load_consecutive_matches

# ---------------------------------------------------------------------------
# Global Constants & Paths
# ---------------------------------------------------------------------------
ALGORITHM_NAME = "splinec2"
WINDOW_SIZE = 10
COL_OF_INTEREST = "SoT%"


# ---------------------------------------------------------------------------
# Mathematical Algorithm: Natural Cubic Spline Interpolation (C^2)
# ---------------------------------------------------------------------------
def run_cubic_spline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_real: float,
    y_real: float,
) -> Dict[str, Any]:
    """
    Fits a natural cubic spline on training points and predicts the target out-of-sample point.

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

    Returns
    -------
    Dict[str, Any]
        Dictionary containing spline object, dense curve points, prediction, and error.
    """
    # 1. Fit Natural Cubic Spline: bc_type='natural' enforces f''(x[0]) = 0 and f''(x[-1]) = 0
    cs = CubicSpline(x_train, y_train, bc_type="natural")

    # 2. Evaluate prediction at the out-of-sample point
    y_approx = float(cs(x_real))
    prediction_error = abs(y_real - y_approx)

    # 3. Dense curve for smooth plotting spanning from match 1 to beyond match 10
    x_curve = np.linspace(min(x_train), max(x_train) + 1.2, 200)
    y_curve = cs(x_curve)

    return {
        "cs": cs,
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
    Renders and saves the Natural Cubic Spline interpolation plot.
    """
    team_name = df["Team"].iloc[0] if "Team" in df.columns else "Selected Window"
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
        color="red",
        s=100,
        label="Data Points (Matches 1-9)",
        zorder=5,
    )

    # Plot natural cubic spline curve
    plt.plot(
        x_curve,
        y_curve,
        color="blue",
        linewidth=2,
        label="Natural Cubic Spline ($C^2$)",
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

    # Plot approximated point on the spline
    plt.scatter(
        [x_real],
        [y_approx],
        color="green",
        marker="X",
        s=150,
        label=f"Approximated Point (Spline: {y_approx:.1f}%)",
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
        f"Natural Cubic Spline Interpolation ($C^2$) - {team_name}\n"
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
        description="Natural Cubic Spline (C2) interpolation on consecutive match metrics."
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

    print("=" * 60)
    print("      NATURAL CUBIC SPLINE (C2) INTERPOLATION & FORECAST      ")
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

    results = run_cubic_spline(x_train, y_train, x_real, y_real)

    print("Algorithm Executed  : Natural Cubic Spline (bc_type='natural')")
    print("Boundary Condition  : f''(x_1) = 0 and f''(x_9) = 0")
    print(f"Match #10 Opponent  : {df['Opponent'].iloc[-1]} (Date: {df['Date'].iloc[-1]})")
    print(f"ACTUAL SOT% FOR MATCH #10     : {results['y_real']:.2f}%")
    print(f"SPLINE EXTRAPOLATION PREDICTED: {results['y_approx']:.2f}%")
    print(f"ABSOLUTE PREDICTION ERROR     : {results['prediction_error']:.2f}%")
    print("-" * 60)

    # 3. Plot and save results
    plot_and_save(df, results, output_path)
    print(f"Plot Successfully Saved -> {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()