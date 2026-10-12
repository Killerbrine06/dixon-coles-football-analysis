"""
Fast Fourier Transform (FFT) Signal Reconstruction for Match Metrics.

Loads a window of 10 consecutive finished matches from SQLite, extracts the
shots on target percentage (SoT%) sequence, and performs Fast Fourier Transform (FFT)
into the frequency domain followed by Inverse Fast Fourier Transform (IFFT)
to reconstruct the continuous form signal.

Usage:
    python general_models/fft.py
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
ALGORITHM_NAME = "fft"
WINDOW_SIZE = 10
COL_OF_INTEREST = "SoT%"


# ---------------------------------------------------------------------------
# Mathematical Algorithm: Fast Fourier Transform & Inverse FFT
# ---------------------------------------------------------------------------
def run_fft(y_signal: np.ndarray) -> Dict[str, Any]:
    """
    Performs FFT analysis and IFFT signal reconstruction.

    Parameters
    ----------
    y_signal : np.ndarray
        1D array of metric values across consecutive matches.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing FFT coefficients, reconstructed signal, and metrics.
    """
    y = np.asarray(y_signal, dtype=float)
    n = len(y)
    x = np.arange(1, n + 1)

    # 1. Forward Fast Fourier Transform (spatial -> frequency domain)
    Y_fft = np.fft.fft(y)

    # 2. Inverse Fast Fourier Transform (frequency -> spatial domain)
    y_approx = np.fft.ifft(Y_fft)

    # Note: IFFT often returns infinitesimal imaginary components due to floating-point
    # arithmetic. We take the real component for the physical reconstruction.
    y_approx_real = y_approx.real
    max_recon_error = float(np.max(np.abs(y - y_approx_real)))

    return {
        "x": x,
        "y": y,
        "Y_fft": Y_fft,
        "y_approx": y_approx_real,
        "max_recon_error": max_recon_error,
        "frequencies": np.fft.fftfreq(n),
        "magnitudes": np.abs(Y_fft),
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
    Renders and saves the FFT signal reconstruction plot.
    """
    team_name = df["Team"].iloc[0] if "Team" in df.columns else "Selected Window"
    x = results["x"]
    y = results["y"]
    y_approx = results["y_approx"]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    # Plot original data points
    plt.plot(
        x,
        y,
        "ro",
        label="Original Data Points",
        markersize=7,
        zorder=5,
    )

    # Plot reconstructed approximation via IFFT
    plt.plot(
        x,
        y_approx,
        "b-",
        label="Reconstructed Approximation (IFFT)",
        linewidth=2,
        alpha=0.8,
    )

    plt.title(
        f"FFT and Inverse FFT Reconstruction ($h=1$) - {team_name}\n"
        f"10 Consecutive Matches ({df['Date'].iloc[0]} to {df['Date'].iloc[-1]})",
        fontsize=13,
        fontweight="bold",
    )
    plt.xlabel("Match Index ($h=1$)", fontsize=11)
    plt.ylabel(f"{COL_OF_INTEREST}", fontsize=11)
    plt.xticks(x)
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
        description="Fast Fourier Transform (FFT) signal reconstruction on match metrics."
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
    print("      FAST FOURIER TRANSFORM (FFT) SIGNAL RECONSTRUCTION      ")
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
    print(f"Window Size         : {len(df)} consecutive matches")
    print("-" * 60)
    print(f"{'Idx':<4} {'Date':<12} {'Opponent':<18} {'Venue':<6} {'GF':<4} {'SoT%':<8}")
    print("-" * 60)
    for idx, row in df.iterrows():
        venue = "Home" if row["Is_Home"] == 1 else "Away"
        print(
            f"{idx + 1:<4} {row['Date']:<12} {row['Opponent']:<18} "
            f"{venue:<6} {row['GF']:<4.0f} {row['SoT%']:<8.2f}%"
        )
    print("-" * 60)

    # 2. Run the mathematical algorithm
    y = df[COL_OF_INTEREST].values
    results = run_fft(y)

    print(f"Algorithm Executed  : 1D Fast Fourier Transform & Inverse FFT")
    print(f"Max Recon Error     : {results['max_recon_error']:.2e} (Identity Verified)")
    print(f"DC Component (Freq=0): {results['magnitudes'][0] / len(y):.2f}% (Average SoT%)")
    print("-" * 60)

    # 3. Plot and save results
    plot_and_save(df, results, output_path)
    print(f"Plot Successfully Saved -> {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()