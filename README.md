# Dixon-Coles Football Match Prediction and Analytics

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Repository](https://img.shields.io/badge/GitHub-Killerbrine06%2Fdixon--coles--football--analysis-181717?logo=github)](https://github.com/Killerbrine06/dixon-coles-football-analysis)
[![Framework](https://img.shields.io/badge/Model-Dixon--Coles%20(1997)-blue.svg)](docs/Dixon_Coles_Lecture.md)
[![Status](https://img.shields.io/badge/Status-Maintained-brightgreen.svg)]()

A quantitative sports analytics framework implementing the Dixon-Coles (1997) bivariate Poisson model for football match forecasting, time-decay parameter optimization, and out-of-sample evaluation.

---

## About

This repository implements the predictive sports modeling framework established by Mark J. Dixon and Stuart G. Coles (1997). The system forecasts match scoreline distributions while accounting for team attack/defense ratings, home field advantage, low-score event dependence, and temporal form decay.

### Mathematical Framework

- **Expected Goals ($\lambda, \mu$)**: Modeled via log-linear attack ($\alpha$), defense ($\beta$), and home advantage ($\gamma$):
  $$\lambda = \alpha_i \cdot \beta_j \cdot \gamma \quad (\text{Home}), \qquad \mu = \alpha_j \cdot \beta_i \quad (\text{Away})$$
- **Low-Score Interaction ($\tau$)**: Applies correlation parameter $\rho$ to correct for under-represented draws ($0-0, 1-1$) and narrow wins ($1-0, 0-1$):
  $$\tau(0,0) = 1 - \lambda \mu \rho, \quad \tau(1,0) = 1 + \mu \rho, \quad \tau(0,1) = 1 + \lambda \rho, \quad \tau(1,1) = 1 - \rho$$
- **Exponential Time Decay ($w$)**: Discounts older matches using half-life parameter $\xi$ based on days elapsed ($\Delta t$):
  $$w_k = \exp(-\xi \cdot \Delta t_k)$$
- **Regularized Maximum Likelihood**: Solves parameters via L-BFGS-B with sample-size-invariant $L_2$ shrinkage and identifiability constraints.

### Core Modules

- `predict.py`: Computes expected goals ($xG$), win/draw/loss probabilities, and top scorelines for scheduled matches.
- `dixon_coles_multithread.py`: Multi-threaded solver running walk-forward grid search over $\xi$ and exporting calibrated parameters to JSON.
- `tools/db_pipeline.py`: Automated ETL pipeline that downloads match records, manages SQLite databases, and exports formatted CSV datasets.
- `tools/run_test_suite.py`: Multi-tier evaluation engine running mathematical verification, benchmark evaluations, and rendering `Dixon_Coles_Evaluation_Report.pdf`.
- `general_models/`: Exploratory statistical scripts (cubic splines, polynomial fit, CGLS regression, and FFT signal reconstruction).

---

## How to Install

### Prerequisites

- Python 3.10 or higher
- Git

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/Killerbrine06/dixon-coles-football-analysis.git
   cd dixon-coles-football-analysis
   ```

2. Create and activate a virtual environment:
   - On macOS/Linux:
     ```bash
     python3 -m venv env
     source env/bin/activate
     ```
   - On Windows:
     ```bash
     python -m venv env
     env\Scripts\activate
     ```

3. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## How to use/examples of usage

### 1. Ingest Data and Build Databases

The repository does not bundle raw match databases. Before training models or generating predictions, run the database pipeline to fetch historical match data from football-data.co.uk, build local SQLite databases, and export the processed CSV files:

```bash
# Build databases and export processed datasets for both EPL and La Liga
python tools/db_pipeline.py --league all --action build
```

To update active seasons with the latest match results:
```bash
python tools/db_pipeline.py --league all --action update
```

To re-export training CSV files from existing databases:
```bash
python tools/db_pipeline.py --league epl --action export
```

### 2. Train and Optimize Model Parameters

Run `dixon_coles_multithread.py` on the generated CSV file. The script performs multi-threaded walk-forward validation to select the optimal time-decay parameter $\xi$, estimates team parameters, and exports calibrated values to JSON:

```bash
# Train Premier League model
python dixon_coles_multithread.py data/proccessed/epl_ready_data.csv

# Train La Liga model
python dixon_coles_multithread.py data/proccessed/laliga_ready_data.csv
```

The resulting JSON file (`data/proccessed/<league>_ready_data.json`) contains fitted constants and team ratings:
```json
{
    "global_params": {
        "gamma": 1.2514,
        "rho": -0.0412,
        "xi": 0.0019
    },
    "teams": [
        {
            "name": "Arsenal",
            "alpha": 1.3412,
            "beta": 0.7821
        },
        {
            "name": "Chelsea",
            "alpha": 1.1205,
            "beta": 0.9410
        }
    ]
}
```

### 3. Generate Match Predictions

Run `predict.py` with the team names and target league to calculate expected goals ($xG$), outcome probabilities, and most likely scorelines.

#### Example: Premier League Match

```bash
python predict.py Arsenal Chelsea --league epl
```

Output:
```text
--- MATCH PREDICTION: Arsenal vs Chelsea (Home: Arsenal | Away: Chelsea) ---
Database: data/proccessed/epl_ready_data.json
Expected Goals (xG): Arsenal 2.00 - 1.05 Chelsea

MATCH PROBABILITIES:
Arsenal Win: 58.7%
Draw:      22.6%
Chelsea Win: 18.7%

TOP 3 MOST LIKELY EXACT SCORELINES:
  1-1: 10.6%
  2-1: 9.9%
  2-0: 9.4%
```

#### Example: Neutral Venue Match

Pass `--neutral` to disable home field advantage ($\gamma = 1.0$):

```bash
python predict.py "Real Madrid" "Barcelona" --league laliga --neutral
```

Output:
```text
--- MATCH PREDICTION: Real Madrid vs Barcelona (Neutral Venue) ---
Database: data/proccessed/laliga_ready_data.json
Expected Goals (xG): Real Madrid 1.52 - 1.75 Barcelona

MATCH PROBABILITIES:
Real Madrid Win: 33.6%
Draw:      23.1%
Barcelona Win: 43.3%

TOP 3 MOST LIKELY EXACT SCORELINES:
  1-1: 10.2%
  1-2: 8.9%
  2-1: 7.7%
```

### 4. Run Exploratory Statistical Models

The `general_models/` directory contains supplementary curve-fitting and numerical models applied to match metrics:

```bash
# Fast Fourier Transform (FFT) signal reconstruction
python general_models/fft.py

# Multivariate Linear Regression via Conjugate Gradient Least Squares (CGLS)
python general_models/multivar_lin_reg.py

# Natural Cubic Spline (C2) interpolation
python general_models/splinec2.py

# Polynomial Least Squares fit
python general_models/polynomial_fit.py
```

---

## Benchmarks

### What This Model Achieves

- **Superiority Over Independent Poisson**: By incorporating the low-score correlation adjustment factor $\tau(x, y)$, the model captures empirical draw clusters ($0-0$ and $1-1$), outperforming standard unadjusted Poisson baselines on out-of-sample Brier score and Ranked Probability Score (RPS).
- **Adaptive Form Weighting**: The exponential time-decay parameter ($\xi$) allows ratings to evolve with team roster and tactical changes, improving predictive accuracy over static historical averages.
- **Warm-Start Solver Efficiency**: Leveraging parameter warm-starting across rolling walk-forward windows reduces L-BFGS-B convergence time by up to 60% compared to cold starts.
- **Rigorous Mathematical Verification**: Automated tests confirm probability mass conservation ($\sum P = 1.0$) across extreme parameter distributions.

### Running the Evaluation Suite

Execute the comprehensive test harness to verify mathematical properties, run walk-forward benchmarks across EPL and La Liga, and compile diagnostic charts:

```bash
python tools/run_test_suite.py
```

This generates `Dixon_Coles_Evaluation_Report.pdf`, containing:
- Mathematical proof verification tables.
- Cold-start vs. warm-start runtime comparisons.
- Multi-season out-of-sample Brier score tracking against naive baselines and closing market odds.
- Reliability diagrams and probability calibration curves.

### Disclaimer

This repository is developed strictly for educational, scientific, and quantitative research purposes. It is not a betting tool, does not beat bookmakers, and should not be trusted with real money. Modern sports betting markets are highly efficient and incorporate team lineups, tactical developments, injuries, and real-time capital flow that bivariate Poisson models alone cannot capture. Over extended betting simulations, transaction costs and bookmaker margins (overrounds) yield negative expected returns.
