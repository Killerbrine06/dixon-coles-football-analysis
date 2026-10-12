"""
Data loader utility for general models.

Retrieves 10 consecutive matches for a specified team from an SQLite database.
"""

import os
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd

PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"


def load_consecutive_matches(db_path: str, team: str, window_size: int = 10) -> pd.DataFrame:
    """
    Loads a random window of `window_size` consecutive finished matches for a given team.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    query = """
        SELECT date, home_team, away_team, home_goals, away_goals,
               home_shots, away_shots, home_shots_target, away_shots_target
        FROM matches
        WHERE status = 'FINISHED'
          AND (home_team = ? OR away_team = ?)
          AND home_shots IS NOT NULL
          AND away_shots IS NOT NULL
        ORDER BY date ASC, id ASC
    """
    raw_df = pd.read_sql_query(query, conn, params=[team, team])
    conn.close()

    if len(raw_df) < window_size:
        raise ValueError(f"Team '{team}' only has {len(raw_df)} finished matches, need at least {window_size}.")

    rows = []
    for _, row in raw_df.iterrows():
        is_home = 1 if row["home_team"] == team else 0
        opp = row["away_team"] if is_home else row["home_team"]
        gf = float(row["home_goals"]) if is_home else float(row["away_goals"])
        ga = float(row["away_goals"]) if is_home else float(row["home_goals"])
        sh = float(row["home_shots"]) if is_home else float(row["away_shots"])
        sot = float(row["home_shots_target"]) if is_home else float(row["away_shots_target"])
        opp_sh = float(row["away_shots"]) if is_home else float(row["home_shots"])

        sot_pct = (sot / sh * 100.0) if sh > 0 else 0.0
        poss_est = (sh / (sh + opp_sh) * 100.0) if (sh + opp_sh) > 0 else 50.0

        rows.append({
            "Date": str(row["date"]),
            "Team": team,
            "Opponent": opp,
            "Is_Home": is_home,
            "GF": gf,
            "GA": ga,
            "Sh": sh,
            "SoT": sot,
            "SoT%": round(sot_pct, 2),
            "Poss_Est": round(poss_est, 1),
        })

    df = pd.DataFrame(rows)
    df["Date_dt"] = pd.to_datetime(df["Date"])
    df["Rest_Days"] = df["Date_dt"].diff().dt.days.fillna(7.0).clip(1.0, 30.0)
    df["Form_Sh_3M"] = df["Sh"].shift(1).rolling(3, min_periods=1).mean().fillna(df["Sh"])
    df["Form_SoT_3M"] = df["SoT"].shift(1).rolling(3, min_periods=1).mean().fillna(df["SoT"])
    df["Form_Poss_3M"] = df["Poss_Est"].shift(1).rolling(3, min_periods=1).mean().fillna(50.0)
    df = df.drop(columns=["Date_dt"])

    while True:
        start_pos = np.random.randint(0, len(df))
        if start_pos + window_size <= len(df):
            return df.iloc[start_pos : start_pos + window_size].copy().reset_index(drop=True)
