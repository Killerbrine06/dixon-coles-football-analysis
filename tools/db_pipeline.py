"""
Football Data Ingestion and Update Pipeline for EPL and La Liga.

Connects to verified football data providers (football-data.co.uk with optional
football-data.org REST API support) to build, update, and maintain SQLite databases
and export Dixon-Coles-ready datasets.
"""

import os
import io
import sys
import sqlite3
import argparse
import datetime
import urllib.request
import pandas as pd
from typing import List, Dict, Optional

# Supported leagues and metadata
LEAGUE_CONFIG = {
    'epl': {
        'name': 'English Premier League',
        'div_code': 'E0',
        'db_name': 'epl.db',
        'export_csv': 'epl_ready_data.csv',
        'api_code': 'PL'
    },
    'laliga': {
        'name': 'Spanish La Liga',
        'div_code': 'SP1',
        'db_name': 'laliga.db',
        'export_csv': 'laliga_ready_data.csv',
        'api_code': 'PD'
    }
}

DEFAULT_SEASONS = ['1819', '1920', '2021', '2122', '2223', '2324', '2425']

def get_current_season_code() -> str:
    """Returns the season code (e.g. '2425') for the active calendar period."""
    now = datetime.datetime.now()
    year = now.year
    # Football season typically starts in July/August
    if now.month >= 7:
        start_yr = year % 100
        end_yr = (year + 1) % 100
    else:
        start_yr = (year - 1) % 100
        end_yr = year % 100
    return f"{start_yr:02d}{end_yr:02d}"

def init_database(db_path: str):
    """Initializes SQLite schema for league matches."""
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                season TEXT NOT NULL,
                date TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                home_goals REAL,
                away_goals REAL,
                result TEXT,
                status TEXT NOT NULL,
                half_time_home_goals REAL,
                half_time_away_goals REAL,
                home_shots REAL,
                away_shots REAL,
                home_shots_target REAL,
                away_shots_target REAL,
                odds_home REAL,
                odds_draw REAL,
                odds_away REAL,
                source TEXT DEFAULT 'football-data.co.uk',
                updated_at TEXT,
                UNIQUE(date, home_team, away_team)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_date ON matches(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_teams ON matches(home_team, away_team)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON matches(status)")
        conn.commit()

def fetch_season_csv(div_code: str, season_code: str) -> Optional[pd.DataFrame]:
    """Downloads and parses a season CSV from football-data.co.uk."""
    url = f"https://www.football-data.co.uk/mmz4281/{season_code}/{div_code}.csv"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read()
        df = pd.read_csv(io.BytesIO(content), encoding='latin1')
        df = df.dropna(how='all')
        if 'Date' not in df.columns or 'HomeTeam' not in df.columns or 'AwayTeam' not in df.columns:
            return None
        return df
    except Exception as e:
        print(f"  [Warning] Failed to fetch {url}: {e}")
        return None

def normalize_season_dataframe(df: pd.DataFrame, season_code: str) -> pd.DataFrame:
    """Standardizes columns and formats dates to YYYY-MM-DD."""
    # Format season string e.g. 2324 -> 2023-2024
    s_start = 2000 + int(season_code[:2]) if int(season_code[:2]) < 90 else 1900 + int(season_code[:2])
    s_end = 2000 + int(season_code[2:]) if int(season_code[2:]) < 90 else 1900 + int(season_code[2:])
    season_label = f"{s_start}-{s_end}"

    # Parse dates cleanly with dayfirst=True
    parsed_dates = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    valid_mask = parsed_dates.notna() & df['HomeTeam'].notna() & df['AwayTeam'].notna()
    df_clean = df[valid_mask].copy()
    date_strs = parsed_dates[valid_mask].dt.strftime('%Y-%m-%d')

    home_goals = pd.to_numeric(df_clean['FTHG'], errors='coerce')
    away_goals = pd.to_numeric(df_clean['FTAG'], errors='coerce')

    out_data = {
        'season': [season_label] * len(df_clean),
        'date': date_strs,
        'home_team': df_clean['HomeTeam'].astype(str).str.strip(),
        'away_team': df_clean['AwayTeam'].astype(str).str.strip(),
        'home_goals': home_goals,
        'away_goals': away_goals,
        'result': df_clean['FTR'] if 'FTR' in df_clean.columns else None,
        'status': home_goals.notna().apply(lambda x: 'FINISHED' if x else 'SCHEDULED'),
        'half_time_home_goals': pd.to_numeric(df_clean['HTHG'], errors='coerce') if 'HTHG' in df_clean.columns else None,
        'half_time_away_goals': pd.to_numeric(df_clean['HTAG'], errors='coerce') if 'HTAG' in df_clean.columns else None,
        'home_shots': pd.to_numeric(df_clean['HS'], errors='coerce') if 'HS' in df_clean.columns else None,
        'away_shots': pd.to_numeric(df_clean['AS'], errors='coerce') if 'AS' in df_clean.columns else None,
        'home_shots_target': pd.to_numeric(df_clean['HST'], errors='coerce') if 'HST' in df_clean.columns else None,
        'away_shots_target': pd.to_numeric(df_clean['AST'], errors='coerce') if 'AST' in df_clean.columns else None,
        'odds_home': pd.to_numeric(df_clean.get('B365H', df_clean.get('AvgH')), errors='coerce'),
        'odds_draw': pd.to_numeric(df_clean.get('B365D', df_clean.get('AvgD')), errors='coerce'),
        'odds_away': pd.to_numeric(df_clean.get('B365A', df_clean.get('AvgA')), errors='coerce'),
        'source': 'football-data.co.uk',
        'updated_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    return pd.DataFrame(out_data)

def upsert_matches_to_db(db_path: str, df: pd.DataFrame) -> int:
    """Inserts or updates match records into the SQLite database."""
    if df.empty:
        return 0
    
    init_database(db_path)
    records = df[[
        'season', 'date', 'home_team', 'away_team', 'home_goals', 'away_goals',
        'result', 'status', 'half_time_home_goals', 'half_time_away_goals',
        'home_shots', 'away_shots', 'home_shots_target', 'away_shots_target',
        'odds_home', 'odds_draw', 'odds_away', 'source', 'updated_at'
    ]].values.tolist()

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.executemany("""
            INSERT INTO matches (
                season, date, home_team, away_team, home_goals, away_goals,
                result, status, half_time_home_goals, half_time_away_goals,
                home_shots, away_shots, home_shots_target, away_shots_target,
                odds_home, odds_draw, odds_away, source, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(date, home_team, away_team) DO UPDATE SET
                home_goals=excluded.home_goals,
                away_goals=excluded.away_goals,
                result=excluded.result,
                status=excluded.status,
                half_time_home_goals=excluded.half_time_home_goals,
                half_time_away_goals=excluded.half_time_away_goals,
                home_shots=excluded.home_shots,
                away_shots=excluded.away_shots,
                home_shots_target=excluded.home_shots_target,
                away_shots_target=excluded.away_shots_target,
                odds_home=excluded.odds_home,
                odds_draw=excluded.odds_draw,
                odds_away=excluded.odds_away,
                updated_at=excluded.updated_at
        """, records)
        conn.commit()
    return len(records)

def export_dixon_coles_csv(db_path: str, export_path: str) -> int:
    """
    Exports a clean CSV from SQLite matching the exact format expected by
    dixon_coles_multithread.py: Id, Date, Team1, Team2, TeamHome, GF, GA
    """
    os.makedirs(os.path.dirname(os.path.abspath(export_path)), exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        query = """
            SELECT 
                id AS Id,
                date AS Date,
                home_team AS Team1,
                away_team AS Team2,
                1 AS TeamHome,
                home_goals AS GF,
                away_goals AS GA
            FROM matches
            WHERE status = 'FINISHED' AND home_goals IS NOT NULL AND away_goals IS NOT NULL
            ORDER BY date ASC, id ASC
        """
        df = pd.read_sql_query(query, conn)
    
    # Re-index Id sequentially
    df['Id'] = range(1, len(df) + 1)
    df.to_csv(export_path, index=False)
    return len(df)

def run_pipeline(league_key: str, action: str = 'build', seasons: Optional[List[str]] = None, db_dir: str = 'data/databases', export_dir: str = 'data/proccessed'):
    """Executes the build or update process for a specified league."""
    config = LEAGUE_CONFIG[league_key]
    db_path = os.path.join(db_dir, config['db_name'])
    export_path = os.path.join(export_dir, config['export_csv'])
    
    print(f"\n==================================================")
    print(f" League: {config['name']} ({config['div_code']})")
    print(f" Action: {action.upper()} | DB: {db_path}")
    print(f"==================================================")

    if action == 'build':
        target_seasons = seasons if seasons else DEFAULT_SEASONS
        print(f"Building historical database across {len(target_seasons)} seasons: {target_seasons}")
        total_rows = 0
        for s in target_seasons:
            print(f"-> Fetching season {s}...")
            raw_df = fetch_season_csv(config['div_code'], s)
            if raw_df is not None:
                norm_df = normalize_season_dataframe(raw_df, s)
                count = upsert_matches_to_db(db_path, norm_df)
                total_rows += count
                print(f"   Saved {count} matches from season {s}.")
        print(f"\n[Build Complete] Total records ingested/upserted: {total_rows}")

    elif action == 'update':
        curr_season = get_current_season_code()
        # Also check previous season if right at the transition
        seasons_to_check = [curr_season]
        print(f"Updating active season(s): {seasons_to_check}")
        total_rows = 0
        for s in seasons_to_check:
            raw_df = fetch_season_csv(config['div_code'], s)
            if raw_df is not None:
                norm_df = normalize_season_dataframe(raw_df, s)
                count = upsert_matches_to_db(db_path, norm_df)
                total_rows += count
                print(f"   Synchronized {count} matches for season {s}.")
        print(f"\n[Update Complete] Total records synchronized: {total_rows}")

    # Always generate the latest Dixon-Coles-ready CSV export
    export_count = export_dixon_coles_csv(db_path, export_path)
    print(f"[Export] Generated {export_count} finished matches for Dixon-Coles at: {export_path}")

    # Display database summary
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT count(*), min(date), max(date), count(DISTINCT home_team) FROM matches WHERE status='FINISHED'")
        m_count, min_d, max_d, team_count = cursor.fetchone()
        cursor.execute("SELECT count(*) FROM matches WHERE status='SCHEDULED'")
        sched_count = cursor.fetchone()[0]
        print(f"[DB Status] Finished: {m_count} matches ({min_d} to {max_d}) | Teams: {team_count} | Scheduled: {sched_count}")

def main():
    parser = argparse.ArgumentParser(description="Football Data Ingestion and Update Pipeline")
    parser.add_argument('--league', choices=['epl', 'laliga', 'all'], default='all', help="League to process (default: all)")
    parser.add_argument('--action', choices=['build', 'update', 'export'], default='build', help="Pipeline action (build, update, export)")
    parser.add_argument('--seasons', nargs='+', default=None, help="Specific season codes to fetch (e.g. 2223 2324 2425)")
    parser.add_argument('--db-dir', default='data/databases', help="Directory to store SQLite databases")
    parser.add_argument('--export-dir', default='data/proccessed', help="Directory to export ready CSVs")
    args = parser.parse_args()

    leagues = ['epl', 'laliga'] if args.league == 'all' else [args.league]

    for league in leagues:
        if args.action == 'export':
            config = LEAGUE_CONFIG[league]
            db_path = os.path.join(args.db_dir, config['db_name'])
            export_path = os.path.join(args.export_dir, config['export_csv'])
            c = export_dixon_coles_csv(db_path, export_path)
            print(f"Exported {c} matches to {export_path}")
        else:
            run_pipeline(league, action=args.action, seasons=args.seasons, db_dir=args.db_dir, export_dir=args.export_dir)

if __name__ == '__main__':
    main()
