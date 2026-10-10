import pandas as pd
import country_converter as coco

# 1. Load the raw dataset
df = pd.read_csv('data/raw/world_results.csv')

# 2. Define your relevant tournaments
relevant_tournaments = [
    'FIFA World Cup',
    'FIFA World Cup qualification',
    'UEFA Euro',
    'UEFA Euro qualification',
    'Copa América',
    'Copa América qualification',
    'African Cup of Nations',
    'African Cup of Nations qualification',
    'AFC Asian Cup',
    'AFC Asian Cup qualification',
    'CONCACAF Gold Cup',
    'CONCACAF Gold Cup qualification',
    'UEFA Nations League',
    'CONCACAF Nations League',
    # 'Friendly'
]

# 3. FILTER FIRST: Keep only the relevant tournaments
# This immediately drops Alderney, Jersey, and other non-FIFA regional games
df = df[df['tournament'].isin(relevant_tournaments)].copy()
df['date'] = pd.to_datetime(df['date'])
df = df[df['date'].dt.year >= 2022].copy()

# 3. Map the columns to your requested schema
df['Id'] = range(1, len(df) + 1)
df['Date'] = df['date'].dt.strftime('%Y-%m-%d')
df['Team1'] = df['home_team']
df['Team2'] = df['away_team']
df['GF'] = df['home_score']
df['GA'] = df['away_score']

# 4. Handle the 'TeamHome' logic
# If 'neutral' is True, neither is home (0). Otherwise, Team 1 is home (1).
df['TeamHome'] = df['neutral'].apply(lambda is_neutral: 0 if is_neutral else 1)
df = df[['Id', 'Date', 'Team1', 'Team2', 'TeamHome', 'GF', 'GA']]

# 4. Enriched custom dictionary for FIFA edge cases
custom_football_codes = {
    "England": "ENG",
    "Scotland": "SCO",
    "Wales": "WAL",
    "Northern Ireland": "NIR",
    "Kosovo": "KOS",
    "Cape Verde": "CPV",
    "Ivory Coast": "CIV",
    "DR Congo": "COD",
    "Republic of Ireland": "IRL",
    "South Korea": "KOR",
    "North Korea": "PRK",
    "USA": "USA",
    "United States": "USA",
    "Türkiye": "TUR",
    "Turkey": "TUR"
}

# 5. Extract unique names and map
all_teams = pd.Series(df['Team1'].tolist() + df['Team2'].tolist()).dropna().unique()

master_mapping = {}
teams_for_coco = []

for team in all_teams:
    clean_team = str(team).strip()
    if clean_team in custom_football_codes:
        master_mapping[clean_team] = custom_football_codes[clean_team]
    else:
        teams_for_coco.append(clean_team)

if teams_for_coco:
    print(f"Sending {len(teams_for_coco)} standard countries to coco...")
    coco_results = coco.convert(names=teams_for_coco, to='FIFA', not_found=None)
    for original_name, converted_code in zip(teams_for_coco, coco_results):
        master_mapping[original_name] = converted_code

# 6. Apply the mapping and drop the tournament column if you only wanted it for filtering
df['Team1'] = df['Team1'].map(master_mapping).fillna(df['Team1'])
df['Team2'] = df['Team2'].map(master_mapping).fillna(df['Team2'])

df.to_csv('data/proccessed/smaller_world_cup_ready_data.csv', index=False)
print("Data filtered by tournament, mapped to FIFA codes, and saved!")