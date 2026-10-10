import pandas as pd

# 1. Load the raw dataset directly from the GitHub repository
# url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
df = pd.read_csv('data/raw/world_results.csv')

# 2. Filter for the last decade (e.g., 2016 onwards)
df['date'] = pd.to_datetime(df['date'])
df = df[df['date'].dt.year >= 2010].copy()

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

# 5. Extract only your required columns
final_df = df[['Id', 'Date', 'Team1', 'Team2', 'TeamHome', 'GF', 'GA']]

# 6. Export to CSV
final_df.to_csv('data/proccessed/official_football_data_decade.csv', index=False)
final_df.info()
print("Data successfully formatted and exported!")