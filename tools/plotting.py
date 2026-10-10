import pandas as pd
import matplotlib.pyplot as plt

# Load the previously generated CSV
file_path = 'PSG_UCL_2023_24_Merged_Stats.csv'
df = pd.read_csv(file_path)

# Injecting Possession data (Group Stage + Knockout estimates)
df['Possession'] = [68, 73, 62, 69, 72, 59, 58, 54, 54, 46, 65, 67] 

# 1. Plot Possession (Line Chart)
plt.figure(figsize=(10, 5))
plt.plot(df['Date'], df['Possession'], marker='o', linestyle='-', color='#1f77b4', linewidth=2)
plt.title('PSG Ball Possession per Match (UCL 23/24)', fontsize=14)
plt.xlabel('Match Date', fontsize=12)
plt.ylabel('Possession (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('possession_plot.png')
plt.close()

# 2. Plot SoT% (Line Chart)
plt.figure(figsize=(10, 5))
plt.plot(df['Date'], df['SoT%'], marker='o', linestyle='-', color='#1f77b4', linewidth=2)
plt.title('PSG Shots on Target percentage per Match (UCL 23/24)', fontsize=14)
plt.xlabel('Match Date', fontsize=12)
plt.ylabel('SoT (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('sot_plot.png')
plt.close()

# 3. Plot Crosses (Bar Chart)
plt.figure(figsize=(10, 5))
plt.bar(df['Date'], df['Crs'], color='#2ca02c', alpha=0.8)
plt.title('PSG Crosses Attempted per Match (UCL 23/24)', fontsize=14)
plt.xlabel('Match Date', fontsize=12)
plt.ylabel('Number of Crosses', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('crosses_plot.png')
plt.close()

# 4. Plot Offsides (Step/Line Chart)
plt.figure(figsize=(10, 5))
plt.plot(df['Date'], df['Off'], marker='s', linestyle='-.', color='#d62728', linewidth=2)
plt.title('PSG Offsides per Match (UCL 23/24)', fontsize=14)
plt.xlabel('Match Date', fontsize=12)
plt.ylabel('Number of Offsides', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('offsides_plot.png')
plt.close()