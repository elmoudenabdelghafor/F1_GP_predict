import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ==============================================================================
# LOAD — reads everything saved by f1_data_pipeline.py from disk
# Run f1_data_pipeline.py first (only once), then this file runs independently
# ==============================================================================

SAVE_DIR = 'f1_pipeline_outputs'

if not os.path.exists(SAVE_DIR):
    raise FileNotFoundError(
        f"'{SAVE_DIR}/' not found. Run f1_data_pipeline.py first to generate it."
    )

df           = pd.read_parquet(os.path.join(SAVE_DIR, 'df_prepared.parquet'))
X_train      = pd.read_parquet(os.path.join(SAVE_DIR, 'X_train.parquet'))
X_test       = pd.read_parquet(os.path.join(SAVE_DIR, 'X_test.parquet'))
y_train      = pd.read_parquet(os.path.join(SAVE_DIR, 'y_train.parquet')).squeeze()
y_test       = pd.read_parquet(os.path.join(SAVE_DIR, 'y_test.parquet')).squeeze()
X_train_res  = pd.read_parquet(os.path.join(SAVE_DIR, 'X_train_smote.parquet'))
y_train_res  = pd.read_parquet(os.path.join(SAVE_DIR, 'y_train_smote.parquet')).squeeze()

with open(os.path.join(SAVE_DIR, 'label_encoders.pkl'), 'rb') as f:
    label_encoders = pickle.load(f)

print("Loaded from disk:")
print(f"  df           : {df.shape}")
print(f"  X_train      : {X_train.shape}  |  y_train : {y_train.shape}")
print(f"  X_test       : {X_test.shape}   |  y_test  : {y_test.shape}")
print(f"  X_train_smote: {X_train_res.shape}")
print(f"  label_encoders: {list(label_encoders.keys())}")

# Style
plt.rcParams.update({
    'figure.facecolor': '#0f0f0f',
    'axes.facecolor':   '#1a1a2e',
    'axes.edgecolor':   '#444466',
    'axes.labelcolor':  '#ccccdd',
    'xtick.color':      '#ccccdd',
    'ytick.color':      '#ccccdd',
    'text.color':       '#ccccdd',
    'grid.color':       '#2a2a4a',
    'grid.linestyle':   '--',
    'grid.alpha':       0.5,
    'font.family':      'monospace',
    'axes.titlesize':   13,
    'axes.labelsize':   11,
})

RED    = '#e8003d'
GOLD   = '#ffd700'
SILVER = '#c0c0c0'
TEAL   = '#00c9a7'
PURPLE = '#9b59b6'
BLUE   = '#3498db'

OUTPUT_DIR = 'eda_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# 4.1  DESCRIPTIVE STATISTICS
# ==============================================================================

print("=" * 60)
print("4.1 — DESCRIPTIVE STATISTICS")
print("=" * 60)

numeric_features = ['position', 'grid', 'q1', 'q2', 'q3',
                    'q1_gap_to_pole', 'driver_win_rate_10',
                    'driver_win_rate_all', 'constructor_win_rate_20',
                    'driver_circuit_win_rate', 'alt']

# Replace penalty values with NaN for descriptive stats so they don't distort
df_stats = df.copy()
for col in ['q1', 'q2', 'q3', 'q1_gap_to_pole']:
    df_stats[col] = df_stats[col].replace(999_999, np.nan)

stats = df_stats[numeric_features].describe().T
stats['median'] = df_stats[numeric_features].median()
stats['skew']   = df_stats[numeric_features].skew()
print(stats[['count', 'mean', 'median', 'std', 'min', 'max', 'skew']].round(3))

print(f"\nTotal races in dataset  : {df['round'].nunique() * df['year'].nunique()}")
print(f"Total driver-race rows  : {len(df)}")
print(f"Seasons covered         : {df['year'].min()} – {df['year'].max()}")
print(f"Unique drivers          : {df['code'].nunique()}")
print(f"Unique constructors     : {df['name_constructor'].nunique()}")
print(f"Unique circuits         : {df['name_circuit'].nunique()}")
print(f"\nWinners (is_winner=1)   : {df['is_winner'].sum()}  ({df['is_winner'].mean()*100:.1f}%)")
print(f"Non-winners             : {(df['is_winner']==0).sum()}  ({(1-df['is_winner'].mean())*100:.1f}%)")

# ==============================================================================
# 4.2  PLOT 1 — Class imbalance
# ==============================================================================

fig, ax = plt.subplots(figsize=(6, 4))
fig.patch.set_facecolor('#0f0f0f')
counts = df['is_winner'].value_counts()
bars = ax.bar(['Non-winner (0)', 'Winner (1)'],
              counts.values,
              color=[SILVER, GOLD], width=0.5, edgecolor='none')
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 80,
            f'{val:,}\n({val/len(df)*100:.1f}%)',
            ha='center', va='bottom', fontsize=10, color='#ccccdd')
ax.set_title('Class distribution — is_winner', pad=12)
ax.set_ylabel('Number of rows')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax.grid(axis='x', alpha=0)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/01_class_imbalance.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print("Saved: 01_class_imbalance.png")

# ==============================================================================
# 4.3  PLOT 2 — Win rate by grid position (P1 to P20)
# ==============================================================================

win_by_grid = (df.groupby('grid')['is_winner']
                 .agg(['sum', 'count'])
                 .assign(win_rate=lambda x: x['sum'] / x['count'] * 100)
                 .loc[1:20])

fig, ax = plt.subplots(figsize=(10, 4))
fig.patch.set_facecolor('#0f0f0f')
colors = [GOLD if i == 0 else RED if i == 1 else TEAL if i < 5 else SILVER
          for i in range(len(win_by_grid))]
bars = ax.bar(win_by_grid.index, win_by_grid['win_rate'], color=colors, edgecolor='none')
ax.set_title('Win rate (%) by starting grid position', pad=12)
ax.set_xlabel('Grid position')
ax.set_ylabel('Win rate (%)')
ax.set_xticks(range(1, 21))
ax.grid(axis='y')
for bar in bars[:5]:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=8, color=GOLD)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_win_rate_by_grid.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print("Saved: 02_win_rate_by_grid.png")

# ==============================================================================
# 4.4  PLOT 3 — Pole position win rate by decade
# ==============================================================================

df['decade'] = (df['year'] // 10 * 10).astype(str) + 's'
pole_df = df[df['grid'] == 1].copy()
pole_win_by_decade = (pole_df.groupby('decade')['is_winner']
                              .mean() * 100).reset_index()
pole_win_by_decade.columns = ['decade', 'pole_win_rate']

fig, ax = plt.subplots(figsize=(8, 4))
fig.patch.set_facecolor('#0f0f0f')
ax.plot(pole_win_by_decade['decade'], pole_win_by_decade['pole_win_rate'],
        color=GOLD, marker='o', linewidth=2, markersize=8)
ax.fill_between(range(len(pole_win_by_decade)), pole_win_by_decade['pole_win_rate'],
                alpha=0.15, color=GOLD)
ax.set_xticks(range(len(pole_win_by_decade)))
ax.set_xticklabels(pole_win_by_decade['decade'])
ax.set_title('Pole position → race win conversion rate by decade', pad=12)
ax.set_ylabel('Pole-to-win rate (%)')
ax.set_ylim(0, 100)
ax.grid(axis='y')
for i, row in pole_win_by_decade.iterrows():
    ax.annotate(f"{row['pole_win_rate']:.0f}%",
                (i, row['pole_win_rate']),
                textcoords='offset points', xytext=(0, 8),
                ha='center', fontsize=9, color=GOLD)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03_pole_win_rate_by_decade.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print("Saved: 03_pole_win_rate_by_decade.png")

# ==============================================================================
# 4.5  PLOT 4 — Distribution of qualifying position for winners vs non-winners
# ==============================================================================

fig, ax = plt.subplots(figsize=(10, 4))
fig.patch.set_facecolor('#0f0f0f')
winners     = df[df['is_winner'] == 1]['position'].clip(upper=20)
non_winners = df[df['is_winner'] == 0]['position'].clip(upper=20)
bins = np.arange(0.5, 21.5, 1)
ax.hist(non_winners, bins=bins, color=SILVER, alpha=0.5, label='Non-winner', density=True)
ax.hist(winners,     bins=bins, color=GOLD,   alpha=0.8, label='Winner',     density=True)
ax.set_title('Qualifying position distribution — winners vs non-winners', pad=12)
ax.set_xlabel('Qualifying position')
ax.set_ylabel('Density')
ax.set_xticks(range(1, 21))
ax.legend(facecolor='#1a1a2e', edgecolor='#444466')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/04_qual_position_distribution.png', dpi=150,
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.show()
print("Saved: 04_qual_position_distribution.png")

# ==============================================================================
# 4.6  PLOT 5 — Correlation heatmap of numeric features
# ==============================================================================

corr_cols = ['position', 'grid', 'q1_gap_to_pole',
             'driver_win_rate_10', 'driver_win_rate_all',
             'constructor_win_rate_20', 'driver_circuit_win_rate',
             'is_pole', 'is_front_row', 'is_winner']

df_corr = df[corr_cols].copy()
df_corr['q1_gap_to_pole'] = df_corr['q1_gap_to_pole'].replace(999_999, np.nan)
corr_matrix = df_corr.corr()

fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor('#0f0f0f')
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0,
            annot=True, fmt='.2f', annot_kws={'size': 8},
            linewidths=0.5, linecolor='#0f0f0f',
            ax=ax, cbar_kws={'shrink': 0.8})
ax.set_title('Feature correlation matrix', pad=12)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/05_correlation_heatmap.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print("Saved: 05_correlation_heatmap.png")

# ==============================================================================
# 4.7  PLOT 6 — Top 10 constructors by win count (modern era)
# ==============================================================================

# Decode constructor labels back using the label encoder
constructor_wins = (df[df['is_winner'] == 1]
                    .groupby('name_constructor')
                    .size()
                    .sort_values(ascending=False)
                    .head(10))

# Map encoded int back to name using the stored label encoder
inv_constructor = {v: k for k, v in
                   enumerate(label_encoders['name_constructor'].classes_)}
constructor_wins.index = [label_encoders['name_constructor']
                          .classes_[i] for i in constructor_wins.index]

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor('#0f0f0f')
colors_c = [GOLD, SILVER, RED] + [TEAL] * 7
bars = ax.barh(constructor_wins.index[::-1],
               constructor_wins.values[::-1],
               color=colors_c[::-1], edgecolor='none')
ax.set_title('Top 10 constructors by race wins (1983–2023)', pad=12)
ax.set_xlabel('Number of wins')
for bar in bars:
    ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
            str(int(bar.get_width())), va='center', fontsize=9)
ax.grid(axis='y', alpha=0)
ax.grid(axis='x')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/06_constructor_wins.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print("Saved: 06_constructor_wins.png")

# ==============================================================================
# 4.8  PLOT 7 — Constructor win rate over time (rolling, top 4 teams)
# ==============================================================================

top4_names = constructor_wins.index[:4].tolist()

# Get encoded IDs for top 4
top4_ids = [np.where(label_encoders['name_constructor'].classes_ == name)[0][0]
            for name in top4_names]

df_top4 = df[df['name_constructor'].isin(top4_ids)].copy()
df_top4['team_name'] = df_top4['name_constructor'].map(
    dict(zip(top4_ids, top4_names)))

yearly = (df_top4.groupby(['year', 'team_name'])['is_winner']
                 .mean() * 100).reset_index()
yearly.columns = ['year', 'team', 'win_rate']

fig, ax = plt.subplots(figsize=(11, 5))
fig.patch.set_facecolor('#0f0f0f')
palette = [GOLD, RED, SILVER, TEAL]
for team, color in zip(top4_names, palette):
    data = yearly[yearly['team'] == team]
    ax.plot(data['year'], data['win_rate'], label=team,
            color=color, linewidth=1.8, alpha=0.9)
ax.set_title('Win rate per year — top 4 constructors', pad=12)
ax.set_xlabel('Year')
ax.set_ylabel('Win rate per race (%)')
ax.legend(facecolor='#1a1a2e', edgecolor='#444466', fontsize=9)
ax.grid()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/07_constructor_win_rate_over_time.png', dpi=150,
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.show()
print("Saved: 07_constructor_win_rate_over_time.png")

# ==============================================================================
# 4.9  PLOT 8 — Q1 gap to pole: winners vs non-winners (boxplot)
# ==============================================================================

df_gap = df.copy()
df_gap['q1_gap_to_pole'] = df_gap['q1_gap_to_pole'].replace(999_999, np.nan)
df_gap = df_gap.dropna(subset=['q1_gap_to_pole'])
df_gap['result'] = df_gap['is_winner'].map({0: 'Non-winner', 1: 'Winner'})

fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor('#0f0f0f')
sns.boxplot(data=df_gap, x='result', y='q1_gap_to_pole',
            palette={'Non-winner': SILVER, 'Winner': GOLD},
            width=0.45, linewidth=1.2, fliersize=2,
            flierprops=dict(markerfacecolor=TEAL, alpha=0.3), ax=ax)
ax.set_title('Q1 lap time gap to pole — winners vs non-winners', pad=12)
ax.set_xlabel('')
ax.set_ylabel('Gap to pole time (ms)')
ax.grid(axis='y')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/08_q1_gap_boxplot.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print("Saved: 08_q1_gap_boxplot.png")

# ==============================================================================
# 4.10 PLOT 9 — Missing value heatmap (Q1 / Q2 / Q3)
# ==============================================================================

missing_by_pos = pd.DataFrame({
    'Q2 missing': df.groupby('position')['q2_missing'].mean() * 100,
    'Q3 missing': df.groupby('position')['q3_missing'].mean() * 100,
}).loc[1:20]

fig, ax = plt.subplots(figsize=(10, 4))
fig.patch.set_facecolor('#0f0f0f')
ax.plot(missing_by_pos.index, missing_by_pos['Q2 missing'],
        color=BLUE, marker='o', linewidth=2, label='Q2 missing')
ax.plot(missing_by_pos.index, missing_by_pos['Q3 missing'],
        color=RED, marker='s', linewidth=2, label='Q3 missing')
ax.axvline(x=15.5, color=SILVER, linestyle='--', alpha=0.5, label='Q1 cutoff (P16)')
ax.axvline(x=10.5, color=GOLD,   linestyle='--', alpha=0.5, label='Q2 cutoff (P11)')
ax.set_title('Rate of missing Q2/Q3 times by qualifying position', pad=12)
ax.set_xlabel('Qualifying position')
ax.set_ylabel('% of rows with missing time')
ax.set_xticks(range(1, 21))
ax.legend(facecolor='#1a1a2e', edgecolor='#444466', fontsize=9)
ax.grid()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/09_missing_by_position.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print("Saved: 09_missing_by_position.png")

# ==============================================================================
# 4.11 PLOT 10 — Driver win rate (historical) vs is_winner scatter
# ==============================================================================

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor('#0f0f0f')
non_win = df[df['is_winner'] == 0]
win     = df[df['is_winner'] == 1]
ax.scatter(non_win['driver_win_rate_all'], non_win['constructor_win_rate_20'],
           alpha=0.08, s=8, color=SILVER, label='Non-winner')
ax.scatter(win['driver_win_rate_all'],     win['constructor_win_rate_20'],
           alpha=0.5,  s=20, color=GOLD,  label='Winner')
ax.set_title('Driver historical win rate vs constructor win rate', pad=12)
ax.set_xlabel('Driver all-time win rate')
ax.set_ylabel('Constructor win rate (last 20 races)')
ax.legend(facecolor='#1a1a2e', edgecolor='#444466')
ax.grid()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/10_driver_vs_constructor_winrate.png', dpi=150,
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.show()
print("Saved: 10_driver_vs_constructor_winrate.png")

# ==============================================================================
# 4.12 SUMMARY — Correlation of each feature with is_winner
# ==============================================================================

print("\n" + "=" * 60)
print("4.12 — FEATURE CORRELATION WITH TARGET (is_winner)")
print("=" * 60)

target_corr = df_corr.corr()['is_winner'].drop('is_winner').sort_values(key=abs,
                                                                         ascending=False)
print(target_corr.round(4).to_string())

print(f"\nAll plots saved to: ./{OUTPUT_DIR}/")
print("EDA complete — ready for Step 5 (Project Schema) and Step 6 (Model Building).")