import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
 
import kagglehub
 
# ==============================================================================
# STEP 1 — DATA SELECTION
# Download the dataset from Kaggle
# ==============================================================================
 
path = kagglehub.dataset_download("rohanrao/formula-1-world-championship-1950-2020")
print("Path to dataset files:", path)
 
# Load the six selected CSV files
qual         = pd.read_csv(os.path.join(path, 'qualifying.csv'))
results      = pd.read_csv(os.path.join(path, 'results.csv'))
races        = pd.read_csv(os.path.join(path, 'races.csv'))
circuits     = pd.read_csv(os.path.join(path, 'circuits.csv'))
constructors = pd.read_csv(os.path.join(path, 'constructors.csv'))
drivers      = pd.read_csv(os.path.join(path, 'drivers.csv'))
 
print("\n--- Raw shapes ---")
for name, frame in [('qualifying', qual), ('results', results), ('races', races),
                    ('circuits', circuits), ('constructors', constructors), ('drivers', drivers)]:
    print(f"  {name:20s}: {frame.shape}")
 
 
# ==============================================================================
# STEP 2 — DATA UNDERSTANDING
# Quick overview of each file's structure
# ==============================================================================
 
print("\n--- Column overview ---")
for name, frame in [('qualifying', qual), ('results', results), ('races', races),
                    ('circuits', circuits), ('constructors', constructors), ('drivers', drivers)]:
    print(f"\n{name}: {list(frame.columns)}")
 
 
# ==============================================================================
# STEP 3 — DATA PREPARATION
# ==============================================================================
 
# ------------------------------------------------------------------------------
# 3.1  Merge all tables into a single dataframe
# ------------------------------------------------------------------------------
 
# Keep only the columns we need from results to avoid leakage from the start
results_cols = ['raceId', 'driverId', 'constructorId', 'grid', 'positionOrder', 'statusId']
 
df = qual.merge(results[results_cols], on=['raceId', 'driverId', 'constructorId'])
df = df.merge(races[['raceId', 'year', 'round', 'circuitId', 'date']], on='raceId')
df = df.merge(circuits[['circuitId', 'name', 'country', 'alt']],       on='circuitId')
df = df.merge(constructors[['constructorId', 'name']],                  on='constructorId',
              suffixes=('_circuit', '_constructor'))
df = df.merge(drivers[['driverId', 'code']],                            on='driverId')
 
print(f"\nMerged shape: {df.shape}")
 
# ------------------------------------------------------------------------------
# 3.2  Filter to modern era (1983+)
# ------------------------------------------------------------------------------
 
df = df[df['year'] >= 1983].reset_index(drop=True)
print(f"After year filter (>=1983): {df.shape}")
 
# ------------------------------------------------------------------------------
# 3.3  Create target variable, then drop post-race columns
# ------------------------------------------------------------------------------
 
# Target: 1 if the driver won (P1), 0 otherwise
df['is_winner'] = (df['positionOrder'] == 1).astype(int)
 
# Drop leaky columns — only known after the race
df.drop(columns=['positionOrder', 'statusId'], inplace=True)
 
print(f"\nClass distribution:\n{df['is_winner'].value_counts(normalize=True).round(3)}")
 
# ------------------------------------------------------------------------------
# 3.4  Parse qualifying times and handle missing values
# ------------------------------------------------------------------------------
 
def parse_laptime(t):
    """Convert lap time string '1:20.456' to milliseconds. Returns NaN if invalid."""
    try:
        if pd.isna(t) or str(t).strip() in ['\\N', '', 'nan']:
            return np.nan
        s = str(t).strip()
        if ':' in s:
            mins, secs = s.split(':')
            return float(mins) * 60000 + float(secs) * 1000
        return float(s) * 1000
    except Exception:
        return np.nan
 
for col in ['q1', 'q2', 'q3']:
    df[col] = df[col].apply(parse_laptime)
    # Binary flag: was this time missing? (informative absence)
    df[f'{col}_missing'] = df[col].isna().astype(int)
 
# Impute missing lap times with a large penalty value
# Rationale: absence means the driver was eliminated — penalizing is better than mean imputation
PENALTY_MS = 999_999
df[['q1', 'q2', 'q3']] = df[['q1', 'q2', 'q3']].fillna(PENALTY_MS)
 
# Circuit altitude: rare missing values filled with median
df['alt'] = pd.to_numeric(df['alt'], errors='coerce')
df['alt'] = df['alt'].fillna(df['alt'].median())
 
# Qualifying position: rare \N values
df['position'] = pd.to_numeric(df['position'], errors='coerce')
df['position'] = df['position'].fillna(df['position'].median())
 
# Grid position
df['grid'] = pd.to_numeric(df['grid'], errors='coerce')
df['grid'] = df['grid'].fillna(df['grid'].median())
 
print(f"\nMissing values after imputation:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
 
# ------------------------------------------------------------------------------
# 3.5  Feature engineering
# ------------------------------------------------------------------------------
 
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)
 
# --- Binary position flags ---
df['is_pole']       = (df['grid'] == 1).astype(int)
df['is_front_row']  = (df['grid'] <= 2).astype(int)
df['is_top5_grid']  = (df['grid'] <= 5).astype(int)
 
# --- Qualifying gap to session pole time ---
# How many milliseconds slower was this driver vs the fastest Q1 in that race?
valid_q1 = df['q1'].replace(PENALTY_MS, np.nan)
session_best = valid_q1.groupby(df['raceId']).transform('min')
df['q1_gap_to_pole'] = (valid_q1 - session_best).fillna(PENALTY_MS)
 
# --- Rolling driver win rate (last 10 races) ---
# shift(1) ensures we only use past races — never the current one
df = df.sort_values(['driverId', 'date']).reset_index(drop=True)
df['driver_win_rate_10'] = (
    df.groupby('driverId')['is_winner']
      .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
).fillna(0)
 
# --- Driver historical win rate (all time, up to but not including current race) ---
df['driver_win_rate_all'] = (
    df.groupby('driverId')['is_winner']
      .transform(lambda x: x.shift(1).expanding().mean())
).fillna(0)
 
# --- Constructor (team) win rate over last 20 races ---
df = df.sort_values(['constructorId', 'date']).reset_index(drop=True)
df['constructor_win_rate_20'] = (
    df.groupby('constructorId')['is_winner']
      .transform(lambda x: x.shift(1).rolling(20, min_periods=1).mean())
).fillna(0)
 
# --- Driver win rate on this specific circuit ---
df = df.sort_values(['driverId', 'circuitId', 'date']).reset_index(drop=True)
df['driver_circuit_win_rate'] = (
    df.groupby(['driverId', 'circuitId'])['is_winner']
      .transform(lambda x: x.shift(1).expanding().mean())
).fillna(0)
 
# Re-sort by date for clean downstream use
df = df.sort_values('date').reset_index(drop=True)
 
print(f"\nEngineered features added. New shape: {df.shape}")
 
# ------------------------------------------------------------------------------
# 3.6  Encode categorical variables
# ------------------------------------------------------------------------------
 
cat_cols = ['name_constructor', 'name_circuit', 'code', 'country']
label_encoders = {}
 
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le  # store for inverse transform if needed later
 
# ------------------------------------------------------------------------------
# 3.7  Drop identifier and date columns (not model inputs)
# ------------------------------------------------------------------------------
 
drop_cols = ['raceId', 'driverId', 'constructorId', 'circuitId', 'qualifyId', 'date', 'number']
drop_cols = [c for c in drop_cols if c in df.columns]
df.drop(columns=drop_cols, inplace=True)
 
# Final fill for any residual NaN in engineered features
df.fillna(0, inplace=True)
 
print(f"\nFinal columns ({len(df.columns)}):\n{list(df.columns)}")
print(f"\nFinal dataset shape: {df.shape}")
 
# ------------------------------------------------------------------------------
# 3.8  Train / test split (by year — NOT random)
# Rationale: simulates real forecasting — model never sees future seasons during training
# ------------------------------------------------------------------------------
 
TRAIN_CUTOFF = 2021
 
train_df = df[df['year'] <= TRAIN_CUTOFF].copy()
test_df  = df[df['year'] >  TRAIN_CUTOFF].copy()
 
X_train = train_df.drop(columns=['is_winner', 'year'])
y_train = train_df['is_winner']
X_test  = test_df.drop(columns=['is_winner', 'year'])
y_test  = test_df['is_winner']
 
print(f"\nTrain: {X_train.shape} | Test: {X_test.shape}")
print(f"Train class balance: {y_train.value_counts(normalize=True).round(3).to_dict()}")
print(f"Test  class balance: {y_test.value_counts(normalize=True).round(3).to_dict()}")
 
# ------------------------------------------------------------------------------
# 3.9  Handle class imbalance with SMOTE (training set only)
# Rationale: applying SMOTE to test set would misrepresent real-world distribution
# ------------------------------------------------------------------------------
 
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
 
print(f"\nAfter SMOTE — X_train: {X_train_res.shape}")
print(f"After SMOTE — class balance: {pd.Series(y_train_res).value_counts(normalize=True).round(3).to_dict()}")
 
print("\n--- Data preparation complete. Ready for EDA and model building. ---")
 
# ==============================================================================
# SAVE ALL OUTPUTS TO DISK
# Everything downstream (EDA, model) loads from these files — no shared session needed
# ==============================================================================
 
SAVE_DIR = 'f1_pipeline_outputs'
os.makedirs(SAVE_DIR, exist_ok=True)
 
# 1. Full prepared dataframe (used by EDA for descriptive stats and plots)
df.to_parquet(os.path.join(SAVE_DIR, 'df_prepared.parquet'), index=False)
 
# 2. Train / test splits — raw (before SMOTE), used by EDA and baseline models
X_train.to_parquet(os.path.join(SAVE_DIR, 'X_train.parquet'), index=False)
X_test.to_parquet(os.path.join(SAVE_DIR,  'X_test.parquet'),  index=False)
y_train.to_frame().to_parquet(os.path.join(SAVE_DIR, 'y_train.parquet'), index=False)
y_test.to_frame().to_parquet(os.path.join(SAVE_DIR,  'y_test.parquet'),  index=False)
 
# 3. SMOTE-resampled training set — used by model training
X_train_res_df = pd.DataFrame(X_train_res, columns=X_train.columns)
y_train_res_df = pd.Series(y_train_res, name='is_winner')
X_train_res_df.to_parquet(os.path.join(SAVE_DIR, 'X_train_smote.parquet'), index=False)
y_train_res_df.to_frame().to_parquet(os.path.join(SAVE_DIR, 'y_train_smote.parquet'), index=False)
 
# 4. Label encoders — needed to decode predictions back to team/circuit names
with open(os.path.join(SAVE_DIR, 'label_encoders.pkl'), 'wb') as f:
    pickle.dump(label_encoders, f)
 
print(f"\nSaved to ./{SAVE_DIR}/:")
for fname in sorted(os.listdir(SAVE_DIR)):
    size = os.path.getsize(os.path.join(SAVE_DIR, fname))
    print(f"  {fname:35s} {size/1024:.1f} KB")