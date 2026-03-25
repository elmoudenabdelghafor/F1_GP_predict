import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score,
    precision_score, recall_score
)
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

# ==============================================================================
# LOAD — everything saved by f1_data_pipeline.py
# ==============================================================================

SAVE_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'f1_pipeline_outputs')
SAVE_PATH = os.path.join(SAVE_DIR, 'f1_pipeline.joblib')

if not os.path.exists(SAVE_PATH):
    raise FileNotFoundError(
        f"'{SAVE_PATH}' not found. Run f1_data_pipeline.py first."
    )

data = joblib.load(SAVE_PATH)

X_train        = data['X_train']
y_train        = data['y_train']
X_test         = data['X_test']
y_test         = data['y_test']
X_train_smote  = data['X_train_smote']
y_train_smote  = data['y_train_smote']
label_encoders = data['label_encoders']

print("Loaded successfully.")
print(f"  X_train (raw)    : {X_train.shape}")
print(f"  X_train (SMOTE)  : {X_train_smote.shape}")
print(f"  X_test           : {X_test.shape}")

# ==============================================================================
# PLOT STYLE
# ==============================================================================

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

RED   = '#e8003d'
GOLD  = '#ffd700'
TEAL  = '#00c9a7'
BLUE  = '#3498db'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'model_plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# HELPER — print and plot evaluation
# ==============================================================================

def evaluate(name, model, X_te, y_te, color, results_store):
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    auc   = roc_auc_score(y_te, y_prob)
    f1    = f1_score(y_te, y_pred)
    prec  = precision_score(y_te, y_pred, zero_division=0)
    rec   = recall_score(y_te, y_pred)
    cm    = confusion_matrix(y_te, y_pred)
    fpr, tpr, _ = roc_curve(y_te, y_prob)

    results_store[name] = {
        'model': model, 'y_pred': y_pred, 'y_prob': y_prob,
        'auc': auc, 'f1': f1, 'precision': prec, 'recall': rec,
        'cm': cm, 'fpr': fpr, 'tpr': tpr,
    }

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"  F1-score  : {f1:.4f}  (winner class)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"\n{classification_report(y_te, y_pred, target_names=['Non-winner','Winner'])}")

    # ---------- Confusion matrix plot (enhanced) ----------
    class_labels = ['Non-winner', 'Winner']
    cm_total = cm.sum()
    cm_pct = cm / cm_total * 100
    accuracy = np.trace(cm) / cm_total * 100

    # Build annotation strings: count + percentage
    annot_text = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot_text[i, j] = f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)"

    # Custom dark colormap matching the model colour
    cmap_cm = mcolors.LinearSegmentedColormap.from_list(
        'custom', ['#1a1a2e', color], N=256
    )

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    fig.patch.set_facecolor('#0f0f0f')

    sns.heatmap(
        cm, annot=annot_text, fmt='', cmap=cmap_cm,
        xticklabels=class_labels, yticklabels=class_labels,
        linewidths=2, linecolor='#0f0f0f',
        annot_kws={'size': 14, 'weight': 'bold', 'color': '#ffffff'},
        cbar=False, ax=ax,
    )

    ax.set_xlabel('Predicted Label', fontsize=12, labelpad=10)
    ax.set_ylabel('True Label', fontsize=12, labelpad=10)
    ax.set_title(f'Confusion Matrix — {name}', fontsize=14, pad=14, weight='bold')
    ax.tick_params(axis='both', labelsize=11)

    # Accuracy footer
    fig.text(
        0.5, 0.01, f'Overall Accuracy: {accuracy:.1f}%',
        ha='center', fontsize=11, style='italic',
        color=color, weight='bold',
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    safe = name.replace(' ', '_').lower()
    plt.savefig(f'{OUTPUT_DIR}/cm_{safe}.png', dpi=180,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.show()
    print(f"Saved: cm_{safe}.png")

    return results_store


# ==============================================================================
# MODEL 1 — RANDOM FOREST
# ==============================================================================

print("\n" + "="*55)
print("  TRAINING — Random Forest")
print("="*55)

# --- Hyperparameter search ---
rf_param_grid = {
    'n_estimators':      [100, 200, 300],
    'max_depth':         [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf':  [1, 2, 4],
    'max_features':      ['sqrt', 'log2'],
}

rf_base = RandomForestClassifier(
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_search = RandomizedSearchCV(
    rf_base,
    rf_param_grid,
    n_iter=20,
    scoring='f1',          # optimise for winner F1
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

rf_search.fit(X_train_smote, y_train_smote)

print(f"\nBest RF params : {rf_search.best_params_}")
print(f"Best CV F1     : {rf_search.best_score_:.4f}")

best_rf = rf_search.best_estimator_

# ==============================================================================
# MODEL 2 — XGBOOST
# ==============================================================================

print("\n" + "="*55)
print("  TRAINING — XGBoost")
print("="*55)

# class imbalance ratio for scale_pos_weight
neg   = int((y_train == 0).sum())
pos   = int((y_train == 1).sum())
ratio = neg / pos
print(f"scale_pos_weight = {ratio:.1f}  (neg/pos ratio in raw train set)")

xgb_param_grid = {
    'n_estimators':    [100, 200, 300, 400],
    'max_depth':       [3, 4, 5, 6],
    'learning_rate':   [0.01, 0.05, 0.1, 0.2],
    'subsample':       [0.6, 0.8, 1.0],
    'colsample_bytree':[0.6, 0.8, 1.0],
    'min_child_weight':[1, 3, 5],
}

xgb_base = XGBClassifier(
    scale_pos_weight=ratio,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)

xgb_search = RandomizedSearchCV(
    xgb_base,
    xgb_param_grid,
    n_iter=20,
    scoring='f1',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# XGBoost trained on raw (non-SMOTE) train — scale_pos_weight handles imbalance
xgb_search.fit(X_train, y_train)

print(f"\nBest XGB params : {xgb_search.best_params_}")
print(f"Best CV F1      : {xgb_search.best_score_:.4f}")

best_xgb = xgb_search.best_estimator_

# ==============================================================================
# EVALUATION — both models on test set
# ==============================================================================

results = {}
results = evaluate('Random Forest', best_rf,  X_test, y_test, GOLD,  results)
results = evaluate('XGBoost',       best_xgb, X_test, y_test, TEAL,  results)

# ==============================================================================
# PLOT — ROC curves (both models on same axes)
# ==============================================================================

fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor('#0f0f0f')
ax.plot([0, 1], [0, 1], color='#555', linestyle='--', linewidth=1, label='Random baseline')

for name, color in [('Random Forest', GOLD), ('XGBoost', TEAL)]:
    r = results[name]
    ax.plot(r['fpr'], r['tpr'], color=color, linewidth=2,
            label=f"{name}  (AUC = {r['auc']:.3f})")

ax.set_title('ROC curves — Random Forest vs XGBoost', pad=12)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(facecolor='#1a1a2e', edgecolor='#444466', fontsize=10)
ax.grid()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/roc_curves.png', dpi=150,
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.show()
print("Saved: roc_curves.png")

# ==============================================================================
# PLOT — Metric comparison bar chart
# ==============================================================================

metrics      = ['f1', 'precision', 'recall', 'auc']
metric_labels = ['F1-score', 'Precision', 'Recall', 'ROC-AUC']
x = np.arange(len(metrics))
width = 0.35

rf_vals  = [results['Random Forest'][m] for m in metrics]
xgb_vals = [results['XGBoost'][m]       for m in metrics]

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor('#0f0f0f')
bars1 = ax.bar(x - width/2, rf_vals,  width, label='Random Forest', color=GOLD,  alpha=0.85)
bars2 = ax.bar(x + width/2, xgb_vals, width, label='XGBoost',       color=TEAL, alpha=0.85)

for bar in list(bars1) + list(bars2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

ax.set_title('Model comparison — evaluation metrics on test set', pad=12)
ax.set_xticks(x)
ax.set_xticklabels(metric_labels)
ax.set_ylim(0, 1.1)
ax.set_ylabel('Score')
ax.legend(facecolor='#1a1a2e', edgecolor='#444466')
ax.grid(axis='y')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/metric_comparison.png', dpi=150,
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.show()
print("Saved: metric_comparison.png")

# ==============================================================================
# PLOT — Feature importance (both models)
# ==============================================================================

feature_names = X_train.columns.tolist()
top_n = 15

# Random Forest importances
rf_imp = pd.Series(best_rf.feature_importances_, index=feature_names)\
           .sort_values(ascending=False).head(top_n)

# XGBoost importances
xgb_imp = pd.Series(best_xgb.feature_importances_, index=feature_names)\
            .sort_values(ascending=False).head(top_n)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('#0f0f0f')

for ax, imp, title, color in zip(
    axes,
    [rf_imp, xgb_imp],
    ['Random Forest — feature importance', 'XGBoost — feature importance'],
    [GOLD, TEAL]
):
    ax.barh(imp.index[::-1], imp.values[::-1], color=color, alpha=0.85)
    ax.set_title(title, pad=10)
    ax.set_xlabel('Importance score')
    ax.grid(axis='x')
    for i, v in enumerate(imp.values[::-1]):
        ax.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/feature_importance.png', dpi=150,
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.show()
print("Saved: feature_importance.png")

# ==============================================================================
# SAVE MODELS
# ==============================================================================

models_data = {
    'random_forest': best_rf,
    'xgboost':       best_xgb,
    'results':       {k: {m: v for m, v in r.items() if m != 'model'}
                      for k, r in results.items()},
    'feature_names': feature_names,
}
models_path = os.path.join(SAVE_DIR, 'f1_models.joblib')
joblib.dump(models_data, models_path, compress=3)
print(f"\nModels saved to: {models_path}")

# ==============================================================================
# FINAL SUMMARY TABLE
# ==============================================================================

print("\n" + "="*55)
print("  FINAL COMPARISON SUMMARY")
print("="*55)
summary = pd.DataFrame({
    'Metric':         metric_labels,
    'Random Forest':  [f'{v:.4f}' for v in rf_vals],
    'XGBoost':        [f'{v:.4f}' for v in xgb_vals],
})
print(summary.to_string(index=False))

winner_model = 'XGBoost' if results['XGBoost']['f1'] >= results['Random Forest']['f1'] \
               else 'Random Forest'
print(f"\nBest model by F1-score: {winner_model}")

# ---------- Export summary table as styled plot ----------

# Build the table data (header + metric rows + best-model footer)
col_labels = ['Metric', 'Random Forest', 'XGBoost']
cell_data  = [[m, rf, xg] for m, rf, xg in
              zip(metric_labels,
                  [f'{v:.4f}' for v in rf_vals],
                  [f'{v:.4f}' for v in xgb_vals])]
cell_data.append(['Best Model (F1)', winner_model, ''])

fig, ax = plt.subplots(figsize=(8, 3.5))
fig.patch.set_facecolor('#0f0f0f')
ax.set_facecolor('#0f0f0f')
ax.axis('off')
ax.set_title('Final Comparison Summary',
             fontsize=16, weight='bold', color='#ffffff', pad=18)

table = ax.table(
    cellText=cell_data,
    colLabels=col_labels,
    cellLoc='center',
    loc='center',
)
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 1.6)

# Determine which column index is the winner (1 = RF, 2 = XGB)
winner_col = 1 if winner_model == 'Random Forest' else 2
winner_color = GOLD if winner_model == 'Random Forest' else TEAL

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor('#444466')
    cell.set_linewidth(0.8)

    if row == 0:  # header row
        cell.set_facecolor(RED)
        cell.set_text_props(color='#ffffff', weight='bold', fontsize=12)
        cell.set_height(0.12)
    elif row == len(cell_data):  # best-model footer
        cell.set_facecolor('#2a2a4a')
        cell.set_text_props(color=winner_color, weight='bold', fontsize=12)
    else:  # data rows — alternating colours
        bg = '#1a1a2e' if row % 2 == 1 else '#222244'
        cell.set_facecolor(bg)
        cell.set_text_props(color='#ccccdd', fontsize=11)

        # Highlight the winning model's column cells
        if col == winner_col:
            cell.set_text_props(color=winner_color, weight='bold', fontsize=11)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/summary_table.png', dpi=180,
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.show()
print("Saved: summary_table.png")

print("\n--- Model building complete. Ready for Step 7 (Discussion). ---")

print(f"\nAll outputs saved to:")
print(f"  Models  : {models_path}")
for fname in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, fname)
    size  = os.path.getsize(fpath) / 1024
    print(f"  Plot    : {fpath}  ({size:.0f} KB)")