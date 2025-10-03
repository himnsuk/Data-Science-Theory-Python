Absolutely, Himanshu — here’s a production-grade, **end-to-end, step-by-step Python pipeline** for **multiclass classification with mixed numerical & categorical features**. I’ve prioritized:

- **Fast and safe preprocessing** (no leakage, modular, scalable)
- **Handling missing data**
- **Deduplication & correlation-based pruning early**
- **(Optional) Unsupervised clustering to estimate natural classes**
- **Model comparison & hyperparameter tuning**
- **Validation rigor** to ensure a chosen model (e.g., **XGBoost**) “makes the cut”
- **Meaningful metrics** (accuracy, macro-F1, per-class precision/recall, ROC-AUC OvR, log loss)
- **Interpretability & stability checks**
- **Reproducibility** and **artifact saving**

> **Note**: This is written to be copy-paste runnable (with minor path edits). It uses standard Python stack: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `xgboost`. Optional: `imblearn`, `shap`.

---

## 0) Setup & Imports

```python
# If needed, install optional packages in your environment:
# pip install xgboost imbalanced-learn shap mlflow

import os
import numpy as np
import pandas as pd

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix, roc_auc_score,
                             log_loss)
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_class_weight

# Optional (guarded) imports
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMB_AVAILABLE = True
except Exception:
    IMB_AVAILABLE = False

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
```

---

## 1) Load Data & Basic Sanity Checks

Replace `data.csv`, `target_col` with your actuals.

```python
# --- Load ---
df = pd.read_csv("data.csv")  # TODO: point to your dataset
target_col = "target"         # TODO: set your target column name

# --- Separate target ---
assert target_col in df.columns, f"Target column {target_col} not found"
y = df[target_col]
X = df.drop(columns=[target_col])

# --- Identify column types ---
cat_cols = [c for c in X.columns if X[c].dtype == 'object' or str(X[c].dtype).startswith('category')]
num_cols = [c for c in X.columns if c not in cat_cols]

print(f"Shape: {df.shape}, #num: {len(num_cols)}, #cat: {len(cat_cols)}, classes: {y.nunique()}")
print("Class distribution:\n", y.value_counts(normalize=True).round(3))
```

---

## 2) FAST Data Quality: Missingness, Duplicates, High-Cardinality/Constant/ID-like Columns

- **Missing values**: impute later in pipeline (no leakage).
- **Duplicates**: Quick removal upfront (`drop_duplicates`).
- **Constant columns** and **near-constant**: remove.
- **ID-like columns** (unique ratio ~1.0): remove (they add noise/leakage risk).

```python
# --- Drop exact duplicate rows ---
before = len(X)
Xy = pd.concat([X, y], axis=1).drop_duplicates()
after = len(Xy)
print(f"Dropped {before - after} duplicate rows")
X, y = Xy.drop(columns=[target_col]), Xy[target_col]

# --- Drop columns with too many missing values (>95%) ---
missing_ratio = X.isna().mean()
cols_high_na = missing_ratio[missing_ratio > 0.95].index.tolist()
if cols_high_na:
    print("Dropping high-missing columns:", cols_high_na)
    X = X.drop(columns=cols_high_na)
    cat_cols = [c for c in cat_cols if c not in cols_high_na]
    num_cols = [c for c in num_cols if c not in cols_high_na]

# --- Drop constant / near-constant columns ---
nunique = X.nunique(dropna=False)
const_cols = nunique[nunique <= 1].index.tolist()
if const_cols:
    print("Dropping constant columns:", const_cols)
    X = X.drop(columns=const_cols)
    cat_cols = [c for c in cat_cols if c not in const_cols]
    num_cols = [c for c in num_cols if c not in const_cols]

# --- Drop ID-like columns (very high uniqueness) ---
n = len(X)
uniq_ratio = nunique / n
id_like_cols = uniq_ratio[uniq_ratio > 0.98].index.tolist()
# Keep target-like keys (rare) or known keys if needed; otherwise drop:
if id_like_cols:
    print("Dropping ID-like columns:", id_like_cols)
    X = X.drop(columns=id_like_cols)
    cat_cols = [c for c in cat_cols if c not in id_like_cols]
    num_cols = [c for c in num_cols if c not in id_like_cols]
```

---

## 3) Train/Test Split (No Leakage) + Stratification

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
```

---

## 4) Preprocessing Pipelines (Missing Data, Encoding, Scaling)

- **Numeric**: median imputation + standard scaling  
- **Categorical**: most frequent imputation + one-hot (`handle_unknown="ignore"`)

```python
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols)
    ],
    remainder="drop",
    verbose_feature_names_out=False
)
```

---

## 5) Remove Correlated Features **Quickly** (train-only, numeric-only to avoid 1-hot artifacts)

This is **fast** and **leak-safe**: fit on training only, remove at transform.

```python
class CorrelatedFeatureRemover(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: float = 0.9, feature_names: Optional[List[str]] = None):
        self.threshold = threshold
        self.feature_names_ = feature_names
        self.to_keep_idx_ = None

    def fit(self, X, y=None):
        # X is expected to be a dense 2D array after ColumnTransformer
        X = np.asarray(X)
        corr = np.corrcoef(X, rowvar=False)
        n = corr.shape[0]
        to_drop = set()
        for i in range(n):
            if i in to_drop:
                continue
            for j in range(i+1, n):
                if abs(corr[i, j]) > self.threshold and j not in to_drop:
                    # Drop j to keep i
                    to_drop.add(j)
        self.to_keep_idx_ = np.array([i for i in range(n) if i not in to_drop], dtype=int)
        return self

    def transform(self, X):
        X = np.asarray(X)
        return X[:, self.to_keep_idx_]

    def get_feature_names_out(self, input_features=None):
        if self.feature_names_ is None:
            if input_features is not None:
                self.feature_names_ = list(input_features)
            else:
                # fallback to index names
                self.feature_names_ = [f"f_{i}" for i in range(max(self.to_keep_idx_) + 1)]
        return [self.feature_names_[i] for i in self.to_keep_idx_]
```

> **Note**: This removes highly correlated columns after preprocessing, **speeding up** training & improving stability.

---

## 6) Baseline Models: Quick Compare with Cross-Validation

We’ll try:
- **Logistic Regression (multinomial)** — strong baseline
- **Random Forest** — robust
- **XGBoost** — strong tabular default

We’ll use a pipeline: `preprocessor` → `corr_pruner` → `model`.

```python
corr_pruner = CorrelatedFeatureRemover(threshold=0.95)

def make_pipeline(model):
    return Pipeline(steps=[
        ("pre", preprocessor),
        ("corr", corr_pruner),
        ("model", model)
    ])

models = {
    "logreg": LogisticRegression(
        multi_class="multinomial", solver="lbfgs", max_iter=200, n_jobs=-1, random_state=RANDOM_STATE
    ),
    "rf": RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_split=2, class_weight="balanced_subsample",
        n_jobs=-1, random_state=RANDOM_STATE
    )
}

if XGB_AVAILABLE:
    models["xgb"] = XGBClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=RANDOM_STATE
    )

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = {
    "accuracy": "accuracy",
    "f1_macro": "f1_macro",
    "f1_weighted": "f1_weighted",
    "log_loss": "neg_log_loss"
}

cv_results = {}
for name, model in models.items():
    pipe = make_pipeline(model)
    scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)
    cv_results[name] = {k: v for k, v in scores.items()}
    print(f"\nModel: {name}")
    for metric in scoring.keys():
        vals = scores[f"test_{metric}"]
        print(f"  {metric}: {vals.mean():.4f} ± {vals.std():.4f}")
```

> **Pick top 1–2 models** by macro-F1 (robust to imbalance) and log loss.

---

## 7) (Optional) Handling Class Imbalance

Two safe options:

1) **Class weights** (no leakage, minimal tuning):
```python
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight_map = {cls: w for cls, w in zip(classes, class_weights)}
print("Class weights:", class_weight_map)

# Example: logistic regression with class_weight
logreg_balanced = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=200, n_jobs=-1,
                                     class_weight=class_weight_map, random_state=RANDOM_STATE)
```

2) **SMOTE** inside CV (requires imblearn):
```python
if IMB_AVAILABLE:
    smote = SMOTE(random_state=RANDOM_STATE)
    pipe_smote_rf = ImbPipeline(steps=[
        ("pre", preprocessor),
        ("smote", smote),        # applied within CV folds
        ("corr", corr_pruner),
        ("model", RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=RANDOM_STATE))
    ])
    scores_smote = cross_validate(pipe_smote_rf, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    print("SMOTE RF macro-F1:", scores_smote["test_f1_macro"].mean())
```

---

## 8) Hyperparameter Tuning (RandomizedSearchCV on Top Models)

We’ll tune **XGBoost** and **RandomForest**. Avoid grid search; randomized search is faster.

```python
search_spaces = {}

if XGB_AVAILABLE:
    search_spaces["xgb"] = {
        "model__n_estimators": [300, 600, 900],
        "model__learning_rate": [0.02, 0.05, 0.1],
        "model__max_depth": [4, 6, 8, 10],
        "model__min_child_weight": [1, 3, 5],
        "model__subsample": [0.7, 0.85, 1.0],
        "model__colsample_bytree": [0.7, 0.85, 1.0],
        "model__reg_lambda": [1.0, 2.0, 5.0],
    }

search_spaces["rf"] = {
    "model__n_estimators": [300, 500, 800],
    "model__max_depth": [None, 10, 20, 40],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", "log2", None],
}

best_models = {}
for name in list(search_spaces.keys()):
    model = models[name]
    pipe = make_pipeline(model)
    rs = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=search_spaces[name],
        n_iter=25,
        scoring="f1_macro",
        n_jobs=-1,
        cv=cv,
        verbose=1,
        random_state=RANDOM_STATE,
        refit=True
    )
    rs.fit(X_train, y_train)
    best_models[name] = rs.best_estimator_
    print(f"\nTuned {name}: best f1_macro={rs.best_score_:.4f}")
    print("Best params:\n", rs.best_params_)
```

---

## 9) Evaluate on Holdout: Accuracy, F1 (macro/weighted), Precision/Recall per Class, Log Loss, ROC-AUC (OvR)

```python
def evaluate_model(estimator, X_test, y_test, label="model"):
    proba = None
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X_test)
    y_pred = estimator.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    pr, rc, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None, zero_division=0)
    f1_macro_ = f1_score(y_test, y_pred, average="macro")
    f1_weighted_ = f1_score(y_test, y_pred, average="weighted")
    ll = log_loss(y_test, proba) if proba is not None else np.nan

    print(f"\n[{label}] Holdout results")
    print(f"  Accuracy:     {acc:.4f}")
    print(f"  F1 (macro):   {f1_macro_:.4f}")
    print(f"  F1 (weighted):{f1_weighted_:.4f}")
    if proba is not None:
        # For multiclass ROC-AUC OvR
        try:
            auc_ovr = roc_auc_score(y_test, proba, multi_class="ovr")
            print(f"  ROC-AUC (OvR):{auc_ovr:.4f}")
        except Exception:
            pass
        print(f"  Log Loss:     {ll:.4f}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d",
                xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title(f"Confusion Matrix - {label}")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.show()

    return {"accuracy": acc, "f1_macro": f1_macro_, "f1_weighted": f1_weighted_, "log_loss": ll}

# Evaluate tuned models
for name, model in best_models.items():
    _ = evaluate_model(model, X_test, y_test, label=f"{name}_tuned")
```

---

## 10) (Optional) Estimate Natural Number of Classes via Clustering

- Use the **preprocessor** to transform features to numeric space.
- Reduce dimension (PCA) for speed/stability.
- Evaluate **KMeans** with multiple `k` via **silhouette**, **Calinski-Harabasz**, **Davies-Bouldin**.

```python
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Fit preprocessor on full data to observe structure (or on train)
Z = preprocessor.fit_transform(X)
pca = PCA(n_components=min(30, Z.shape[1]), random_state=RANDOM_STATE)
Zp = pca.fit_transform(Z)

ks = list(range(2, min(15, len(np.unique(y)) + 7)))  # heuristic range
sil_scores, ch_scores, db_scores = [], [], []
for k in ks:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
    labels_k = km.fit_predict(Zp)
    sil_scores.append(silhouette_score(Zp, labels_k))
    ch_scores.append(calinski_harabasz_score(Zp, labels_k))
    db_scores.append(davies_bouldin_score(Zp, labels_k))

plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.plot(ks, sil_scores, "-o"); plt.title("Silhouette ↑"); plt.xlabel("k")
plt.subplot(1,3,2); plt.plot(ks, ch_scores, "-o"); plt.title("Calinski-Harabasz ↑"); plt.xlabel("k")
plt.subplot(1,3,3); plt.plot(ks, db_scores, "-o"); plt.title("Davies-Bouldin ↓"); plt.xlabel("k")
plt.tight_layout(); plt.show()

best_k = ks[int(np.argmax(sil_scores))]
print("Suggested k by Silhouette:", best_k)
```

> Use this to **sanity-check** your label cardinality vs. natural clusters. If mismatch is large, re-check labeling or consider relabeling tasks.

---

## 11) Choose Final Model (e.g., **XGBoost**) & Make Sure It “Makes the Cut”

Key checks:
- **Cross-validated performance** (already done)
- **Holdout performance & uncertainty** (bootstrap CI)
- **Calibration** (probabilities)
- **Stability** (small noise/seed changes)
- **Interpretability** (feature importance, PDP)
- **No leakage** (pipelines/splits already prevent most)
- **Speed/Memory** (fit time, predict latency) — ensure meets SLAs

### 11.1 Calibrate Probabilities

```python
from sklearn.calibration import CalibratedClassifierCV

# Start from the tuned XGB if available, else best other model
final_name = "xgb" if ("xgb" in best_models) else list(best_models.keys())[0]
final_base = best_models[final_name]

# Calibrate with 5-fold CV on training set
calibrated_final = CalibratedClassifierCV(
    base_estimator=final_base, method="isotonic", cv=5
)
calibrated_final.fit(X_train, y_train)

_ = evaluate_model(calibrated_final, X_test, y_test, label=f"{final_name}_calibrated")
```

### 11.2 Bootstrap Confidence Intervals on Holdout

```python
def bootstrap_metric_ci(estimator, X_test, y_test, n_boot=200, metric="f1_macro"):
    rng = np.random.default_rng(RANDOM_STATE)
    preds = estimator.predict(X_test)
    proba = estimator.predict_proba(X_test) if hasattr(estimator, "predict_proba") else None
    idx = np.arange(len(y_test))

    vals = []
    for _ in range(n_boot):
        sample = rng.choice(idx, size=len(idx), replace=True)
        yt = y_test.iloc[sample]
        yp = preds[sample]
        if metric == "f1_macro":
            vals.append(f1_score(yt, yp, average="macro"))
        elif metric == "accuracy":
            vals.append(accuracy_score(yt, yp))
        elif metric == "log_loss" and proba is not None:
            vals.append(log_loss(yt, proba[sample]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return np.mean(vals), (lo, hi)

mean_f1, (lo, hi) = bootstrap_metric_ci(calibrated_final, X_test, y_test, n_boot=300, metric="f1_macro")
print(f"Bootstrapped F1-macro mean={mean_f1:.4f}, 95% CI=({lo:.4f},{hi:.4f})")
```

### 11.3 Feature Importance & Permutation Importance

```python
# Permutation importance on holdout (safer than impurity importances)
res = permutation_importance(calibrated_final, X_test, y_test, n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)
imp_means = res.importances_mean
imp_idx = np.argsort(imp_means)[::-1]

# Recover feature names after preprocessor & corr pruning
pre = calibrated_final.base_estimator.named_steps["pre"]
corr = calibrated_final.base_estimator.named_steps["corr"]
feature_names_pre = pre.get_feature_names_out()
try:
    feature_names_final = corr.get_feature_names_out(feature_names_pre)
except Exception:
    # if not implemented, fallback
    feature_names_final = [f"f_{i}" for i in range(len(imp_means))]

top_k = 20
plt.figure(figsize=(6, max(4, top_k * 0.3)))
sns.barplot(x=imp_means[imp_idx][:top_k], y=np.array(feature_names_final)[imp_idx][:top_k], orient="h")
plt.title("Permutation Importance (Top 20)")
plt.tight_layout(); plt.show()
```

> Optionally use **SHAP** for tree models (requires `shap`).

```python
if XGB_AVAILABLE:
    try:
        import shap
        # Extract underlying fitted XGBClassifier from pipeline inside CalibratedClassifierCV
        fitted_pipe = calibrated_final.base_estimator
        xgb_model = fitted_pipe.named_steps["model"]
        X_test_transformed = fitted_pipe.named_steps["corr"].transform(
            fitted_pipe.named_steps["pre"].transform(X_test)
        )
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_test_transformed)
        shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names_final, plot_type="bar")
    except Exception as e:
        print("SHAP not available or failed:", e)
```

---

## 12) (Optional) Per-Class Threshold Optimization (One-vs-Rest)

Multiclass default is `argmax`. If you have asymmetric costs or a “reject” option, you can **tune thresholds** per class:

```python
def optimize_thresholds_ovr(estimator, X_val, y_val):
    proba = estimator.predict_proba(X_val)
    classes = estimator.classes_
    thresholds = np.full(len(classes), 1.0 / len(classes))  # start
    best_thresholds = thresholds.copy()
    best_macro_f1 = -np.inf

    # Simple grid search per class over thresholds
    grid = np.linspace(0.2, 0.9, 15)
    for t_vec in np.array(np.meshgrid(*([grid]*len(classes)))).T.reshape(-1, len(classes)):
        y_pred_custom = np.argmax(proba - (1 - t_vec), axis=1)  # lift probs above threshold
        f1m = f1_score(y_val, classes[y_pred_custom], average="macro")
        if f1m > best_macro_f1:
            best_macro_f1 = f1m
            best_thresholds = t_vec
    return best_thresholds

# Example usage: split train into train/val to find thresholds, then apply on test.
```

---

## 13) Save Artifacts for Reuse

```python
import joblib

os.makedirs("artifacts", exist_ok=True)
joblib.dump(calibrated_final, "artifacts/final_model.joblib")

# Save columns info to make inference robust
meta = {
    "num_cols": num_cols,
    "cat_cols": cat_cols,
    "target_classes": list(np.unique(y_train))
}
joblib.dump(meta, "artifacts/metadata.joblib")

# Inference wrapper
def predict_proba_from_csv(model_path, metadata_path, csv_path):
    model = joblib.load(model_path)
    meta = joblib.load(metadata_path)
    df_new = pd.read_csv(csv_path)
    X_new = df_new[meta["num_cols"] + meta["cat_cols"]]
    return model.predict_proba(X_new)

# Example:
# proba = predict_proba_from_csv("artifacts/final_model.joblib", "artifacts/metadata.joblib", "new_data.csv")
```

---

## 14) What I’d Show as **Resulting KPIs**

For multiclass, I’d report:

- **Overall**: Accuracy, **F1 (macro)**, F1 (weighted), **Log Loss**, **ROC-AUC (OvR)**
- **Per class**: Precision, Recall, F1, **Support**
- **Confusion matrix**
- **Calibration curve** (if probabilities matter)
- **CI** (95%) for key metrics (bootstrap)
- **Feature importance** (Permutation importance or SHAP for trees)

These are already produced in sections 9, 11.1–11.3.

---

## 15) Quick Summary of the Approach (Answers to Your Points)

1. **Dealing with missing data**  
   - Numeric: median imputation  
   - Categorical: `most_frequent` + `OneHotEncoder(handle_unknown='ignore')`  
   - Done **inside pipeline** → no leakage.

2. **How quickly remove duplicates & correlated items and create model**  
   - Immediate `drop_duplicates`  
   - Drop **high-missing**, **constant**, **ID-like** columns  
   - Add **CorrelatedFeatureRemover** after preprocessing to drop highly correlated features (e.g., 0.95 threshold) — **fast**, removes redundant columns before model fit.

3. **Creating clusters & figuring how many classes exist**  
   - Preprocess → PCA → **KMeans** across `k` with **Silhouette**, **Calinski-Harabasz**, **Davies-Bouldin** to pick a natural `k`  
   - Useful to **audit** whether label cardinality matches latent structure.

4. **Selecting multiple algorithms & fine-tuning**  
   - Compare **LogReg**, **RandomForest**, **XGBoost** with CV on the same pipeline.  
   - Tune top models via **RandomizedSearchCV** (macro-F1).

5. **Ensuring XGBoost “makes the cut”**  
   - **Cross-validated performance + holdout**  
   - **Calibration** with CalibratedClassifierCV  
   - **Bootstrapped CIs** for macro-F1  
   - **Permutation importance / SHAP** for interpretability  
   - Check **confusion matrix**, per-class metrics, and (optional) **threshold tuning** if costs dictate.

6. **Result parameters to show**  
   - **Accuracy**, **F1-macro**, **F1-weighted**, **Precision/Recall** per class  
   - **Log Loss**, **ROC-AUC (OvR)**  
   - **Confusion Matrix**  
   - (Optionally) **Calibration**, **Feature importances**, **CIs**.

---

## 16) Extras You Can Add (if time allows)

- **Mutual Information** for initial feature ranking:
  ```python
  mi = mutual_info_classif(
      preprocessor.fit_transform(X_train), y_train, discrete_features=False, random_state=RANDOM_STATE
  )
  ```
- **Drift checks** (KS for numeric, chi-square for categorical) between train/test.
- **MLflow** to track experiments, params, metrics, and artifacts.
- **Model Cards** for documentation.

---

## Final Notes

- Everything is **wrapped in pipelines** to avoid leakage and ensure consistent transformations between train and inference.  
- The **correlation pruning** sits after preprocessing to operate on the final numeric space (fast and effective).  
- **Clustering** is optional but helpful when you’re unsure about the true number of classes.  
- For **high-cardinality categoricals**, consider **target encoding with CV** (via `category_encoders`) — but handle carefully to avoid leakage.

---

If you share a sample schema or the actual dataset (even a small anonymized slice), I can tailor the thresholds, hyperparameter spaces, and add specific diagnostics (e.g., drift charts, classwise PR curves).
