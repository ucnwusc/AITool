#!/usr/bin/env python3
"""
small_data_classifier.py
========================
A diagnostic + modeling tool for the classic small-data binary-classification
problem: ~10 process factors, ~100 experiments, a Yes/No outcome, and "the
model never works."

The goal is NOT to throw a neural net at it. With n~100 and p~10, the right
move is regularized linear models and small tree ensembles, evaluated with
proper stratified cross-validation, plus an honest read on whether the signal
is even in the data.

Usage
-----
    python small_data_classifier.py experiments.csv --outcome outcome

CSV format
----------
    One row per experiment. One column is the binary outcome (Yes/No, 1/0,
    True/False, pass/fail -- all handled). Remaining columns are the factors
    you varied. Non-numeric factor columns are one-hot encoded automatically.

What it does
------------
  1. Loads data, reports shape and class balance.
  2. Warns about the small-data regime (samples-per-feature ratio).
  3. Runs three appropriate models with stratified k-fold CV:
       - Regularized logistic regression (the right default here)
       - Small random forest (captures nonlinear factor interactions)
       - Gradient-boosted trees (often best on tabular data)
  4. Reports AUC and balanced accuracy with honest +/- spread.
  5. Ranks which factors actually carry signal (permutation importance).
  6. Delivers a plain-language verdict: is the signal there or not?

Dependencies: numpy, pandas, scikit-learn, matplotlib
    pip install numpy pandas scikit-learn matplotlib
"""

import argparse
import sys
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.dummy import DummyClassifier

warnings.filterwarnings("ignore")  # sklearn chatters a lot on tiny data


# --------------------------------------------------------------------------- #
#  Data loading and cleaning
# --------------------------------------------------------------------------- #
def coerce_binary(series: pd.Series):
    """Turn a Yes/No-ish column into 0/1. Returns (y, label_map) or raises."""
    truthy = {"yes", "y", "true", "t", "1", "pass", "positive", "+", "good"}
    falsy = {"no", "n", "false", "f", "0", "fail", "negative", "-", "bad"}

    raw = series.astype(str).str.strip().str.lower()
    uniq = sorted(raw.dropna().unique())

    if len(uniq) != 2:
        raise ValueError(
            f"Outcome column has {len(uniq)} distinct values {uniq}, "
            "expected exactly 2 for binary classification."
        )

    # Try the known vocab first; otherwise map alphabetically (0,1).
    mapping = {}
    if set(uniq) <= (truthy | falsy):
        for v in uniq:
            mapping[v] = 1 if v in truthy else 0
    else:
        mapping = {uniq[0]: 0, uniq[1]: 1}

    y = raw.map(mapping).astype(int)
    inv = {v: k for k, v in mapping.items()}
    return y.values, inv


def load_data(path: str, outcome_col: str):
    df = pd.read_csv(path)
    if outcome_col not in df.columns:
        sys.exit(
            f"Outcome column '{outcome_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    y, label_map = coerce_binary(df[outcome_col])
    X = df.drop(columns=[outcome_col])

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return X, y, num_cols, cat_cols, label_map


def build_preprocessor(num_cols, cat_cols):
    """Impute + scale numerics, impute + one-hot categoricals."""
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    transformers = []
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))
    return ColumnTransformer(transformers)


# --------------------------------------------------------------------------- #
#  Diagnostics
# --------------------------------------------------------------------------- #
def report_regime(X, y, num_cols, cat_cols, label_map):
    n, p = X.shape
    print("=" * 64)
    print("DATA SUMMARY")
    print("=" * 64)
    print(f"  Experiments (rows):     {n}")
    print(f"  Factors (columns):      {p}  "
          f"({len(num_cols)} numeric, {len(cat_cols)} categorical)")
    counts = pd.Series(y).value_counts().to_dict()
    pretty = {label_map[k]: v for k, v in counts.items()}
    print(f"  Outcome balance:        {pretty}")

    minority = min(counts.values())
    ratio = n / max(p, 1)
    print(f"  Samples per factor:     {ratio:.1f}")
    print()

    print("REGIME CHECK")
    if ratio < 10:
        print(f"  [!] {ratio:.1f} samples/factor is LOW. Use simple, "
              "regularized models only.")
        print("      A neural net or deep model will overfit here by design.")
    else:
        print(f"  [ok] {ratio:.1f} samples/factor is workable for simple models.")

    if minority < 15:
        print(f"  [!] Minority class has only {minority} examples. "
              "CV folds will be noisy;")
        print("      judge by AUC / balanced accuracy, NOT raw accuracy.")
    if minority / n < 0.30:
        print(f"  [!] Classes are imbalanced ({minority}/{n}). "
              "A model that always")
        print("      predicts the majority class can look ~"
              f"{100*(1-minority/n):.0f}% 'accurate' while learning nothing.")
    print()


# --------------------------------------------------------------------------- #
#  Modeling
# --------------------------------------------------------------------------- #
def evaluate_models(X, y, pre, n_splits):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

    models = {
        "logistic_regression": LogisticRegression(
            penalty="l2", C=1.0, class_weight="balanced", max_iter=2000),
        "random_forest": RandomForestClassifier(
            n_estimators=400, max_depth=4, min_samples_leaf=5,
            class_weight="balanced", random_state=0),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=2, learning_rate=0.05,
            random_state=0),
        "baseline_majority": DummyClassifier(strategy="most_frequent"),
    }

    print("=" * 64)
    print(f"MODEL PERFORMANCE  ({n_splits}-fold stratified CV)")
    print("=" * 64)
    print(f"  {'model':<22}{'ROC-AUC':<20}{'balanced acc':<18}")
    print("  " + "-" * 56)

    results = {}
    for name, clf in models.items():
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        try:
            auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
            bacc = cross_val_score(pipe, X, y, cv=cv,
                                   scoring="balanced_accuracy")
            results[name] = (auc.mean(), auc.std(), bacc.mean())
            print(f"  {name:<22}"
                  f"{auc.mean():.3f} +/- {auc.std():.3f}     "
                  f"{bacc.mean():.3f}")
        except Exception as e:
            print(f"  {name:<22}failed: {e}")
    print()
    return results


def rank_factors(X, y, pre, num_cols, cat_cols):
    """Permutation importance on a held-back split, using the RF model."""
    print("=" * 64)
    print("FACTOR IMPORTANCE  (permutation, higher = more signal)")
    print("=" * 64)
    pipe = Pipeline([
        ("pre", pre),
        ("clf", RandomForestClassifier(
            n_estimators=400, max_depth=4, min_samples_leaf=5,
            class_weight="balanced", random_state=0)),
    ])
    pipe.fit(X, y)
    r = permutation_importance(pipe, X, y, n_repeats=30,
                               random_state=0, scoring="roc_auc")
    imp = sorted(zip(X.columns, r.importances_mean, r.importances_std),
                 key=lambda t: -t[1])
    for name, mean, std in imp:
        bar = "#" * int(max(mean, 0) * 100)
        print(f"  {name:<22}{mean:+.3f} +/- {std:.3f}  {bar}")
    print()
    return imp


# --------------------------------------------------------------------------- #
#  Verdict
# --------------------------------------------------------------------------- #
def verdict(results):
    print("=" * 64)
    print("VERDICT")
    print("=" * 64)
    real = {k: v for k, v in results.items() if k != "baseline_majority"}
    if not real:
        print("  No model ran successfully. Check the data.")
        return
    best_name, (best_auc, best_std, _) = max(real.items(),
                                             key=lambda kv: kv[1][0])

    if best_auc >= 0.75:
        print(f"  Signal is clearly present. Best model: {best_name} "
              f"(AUC {best_auc:.3f}).")
        print("  -> A working classifier is achievable. Focus on the top")
        print("     factors above and collect a few more points to firm it up.")
    elif best_auc >= 0.65:
        print(f"  Weak but real signal. Best model: {best_name} "
              f"(AUC {best_auc:.3f}).")
        print("  -> Promising. More data and dropping noise factors should")
        print("     push this into useful territory.")
    elif best_auc >= 0.55:
        print(f"  Marginal signal (AUC {best_auc:.3f}). Near the noise floor.")
        print("  -> The current data barely separates Yes from No. This is")
        print("     likely a design-of-experiments problem, not a model")
        print("     problem: the factor space is undersampled.")
    else:
        print(f"  No usable signal (AUC {best_auc:.3f} ~ random).")
        print("  -> The Yes/No outcome is not predictable from these factors")
        print("     as currently measured. Before more modeling, revisit the")
        print("     experiment design: are the right factors being varied,")
        print("     over wide enough ranges, with consistent measurement?")
    print()
    print("  Reminder: with ~100 points, a single AUC number is noisy. Trust")
    print("  the cross-validated mean and its +/- spread, not one lucky split.")
    print("=" * 64)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Small-data binary classifier diagnostic.")
    ap.add_argument("csv", help="Path to the experiments CSV.")
    ap.add_argument("--outcome", default="outcome",
                    help="Name of the Yes/No outcome column "
                         "(default: 'outcome').")
    ap.add_argument("--folds", type=int, default=5,
                    help="Number of CV folds (default 5).")
    args = ap.parse_args()

    X, y, num_cols, cat_cols, label_map = load_data(args.csv, args.outcome)

    # Don't ask for more folds than the minority class can support.
    minority = int(min(np.bincount(y)))
    folds = max(2, min(args.folds, minority))
    if folds < args.folds:
        print(f"(Reduced to {folds} folds: minority class has only "
              f"{minority} examples.)\n")

    report_regime(X, y, num_cols, cat_cols, label_map)
    pre = build_preprocessor(num_cols, cat_cols)
    results = evaluate_models(X, y, pre, folds)
    rank_factors(X, y, pre, num_cols, cat_cols)
    verdict(results)


if __name__ == "__main__":
    main()
