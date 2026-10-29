#optimization model
# ============================================================
# OPTIMIZED LOSO OPEN-SET MODEL
# Fresh model per LOSO fold:
#   - fresh train/val/test split
#   - fresh ANOVA k selection
#   - fresh SVM hyperparameter tuning
#   - train-only score normalization
#   - tau optimized per fold by maximizing F1_unknown
#
# Outputs:
#   - per-fold OSR metrics
#   - averaged LOSO metrics
#   - unknown false-acceptance matrix
#   - OSCR curve
#   - ROC curve
#   - PR curve
#   - per-species bar plots
# ============================================================

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix
)

warnings.filterwarnings("ignore")

# ============================================================
# 0. USER SETTINGS
# ============================================================

FILE_PATH = "final ML master file.csv"

TARGET_COLUMN = "CleanSpecies"
STRAIN_COLUMN = "CleanStrain"
COLONY_COLUMN = "CleanColony"

RANDOM_STATE = 42

# Same merged Yersinia group as your current pipeline
MERGE_3_SPECIES = ["Y.intermedia", "Y.kristensenii", "Y.frederiksenii"]
MERGE_3_LABEL = "Y.intermedia_kristensenii_frederiksenii_GROUP"

# LOSO split inside known taxa
LOSO_TRAIN_FRAC = 0.60
LOSO_VAL_FRAC = 0.20
LOSO_TEST_FRAC = 0.20

# Unknown species split: one part for tau optimization, one part for final test
UNKNOWN_CALIB_FRAC = 0.50

# Feature families
USE_EXPERIMENTAL = True
USE_PSEUDOZERNIKE = True
USE_ZERNIKE = False

EXPERIMENTAL_PREFIX = "Experimental "
PSEUDOZERNIKE_PREFIX = "PseudoZernike "
ZERNIKE_PREFIX = "Zernike "

# k sweep
K_SWEEP_VALUES = [226, 338, 564, 677, 790, 902]

# SVM tuning
CV_FOLDS = 5
N_ITER_SVM = 40

PARAM_DIST_SVM = {
    "svm__C": np.logspace(-1, 2, 40),
    "svm__gamma": np.logspace(-4, 0, 40),
    "svm__class_weight": [None, "balanced"]
}

# Output folder
DESKTOP = os.path.join(os.path.expanduser("~"), "Desktop")
OUT_ROOT = os.path.join(DESKTOP, "OPTIMIZED_LOSO_OSR_FULL")
TABLES_DIR = os.path.join(OUT_ROOT, "tables")
PLOTS_DIR = os.path.join(OUT_ROOT, "plots")
REPORTS_DIR = os.path.join(OUT_ROOT, "reports")

for d in [OUT_ROOT, TABLES_DIR, PLOTS_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# 1. HELPER FUNCTIONS
# ============================================================

def print_block(title):
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)

def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def save_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def apply_merge3_labels(y):
    y = y.astype(str).copy()
    return y.replace(MERGE_3_SPECIES, MERGE_3_LABEL)

def safe_div(a, b):
    return a / b if b != 0 else 0.0

def normalize_rows(df):
    out = df.copy().astype(float)
    row_sums = out.sum(axis=1).replace(0, np.nan)
    return out.div(row_sums, axis=0).fillna(0.0)

def get_feature_columns(df):
    experimental_cols = sorted(
        [c for c in df.columns if c.startswith(EXPERIMENTAL_PREFIX)],
        key=lambda x: int(x.replace(EXPERIMENTAL_PREFIX, "").strip())
    )

    pseudozernike_cols = sorted(
        [c for c in df.columns if c.startswith(PSEUDOZERNIKE_PREFIX)],
        key=lambda x: int(x.replace(PSEUDOZERNIKE_PREFIX, "").strip())
    )

    zernike_cols = sorted(
        [c for c in df.columns if c.startswith(ZERNIKE_PREFIX)],
        key=lambda x: int(x.replace(ZERNIKE_PREFIX, "").strip())
    )

    selected = []
    if USE_EXPERIMENTAL:
        selected.extend(experimental_cols)
    if USE_PSEUDOZERNIKE:
        selected.extend(pseudozernike_cols)
    if USE_ZERNIKE:
        selected.extend(zernike_cols)

    print_block("FEATURE FAMILY SUMMARY")
    print(f"Experimental columns found : {len(experimental_cols)}")
    print(f"PseudoZernike columns found: {len(pseudozernike_cols)}")
    print(f"Zernike columns found      : {len(zernike_cols)}")
    print(f"Zernike included?          : {USE_ZERNIKE}")
    print(f"Final selected features    : {len(selected)}")

    if len(selected) == 0:
        raise ValueError("No feature columns selected. Check column names.")

    return selected

def decision_details(model, X):
    margins = model.decision_function(X)

    if margins.ndim == 1:
        margins = np.vstack([-margins, margins]).T

    pred_index = np.argmax(margins, axis=1)
    max_margin = np.max(margins, axis=1)

    if margins.shape[1] >= 2:
        second_margin = np.partition(margins, -2, axis=1)[:, -2]
    else:
        second_margin = np.zeros_like(max_margin)

    gap = max_margin - second_margin

    return pred_index, margins, max_margin, second_margin, gap

def z_normalize_from_train(train_max_margin, target_max_margin):
    mu = float(np.mean(train_max_margin))
    sd = float(np.std(train_max_margin, ddof=0))

    if sd == 0:
        sd = 1.0

    z = (target_max_margin - mu) / sd
    return z, mu, sd

def compute_osr_metrics(z_known, known_correct_mask, z_unknown, tau):
    known_accept_mask = z_known >= tau
    unknown_reject_mask = z_unknown < tau

    unknown_accept_mask = ~unknown_reject_mask

    known_accept = float(np.mean(known_accept_mask))
    known_reject = float(np.mean(~known_accept_mask))

    ccr = float(np.mean(known_accept_mask & known_correct_mask))
    incorrectly_accepted_known = float(np.mean(known_accept_mask & (~known_correct_mask)))

    unknown_reject = float(np.mean(unknown_reject_mask))
    unknown_accept = float(np.mean(unknown_accept_mask))

    tp = np.sum(unknown_reject_mask)
    fn = np.sum(unknown_accept_mask)
    fp = np.sum(~known_accept_mask)

    precision_unknown = safe_div(tp, tp + fp)
    recall_unknown = safe_div(tp, tp + fn)
    f1_unknown = safe_div(
        2 * precision_unknown * recall_unknown,
        precision_unknown + recall_unknown
    )

    balanced_osr = (known_accept + unknown_reject) / 2.0

    return {
        "Known Acceptance": known_accept,
        "Known Rejection Rate": known_reject,
        "CCR (Known correctly classified & accepted)": ccr,
        "Incorrectly classified but accepted knowns": incorrectly_accepted_known,
        "Unknown Rejection Rate": unknown_reject,
        "Unknown Acceptance Rate": unknown_accept,
        "Precision (Unknown detection)": precision_unknown,
        "Recall (Unknown detection)": recall_unknown,
        "F1 Unknown": f1_unknown,
        "Balanced OSR": balanced_osr
    }

def optimize_tau_by_f1_unknown(z_known_calib, z_unknown_calib):
    all_scores = np.concatenate([z_known_calib, z_unknown_calib])

    tau_grid = np.unique(np.quantile(all_scores, np.linspace(0, 1, 600)))

    best_tau = None
    best_metrics = None
    best_f1 = -1

    known_correct_dummy = np.ones(len(z_known_calib), dtype=bool)

    for tau in tau_grid:
        metrics = compute_osr_metrics(
            z_known=z_known_calib,
            known_correct_mask=known_correct_dummy,
            z_unknown=z_unknown_calib,
            tau=tau
        )

        if metrics["F1 Unknown"] > best_f1:
            best_f1 = metrics["F1 Unknown"]
            best_tau = float(tau)
            best_metrics = metrics

    return best_tau, best_metrics

def compute_oscr_curve(z_known, known_correct_mask, z_unknown):
    all_scores = np.concatenate([z_known, z_unknown])
    grid = np.unique(np.quantile(all_scores, np.linspace(0, 1, 500)))

    x_unknown_accept = []
    y_ccr = []

    for tau in grid:
        known_accept_mask = z_known >= tau
        unknown_accept_mask = z_unknown >= tau

        ccr = np.mean(known_correct_mask & known_accept_mask)
        unknown_accept = np.mean(unknown_accept_mask)

        x_unknown_accept.append(unknown_accept)
        y_ccr.append(ccr)

    x_unknown_accept = np.array(x_unknown_accept)
    y_ccr = np.array(y_ccr)

    order = np.argsort(x_unknown_accept)

    return x_unknown_accept[order], y_ccr[order]

def save_heatmap(df, title, outpath, normalize=False, figsize=(13, 9)):
    plot_df = normalize_rows(df) if normalize else df.copy()

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(plot_df.values, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ax.set_title(title)
    ax.set_xticks(np.arange(plot_df.shape[1]))
    ax.set_yticks(np.arange(plot_df.shape[0]))
    ax.set_xticklabels(plot_df.columns, rotation=90)
    ax.set_yticklabels(plot_df.index)

    for i in range(plot_df.shape[0]):
        for j in range(plot_df.shape[1]):
            val = plot_df.values[i, j]
            txt = f"{val:.2f}" if normalize else f"{int(val)}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

def save_barplot(series, title, ylabel, outpath, figsize=(12, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    series.plot(kind="bar", ax=ax)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    plt.xticks(rotation=75, ha="right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

def save_roc_curve(y_binary, score_unknown, outpath):
    fpr, tpr, _ = roc_curve(y_binary, score_unknown)
    auc_val = roc_auc_score(y_binary, score_unknown)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f"AUROC = {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Optimized LOSO Unknown-vs-Known ROC")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

def save_pr_curve(y_binary, score_unknown, outpath):
    precision, recall, _ = precision_recall_curve(y_binary, score_unknown)
    ap = average_precision_score(y_binary, score_unknown)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, label=f"AUPRC = {ap:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Optimized LOSO Unknown-vs-Known PR Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

def save_oscr_curve(x_unknown_accept, y_ccr, outpath):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(x_unknown_accept, y_ccr, linewidth=2)

    ax.set_xlabel("Unknown Acceptance Rate")
    ax.set_ylabel("CCR (Known correctly classified & accepted)")
    ax.set_title("Optimized LOSO OSCR Curve")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

# ============================================================
# 2. LOAD DATA
# ============================================================

print_block("LOAD DATA")

df = pd.read_csv(FILE_PATH)

df[TARGET_COLUMN] = apply_merge3_labels(df[TARGET_COLUMN])

feature_cols = get_feature_columns(df)

X_all = df[feature_cols].copy()
y_all = df[TARGET_COLUMN].astype(str).copy()

species_list = sorted(y_all.unique())

print(f"Total samples: {len(df)}")
print(f"Total classes after merge: {len(species_list)}")
print(species_list)

# ============================================================
# 3. OPTIMIZED LOSO
# ============================================================

fold_rows = []

all_z_known_test = []
all_z_unknown_test = []
all_known_correct_test = []
all_binary_unknown_labels = []
all_unknown_scores_for_auc = []

unknown_true_labels_all = []
unknown_predicted_as_all = []

tau_records = []

for heldout_species in species_list:

    print_block(f"LOSO FOLD: HELD-OUT UNKNOWN = {heldout_species}")

    known_mask = y_all != heldout_species
    unknown_mask = y_all == heldout_species

    X_known = X_all.loc[known_mask].reset_index(drop=True)
    y_known = y_all.loc[known_mask].reset_index(drop=True)

    X_unknown = X_all.loc[unknown_mask].reset_index(drop=True)

    if len(X_unknown) < 4:
        print(f"Skipping {heldout_species}: too few unknown samples.")
        continue

    # --------------------------------------------------------
    # Known split: train / val / known-test
    # --------------------------------------------------------

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_known,
        y_known,
        train_size=LOSO_TRAIN_FRAC,
        stratify=y_known,
        random_state=RANDOM_STATE
    )

    relative_val_frac = LOSO_VAL_FRAC / (LOSO_VAL_FRAC + LOSO_TEST_FRAC)

    X_val, X_test_known, y_val, y_test_known = train_test_split(
        X_temp,
        y_temp,
        train_size=relative_val_frac,
        stratify=y_temp,
        random_state=RANDOM_STATE
    )

    # --------------------------------------------------------
    # Unknown split: calibration for tau / final unknown-test
    # --------------------------------------------------------

    X_unknown_calib, X_unknown_test = train_test_split(
        X_unknown,
        train_size=UNKNOWN_CALIB_FRAC,
        random_state=RANDOM_STATE
    )

    # --------------------------------------------------------
    # Encode labels inside fold
    # --------------------------------------------------------

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_known_enc = le.transform(y_test_known)

    # --------------------------------------------------------
    # Fresh k selection + fresh SVM tuning
    # --------------------------------------------------------

    best_model = None
    best_k = None
    best_cv_score = -1
    best_val_macro_f1 = -1
    best_params = None

    k_records = []

    for k in K_SWEEP_VALUES:

        if k >= X_train.shape[1]:
            k_use = "all"
        else:
            k_use = k

        pipe = Pipeline([
            ("var", VarianceThreshold()),
            ("anova", SelectKBest(score_func=f_classif, k=k_use)),
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", decision_function_shape="ovr"))
        ])

        cv = StratifiedKFold(
            n_splits=CV_FOLDS,
            shuffle=True,
            random_state=RANDOM_STATE
        )

        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=PARAM_DIST_SVM,
            n_iter=N_ITER_SVM,
            scoring="f1_macro",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0
        )

        search.fit(X_train, y_train_enc)

        val_pred = search.best_estimator_.predict(X_val)
        val_macro_f1 = f1_score(y_val_enc, val_pred, average="macro")

        k_records.append({
            "Held-out Unknown Species": heldout_species,
            "k": k,
            "CV Macro-F1": search.best_score_,
            "VAL Macro-F1": val_macro_f1,
            "Best Params": str(search.best_params_)
        })

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_cv_score = search.best_score_
            best_model = search.best_estimator_
            best_k = k
            best_params = search.best_params_

    pd.DataFrame(k_records).to_csv(
        os.path.join(TABLES_DIR, f"k_sweep_{heldout_species}.csv"),
        index=False
    )

    print(f"Best k: {best_k}")
    print(f"Best CV Macro-F1: {best_cv_score:.4f}")
    print(f"Best VAL Macro-F1: {best_val_macro_f1:.4f}")
    print(f"Best params: {best_params}")

    # --------------------------------------------------------
    # Train-only margin normalization
    # --------------------------------------------------------

    _, _, train_max_margin, _, _ = decision_details(best_model, X_train)

    # Known calibration scores
    val_pred_index, _, val_max_margin, _, _ = decision_details(best_model, X_val)
    z_known_calib, margin_mu, margin_sd = z_normalize_from_train(
        train_max_margin,
        val_max_margin
    )

    # Unknown calibration scores
    _, _, unknown_calib_max_margin, _, _ = decision_details(
        best_model,
        X_unknown_calib
    )
    z_unknown_calib, _, _ = z_normalize_from_train(
        train_max_margin,
        unknown_calib_max_margin
    )

    # --------------------------------------------------------
    # Optimize tau using F1_unknown
    # --------------------------------------------------------

    best_tau, calib_tau_metrics = optimize_tau_by_f1_unknown(
        z_known_calib,
        z_unknown_calib
    )

    tau_records.append({
        "Held-out Unknown Species": heldout_species,
        "Optimized Tau": best_tau,
        "Calibration F1 Unknown": calib_tau_metrics["F1 Unknown"],
        "Calibration Balanced OSR": calib_tau_metrics["Balanced OSR"],
        "Calibration Known Acceptance": calib_tau_metrics["Known Acceptance"],
        "Calibration Unknown Rejection Rate": calib_tau_metrics["Unknown Rejection Rate"],
        "Margin Train Mean": margin_mu,
        "Margin Train SD": margin_sd
    })

    print(f"Optimized tau: {best_tau:.4f}")
    print(f"Calibration F1_unknown: {calib_tau_metrics['F1 Unknown']:.4f}")
    print(f"Calibration Balanced OSR: {calib_tau_metrics['Balanced OSR']:.4f}")

    # --------------------------------------------------------
    # Final known-test scores
    # --------------------------------------------------------

    known_pred_index, _, known_max_margin, _, _ = decision_details(
        best_model,
        X_test_known
    )

    z_known_test, _, _ = z_normalize_from_train(
        train_max_margin,
        known_max_margin
    )

    known_correct_mask = known_pred_index == y_test_known_enc

    known_test_accuracy = accuracy_score(y_test_known_enc, known_pred_index)
    known_test_bal_acc = balanced_accuracy_score(y_test_known_enc, known_pred_index)
    known_test_macro_f1 = f1_score(y_test_known_enc, known_pred_index, average="macro")

    # --------------------------------------------------------
    # Final unknown-test scores
    # --------------------------------------------------------

    unknown_pred_index, _, unknown_max_margin, _, _ = decision_details(
        best_model,
        X_unknown_test
    )

    z_unknown_test, _, _ = z_normalize_from_train(
        train_max_margin,
        unknown_max_margin
    )

    unknown_pred_labels = le.inverse_transform(unknown_pred_index)

    # Unknown accepted by model if z >= tau
    unknown_accepted_mask = z_unknown_test >= best_tau

    accepted_unknown_pred_labels = unknown_pred_labels[unknown_accepted_mask]

    for pred_label in accepted_unknown_pred_labels:
        unknown_true_labels_all.append(heldout_species)
        unknown_predicted_as_all.append(pred_label)

    # --------------------------------------------------------
    # Evaluate final test with optimized tau
    # --------------------------------------------------------

    osr_metrics = compute_osr_metrics(
        z_known=z_known_test,
        known_correct_mask=known_correct_mask,
        z_unknown=z_unknown_test,
        tau=best_tau
    )

    y_binary_fold = np.concatenate([
        np.zeros(len(z_known_test)),
        np.ones(len(z_unknown_test))
    ])

    unknown_score_fold = np.concatenate([
        -z_known_test,
        -z_unknown_test
    ])

    try:
        auroc = roc_auc_score(y_binary_fold, unknown_score_fold)
    except:
        auroc = np.nan

    try:
        auprc = average_precision_score(y_binary_fold, unknown_score_fold)
    except:
        auprc = np.nan

    row = {
        "Held-out Unknown Species": heldout_species,
        "Best k": best_k,
        "Best CV Macro-F1": best_cv_score,
        "Known VAL Macro-F1": best_val_macro_f1,
        "Optimized Tau": best_tau,
        "Tau Selection": "Max F1_unknown on fold calibration",
        "Known TEST Accuracy": known_test_accuracy,
        "Known TEST Balanced Accuracy": known_test_bal_acc,
        "Known TEST Macro-F1": known_test_macro_f1,
        "AUROC Unknown-vs-Known": auroc,
        "AUPRC Unknown-vs-Known": auprc,
        **osr_metrics
    }

    fold_rows.append(row)

    # Store pooled values for global curves
    all_z_known_test.extend(z_known_test.tolist())
    all_z_unknown_test.extend(z_unknown_test.tolist())
    all_known_correct_test.extend(known_correct_mask.tolist())

    all_binary_unknown_labels.extend(y_binary_fold.tolist())
    all_unknown_scores_for_auc.extend(unknown_score_fold.tolist())

# ============================================================
# 4. SAVE METRIC TABLES
# ============================================================

print_block("SAVE TABLES")

results_df = pd.DataFrame(fold_rows)
tau_df = pd.DataFrame(tau_records)

numeric_means = results_df.select_dtypes(include=[np.number]).mean().to_dict()
numeric_stds = results_df.select_dtypes(include=[np.number]).std().to_dict()

mean_row = {
    "Held-out Unknown Species": "MEAN_ACROSS_LOSO",
    "Tau Selection": "Max F1_unknown on fold calibration"
}
mean_row.update(numeric_means)

std_row = {
    "Held-out Unknown Species": "STD_ACROSS_LOSO",
    "Tau Selection": "Max F1_unknown on fold calibration"
}
std_row.update(numeric_stds)

summary_df = pd.concat(
    [results_df, pd.DataFrame([mean_row, std_row])],
    ignore_index=True
)

results_path = os.path.join(TABLES_DIR, "optimized_loso_per_fold_results.csv")
summary_path = os.path.join(TABLES_DIR, "optimized_loso_summary_mean_std.csv")
tau_path = os.path.join(TABLES_DIR, "optimized_loso_tau_records.csv")

results_df.to_csv(results_path, index=False)
summary_df.to_csv(summary_path, index=False)
tau_df.to_csv(tau_path, index=False)

print(summary_df)
print("\nSaved:")
print(results_path)
print(summary_path)
print(tau_path)

# Reviewer-friendly compact comparison row
compact_metrics = [
    "Known Acceptance",
    "CCR (Known correctly classified & accepted)",
    "Unknown Rejection Rate",
    "Unknown Acceptance Rate",
    "Balanced OSR",
    "F1 Unknown",
    "AUROC Unknown-vs-Known",
    "AUPRC Unknown-vs-Known",
    "Known TEST Macro-F1"
]

compact_row = {
    "Model": "Optimized LOSO OSR",
    "Tau Selection": "Per-fold tau maximizing F1_unknown"
}

for m in compact_metrics:
    compact_row[f"{m} Mean"] = results_df[m].mean()
    compact_row[f"{m} SD"] = results_df[m].std()

compact_df = pd.DataFrame([compact_row])
compact_path = os.path.join(TABLES_DIR, "optimized_model_comparison_row.csv")
compact_df.to_csv(compact_path, index=False)

# ============================================================
# 5. UNKNOWN FALSE-ACCEPTANCE MATRIX
# ============================================================

if len(unknown_true_labels_all) > 0:
    false_accept_matrix = pd.crosstab(
        pd.Series(unknown_true_labels_all, name="Held-out unknown species"),
        pd.Series(unknown_predicted_as_all, name="Falsely accepted as")
    )

    false_accept_path = os.path.join(TABLES_DIR, "unknown_false_acceptance_matrix_counts.csv")
    false_accept_norm_path = os.path.join(TABLES_DIR, "unknown_false_acceptance_matrix_normalized.csv")

    false_accept_matrix.to_csv(false_accept_path)
    normalize_rows(false_accept_matrix).to_csv(false_accept_norm_path)

    save_heatmap(
        false_accept_matrix,
        "Optimized LOSO Unknown False-Acceptance Matrix Counts",
        os.path.join(PLOTS_DIR, "unknown_false_acceptance_matrix_counts.png"),
        normalize=False
    )

    save_heatmap(
        false_accept_matrix,
        "Optimized LOSO Unknown False-Acceptance Matrix Normalized",
        os.path.join(PLOTS_DIR, "unknown_false_acceptance_matrix_normalized.png"),
        normalize=True
    )
else:
    false_accept_matrix = pd.DataFrame()
    print("No unknown false acceptances detected at optimized tau.")

# ============================================================
# 6. GLOBAL CURVES
# ============================================================

print_block("SAVE GLOBAL CURVES")

all_z_known_test = np.array(all_z_known_test)
all_z_unknown_test = np.array(all_z_unknown_test)
all_known_correct_test = np.array(all_known_correct_test).astype(bool)

all_binary_unknown_labels = np.array(all_binary_unknown_labels)
all_unknown_scores_for_auc = np.array(all_unknown_scores_for_auc)

# ROC
save_roc_curve(
    all_binary_unknown_labels,
    all_unknown_scores_for_auc,
    os.path.join(PLOTS_DIR, "optimized_loso_global_roc.png")
)

# PR
save_pr_curve(
    all_binary_unknown_labels,
    all_unknown_scores_for_auc,
    os.path.join(PLOTS_DIR, "optimized_loso_global_pr.png")
)

# OSCR
x_oscr, y_oscr = compute_oscr_curve(
    all_z_known_test,
    all_known_correct_test,
    all_z_unknown_test
)

oscr_df = pd.DataFrame({
    "Unknown Acceptance Rate": x_oscr,
    "CCR": y_oscr
})

oscr_df.to_csv(os.path.join(TABLES_DIR, "optimized_loso_oscr_curve_points.csv"), index=False)

save_oscr_curve(
    x_oscr,
    y_oscr,
    os.path.join(PLOTS_DIR, "optimized_loso_oscr_curve.png")
)

# ============================================================
# 7. BAR PLOTS PER HELD-OUT SPECIES
# ============================================================

print_block("SAVE BAR PLOTS")

bar_metrics = [
    "Known Acceptance",
    "Unknown Rejection Rate",
    "Unknown Acceptance Rate",
    "Balanced OSR",
    "F1 Unknown",
    "CCR (Known correctly classified & accepted)",
    "AUROC Unknown-vs-Known",
    "AUPRC Unknown-vs-Known"
]

plot_df = results_df.set_index("Held-out Unknown Species")

for metric in bar_metrics:
    save_barplot(
        plot_df[metric],
        f"Optimized LOSO: {metric} by Held-out Unknown Species",
        metric,
        os.path.join(PLOTS_DIR, f"bar_{metric.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '').replace('&', 'and')}.png")
    )

# ============================================================
# 8. FINAL TEXT REPORT
# ============================================================

report_lines = []

report_lines.append("OPTIMIZED LOSO OPEN-SET MODEL REPORT")
report_lines.append("=" * 80)
report_lines.append("")
report_lines.append("Model design:")
report_lines.append("- Each LOSO fold holds out one species as unknown.")
report_lines.append("- Feature selection, scaling, and SVM hyperparameter tuning are repeated from scratch within each fold.")
report_lines.append("- Margin scores are z-normalized using training-set max-margin statistics only.")
report_lines.append("- Tau is optimized per fold using known validation samples and unknown calibration samples.")
report_lines.append("- Tau objective: maximize F1_unknown.")
report_lines.append("")
report_lines.append("Important interpretation:")
report_lines.append("- This is an optimized/best-case OSR setting, not deployment-style thresholding.")
report_lines.append("- Deployment-style uses a fixed known-acceptance target from closed-set validation.")
report_lines.append("- Optimized OSR shows the upper-bound trade-off if unknown calibration information is available.")
report_lines.append("")
report_lines.append("Mean metrics across LOSO folds:")
report_lines.append("")

for m in compact_metrics:
    report_lines.append(f"{m}: {results_df[m].mean():.4f} ± {results_df[m].std():.4f}")

report_lines.append("")
report_lines.append("Files saved:")
report_lines.append(f"- {results_path}")
report_lines.append(f"- {summary_path}")
report_lines.append(f"- {compact_path}")
report_lines.append(f"- {tau_path}")
report_lines.append(f"- plots saved in: {PLOTS_DIR}")

report_path = os.path.join(REPORTS_DIR, "optimized_loso_report.txt")
save_text(report_path, "\n".join(report_lines))

print_block("DONE")
print(f"Report saved: {report_path}")
print(f"All outputs saved in: {OUT_ROOT}")
