import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# ============================================================
# USER SETTINGS
# ============================================================

FILE_PATH = "final ML master file.csv"

TARGET_COLUMN = "CleanSpecies"
STRAIN_COLUMN = "CleanStrain"
COLONY_COLUMN = "CleanColony"

DROP_METADATA_COLUMNS = [TARGET_COLUMN, STRAIN_COLUMN, COLONY_COLUMN]

RANDOM_STATE = 42

# MERGE_3 exactly as before
MERGE_3_SPECIES = ["Y.intermedia", "Y.kristensenii", "Y.frederiksenii"]
MERGE_3_LABEL = "Y.intermedia_kristensenii_frederiksenii_GROUP"

# Splits
STAGE1_TRAIN_FRAC = 0.70
STAGE1_VAL_FRAC = 0.15
STAGE1_TEST_FRAC = 0.15

LOSO_TRAIN_FRAC = 0.60
LOSO_VAL_FRAC = 0.20
LOSO_TEST_FRAC = 0.20

# k sweep exactly same family
K_SWEEP_VALUES = [226, 344, 564, 677, 790, 902]

# SVM tuning
CV_FOLDS = 5
N_ITER_SVM = 40
PARAM_DIST_SVM = {
    "svm__C": np.logspace(-1, 2, 40),
    "svm__gamma": np.logspace(-4, 0, 40),
    "svm__class_weight": [None, "balanced"]
}

# Validation CCR targets
TAU_TARGETS = np.round(np.arange(0.90, 0.951, 0.01), 2).tolist()

# Tau is learned to achieve a validation CCR target.
# All targets must be feasible given validation closed-set accuracy.
REQUIRE_ALL_CCR_TARGETS_FEASIBLE = True

# output folder
DESKTOP = os.path.join(os.path.expanduser("~"), "Desktop")
OUT_ROOT = os.path.join(DESKTOP, "ELS_SVM_analysis")
TABLES_DIR = os.path.join(OUT_ROOT, "tables")
PLOTS_DIR = os.path.join(OUT_ROOT, "plots")
REPORTS_DIR = os.path.join(OUT_ROOT, "reports")

for d in [OUT_ROOT, TABLES_DIR, PLOTS_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ------------------------------------------------------------
# FEATURE FAMILY CONTROL
# ------------------------------------------------------------
USE_EXPERIMENTAL = True
USE_PSEUDOZERNIKE = True
USE_ZERNIKE = False

EXPECTED_EXPERIMENTAL_COUNT = 257
EXPECTED_PSEUDOZERNIKE_COUNT = 1325

EXPERIMENTAL_PREFIX = "Experimental "
PSEUDOZERNIKE_PREFIX = "PseudoZernike "
ZERNIKE_PREFIX = "Zernike "


# ============================================================
# HELPERS
# ============================================================

def print_block(title):
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)

def save_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def apply_merge3_labels(y):
    y = y.astype(str).copy()
    y = y.replace(MERGE_3_SPECIES, MERGE_3_LABEL)
    return y

def safe_div(a, b):
    return a / b if b != 0 else 0.0

def normalize_rows(df):
    out = df.copy().astype(float)
    row_sums = out.sum(axis=1).replace(0, np.nan)
    out = out.div(row_sums, axis=0).fillna(0.0)
    return out

def multiclass_metrics(y_true, y_pred):
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Balanced Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "Macro-F1": float(f1_score(y_true, y_pred, average="macro")),
    }

def ensure_2d_decision_function(arr):
    if arr.ndim == 1:
        return np.vstack([-arr, arr]).T
    return arr

def raw_margin_details(model, X):
    margins = ensure_2d_decision_function(model.decision_function(X))
    pred = margins.argmax(axis=1)
    max_margin = margins.max(axis=1)
    if margins.shape[1] >= 2:
        second_margin = np.partition(margins, -2, axis=1)[:, -2]
    else:
        second_margin = np.zeros_like(max_margin)
    gap = max_margin - second_margin
    return pred, margins, max_margin, second_margin, gap

def z_normalize_from_train(train_max_margin, target_max_margin):
    mu = float(np.mean(train_max_margin))
    sd = float(np.std(train_max_margin, ddof=0))
    if sd == 0:
        sd = 1.0
    z = (target_max_margin - mu) / sd
    return z, mu, sd

def pick_tau_for_target_ccr(z_val_known, known_correct_mask, target_ccr):
    """
    Learn the most restrictive threshold that still achieves the requested
    correct-classification rate (CCR) on known validation samples.

    CCR = proportion of all known validation samples that are both:
          (i) accepted as known, and
          (ii) correctly classified.

    Choosing the highest feasible tau maximizes rejection pressure while
    satisfying the requested CCR target.
    """
    z_val_known = np.asarray(z_val_known, dtype=float)
    known_correct_mask = np.asarray(known_correct_mask, dtype=bool)

    if len(z_val_known) == 0:
        raise ValueError("Cannot learn tau from an empty validation set.")

    max_possible_ccr = float(np.mean(known_correct_mask))
    if target_ccr > max_possible_ccr + 1e-12:
        return np.nan, max_possible_ccr, np.nan, False

    candidate_taus = np.unique(z_val_known)
    feasible = []
    for tau in candidate_taus:
        accept_mask = z_val_known >= tau
        achieved_ccr = float(np.mean(known_correct_mask & accept_mask))
        if achieved_ccr + 1e-12 >= target_ccr:
            known_acceptance = float(np.mean(accept_mask))
            feasible.append((float(tau), achieved_ccr, known_acceptance))

    if not feasible:
        return np.nan, max_possible_ccr, np.nan, False

    # Highest tau is the most conservative threshold satisfying target CCR.
    tau, achieved_ccr, known_acceptance = max(feasible, key=lambda x: x[0])
    return tau, achieved_ccr, known_acceptance, True

def compute_osr_metrics(z_known, known_correct_mask, z_unknown, tau):
    known_accept_mask = z_known >= tau
    unknown_reject_mask = z_unknown < tau
    unknown_accept_mask = ~unknown_reject_mask

    known_accept = float(np.mean(known_accept_mask))
    ccr = float(np.mean(known_correct_mask & known_accept_mask))
    known_reject_rate = float(np.mean(~known_accept_mask))
    incorrectly_accepted_known = float(np.mean((~known_correct_mask) & known_accept_mask))

    unknown_reject = float(np.mean(unknown_reject_mask))
    unknown_accept = float(np.mean(unknown_accept_mask))

    tp = np.sum(unknown_reject_mask)
    fn = np.sum(unknown_accept_mask)
    fp = np.sum(~known_accept_mask)

    precision_unknown = safe_div(tp, tp + fp)
    recall_unknown = safe_div(tp, tp + fn)
    f1_unknown = safe_div(2 * precision_unknown * recall_unknown, precision_unknown + recall_unknown)

    balanced_osr = (ccr + unknown_reject) / 2.0

    return {
        "Known Acceptance": known_accept,
        "CCR (Known correctly classified & accepted)": ccr,
        "Known Rejection Rate": known_reject_rate,
        "Incorrectly classified but accepted knowns": incorrectly_accepted_known,
        "Unknown Rejection Rate": unknown_reject,
        "Unknown Acceptance Rate": unknown_accept,
        "Precision (Unknown detection)": precision_unknown,
        "Recall (Unknown detection)": recall_unknown,
        "Balanced OSR": balanced_osr,
        "F1 Unknown": f1_unknown,
    }

def configure_publication_style():
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.bbox": "tight",
    })

def save_vector_figure(fig, outstem):
    """Save editable vector graphics in both SVG and PDF formats."""
    outstem = str(Path(outstem).with_suffix(""))
    fig.savefig(outstem + ".svg", bbox_inches="tight")
    fig.savefig(outstem + ".pdf", bbox_inches="tight")
    plt.close(fig)

def save_barplot_percent(series, title, ylabel, outstem, figsize=(11, 6)):
    series = series.astype(float)
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(np.arange(len(series)), series.values * 100.0)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.set_ylim(0, max(100, float(series.max() * 100.0) + 8))
    ax.set_xticks(np.arange(len(series)))
    ax.set_xticklabels(series.index, rotation=75, ha="right")

    for bar, value in zip(bars, series.values * 100.0):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 1.0,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_vector_figure(fig, outstem)

def save_lineplot_k_sweep(df_k, outstem):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(df_k["k"], df_k["train_cv_macro_f1"], marker="o", label="Train-CV Macro-F1")
    ax.plot(df_k["k"], df_k["val_macro_f1"], marker="o", label="Validation Macro-F1")
    ax.plot(df_k["k"], df_k["test_macro_f1"], marker="o", label="Test Macro-F1")
    ax.set_title("Stage-1 feature-number sweep")
    ax.set_xlabel("Number of ANOVA-selected features")
    ax.set_ylabel("Macro-F1")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_vector_figure(fig, outstem)

def compute_oscr_curve(z_known, known_correct_mask, z_unknown):
    all_scores = np.concatenate([z_known, z_unknown])
    grid = np.unique(np.quantile(all_scores, np.linspace(0, 1, 400)))

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

def save_oscr_curve(x_unknown_accept, y_ccr, operating_points_df, outstem):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(x_unknown_accept, y_ccr, linewidth=1.8, label="OSCR curve")

    if operating_points_df is not None and not operating_points_df.empty:
        x_pts = operating_points_df["Unknown Acceptance Rate"].to_numpy()
        y_pts = operating_points_df["CCR (Known correctly classified & accepted)"].to_numpy()
        ax.scatter(x_pts, y_pts, s=32, zorder=3, label="Evaluated targets")
        for _, row in operating_points_df.iterrows():
            ax.annotate(
                f"{row['Tau Target']:.2f}",
                (row["Unknown Acceptance Rate"], row["CCR (Known correctly classified & accepted)"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )

    ax.set_xlabel("Unknown acceptance rate")
    ax.set_ylabel("Correct classification rate (CCR)")
    ax.set_title("Open-set classification rate curve")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_vector_figure(fig, outstem)

def save_roc_curve(y_binary, score_unknown, outstem):
    fpr, tpr, _ = roc_curve(y_binary, score_unknown)
    auc_val = roc_auc_score(y_binary, score_unknown)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_title("Unknown-versus-known ROC curve")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_vector_figure(fig, outstem)

def save_pr_curve(y_binary, score_unknown, outstem):
    precision, recall, _ = precision_recall_curve(y_binary, score_unknown)
    ap = average_precision_score(y_binary, score_unknown)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, label=f"Average precision = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Unknown-versus-known precision-recall curve")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_vector_figure(fig, outstem)

def save_tau_tradeoff_plot(df_tau, outstem):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        df_tau["Tau Target"],
        df_tau["CCR (Known correctly classified & accepted)"],
        marker="o",
        label="Known CCR",
    )
    ax.plot(
        df_tau["Tau Target"],
        df_tau["Unknown Rejection Rate"],
        marker="o",
        label="Unknown rejection",
    )
    ax.plot(
        df_tau["Tau Target"],
        df_tau["Balanced OSR"],
        marker="o",
        label="Balanced OSR",
    )
    ax.set_xlabel("Target validation CCR")
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1)
    ax.set_xticks(df_tau["Tau Target"])
    ax.set_title("LOSO operating-point trade-off")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_vector_figure(fig, outstem)

def get_selected_feature_columns(df):
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

    if USE_EXPERIMENTAL and len(experimental_cols) != EXPECTED_EXPERIMENTAL_COUNT:
        print(f"WARNING: Expected {EXPECTED_EXPERIMENTAL_COUNT} Experimental features, found {len(experimental_cols)}")

    if USE_PSEUDOZERNIKE and len(pseudozernike_cols) != EXPECTED_PSEUDOZERNIKE_COUNT:
        print(f"WARNING: Expected {EXPECTED_PSEUDOZERNIKE_COUNT} PseudoZernike features, found {len(pseudozernike_cols)}")

    if len(selected) == 0:
        raise ValueError("No feature columns selected. Check column names in the CSV.")

    return selected, experimental_cols, pseudozernike_cols, zernike_cols

configure_publication_style()


# ============================================================
# FEATURE SELECTION
# ============================================================

def select_features_anova(X_train_df, y_train, k):
    """Rank training features by one-way ANOVA F score and retain the top k."""

    # Remove features with no variance in the training subset.
    vt = VarianceThreshold(threshold=0.0)
    X_train_vt = vt.fit_transform(X_train_df)
    vt_names = np.array(X_train_df.columns)[vt.get_support()].tolist()

    # Rank all remaining features using the training subset only.
    anova_all = SelectKBest(score_func=f_classif, k="all")
    anova_all.fit(X_train_vt, y_train)

    f_scores = np.nan_to_num(
        anova_all.scores_,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    ranking_idx = np.argsort(f_scores)[::-1]
    ranked_names = [vt_names[i] for i in ranking_idx]

    k_used = min(int(k), len(ranked_names))
    selected_features = ranked_names[:k_used]

    return {
        "vt": vt,
        "ranked_names": ranked_names,
        "selected_features": selected_features,
        "k_used": k_used,
        "final_feature_count": len(selected_features),
    }


def transform_by_selected_features(X_df, vt, selected_features):
    """Apply the training-fitted variance filter and retain selected features."""
    vt_names = np.array(X_df.columns)[vt.get_support()].tolist()
    X_vt = pd.DataFrame(
        vt.transform(X_df),
        columns=vt_names,
        index=X_df.index,
    )
    return X_vt[selected_features].copy()


def fit_train_only_feature_pipeline(
    X_train_df,
    X_val_df,
    X_test_df,
    y_train,
    k,
):
    """Fit feature selection on training data and apply it to all data subsets."""

    fs = select_features_anova(
        X_train_df=X_train_df,
        y_train=y_train,
        k=k,
    )

    X_train_sel = transform_by_selected_features(
        X_train_df,
        fs["vt"],
        fs["selected_features"],
    )
    X_val_sel = transform_by_selected_features(
        X_val_df,
        fs["vt"],
        fs["selected_features"],
    )
    X_test_sel = transform_by_selected_features(
        X_test_df,
        fs["vt"],
        fs["selected_features"],
    )

    return {
        **fs,
        "X_train_sel": X_train_sel,
        "X_val_sel": X_val_sel,
        "X_test_sel": X_test_sel,
    }


# ============================================================
# SVM TUNING
# ============================================================

def fit_tuned_svm_train_only(X_train_sel, y_train, X_val_sel, y_val, X_test_sel, y_test):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            probability=False,
            decision_function_shape="ovr",
            random_state=RANDOM_STATE
        ))
    ])

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

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
    search.fit(X_train_sel, y_train)

    best_pipe = search.best_estimator_
    best_params = search.best_params_
    best_cv_f1 = float(search.best_score_)

    pred_val = ensure_2d_decision_function(best_pipe.decision_function(X_val_sel)).argmax(axis=1)
    pred_test = ensure_2d_decision_function(best_pipe.decision_function(X_test_sel)).argmax(axis=1)

    val_m = multiclass_metrics(y_val, pred_val)
    test_m = multiclass_metrics(y_test, pred_test)

    return best_pipe, best_params, best_cv_f1, val_m, test_m

# ============================================================
# MODEL / TAU SELECTION HELPERS
# ============================================================

def classify_fit_status(train_cv, val_score):
    gap = train_cv - val_score
    abs_gap = abs(gap)

    if gap > 0.02:
        return "Possible overfitting"
    elif gap < -0.02:
        return "Possible underfitting or split luck"
    else:
        return "No strong over/underfitting signal"

def choose_best_stage1_model(df_k):
    df = df_k.copy()
    df["abs_train_val_gap"] = (df["train_cv_macro_f1"] - df["val_macro_f1"]).abs()

    df = df.sort_values(
        by=["val_macro_f1", "abs_train_val_gap", "train_cv_macro_f1", "final_feature_count"],
        ascending=[False, True, False, True]
    ).reset_index(drop=True)

    best = df.iloc[0].to_dict()
    reason_lines = [
        f"Selected because it achieved the highest VAL Macro-F1 = {best['val_macro_f1']:.4f}.",
        f"Tie-break favored smaller |Train-CV - VAL| gap = {best['abs_train_val_gap']:.4f}.",
        f"Train-CV Macro-F1 = {best['train_cv_macro_f1']:.4f}.",
        f"Generalization check: {classify_fit_status(best['train_cv_macro_f1'], best['val_macro_f1'])}.",
        "TEST metrics were not used to choose the model; they are only reported afterward."
    ]
    return best, reason_lines

def choose_best_tau_tradeoff(results_df):
    """
    Select the evaluated CCR-target operating point with the highest Balanced OSR.
    Balanced OSR gives equal weight to known-sample CCR and unknown rejection.
    """
    eligible = results_df.copy()
    eligible = eligible.sort_values(
        by=[
            "Balanced OSR",
            "F1 Unknown",
            "Unknown Rejection Rate",
            "CCR (Known correctly classified & accepted)",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    best = eligible.iloc[0].to_dict()
    reason_lines = [
        "All candidate thresholds were learned on validation known samples using prespecified CCR targets.",
        f"Selected target CCR {best['Tau Target']:.2f} because it produced the highest LOSO Balanced OSR = {best['Balanced OSR']:.4f}.",
        "Balanced OSR was defined as the arithmetic mean of known-sample CCR and unknown-sample rejection rate.",
        f"At this operating point, LOSO CCR = {best['CCR (Known correctly classified & accepted)']:.4f}, "
        f"unknown rejection = {best['Unknown Rejection Rate']:.4f}, and unknown-detection F1 = {best['F1 Unknown']:.4f}.",
    ]
    return best, reason_lines

# ============================================================
# LOAD DATA
# ============================================================

print_block("STAGE 1: LOAD DATA")

df = pd.read_csv(FILE_PATH)

required_cols = {TARGET_COLUMN, STRAIN_COLUMN, COLONY_COLUMN}
missing_cols = required_cols - set(df.columns)
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

feature_cols, experimental_cols, pseudozernike_cols, zernike_cols = get_selected_feature_columns(df)

df_meta = pd.DataFrame({
    "Taxon": apply_merge3_labels(df[TARGET_COLUMN]),
    "Strain ID": df[STRAIN_COLUMN].astype(str),
    "Colony ID": df[COLONY_COLUMN].astype(str),
})

X_raw = df[feature_cols].copy()
y_raw_merged = df_meta["Taxon"].copy()

unexpected_zernike = [c for c in feature_cols if c.startswith(ZERNIKE_PREFIX)]
if len(unexpected_zernike) > 0:
    raise ValueError("Zernike columns are still present in feature_cols, which should not happen.")


print("Data shape:", df.shape)
print("Feature count:", len(feature_cols))
print("Classes after MERGE_3:", sorted(y_raw_merged.unique().tolist()))
print("Using only Experimental + PseudoZernike features.")

# ============================================================
# STAGE-1 SPLIT
# ============================================================

print_block("STAGE 1A: BUILD CLOSED-SET 70/15/15 SPLIT")

idx_all = np.arange(len(df))
y_all_strat = y_raw_merged.astype(str).values

idx_train, idx_temp = train_test_split(
    idx_all,
    test_size=(1.0 - STAGE1_TRAIN_FRAC),
    stratify=y_all_strat,
    random_state=RANDOM_STATE,
)

temp_labels = y_all_strat[idx_temp]
temp_test_fraction = STAGE1_TEST_FRAC / (STAGE1_VAL_FRAC + STAGE1_TEST_FRAC)

idx_val, idx_test = train_test_split(
    idx_temp,
    test_size=temp_test_fraction,
    stratify=temp_labels,
    random_state=RANDOM_STATE,
)

df_train = pd.concat([df_meta.iloc[idx_train].reset_index(drop=True), X_raw.iloc[idx_train].reset_index(drop=True)], axis=1)
df_val   = pd.concat([df_meta.iloc[idx_val].reset_index(drop=True),   X_raw.iloc[idx_val].reset_index(drop=True)], axis=1)
df_test  = pd.concat([df_meta.iloc[idx_test].reset_index(drop=True),  X_raw.iloc[idx_test].reset_index(drop=True)], axis=1)

X_train_df = df_train[feature_cols].copy()
X_val_df   = df_val[feature_cols].copy()
X_test_df  = df_test[feature_cols].copy()

y_train_labels = df_train["Taxon"].astype(str)
y_val_labels   = df_val["Taxon"].astype(str)
y_test_labels  = df_test["Taxon"].astype(str)

le_stage1 = LabelEncoder()
y_train = le_stage1.fit_transform(y_train_labels)
y_val   = le_stage1.transform(y_val_labels)
y_test  = le_stage1.transform(y_test_labels)
class_names_stage1 = le_stage1.classes_

print("Split sizes:")
print("  Train:", X_train_df.shape, " Val:", X_val_df.shape, " Test:", X_test_df.shape)
print("Classes:", len(class_names_stage1))

# ============================================================
# FEATURE-SET SIZE EVALUATION AND SVM OPTIMISATION
# ============================================================


print_block("STAGE 1B: FEATURE-SET SIZE EVALUATION AND SVM OPTIMISATION")

k_rows = []
k_feature_artifacts = {}

for k in K_SWEEP_VALUES:
    print(f"  Evaluating k = {k}")

    fs = fit_train_only_feature_pipeline(
        X_train_df=X_train_df,
        X_val_df=X_val_df,
        X_test_df=X_test_df,
        y_train=y_train,
        k=k,
        
    )

    best_pipe, best_params, best_cv_f1, val_m, test_m = fit_tuned_svm_train_only(
        X_train_sel=fs["X_train_sel"],
        y_train=y_train,
        X_val_sel=fs["X_val_sel"],
        y_val=y_val,
        X_test_sel=fs["X_test_sel"],
        y_test=y_test,
    )

    gap = abs(best_cv_f1 - val_m["Macro-F1"])

    k_rows.append({
        "k": k,
        "k_used": fs["k_used"],
        "final_feature_count": fs["final_feature_count"],
        "train_cv_macro_f1": best_cv_f1,
        "val_accuracy": val_m["Accuracy"],
        "val_balanced_accuracy": val_m["Balanced Accuracy"],
        "val_macro_f1": val_m["Macro-F1"],
        "test_accuracy": test_m["Accuracy"],
        "test_balanced_accuracy": test_m["Balanced Accuracy"],
        "test_macro_f1": test_m["Macro-F1"],
        "abs_train_val_gap": gap,
        "best_C": best_params["svm__C"],
        "best_gamma": best_params["svm__gamma"],
        "best_class_weight": str(best_params["svm__class_weight"]),
    })

    k_feature_artifacts[k] = {
        "selected_features": fs["selected_features"],
        "best_params": {
            "C": best_params["svm__C"],
            "gamma": best_params["svm__gamma"],
            "class_weight": best_params["svm__class_weight"],
        },
    }

    print(
        f"    done | k={k} | VAL Macro-F1={val_m['Macro-F1']:.4f} | "
        f"TEST Macro-F1={test_m['Macro-F1']:.4f} | TRAIN-CV Macro-F1={best_cv_f1:.4f} | "
        f"FINAL FEATURES={fs['final_feature_count']}"
    )

df_k = pd.DataFrame(k_rows)
df_k.to_csv(os.path.join(TABLES_DIR, "stage1_k_sweep.csv"), index=False)
save_lineplot_k_sweep(df_k, os.path.join(PLOTS_DIR, "stage1_k_sweep.png"))

best_stage1_row, stage1_reason_lines = choose_best_stage1_model(df_k)

BEST_K = int(best_stage1_row["k"])
BEST_SVM_PARAMS = {
    "C": float(best_stage1_row["best_C"]),
    "gamma": float(best_stage1_row["best_gamma"]),
    "class_weight": None if best_stage1_row["best_class_weight"] == "None" else best_stage1_row["best_class_weight"],
}

print_block("BEST STAGE-1 MODEL SELECTED")
print(f"Best k = {BEST_K}")
print(f"Best final feature count = {int(best_stage1_row['final_feature_count'])}")
print(f"Best SVM params = {BEST_SVM_PARAMS}")
print(f"Best VAL Macro-F1 = {best_stage1_row['val_macro_f1']:.4f}")
print(f"Train-CV Macro-F1 = {best_stage1_row['train_cv_macro_f1']:.4f}")
print(f"|Train-CV - VAL| gap = {best_stage1_row['abs_train_val_gap']:.4f}")
print("Selection reasons:")
for line in stage1_reason_lines:
    print(" ", line)

save_text(
    os.path.join(REPORTS_DIR, "STAGE1_MODEL_SELECTION_REASON.txt"),
    "\n".join(stage1_reason_lines)
)

# ============================================================
# REFIT BEST CLOSED-SET MODEL + LEARN TAU
# ============================================================

print_block("STAGE 1C: REFIT BEST CLOSED-SET MODEL + LEARN TAU")

best_fs = fit_train_only_feature_pipeline(
    X_train_df=X_train_df,
    X_val_df=X_val_df,
    X_test_df=X_test_df,
    y_train=y_train,
    k=BEST_K,
    
)

pd.DataFrame({"selected_feature_best_model": best_fs["selected_features"]}).to_csv(
    os.path.join(TABLES_DIR, "BEST_STAGE1_SELECTED_FEATURES.csv"),
    index=False
)

print_block("BEST-MODEL FEATURE DETAILS")
print(f"Selected k = {BEST_K}")
print(f"Final total selected features = {len(best_fs['selected_features'])}")

selected_series = pd.Series(best_fs["selected_features"], dtype=str)
n_experimental_best = int(selected_series.str.startswith(EXPERIMENTAL_PREFIX).sum())
n_pseudozernike_best = int(selected_series.str.startswith(PSEUDOZERNIKE_PREFIX).sum())
n_zernike_best = int(selected_series.str.startswith(ZERNIKE_PREFIX).sum())

print(f"Experimental features = {n_experimental_best}")
print(f"PseudoZernike features = {n_pseudozernike_best}")
print(f"Zernike features = {n_zernike_best}")

pd.DataFrame({
    "feature_family": ["Experimental", "PseudoZernike", "Zernike"],
    "n_selected": [n_experimental_best, n_pseudozernike_best, n_zernike_best],
}).to_csv(
    os.path.join(TABLES_DIR, "BEST_STAGE1_SELECTED_FEATURE_FAMILIES.csv"),
    index=False,
)

scaler_stage1 = StandardScaler()
X_train_svm = scaler_stage1.fit_transform(best_fs["X_train_sel"])
X_val_svm   = scaler_stage1.transform(best_fs["X_val_sel"])
X_test_svm  = scaler_stage1.transform(best_fs["X_test_sel"])

svm_stage1 = SVC(
    kernel="rbf",
    C=BEST_SVM_PARAMS["C"],
    gamma=BEST_SVM_PARAMS["gamma"],
    class_weight=BEST_SVM_PARAMS["class_weight"],
    probability=False,
    decision_function_shape="ovr",
    random_state=RANDOM_STATE,
)
svm_stage1.fit(X_train_svm, y_train)

pred_train, margins_train, maxm_train, _, gap_train = raw_margin_details(svm_stage1, X_train_svm)
pred_val, margins_val, maxm_val, _, gap_val = raw_margin_details(svm_stage1, X_val_svm)
pred_test, margins_test, maxm_test, _, gap_test = raw_margin_details(svm_stage1, X_test_svm)

z_train, mu_train, sd_train = z_normalize_from_train(maxm_train, maxm_train)
z_val, _, _ = z_normalize_from_train(maxm_train, maxm_val)
z_test, _, _ = z_normalize_from_train(maxm_train, maxm_test)

closed_val_metrics = multiclass_metrics(y_val, pred_val)
closed_test_metrics = multiclass_metrics(y_test, pred_test)

tau_rows = []
TAU_MAP = {}
val_correct_mask = (pred_val == y_val)
max_validation_ccr = float(np.mean(val_correct_mask))

for target in TAU_TARGETS:
    tau, achieved_ccr, achieved_known_acceptance, feasible = pick_tau_for_target_ccr(
        z_val_known=z_val,
        known_correct_mask=val_correct_mask,
        target_ccr=target,
    )
    tau_rows.append({
        "Tau Target": target,
        "Threshold τ": tau,
        "Achieved Validation CCR": achieved_ccr,
        "Achieved Validation Known Acceptance": achieved_known_acceptance,
        "Maximum Possible Validation CCR": max_validation_ccr,
        "Feasible": feasible,
    })
    if feasible:
        TAU_MAP[target] = tau

tau_df = pd.DataFrame(tau_rows)
tau_df.to_csv(os.path.join(TABLES_DIR, "stage1_validation_ccr_tau_table.csv"), index=False)

infeasible_targets = tau_df.loc[~tau_df["Feasible"], "Tau Target"].tolist()
if infeasible_targets and REQUIRE_ALL_CCR_TARGETS_FEASIBLE:
    raise ValueError(
        "The following requested CCR targets are not achievable on the Stage-1 validation set: "
        f"{infeasible_targets}. Maximum possible validation CCR is {max_validation_ccr:.4f}. "
        "Reduce the upper target or improve the closed-set model before LOSO evaluation."
    )

TAU_TARGETS = [t for t in TAU_TARGETS if t in TAU_MAP]

print("\nClosed-set classification results:")
print(
    f"VAL  Accuracy={closed_val_metrics['Accuracy']:.4f} | "
    f"Balanced Accuracy={closed_val_metrics['Balanced Accuracy']:.4f} | "
    f"Macro-F1={closed_val_metrics['Macro-F1']:.4f}"
)
print(
    f"TEST Accuracy={closed_test_metrics['Accuracy']:.4f} | "
    f"Balanced Accuracy={closed_test_metrics['Balanced Accuracy']:.4f} | "
    f"Macro-F1={closed_test_metrics['Macro-F1']:.4f}"
)

print("\nValidation CCR-target threshold table:")
print(tau_df.to_string(index=False))

# ============================================================
# LOSO
# ============================================================

print_block("STAGE 2: LEAVE-ONE-TAXON-OUT OPEN-SET EVALUATION")

merged_class_names = sorted(y_raw_merged.unique().tolist())

per_tau_known_z = {t: [] for t in TAU_TARGETS}
per_tau_known_correct = {t: [] for t in TAU_TARGETS}
per_tau_unknown_z = {t: [] for t in TAU_TARGETS}

unknown_accept_detail_rows = []
unknown_reject_summary_rows = []
closed_fold_metrics_rows = []

for left_out_name in merged_class_names:
    print(f"\nLOSO fold: left out {left_out_name}")

    mask_unknown = (y_raw_merged == left_out_name)
    mask_known = ~mask_unknown

    X_known_df = X_raw.loc[mask_known].copy().reset_index(drop=True)
    y_known_labels_raw = y_raw_merged.loc[mask_known].copy().reset_index(drop=True)

    X_unknown_df = X_raw.loc[mask_unknown].copy().reset_index(drop=True)

    le_fold = LabelEncoder()
    y_known = le_fold.fit_transform(y_known_labels_raw)
    known_class_names_fold = le_fold.classes_

    idx_known = np.arange(len(X_known_df))
    idx_train_k, idx_temp_k, y_train_k, y_temp_k = train_test_split(
        idx_known,
        y_known,
        test_size=(1.0 - LOSO_TRAIN_FRAC),
        stratify=y_known,
        random_state=RANDOM_STATE,
    )

    temp_test_fraction = LOSO_TEST_FRAC / (LOSO_VAL_FRAC + LOSO_TEST_FRAC)
    idx_val_k, idx_test_k, y_val_k, y_test_k = train_test_split(
        idx_temp_k,
        y_temp_k,
        test_size=temp_test_fraction,
        stratify=y_temp_k,
        random_state=RANDOM_STATE,
    )

    X_train_df_k = X_known_df.iloc[idx_train_k].copy()
    X_val_df_k = X_known_df.iloc[idx_val_k].copy()
    X_test_known_df_k = X_known_df.iloc[idx_test_k].copy()

    fs_k = fit_train_only_feature_pipeline(
        X_train_df=X_train_df_k,
        X_val_df=X_val_df_k,
        X_test_df=X_test_known_df_k,
        y_train=y_train_k,
        k=BEST_K,
        
    )

    X_unknown_sel = transform_by_selected_features(
        X_df=X_unknown_df,
        vt=fs_k["vt"],
        selected_features=fs_k["selected_features"],
    )

    scaler_fold = StandardScaler()
    X_train_svm_k = scaler_fold.fit_transform(fs_k["X_train_sel"])
    X_test_known_svm_k = scaler_fold.transform(fs_k["X_test_sel"])
    X_unknown_svm_k = scaler_fold.transform(X_unknown_sel)

    svm_fold = SVC(
        kernel="rbf",
        C=BEST_SVM_PARAMS["C"],
        gamma=BEST_SVM_PARAMS["gamma"],
        class_weight=BEST_SVM_PARAMS["class_weight"],
        probability=False,
        decision_function_shape="ovr",
        random_state=RANDOM_STATE,
    )
    svm_fold.fit(X_train_svm_k, y_train_k)

    pred_train_k, margins_train_k, maxm_train_k, _, _ = raw_margin_details(svm_fold, X_train_svm_k)
    pred_test_k, margins_test_k, maxm_test_k, _, _ = raw_margin_details(svm_fold, X_test_known_svm_k)
    pred_unknown_k, margins_unknown_k, maxm_unknown_k, _, _ = raw_margin_details(svm_fold, X_unknown_svm_k)

    z_train_k, mu_k, sd_k = z_normalize_from_train(maxm_train_k, maxm_train_k)
    z_test_k, _, _ = z_normalize_from_train(maxm_train_k, maxm_test_k)
    z_unknown_k, _, _ = z_normalize_from_train(maxm_train_k, maxm_unknown_k)

    known_correct_mask = (pred_test_k == y_test_k)

    closed_fold_metrics_rows.append({
        "left_out_unknown_class": left_out_name,
        "known_test_accuracy": accuracy_score(y_test_k, pred_test_k),
        "known_test_balanced_accuracy": balanced_accuracy_score(y_test_k, pred_test_k),
        "known_test_macro_f1": f1_score(y_test_k, pred_test_k, average="macro"),
    })

    for target in TAU_TARGETS:
        tau = TAU_MAP[target]

        per_tau_known_z[target].append(z_test_k)
        per_tau_known_correct[target].append(known_correct_mask)
        per_tau_unknown_z[target].append(z_unknown_k)

        unknown_accept_mask = z_unknown_k >= tau
        accepted_unknown_preds = pred_unknown_k[unknown_accept_mask]

        for pred_idx in accepted_unknown_preds:
            pred_name = known_class_names_fold[pred_idx]
            unknown_accept_detail_rows.append({
                "Tau Target": target,
                "Left-out unknown class": left_out_name,
                "Accepted as known class": pred_name,
                "Count": 1,
            })

        unknown_reject_summary_rows.append({
            "Tau Target": target,
            "Left-out unknown class": left_out_name,
            "Unknown Rejection Percentage": float(np.mean(z_unknown_k < tau)),
            "Unknown Acceptance Percentage": float(np.mean(z_unknown_k >= tau)),
            "n_unknown_samples": len(z_unknown_k),
        })

# ============================================================
# AGGREGATE TAU RESULTS
# ============================================================

print_block("STAGE 3: AGGREGATE TAU RESULTS AND CHOOSE BEST TAU")

tau_eval_rows = []

for target in TAU_TARGETS:
    z_known_all = np.concatenate(per_tau_known_z[target]) if len(per_tau_known_z[target]) else np.array([])
    known_correct_all = np.concatenate(per_tau_known_correct[target]) if len(per_tau_known_correct[target]) else np.array([])
    z_unknown_all = np.concatenate(per_tau_unknown_z[target]) if len(per_tau_unknown_z[target]) else np.array([])

    tau = TAU_MAP[target]
    osr = compute_osr_metrics(z_known_all, known_correct_all, z_unknown_all, tau)

    row = {
        "Tau Target": target,
        "Threshold τ": tau,
        "Accuracy": closed_test_metrics["Accuracy"],
        "Balanced Accuracy": closed_test_metrics["Balanced Accuracy"],
        "Macro-F1": closed_test_metrics["Macro-F1"],
        **osr
    }

    y_binary = np.concatenate([
        np.zeros(len(z_known_all), dtype=int),
        np.ones(len(z_unknown_all), dtype=int)
    ])
    score_unknown = np.concatenate([
        -z_known_all,
        -z_unknown_all
    ])
    row["ROC AUC"] = float(roc_auc_score(y_binary, score_unknown))
    row["PR AUC"] = float(average_precision_score(y_binary, score_unknown))

    tau_eval_rows.append(row)

tau_results_df = pd.DataFrame(tau_eval_rows)
tau_results_df.to_csv(os.path.join(TABLES_DIR, "loso_tau_results_summary.csv"), index=False)
save_tau_tradeoff_plot(tau_results_df, os.path.join(PLOTS_DIR, "loso_tau_tradeoff"))

best_tau_row, tau_reason_lines = choose_best_tau_tradeoff(tau_results_df)

CHOSEN_TAU_TARGET = float(best_tau_row["Tau Target"])
CHOSEN_TAU = float(best_tau_row["Threshold τ"])

print("Tau selection reasons:")
for line in tau_reason_lines:
    print(" ", line)

save_text(
    os.path.join(REPORTS_DIR, "TAU_SELECTION_REASON.txt"),
    "\n".join(tau_reason_lines)
)

print("\nAll tau target results:")
print(tau_results_df.to_string(index=False))

print("\nChosen tau target:")
print(f"  Tau Target  = {CHOSEN_TAU_TARGET:.2f}")
print(f"  Threshold τ = {CHOSEN_TAU:.6f}")

print("\nMain final metrics at chosen tau:")
print(f"  Accuracy                              = {best_tau_row['Accuracy']:.4f}")
print(f"  Balanced Accuracy                     = {best_tau_row['Balanced Accuracy']:.4f}")
print(f"  Macro-F1                              = {best_tau_row['Macro-F1']:.4f}")
print(f"  Known Acceptance                      = {best_tau_row['Known Acceptance']:.4f}")
print(f"  CCR (Known correctly classified & accepted) = {best_tau_row['CCR (Known correctly classified & accepted)']:.4f}")
print(f"  Known Rejection Rate                  = {best_tau_row['Known Rejection Rate']:.4f}")
print(f"  Incorrectly classified but accepted knowns = {best_tau_row['Incorrectly classified but accepted knowns']:.4f}")
print(f"  Unknown Rejection Rate                = {best_tau_row['Unknown Rejection Rate']:.4f}")
print(f"  Unknown Acceptance Rate               = {best_tau_row['Unknown Acceptance Rate']:.4f}")
print(f"  Precision (Unknown detection)         = {best_tau_row['Precision (Unknown detection)']:.4f}")
print(f"  Recall (Unknown detection)            = {best_tau_row['Recall (Unknown detection)']:.4f}")
print(f"  Balanced OSR                          = {best_tau_row['Balanced OSR']:.4f}")
print(f"  F1 Unknown                            = {best_tau_row['F1 Unknown']:.4f}")
print(f"  ROC AUC                               = {best_tau_row['ROC AUC']:.4f}")
print(f"  PR AUC                                = {best_tau_row['PR AUC']:.4f}")

# ============================================================
# PRIMARY OUTPUTS FOR CHOSEN TAU
# ============================================================

print_block("STAGE 4: SAVE PRIMARY OUTPUTS FOR CHOSEN TAU")

z_known_final = np.concatenate(per_tau_known_z[CHOSEN_TAU_TARGET])
known_correct_final = np.concatenate(per_tau_known_correct[CHOSEN_TAU_TARGET])
z_unknown_final = np.concatenate(per_tau_unknown_z[CHOSEN_TAU_TARGET])

x_oscr, y_oscr = compute_oscr_curve(z_known_final, known_correct_final, z_unknown_final)
pd.DataFrame({
    "unknown_accept_rate": x_oscr,
    "ccr": y_oscr,
}).to_csv(os.path.join(TABLES_DIR, "OSCR_curve_points.csv"), index=False)
save_oscr_curve(x_oscr, y_oscr, tau_results_df, os.path.join(PLOTS_DIR, "OSCR_curve"))

y_binary_final = np.concatenate([
    np.zeros(len(z_known_final), dtype=int),
    np.ones(len(z_unknown_final), dtype=int),
])
score_unknown_final = np.concatenate([
    -z_known_final,
    -z_unknown_final,
])

save_roc_curve(y_binary_final, score_unknown_final, os.path.join(PLOTS_DIR, "unknown_vs_known_ROC"))
save_pr_curve(y_binary_final, score_unknown_final, os.path.join(PLOTS_DIR, "unknown_vs_known_PR"))

unknown_accept_detail_df = pd.DataFrame(unknown_accept_detail_rows)
unknown_accept_detail_df.to_csv(
    os.path.join(TABLES_DIR, "unknown_accepted_as_details_all_thresholds.csv"),
    index=False,
)

unknown_reject_df = pd.DataFrame(unknown_reject_summary_rows)
unknown_reject_df.to_csv(
    os.path.join(TABLES_DIR, "unknown_rejection_summary_all_thresholds.csv"),
    index=False,
)

# One publication-quality, percentage-labelled bar plot for every evaluated CCR target.
for target in TAU_TARGETS:
    target_df = unknown_reject_df[
        unknown_reject_df["Tau Target"] == target
    ].copy().sort_values("Left-out unknown class").reset_index(drop=True)

    if target_df.empty:
        continue

    target_df.to_csv(
        os.path.join(TABLES_DIR, f"unknown_rejection_by_taxon_targetCCR_{target:.2f}.csv"),
        index=False,
    )

    rejection_series = target_df.set_index(
        "Left-out unknown class"
    )["Unknown Rejection Percentage"]

    save_barplot_percent(
        rejection_series,
        title=f"Unknown rejection by left-out taxon (target CCR = {target:.2f})",
        ylabel="Unknown samples correctly rejected (%)",
        outstem=os.path.join(PLOTS_DIR, f"unknown_rejection_by_taxon_targetCCR_{target:.2f}"),
        figsize=(12, 6.5),
    )

closed_fold_metrics_df = pd.DataFrame(closed_fold_metrics_rows)
closed_fold_metrics_df.to_csv(os.path.join(TABLES_DIR, "loso_closed_fold_metrics.csv"), index=False)

# ============================================================
# FINAL REPORTS
# ============================================================

final_report_lines = []
final_report_lines.append("FINAL ANALYSIS SUMMARY")
final_report_lines.append("=" * 70)
final_report_lines.append("")
final_report_lines.append("FEATURE FAMILY SETTING")
final_report_lines.append(f"  Experimental used   = {USE_EXPERIMENTAL}")
final_report_lines.append(f"  PseudoZernike used  = {USE_PSEUDOZERNIKE}")
final_report_lines.append(f"  Zernike used        = {USE_ZERNIKE}")
final_report_lines.append(f"  Experimental count  = {len(experimental_cols)}")
final_report_lines.append(f"  PseudoZernike count = {len(pseudozernike_cols)}")
final_report_lines.append(f"  Total feature count = {len(feature_cols)}")
final_report_lines.append("")
final_report_lines.append("STAGE-1 BEST MODEL")
final_report_lines.append(f"  Best k = {BEST_K}")
final_report_lines.append(f"  Best final feature count = {int(best_stage1_row['final_feature_count'])}")
final_report_lines.append(f"  Experimental features = {n_experimental_best}")
final_report_lines.append(f"  PseudoZernike features = {n_pseudozernike_best}")
final_report_lines.append(f"  Zernike features = {n_zernike_best}")
final_report_lines.append(f"  Best Stage-1 SVM params = {BEST_SVM_PARAMS}")
final_report_lines.append(f"  Train-CV Macro-F1 = {best_stage1_row['train_cv_macro_f1']:.4f}")
final_report_lines.append(f"  VAL Macro-F1      = {best_stage1_row['val_macro_f1']:.4f}")
final_report_lines.append(f"  TEST Macro-F1     = {best_stage1_row['test_macro_f1']:.4f}")
final_report_lines.append("")
final_report_lines.append("CLOSED-SET PERFORMANCE")
final_report_lines.append(f"  VAL  Accuracy = {closed_val_metrics['Accuracy']:.4f}")
final_report_lines.append(f"  VAL  Balanced Accuracy = {closed_val_metrics['Balanced Accuracy']:.4f}")
final_report_lines.append(f"  VAL  Macro-F1 = {closed_val_metrics['Macro-F1']:.4f}")
final_report_lines.append(f"  TEST Accuracy = {closed_test_metrics['Accuracy']:.4f}")
final_report_lines.append(f"  TEST Balanced Accuracy = {closed_test_metrics['Balanced Accuracy']:.4f}")
final_report_lines.append(f"  TEST Macro-F1 = {closed_test_metrics['Macro-F1']:.4f}")
final_report_lines.append("")
final_report_lines.append("CHOSEN OPEN-SET OPERATING POINT")
final_report_lines.append(f"  Tau target = {CHOSEN_TAU_TARGET:.2f}")
final_report_lines.append(f"  Threshold τ = {CHOSEN_TAU:.6f}")
final_report_lines.append(f"  Known Acceptance = {best_tau_row['Known Acceptance']:.4f}")
final_report_lines.append(f"  CCR = {best_tau_row['CCR (Known correctly classified & accepted)']:.4f}")
final_report_lines.append(f"  Unknown Rejection Rate = {best_tau_row['Unknown Rejection Rate']:.4f}")
final_report_lines.append(f"  Unknown Acceptance Rate = {best_tau_row['Unknown Acceptance Rate']:.4f}")
final_report_lines.append(f"  Balanced OSR = {best_tau_row['Balanced OSR']:.4f}")
final_report_lines.append(f"  F1 Unknown = {best_tau_row['F1 Unknown']:.4f}")
final_report_lines.append(f"  ROC AUC = {best_tau_row['ROC AUC']:.4f}")
final_report_lines.append(f"  PR AUC = {best_tau_row['PR AUC']:.4f}")
final_report_lines.append("")

save_text(
    os.path.join(REPORTS_DIR, "FINAL_SUMMARY_REPORT.txt"),
    "\n".join(final_report_lines)
)

final_payload = {
    "feature_family_setting": {
        "use_experimental": USE_EXPERIMENTAL,
        "use_pseudozernike": USE_PSEUDOZERNIKE,
        "use_zernike": USE_ZERNIKE,
        "experimental_count_found": len(experimental_cols),
        "pseudozernike_count_found": len(pseudozernike_cols),
        "zernike_count_found": len(zernike_cols),
        "total_selected_feature_count": len(feature_cols),
    },
    "best_stage1_model": {
        "best_k": BEST_K,
        "best_final_feature_count": int(best_stage1_row["final_feature_count"]),
        "experimental_features": n_experimental_best,
        "pseudozernike_features": n_pseudozernike_best,
        "zernike_features": n_zernike_best,
        "best_svm_params": BEST_SVM_PARAMS,
        "train_cv_macro_f1": float(best_stage1_row["train_cv_macro_f1"]),
        "val_macro_f1": float(best_stage1_row["val_macro_f1"]),
        "test_macro_f1": float(best_stage1_row["test_macro_f1"]),
    },
    "closed_set_metrics": {
        "val": closed_val_metrics,
        "test": closed_test_metrics,
    },
    "chosen_tau": {
        "tau_target": CHOSEN_TAU_TARGET,
        "threshold_tau": CHOSEN_TAU,
        "metrics": dict(best_tau_row),
    },
    "output_root": OUT_ROOT,
}
save_json(os.path.join(REPORTS_DIR, "FINAL_SUMMARY_REPORT.json"), final_payload)

print_block("DONE")
print(f"All outputs saved to:\n{OUT_ROOT}")
