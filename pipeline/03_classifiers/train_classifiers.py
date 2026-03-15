"""
train_classifiers.py

Train COPD (and other respiratory) classifiers from HeAR patient-level embeddings.

Pipeline: StandardScaler → LogisticRegression(C=1.0, class_weight='balanced')
Evaluation: StratifiedGroupKFold(n_splits=5), patient as group, OOF AUROC + bootstrap 95% CI

Datasets: ICBHI (Portugal/Greece), Fraiwan/KAUH (Jordan), RDTR (Turkey)
Embeddings: 512-dim HeAR, averaged per patient across all recordings.
"""
import os

import numpy as np
import pickle
import json
from pathlib import Path
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ─── Paths ────────────────────────────────────────────────────────────────────
# Set DATA_DIR env var to your local embeddings directory, or edit this path
DATA = Path(os.environ.get("DATA_DIR", "data/embeddings"))
ICBHI_NPZ   = DATA / "icbhi_full/icbhi_embeddings_cache.npz"
FRAIWAN_NPZ = DATA / "fraiwan_embeddings.npz"
RDTR_NPZ    = DATA / "rdtr_embeddings.npz"
ICBHI_DIAG  = DATA / "icbhi_full/ICBHI_Challenge_diagnosis.txt"
OUT_DIR     = Path(__file__).parent / "models"
OUT_DIR.mkdir(exist_ok=True)


# ─── Bootstrap CI ─────────────────────────────────────────────────────────────
def bootstrap_auroc_ci(y_true, y_score, n=2000, ci=0.95, seed=0):
    rng = np.random.RandomState(seed)
    scores = []
    for _ in range(n):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(roc_auc_score(y_true[idx], y_score[idx]))
    lo = np.percentile(scores, (1 - ci) / 2 * 100)
    hi = np.percentile(scores, (1 + ci) / 2 * 100)
    return float(lo), float(hi)


# ─── Load embeddings ──────────────────────────────────────────────────────────
def load_all_embeddings():
    """
    Returns (X, labels, groups) arrays for the composite dataset.
    One row = one patient. group = unique patient ID string.
    Labels are harmonised to lowercase.
    """
    # ICBHI — recording-level → patient mean
    icbhi = np.load(ICBHI_NPZ, allow_pickle=True)
    diag_map = {}
    with open(ICBHI_DIAG) as f:
        for line in f:
            p = line.strip().split('\t')
            if len(p) >= 2:
                diag_map[p[0]] = p[1].lower()

    icbhi_patient = {}
    for key, emb in zip(icbhi['keys'], icbhi['embeddings']):
        pid  = str(key).split('_')[0]
        diag = diag_map.get(pid, 'unknown')
        icbhi_patient.setdefault(pid, {'embs': [], 'diag': diag})['embs'].append(emb)

    icbhi_X     = np.array([np.mean(v['embs'], axis=0) for v in icbhi_patient.values()])
    icbhi_diags = np.array([v['diag']                  for v in icbhi_patient.values()])
    icbhi_pids  = np.array([f"ICBHI_{pid}"             for pid in icbhi_patient.keys()])

    # Fraiwan — 3 WAVs per patient (B/D/E), mean all 3
    fr = np.load(FRAIWAN_NPZ, allow_pickle=True)
    fr_pids_raw = np.array([str(x) for x in fr['patient_ids']])
    fr_patient  = {}
    for pid, emb, diag in zip(fr_pids_raw, fr['embeddings'], fr['diagnoses']):
        fr_patient.setdefault(pid, {'embs': [], 'diag': str(diag).lower()})['embs'].append(emb)

    fraiwan_X     = np.array([np.mean(v['embs'], axis=0) for v in fr_patient.values()])
    fraiwan_diags = np.array([v['diag']                  for v in fr_patient.values()])
    fraiwan_pids  = np.array([f"FR_{pid}"                for pid in fr_patient.keys()])

    # RDTR — multiple channels per patient, mean all channels
    rd = np.load(RDTR_NPZ, allow_pickle=True)
    rd_patient = {}
    for pid, emb in zip(rd['patient_ids'], rd['embeddings']):
        pid_str = str(pid)
        rd_patient.setdefault(pid_str, {'embs': []})['embs'].append(emb)

    rdtr_X     = np.array([np.mean(v['embs'], axis=0) for v in rd_patient.values()])
    rdtr_diags = np.array(['copd']                     * len(rd_patient))
    rdtr_pids  = np.array([f"RDTR_{pid}"               for pid in rd_patient.keys()])

    X      = np.vstack([icbhi_X,    fraiwan_X,    rdtr_X])
    labels = np.concatenate([icbhi_diags, fraiwan_diags, rdtr_diags])
    groups = np.concatenate([icbhi_pids,  fraiwan_pids,  rdtr_pids])

    return X, labels, groups


# ─── Train one classifier ─────────────────────────────────────────────────────
def train_task(name, X, labels, groups, pos_label, neg_labels):
    """
    pos_label  : string label for positive class (e.g. 'copd')
    neg_labels : list of strings for negative class (e.g. ['healthy','normal'])
    Returns result dict + fitted pipeline.
    """
    mask = np.array([(l == pos_label or l in neg_labels) for l in labels])
    Xt   = X[mask]
    yt   = (labels[mask] == pos_label).astype(int)
    gt   = groups[mask]

    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('lr',     LogisticRegression(C=1.0, class_weight='balanced',
                                       max_iter=2000, solver='lbfgs')),
    ])
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # OOF predictions
    oof_proba = cross_val_predict(clf, Xt, yt, groups=gt, cv=cv, method='predict_proba')[:, 1]
    oof_auroc = roc_auc_score(yt, oof_proba)
    ci_lo, ci_hi = bootstrap_auroc_ci(yt, oof_proba)

    # Fold-level AUROCs
    fold_aurocs = []
    for train_idx, test_idx in cv.split(Xt, yt, gt):
        clf_fold = Pipeline([
            ('scaler', StandardScaler()),
            ('lr',     LogisticRegression(C=1.0, class_weight='balanced',
                                           max_iter=2000, solver='lbfgs')),
        ])
        clf_fold.fit(Xt[train_idx], yt[train_idx])
        p = clf_fold.predict_proba(Xt[test_idx])[:, 1]
        if len(np.unique(yt[test_idx])) == 2:
            fold_aurocs.append(roc_auc_score(yt[test_idx], p))

    # Fit final model on ALL data
    clf.fit(Xt, yt)

    # Dataset breakdown
    breakdown = {}
    for ds_prefix in ['ICBHI', 'FR', 'RDTR']:
        ds_mask = np.array([g.startswith(ds_prefix) for g in gt])
        n_pos = int(yt[ds_mask].sum())
        n_neg = int((1 - yt[ds_mask]).sum())
        if n_pos + n_neg > 0:
            breakdown[ds_prefix] = {'n_pos': n_pos, 'n_neg': n_neg, 'n_total': n_pos + n_neg}

    result = {
        'task':          name,
        'pos_label':     pos_label,
        'neg_labels':    neg_labels,
        'n_positive':    int(yt.sum()),
        'n_negative':    int((1 - yt).sum()),
        'n_total':       int(len(yt)),
        'oof_auroc':     round(oof_auroc, 4),
        'ci_95':         [round(ci_lo, 4), round(ci_hi, 4)],
        'fold_auroc_mean': round(float(np.mean(fold_aurocs)), 4),
        'fold_auroc_std':  round(float(np.std(fold_aurocs)), 4),
        'n_folds':       len(fold_aurocs),
        'dataset_breakdown': breakdown,
    }
    return result, clf


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Loading embeddings...")
    X, labels, groups = load_all_embeddings()
    print(f"  {len(X)} patients, {X.shape[1]}-dim embeddings")

    from collections import Counter
    label_counts = Counter(labels)
    print("  Label distribution:", dict(sorted(label_counts.items())))

    tasks = [
        # (name,                    pos_label,        neg_labels)
        ('copd_vs_healthy',        'copd',            ['healthy', 'normal']),
        ('copd_vs_all',            'copd',            [l for l in label_counts if l != 'copd']),
        ('pneumonia_vs_healthy',   'pneumonia',       ['healthy', 'normal']),
        ('asthma_vs_healthy',      'asthma',          ['healthy', 'normal']),
        ('heart_failure_vs_healthy','heart_failure',  ['healthy', 'normal']),
    ]

    all_results = []
    print("\nTraining classifiers...")
    for name, pos, neg in tasks:
        # Check if enough data
        mask = np.array([(l == pos or l in neg) for l in labels])
        yt   = (labels[mask] == pos).astype(int)
        if yt.sum() < 5 or (1 - yt).sum() < 5:
            print(f"  SKIP {name}: not enough samples (pos={yt.sum()}, neg={(1-yt).sum()})")
            continue

        print(f"  Training {name} (pos={yt.sum()}, neg={(1-yt).sum()})...")
        result, model = train_task(name, X, labels, groups, pos, neg)
        all_results.append(result)

        # Save model
        model_path = OUT_DIR / f"{name}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        print(f"    OOF AUROC: {result['oof_auroc']} [{result['ci_95'][0]}–{result['ci_95'][1]}] "
              f"  fold: {result['fold_auroc_mean']}±{result['fold_auroc_std']}")

    # Save results JSON
    out_path = Path(__file__).parent.parent / "04_results" / "classifier_performance.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print table
    print("\n" + "="*90)
    print(f"{'Task':<30} {'OOF AUROC':<12} {'95% CI':<18} {'Fold':<14} {'n+':<6} {'n-':<6} {'N'}")
    print("-"*90)
    for r in all_results:
        ci = f"[{r['ci_95'][0]:.3f}–{r['ci_95'][1]:.3f}]"
        fold = f"{r['fold_auroc_mean']:.3f}±{r['fold_auroc_std']:.3f}"
        print(f"{r['task']:<30} {r['oof_auroc']:<12.3f} {ci:<18} {fold:<14} "
              f"{r['n_positive']:<6} {r['n_negative']:<6} {r['n_total']}")
