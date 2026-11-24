# -*- coding: utf-8 -*-
"""Builder for d_notebook_01_data_prep_ecfp_xgboost.ipynb (Direction D Stage 1, CPU ~0 CU)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CELLS = []
_ID = 0


def _next_id():
    global _ID
    _ID += 1
    return f"d-nb01-{_ID:02d}"


def md(src):
    CELLS.append({"cell_type": "markdown", "id": _next_id(), "metadata": {}, "source": src})


def code(src):
    CELLS.append({"cell_type": "code", "id": _next_id(), "execution_count": None, "metadata": {}, "outputs": [], "source": src})


md("""# D-01 — TDC ADMET & ChEMBL Data Prep, Scaffold-OOD Diagnostics & Tabular Baselines

**Resource tier: CPU runtime (~0 CU).**
- Downloads & verifies official **TDC `ADMET_Group`** (22 tasks) + **ChEMBL IC50** target case studies (`CHEMBL203` EGFR, `CHEMBL4822` BACE1, `CHEMBL240` hERG).
- Reproduces exact PyTDC Bemis-Murcko `create_scaffold_split(..., seed=seed, frac=[0.875, 0.125, 0.0])` + 5-fold scaffold CV+ groups.
- Computes 2,231-dim molecular descriptors (`2048 ECFP4 + 167 MACCS + 16 RDKit 2D`) and exact nearest-neighbor Tanimoto OOD distances.
- Fits & caches **Random Forest** (tree-std UQ) and **XGBoost** (point + `reg:quantileerror` CQR quantiles) across `split_iid`, `split_scaffold`, and `5-fold scaffold CV+`.
""")

code('''# Cell 1 — Setup, workspace resolution & scope configuration [CPU, ~0 CU]
import os, sys, json, re, gc, time, warnings
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# Locate Direction D modules (supports local repo, Colab cloned repo, or Drive folder)
IN_COLAB = "google.colab" in sys.modules
def _find_module_dir():
    cands = [
        Path.cwd(),
        Path.cwd() / "direction_d_admet_conformal",
        Path("/content"),
        Path("/content/direction_d_admet_conformal"),
        Path("/content/drive/MyDrive/direction_d_admet_conformal"),
        Path("/content/drive/MyDrive/Colab Notebooks"),
        Path("/content/drive/MyDrive/Colab Notebooks/direction_d_admet_conformal"),
        Path("/content/drive/MyDrive/flip2_benchmark/direction_d_admet"),
        Path("/content/drive/MyDrive/tdc_admet_benchmark"),
    ]
    for cand in cands:
        if (cand / "admet_core.py").exists():
            return cand
    return None

_mod_dir = _find_module_dir()
if _mod_dir is None and IN_COLAB:
    from google.colab import drive
    if not Path("/content/drive/MyDrive").exists():
        drive.mount("/content/drive")
    _mod_dir = _find_module_dir()

if _mod_dir is None:
    raise FileNotFoundError(
        "Cannot find admet_core.py! Please upload admet_core.py (and conformal_ood.py, gin_gps_model.py) "
        "to /content/ (Colab sidebar Files), /content/drive/MyDrive/direction_d_admet_conformal/, "
        "or your Drive workspace folder, then re-run Cell 1."
    )
if str(_mod_dir) not in sys.path:
    sys.path.insert(0, str(_mod_dir))

try:
    import rdkit, xgboost  # noqa: F401
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "rdkit", "xgboost"])

from admet_core import (
    ADMET_TASKS, CHEMBL_TARGETS, ALL_TASKS,
    resolve_workspace, download_tdc_admet_archive, fetch_chembl_target_ic50,
    canonicalize_smiles, get_scaffold_smiles, create_scaffold_split_pytdc,
    scaffold_group_kfold, featurize_ecfp_rdkit, compute_nn_tanimoto_distance,
    compute_task_metrics, atomic_save_npz, sha256_text, confine_path,
)

BASE_DIR = resolve_workspace(in_colab=IN_COLAB)
RAW_DIR  = (BASE_DIR / "data" / "raw").resolve()
PROC_DIR = (BASE_DIR / "data" / "processed").resolve()
FEAT_DIR = (BASE_DIR / "cache" / "features").resolve()
PRED_DIR = (BASE_DIR / "cache" / "predictions").resolve()
TAB_DIR  = (BASE_DIR / "outputs" / "tables").resolve()
for d in (RAW_DIR, PROC_DIR, FEAT_DIR, PRED_DIR, TAB_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---- Configuration ----
RUN_SCOPE = "all"   # "pilot" (6 TDC + 2 ChEMBL) | "all" (22 TDC + 3 ChEMBL)
SEEDS = [1, 2, 3]
INCLUDE_CHEMBL = True
OVERWRITE_PRED_CACHE = False

if RUN_SCOPE == "pilot":
    ACTIVE_TDC = [k for k, v in ADMET_TASKS.items() if v["pilot"]]
    ACTIVE_CHEMBL = [k for k, v in CHEMBL_TARGETS.items() if v["pilot"]] if INCLUDE_CHEMBL else []
elif RUN_SCOPE == "all":
    ACTIVE_TDC = list(ADMET_TASKS.keys())
    ACTIVE_CHEMBL = list(CHEMBL_TARGETS.keys()) if INCLUDE_CHEMBL else []
else:
    raise ValueError(f"Unknown RUN_SCOPE: {RUN_SCOPE}")

ACTIVE_TASKS = ACTIVE_TDC + ACTIVE_CHEMBL
print(f"[CFG] BASE_DIR={BASE_DIR}")
print(f"[CFG] RUN_SCOPE={RUN_SCOPE} | {len(ACTIVE_TDC)} TDC + {len(ACTIVE_CHEMBL)} ChEMBL = {len(ACTIVE_TASKS)} tasks | seeds={SEEDS}")''')

code('''# Cell 2 — Fetch real TDC ADMET_Group + ChEMBL targets & build global molecule registry [CPU, ~0 CU]
group_dir, tdc_meta = download_tdc_admet_archive(RAW_DIR)

task_frames = {}
all_smiles_records = []

for task in ACTIVE_TDC:
    tdir = confine_path(group_dir, group_dir / task)
    tv = pd.read_csv(tdir / "train_val.csv")
    te = pd.read_csv(tdir / "test.csv")
    assert {"Drug", "Y"}.issubset(tv.columns) and {"Drug", "Y"}.issubset(te.columns)
    tv = tv.dropna(subset=["Drug", "Y"]).reset_index(drop=True)
    te = te.dropna(subset=["Drug", "Y"]).reset_index(drop=True)
    tv["canonical_smiles"] = tv["Drug"].map(canonicalize_smiles)
    te["canonical_smiles"] = te["Drug"].map(canonicalize_smiles)
    tv = tv.dropna(subset=["canonical_smiles"]).reset_index(drop=True)
    te = te.dropna(subset=["canonical_smiles"]).reset_index(drop=True)
    tv["mol_hash"] = tv["canonical_smiles"].map(sha256_text)
    te["mol_hash"] = te["canonical_smiles"].map(sha256_text)
    task_frames[task] = (tv, te)
    all_smiles_records.extend(zip(tv["canonical_smiles"], tv["mol_hash"]))
    all_smiles_records.extend(zip(te["canonical_smiles"], te["mol_hash"]))

for task in ACTIVE_CHEMBL:
    spec = CHEMBL_TARGETS[task]
    tv, te = fetch_chembl_target_ic50(task, spec["chembl_id"], RAW_DIR, max_records=2000)
    tv["canonical_smiles"] = tv["Drug"].map(canonicalize_smiles)
    te["canonical_smiles"] = te["Drug"].map(canonicalize_smiles)
    tv = tv.dropna(subset=["canonical_smiles"]).reset_index(drop=True)
    te = te.dropna(subset=["canonical_smiles"]).reset_index(drop=True)
    tv["mol_hash"] = tv["canonical_smiles"].map(sha256_text)
    te["mol_hash"] = te["canonical_smiles"].map(sha256_text)
    task_frames[task] = (tv, te)
    all_smiles_records.extend(zip(tv["canonical_smiles"], tv["mol_hash"]))
    all_smiles_records.extend(zip(te["canonical_smiles"], te["mol_hash"]))

uniq_df = (
    pd.DataFrame(all_smiles_records, columns=["canonical_smiles", "mol_hash"])
    .drop_duplicates(subset=["mol_hash"])
    .sort_values("mol_hash")
    .reset_index(drop=True)
)
uniq_sha = sha256_text("".join(uniq_df["mol_hash"].tolist()))
uniq_df.to_csv(PROC_DIR / "unique_molecules.csv", index=False)

manifest = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "run_scope": RUN_SCOPE,
    "seeds": SEEDS,
    "tdc_archive": tdc_meta,
    "n_unique_molecules": int(len(uniq_df)),
    "unique_sha256": uniq_sha,
    "tasks": {
        t: {
            **ALL_TASKS[t],
            "n_train_val": int(len(task_frames[t][0])),
            "n_test": int(len(task_frames[t][1])),
        }
        for t in ACTIVE_TASKS
    },
}
(PROC_DIR / "admet_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"[SAVED] unique_molecules.csv ({len(uniq_df)} unique SMILES, sha={uniq_sha[:12]}) & admet_manifest.json")''')

code('''# Cell 3 — Featurize global ECFP4+MACCS+RDKit16 cache & scaffold OOD shift diagnostics [CPU, ~0 CU]
feat_cache = FEAT_DIR / f"ecfp_rdkit__global__{uniq_sha[:12]}.npz"
if feat_cache.exists():
    print(f"[SKIP] Global feature cache exists: {feat_cache.name}")
    fc = np.load(feat_cache)
    X_global = fc["X_feat"]
    ecfp_global = fc["ecfp_bits"]
    hash_to_idx = {h: i for i, h in enumerate(fc["mol_hashes"].astype(str))}
else:
    t0 = time.time()
    X_global, ecfp_global, valid_m = featurize_ecfp_rdkit(uniq_df["canonical_smiles"].tolist())
    assert valid_m.all(), f"{int((~valid_m).sum())} invalid canonical SMILES"
    atomic_save_npz(
        feat_cache,
        X_feat=X_global,
        ecfp_bits=ecfp_global,
        mol_hashes=np.asarray(uniq_df["mol_hash"].tolist(), dtype=np.str_),
    )
    hash_to_idx = {h: i for i, h in enumerate(uniq_df["mol_hash"].tolist())}
    print(f"[SAVED] {feat_cache.name} | shape={X_global.shape} | {time.time()-t0:.1f}s")

diag_rows = []
for task in ACTIVE_TASKS:
    tv, te = task_frames[task]
    tv_scafs = set(tv["canonical_smiles"].map(get_scaffold_smiles))
    te_scafs = set(te["canonical_smiles"].map(get_scaffold_smiles))
    overlap = len(tv_scafs & te_scafs)

    tv_idx = [hash_to_idx[h] for h in tv["mol_hash"]]
    te_idx = [hash_to_idx[h] for h in te["mol_hash"]]
    d_nn = compute_nn_tanimoto_distance(ecfp_global[tv_idx], ecfp_global[te_idx])
    sp = ALL_TASKS[task]
    diag_rows.append({
        "task": task,
        "source": "ChEMBL" if task in CHEMBL_TARGETS else "TDC",
        "category": sp.get("category", sp.get("name", "Potency")),
        "task_type": sp["task_type"],
        "metric": sp["metric"],
        "n_train_val": int(len(tv)),
        "n_test": int(len(te)),
        "pos_rate_tv": float(tv["Y"].mean()) if sp["task_type"] == "classification" else float("nan"),
        "pos_rate_test": float(te["Y"].mean()) if sp["task_type"] == "classification" else float("nan"),
        "n_scaffolds_tv": int(len(tv_scafs)),
        "n_scaffolds_test": int(len(te_scafs)),
        "shared_scaffolds": int(overlap),
        "nn_tanimoto_dist_mean": float(np.mean(d_nn)),
        "nn_tanimoto_dist_median": float(np.median(d_nn)),
        "frac_far_ood_ge065": float(np.mean(d_nn >= 0.65)),
    })

diag_df = pd.DataFrame(diag_rows)
diag_df.to_csv(TAB_DIR / "scaffold_shift_diagnostics.csv", index=False)
print(f"[SAVED] scaffold_shift_diagnostics.csv ({len(diag_df)} tasks)")
diag_df[["task", "task_type", "n_train_val", "n_test", "shared_scaffolds", "nn_tanimoto_dist_mean", "frac_far_ood_ge065"]]''')

code('''# Cell 4 — Fit & cache Tabular OOD Baselines (RandomForest UQ + XGBoost + CQR Quantiles) [CPU, ~0 CU]
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

def _fit_predict_rf(X_tr, y_tr, X_ev_list, task_type, seed):
    if task_type == "regression":
        clf = RandomForestRegressor(
            n_estimators=100, max_features="sqrt", min_samples_leaf=2, n_jobs=-1, random_state=seed
        ).fit(X_tr, y_tr)
        means, stds = [], []
        for Xe in X_ev_list:
            tree_preds = np.stack([t.predict(Xe) for t in clf.estimators_], axis=0)
            means.append(tree_preds.mean(axis=0))
            stds.append(tree_preds.std(axis=0))
        return means, stds
    clf = RandomForestClassifier(
        n_estimators=120, max_features="sqrt", class_weight="balanced_subsample", n_jobs=-1, random_state=seed
    ).fit(X_tr, y_tr.astype(int))
    means, stds = [], []
    for Xe in X_ev_list:
        tree_probs = np.stack([t.predict_proba(Xe)[:, 1] for t in clf.estimators_], axis=0)
        means.append(tree_probs.mean(axis=0))
        stds.append(tree_probs.std(axis=0))
    return means, stds

def _fit_predict_xgb(X_tr, y_tr, X_ev_list, task_type, seed):
    if task_type == "regression":
        model = xgb.XGBRegressor(
            n_estimators=250, max_depth=6, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.35, reg_lambda=2.0,
            tree_method="hist", n_jobs=-1, random_state=seed, verbosity=0,
        )
        model.fit(X_tr, y_tr)
        return [model.predict(Xe) for Xe in X_ev_list]
    pos = float((y_tr == 1).sum()); neg = float((y_tr == 0).sum())
    spw = min(10.0, max(1.0, neg / max(1.0, pos)))
    model = xgb.XGBClassifier(
        n_estimators=250, max_depth=5, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.35, reg_lambda=2.0,
        scale_pos_weight=spw, tree_method="hist", eval_metric="logloss",
        n_jobs=-1, random_state=seed, verbosity=0,
    )
    model.fit(X_tr, y_tr.astype(int))
    return [model.predict_proba(Xe)[:, 1] for Xe in X_ev_list]

def _fit_predict_xgb_cqr(X_tr, y_tr, X_cal, X_te, alpha, seed):
    q_lo_m = xgb.XGBRegressor(
        n_estimators=220, max_depth=5, learning_rate=0.05, subsample=0.85,
        colsample_bytree=0.35, objective="reg:quantileerror", quantile_alpha=alpha / 2.0,
        tree_method="hist", n_jobs=-1, random_state=seed, verbosity=0,
    ).fit(X_tr, y_tr)
    q_hi_m = xgb.XGBRegressor(
        n_estimators=220, max_depth=5, learning_rate=0.05, subsample=0.85,
        colsample_bytree=0.35, objective="reg:quantileerror", quantile_alpha=1.0 - alpha / 2.0,
        tree_method="hist", n_jobs=-1, random_state=seed, verbosity=0,
    ).fit(X_tr, y_tr)
    c_lo, c_hi = q_lo_m.predict(X_cal), q_hi_m.predict(X_cal)
    t_lo, t_hi = q_lo_m.predict(X_te), q_hi_m.predict(X_te)
    return np.minimum(c_lo, c_hi), np.maximum(c_lo, c_hi), np.minimum(t_lo, t_hi), np.maximum(t_lo, t_hi)

metric_rows = []
t_start = time.time()

for task in ACTIVE_TASKS:
    tv, te = task_frames[task]
    spec = ALL_TASKS[task]
    ttype = spec["task_type"]
    te_idx = [hash_to_idx[h] for h in te["mol_hash"]]
    X_te_raw = X_global[te_idx]
    y_te = te["Y"].values.astype(np.float64)

    for seed in SEEDS:
        out_npz = PRED_DIR / f"tabular__{task}__seed{seed}.npz"
        if out_npz.exists() and not OVERWRITE_PRED_CACHE:
            d = np.load(out_npz)
            rf_m = compute_task_metrics(y_te, d["rf_scaf_test"], ttype)
            xg_m = compute_task_metrics(y_te, d["xgb_scaf_test"], ttype)
            metric_rows.append({"task": task, "seed": seed, "model": "ecfp4_rdkit_rf", **rf_m})
            metric_rows.append({"task": task, "seed": seed, "model": "ecfp4_rdkit_xgb", **xg_m})
            continue

        # 1) Official scaffold train/valid split (87.5% / 12.5%)
        sp = create_scaffold_split_pytdc(tv, seed=seed, frac=(0.875, 0.125, 0.0), entity="Drug")
        tr_df, va_df = sp["train"], sp["valid"]
        tr_idx = [hash_to_idx[h] for h in tr_df["mol_hash"]]
        va_idx = [hash_to_idx[h] for h in va_df["mol_hash"]]
        tv_idx = [hash_to_idx[h] for h in tv["mol_hash"]]

        # Standardize only the trailing 16 RDKit continuous descriptors on train
        X_tr_raw = X_global[tr_idx].copy()
        X_va_raw = X_global[va_idx].copy()
        X_te_s = X_te_raw.copy()
        scaler = StandardScaler().fit(X_tr_raw[:, 2215:])
        X_tr_raw[:, 2215:] = scaler.transform(X_tr_raw[:, 2215:])
        X_va_raw[:, 2215:] = scaler.transform(X_va_raw[:, 2215:])
        X_te_s[:, 2215:] = scaler.transform(X_te_s[:, 2215:])
        y_tr = tr_df["Y"].values.astype(np.float64)
        y_va = va_df["Y"].values.astype(np.float64)

        # 2) Random IID holdout (15% of train) to demonstrate IID calibration collapse under scaffold shift
        rng = np.random.RandomState(seed * 100 + 7)
        perm = rng.permutation(len(X_tr_raw))
        n_iid_cal = max(15, int(round(0.15 * len(X_tr_raw))))
        iid_cal_i, iid_fit_i = perm[:n_iid_cal], perm[n_iid_cal:]

        [rf_iid_cal, rf_iid_test], _ = _fit_predict_rf(X_tr_raw[iid_fit_i], y_tr[iid_fit_i], [X_tr_raw[iid_cal_i], X_te_s], ttype, seed)
        xgb_iid_cal, xgb_iid_test = _fit_predict_xgb(X_tr_raw[iid_fit_i], y_tr[iid_fit_i], [X_tr_raw[iid_cal_i], X_te_s], ttype, seed)

        # 3) Full train -> scaffold valid + test
        [rf_scaf_cal, rf_scaf_test], [rf_sigma_cal, rf_sigma_test] = _fit_predict_rf(X_tr_raw, y_tr, [X_va_raw, X_te_s], ttype, seed)
        xgb_scaf_cal, xgb_scaf_test = _fit_predict_xgb(X_tr_raw, y_tr, [X_va_raw, X_te_s], ttype, seed)

        # 4) 5-fold scaffold CV+ on train_val (regression)
        X_tv_raw = X_global[tv_idx].copy()
        scaler_tv = StandardScaler().fit(X_tv_raw[:, 2215:])
        X_tv_raw[:, 2215:] = scaler_tv.transform(X_tv_raw[:, 2215:])
        X_te_tv = X_te_raw.copy()
        X_te_tv[:, 2215:] = scaler_tv.transform(X_te_tv[:, 2215:])
        y_tv = tv["Y"].values.astype(np.float64)
        fold_ids = scaffold_group_kfold(tv["canonical_smiles"].tolist(), n_splits=5, seed=seed)

        rf_cv_oof = np.zeros(len(y_tv), dtype=np.float64)
        xgb_cv_oof = np.zeros(len(y_tv), dtype=np.float64)
        rf_cv_test = np.zeros((5, len(y_te)), dtype=np.float64)
        xgb_cv_test = np.zeros((5, len(y_te)), dtype=np.float64)
        if ttype == "regression":
            for k in range(5):
                m_va = fold_ids == k
                m_tr = ~m_va
                [rf_o, rf_t], _ = _fit_predict_rf(X_tv_raw[m_tr], y_tv[m_tr], [X_tv_raw[m_va], X_te_tv], ttype, seed + k)
                xg_o, xg_t = _fit_predict_xgb(X_tv_raw[m_tr], y_tv[m_tr], [X_tv_raw[m_va], X_te_tv], ttype, seed + k)
                rf_cv_oof[m_va] = rf_o; rf_cv_test[k] = rf_t
                xgb_cv_oof[m_va] = xg_o; xgb_cv_test[k] = xg_t

            c10_clo, c10_chi, c10_tlo, c10_thi = _fit_predict_xgb_cqr(X_tr_raw, y_tr, X_va_raw, X_te_s, 0.10, seed)
            c20_clo, c20_chi, c20_tlo, c20_thi = _fit_predict_xgb_cqr(X_tr_raw, y_tr, X_va_raw, X_te_s, 0.20, seed)
        else:
            c10_clo = c10_chi = c10_tlo = c10_thi = np.zeros(0)
            c20_clo = c20_chi = c20_tlo = c20_thi = np.zeros(0)

        d_nn_test = compute_nn_tanimoto_distance(ecfp_global[tr_idx], ecfp_global[te_idx])

        atomic_save_npz(
            out_npz,
            y_iid_cal=y_tr[iid_cal_i],
            y_scaf_cal=y_va,
            y_test=y_te,
            y_train_val=y_tv,
            y_train_std=np.array([float(np.std(y_tr))], dtype=np.float64),
            d_nn_test=d_nn_test,
            cv_fold_ids=fold_ids,
            va_hashes=np.asarray(va_df["mol_hash"].tolist(), dtype=np.str_),
            te_hashes=np.asarray(te["mol_hash"].tolist(), dtype=np.str_),
            rf_iid_cal=rf_iid_cal, rf_iid_test=rf_iid_test,
            rf_scaf_cal=rf_scaf_cal, rf_scaf_test=rf_scaf_test,
            rf_sigma_cal=rf_sigma_cal, rf_sigma_test=rf_sigma_test,
            rf_cv_oof=rf_cv_oof, rf_cv_test_preds=rf_cv_test,
            xgb_iid_cal=xgb_iid_cal, xgb_iid_test=xgb_iid_test,
            xgb_scaf_cal=xgb_scaf_cal, xgb_scaf_test=xgb_scaf_test,
            xgb_cv_oof=xgb_cv_oof, xgb_cv_test_preds=xgb_cv_test,
            xgb_cqr10_cal_lo=c10_clo, xgb_cqr10_cal_hi=c10_chi,
            xgb_cqr10_test_lo=c10_tlo, xgb_cqr10_test_hi=c10_thi,
            xgb_cqr20_cal_lo=c20_clo, xgb_cqr20_cal_hi=c20_chi,
            xgb_cqr20_test_lo=c20_tlo, xgb_cqr20_test_hi=c20_thi,
        )
        rf_m = compute_task_metrics(y_te, rf_scaf_test, ttype)
        xg_m = compute_task_metrics(y_te, xgb_scaf_test, ttype)
        metric_rows.append({"task": task, "seed": seed, "model": "ecfp4_rdkit_rf", **rf_m})
        metric_rows.append({"task": task, "seed": seed, "model": "ecfp4_rdkit_xgb", **xg_m})

    sub = [r for r in metric_rows if r["task"] == task]
    mkey = spec["metric"]
    rf_avg = np.mean([r[mkey] for r in sub if r["model"] == "ecfp4_rdkit_rf"])
    xg_avg = np.mean([r[mkey] for r in sub if r["model"] == "ecfp4_rdkit_xgb"])
    print(f"[{task:30s}] {mkey:8s} | RF={rf_avg:.4f} | XGB={xg_avg:.4f}")

tab_df = pd.DataFrame(metric_rows)
tab_df.to_csv(TAB_DIR / "tabular_baseline_metrics.csv", index=False)
print(f"\\n[DONE] Tabular baselines & prediction caches saved in {time.time()-t_start:.1f}s")''')

nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
TEST = "--pilot" in sys.argv
if TEST:
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["source"] = c["source"].replace('RUN_SCOPE = "all"', 'RUN_SCOPE = "pilot"')
    out = ROOT / "d_nb01_pilot_test.ipynb"
else:
    out = ROOT / "d_notebook_01_data_prep_ecfp_xgboost.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {out} ({len(CELLS)} cells){' [PILOT]' if TEST else ''}")
