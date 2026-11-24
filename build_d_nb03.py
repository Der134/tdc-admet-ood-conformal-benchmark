# -*- coding: utf-8 -*-
"""Builder for d_notebook_03_conformal_ood_virtual_screening.ipynb (Direction D Stage 3, CPU ~0 CU)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CELLS = []
_ID = 0


def _next_id():
    global _ID
    _ID += 1
    return f"d-nb03-{_ID:02d}"


def md(src):
    CELLS.append({"cell_type": "markdown", "id": _next_id(), "metadata": {}, "source": src})


def code(src):
    CELLS.append({"cell_type": "code", "id": _next_id(), "execution_count": None, "metadata": {}, "outputs": [], "source": src})


md("""# D-03 — Conformal OOD Calibration, Virtual Screening Decision Simulation & Figures

**Resource tier: CPU runtime (~0 CU).**
Evaluates all available representations (Tabular `ecfp4_rdkit_rf` / `ecfp4_rdkit_xgb`, plus GPU-cached Frozen CLMs `chemberta2_mtr` / `chemberta2_mlm` / `molformer_xl` and GNNs `gin_vn` / `gps_lite` if present) across:
1. **Regression Conformal Benchmark**: `split_iid`, `split_scaffold`, `cv_plus_scaffold`, `locally_adaptive_ncp`, `cqr`, `weighted_covariate_shift_cp` + Tanimoto OOD distance stratification (`Near`, `Moderate`, `Far`).
2. **Classification Conformal & Selective Prediction**: `icp_iid`, `icp_scaffold`, `mondrian_scaffold`, `weighted_mondrian_cp` + Risk-Coverage `AURC` curves.
3. **"Buy Top-K Molecules" Virtual Screening & Multi-Parameter EGFR+hERG Case Study**: Greedy vs Conformal Lower-Confidence-Bound / Mondrian Singleton screening (`FP_Saved@K`, `Relative_FP_Reduction_%`).
4. **5 Publication-Grade 300-DPI Figures** in `outputs/figures/`.
""")

code('''# Cell 1 — Setup, workspace & cached representation discovery [CPU, ~0 CU]
import os, sys, io, json, gc, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning)

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
        "Cannot find admet_core.py! Please upload admet_core.py and conformal_ood.py "
        "to /content/ or your Google Drive folder, then re-run Cell 1."
    )
if str(_mod_dir) not in sys.path:
    sys.path.insert(0, str(_mod_dir))

try:
    import rdkit  # noqa: F401
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "rdkit"])

from admet_core import (
    ALL_TASKS, CHEMBL_TARGETS, resolve_workspace, create_scaffold_split_pytdc,
    scaffold_group_kfold, compute_task_metrics, atomic_save_npz,
    canonicalize_smiles, sha256_text, confine_path,
)
from conformal_ood import (
    evaluate_regression_conformal,
    evaluate_classification_conformal,
    simulate_topk_screening,
)

BASE_DIR = resolve_workspace(in_colab=IN_COLAB)
RAW_DIR  = (BASE_DIR / "data" / "raw").resolve()
PROC_DIR = (BASE_DIR / "data" / "processed").resolve()
FEAT_DIR = (BASE_DIR / "cache" / "features").resolve()
EMB_DIR  = (BASE_DIR / "cache" / "embeddings").resolve()
PRED_DIR = (BASE_DIR / "cache" / "predictions").resolve()
TAB_DIR  = (BASE_DIR / "outputs" / "tables").resolve()
FIG_DIR  = (BASE_DIR / "outputs" / "figures").resolve()
for d in (TAB_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

manifest = json.loads((PROC_DIR / "admet_manifest.json").read_text(encoding="utf-8"))
uniq_sha = manifest["unique_sha256"]
ACTIVE_TASKS = list(manifest["tasks"].keys())
SEEDS = manifest["seeds"]
ALPHAS = (0.10, 0.20)

# Load global ECFP+RDKit matrix once for covariate-shift discriminator & CLM alignment
fc = np.load(FEAT_DIR / f"ecfp_rdkit__global__{uniq_sha[:12]}.npz")
X_ecfp_global = fc["X_feat"]
hash_to_idx = {h: i for i, h in enumerate(fc["mol_hashes"].astype(str))}

CLM_MODELS = ["chemberta2_mtr", "chemberta2_mlm", "molformer_xl"]
clm_caches = {}
for m in CLM_MODELS:
    cf = EMB_DIR / f"{m}__global__{uniq_sha[:12]}.npz"
    if cf.exists():
        d = np.load(cf)
        clm_caches[m] = (d["embeddings"], {h: i for i, h in enumerate(d["mol_hashes"].astype(str))})
        print(f"[FOUND CLM] {m}: {d['embeddings'].shape}")
    else:
        print(f"[SKIP  CLM] {m}: cache not present (run D-02 GPU notebook to enable)")

print(f"[CFG] {len(ACTIVE_TASKS)} active tasks | seeds={SEEDS}")''')

code('''# Cell 2 — Train Ridge + 5x MLP Deep Ensemble probes for any cached CLM embeddings [CPU, ~0 CU]
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

class CLMProbeMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.SiLU(), nn.Dropout(0.15),
            nn.Linear(128, 64), nn.SiLU(), nn.Dropout(0.10),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def _train_mlp_ensemble(X_tr, y_tr, X_va, y_va, eval_list, ttype, seed, n_ens=5, max_epochs=80):
    sc = StandardScaler().fit(X_tr)
    Xt = torch.tensor(sc.transform(X_tr), dtype=torch.float32)
    Xv = torch.tensor(sc.transform(X_va), dtype=torch.float32)
    Xev = [torch.tensor(sc.transform(Xe), dtype=torch.float32) for Xe in eval_list]
    if ttype == "regression":
        ymu, ysd = float(np.mean(y_tr)), max(float(np.std(y_tr)), 1e-6)
        yt = torch.tensor((y_tr - ymu) / ysd, dtype=torch.float32)
        yv = torch.tensor((y_va - ymu) / ysd, dtype=torch.float32)
        loss_fn = nn.SmoothL1Loss()
    else:
        ymu, ysd = 0.0, 1.0
        yt = torch.tensor(y_tr, dtype=torch.float32)
        yv = torch.tensor(y_va, dtype=torch.float32)
        pos = float((y_tr == 1).sum()); neg = float((y_tr == 0).sum())
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([min(10.0, max(1.0, neg / max(1.0, pos)))]))

    preds_all = [[] for _ in eval_list]
    for m in range(n_ens):
        torch.manual_seed(seed * 50 + m)
        net = CLMProbeMLP(X_tr.shape[1])
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
        best_v, best_st, bad = float("inf"), None, 0
        for _ in range(max_epochs):
            net.train()
            opt.zero_grad(); loss = loss_fn(net(Xt), yt); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                vl = float(loss_fn(net(Xv), yv).item())
            if vl < best_v - 1e-5:
                best_v, best_st, bad = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 8: break
        if best_st is not None:
            net.load_state_dict(best_st)
        net.eval()
        with torch.no_grad():
            for i, X_e in enumerate(Xev):
                raw = net(X_e).numpy()
                preds_all[i].append(raw * ysd + ymu if ttype == "regression" else 1.0 / (1.0 + np.exp(-raw)))
    means = [np.stack(p).mean(0) for p in preds_all]
    stds = [np.stack(p).std(0) for p in preds_all]
    return means, stds

clm_metric_rows = []
for mkey, (E_mat, hmap) in clm_caches.items():
    for task in ACTIVE_TASKS:
        spec = ALL_TASKS[task]
        ttype = spec["task_type"]
        tdir = confine_path(RAW_DIR, RAW_DIR / ("chembl" if task in CHEMBL_TARGETS else "admet_group") / task)
        tv = pd.read_csv(tdir / "train_val.csv").dropna(subset=["Drug", "Y"]).reset_index(drop=True)
        te = pd.read_csv(tdir / "test.csv").dropna(subset=["Drug", "Y"]).reset_index(drop=True)
        tv["canonical_smiles"] = tv["Drug"].map(canonicalize_smiles)
        te["canonical_smiles"] = te["Drug"].map(canonicalize_smiles)
        tv = tv.dropna(subset=["canonical_smiles"]).reset_index(drop=True)
        te = te.dropna(subset=["canonical_smiles"]).reset_index(drop=True)
        tv["mol_hash"] = tv["canonical_smiles"].map(sha256_text)
        te["mol_hash"] = te["canonical_smiles"].map(sha256_text)
        E_te = E_mat[[hmap[h] for h in te["mol_hash"]]]
        y_te = te["Y"].values.astype(np.float64)

        for seed in SEEDS:
            out_npz = PRED_DIR / f"clm_{mkey}__{task}__seed{seed}.npz"
            if out_npz.exists():
                continue
            sp = create_scaffold_split_pytdc(tv, seed=seed, frac=(0.875, 0.125, 0.0), entity="Drug")
            tr_df, va_df = sp["train"], sp["valid"]
            E_tr = E_mat[[hmap[h] for h in tr_df["mol_hash"]]]
            E_va = E_mat[[hmap[h] for h in va_df["mol_hash"]]]
            y_tr = tr_df["Y"].values.astype(np.float64)
            y_va = va_df["Y"].values.astype(np.float64)

            rng = np.random.RandomState(seed * 100 + 7)
            perm = rng.permutation(len(E_tr))
            n_iid = max(15, int(round(0.15 * len(E_tr))))
            iid_cal_i, iid_fit_i = perm[:n_iid], perm[n_iid:]

            [iid_mu_c, iid_mu_t], _ = _train_mlp_ensemble(
                E_tr[iid_fit_i], y_tr[iid_fit_i], E_va, y_va, [E_tr[iid_cal_i], E_te], ttype, seed
            )
            [scaf_mu_c, scaf_mu_t], [scaf_sd_c, scaf_sd_t] = _train_mlp_ensemble(
                E_tr, y_tr, E_va, y_va, [E_va, E_te], ttype, seed + 10
            )
            atomic_save_npz(
                out_npz,
                y_iid_cal=y_tr[iid_cal_i], y_scaf_cal=y_va, y_test=y_te,
                iid_cal=iid_mu_c, iid_test=iid_mu_t,
                scaf_cal=scaf_mu_c, scaf_test=scaf_mu_t,
                sigma_cal=scaf_sd_c, sigma_test=scaf_sd_t,
            )
            met = compute_task_metrics(y_te, scaf_mu_t, ttype)
            clm_metric_rows.append({"task": task, "seed": seed, "model": mkey, **met})

if clm_metric_rows:
    pd.DataFrame(clm_metric_rows).to_csv(TAB_DIR / "clm_baseline_metrics.csv", index=False)
    print(f"[SAVED] clm_baseline_metrics.csv ({len(clm_metric_rows)} rows)")
else:
    print("[INFO] No new CLM probes trained.")''')

code('''# Cell 3 — Unified Conformal OOD Evaluation across all cached models & tasks [CPU, ~0 CU]
reg_conf_rows, reg_bin_rows = [], []
cls_conf_rows, sel_curve_rows = [], []
topk_rows = []
mpo_records = {}  # Store EGFR & hERG test predictions for Case Study Fig 5

def _load_model_records(task, seed):
    tab_path = PRED_DIR / f"tabular__{task}__seed{seed}.npz"
    if not tab_path.exists():
        return {}
    td = np.load(tab_path)
    va_hashes = td["va_hashes"].astype(str)
    te_hashes = td["te_hashes"].astype(str)
    X_cal = X_ecfp_global[[hash_to_idx[h] for h in va_hashes]]
    X_te  = X_ecfp_global[[hash_to_idx[h] for h in te_hashes]]
    base = {
        "y_iid_cal": td["y_iid_cal"], "y_scaf_cal": td["y_scaf_cal"], "y_test": td["y_test"],
        "y_train_std": float(td["y_train_std"][0]), "d_nn": td["d_nn_test"],
        "y_train_val": td["y_train_val"], "cv_fold_ids": td["cv_fold_ids"],
        "X_cal": X_cal, "X_te": X_te, "te_hashes": te_hashes,
    }
    models = {
        "ecfp4_rdkit_rf": {
            "iid_cal": td["rf_iid_cal"], "iid_test": td["rf_iid_test"],
            "scaf_cal": td["rf_scaf_cal"], "scaf_test": td["rf_scaf_test"],
            "sigma_cal": td["rf_sigma_cal"], "sigma_test": td["rf_sigma_test"],
            "cv_oof": td["rf_cv_oof"], "cv_test_preds": td["rf_cv_test_preds"],
        },
        "ecfp4_rdkit_xgb": {
            "iid_cal": td["xgb_iid_cal"], "iid_test": td["xgb_iid_test"],
            "scaf_cal": td["xgb_scaf_cal"], "scaf_test": td["xgb_scaf_test"],
            "cv_oof": td["xgb_cv_oof"], "cv_test_preds": td["xgb_cv_test_preds"],
            "cqr_10": (td["xgb_cqr10_cal_lo"], td["xgb_cqr10_cal_hi"], td["xgb_cqr10_test_lo"], td["xgb_cqr10_test_hi"]),
            "cqr_20": (td["xgb_cqr20_cal_lo"], td["xgb_cqr20_cal_hi"], td["xgb_cqr20_test_lo"], td["xgb_cqr20_test_hi"]),
        },
    }
    for mkey in CLM_MODELS:
        cp = PRED_DIR / f"clm_{mkey}__{task}__seed{seed}.npz"
        if cp.exists():
            cd = np.load(cp)
            models[mkey] = {
                "iid_cal": cd["iid_cal"], "iid_test": cd["iid_test"],
                "scaf_cal": cd["scaf_cal"], "scaf_test": cd["scaf_test"],
                "sigma_cal": cd["sigma_cal"], "sigma_test": cd["sigma_test"],
            }
    gp = PRED_DIR / f"gnn__{task}__seed{seed}.npz"
    if gp.exists():
        gd = np.load(gp)
        for arch in ("gin_vn", "gps_lite"):
            if f"{arch}_scaf_test" in gd:
                models[arch] = {
                    "iid_cal": gd[f"{arch}_iid_cal"], "iid_test": gd[f"{arch}_iid_test"],
                    "scaf_cal": gd[f"{arch}_scaf_cal"], "scaf_test": gd[f"{arch}_scaf_test"],
                    "sigma_cal": gd[f"{arch}_sigma_cal"], "sigma_test": gd[f"{arch}_sigma_test"],
                }
    return {"base": base, "models": models}

for task in ACTIVE_TASKS:
    spec = ALL_TASKS[task]
    ttype = spec["task_type"]
    fav = spec["favorable"]
    hit_thresh = spec.get("hit_threshold", None)

    for seed in SEEDS:
        rec = _load_model_records(task, seed)
        if not rec:
            continue
        b = rec["base"]

        for mname, mp in rec["models"].items():
            if ttype == "regression":
                # Run CQR per alpha cleanly when quantile predictions exist
                all_s, all_b, all_iv = [], [], {}
                for alpha in ALPHAS:
                    cqr_key = f"cqr_{int(round(alpha * 100))}"
                    c_clo, c_chi, c_tlo, c_thi = mp.get(cqr_key, (None, None, None, None))
                    s_rows, b_rows, iv_dict = evaluate_regression_conformal(
                        y_test=b["y_test"], y_train_std=b["y_train_std"],
                        mu_iid_cal=mp["iid_cal"], y_iid_cal=b["y_iid_cal"], mu_iid_test=mp["iid_test"],
                        mu_scaf_cal=mp["scaf_cal"], y_scaf_cal=b["y_scaf_cal"], mu_scaf_test=mp["scaf_test"],
                        sigma_scaf_cal=mp.get("sigma_cal"), sigma_scaf_test=mp.get("sigma_test"),
                        cv_oof_mu=mp.get("cv_oof"), cv_oof_y=b["y_train_val"],
                        cv_test_preds=mp.get("cv_test_preds"), cv_fold_ids=b["cv_fold_ids"],
                        cqr_cal_lo=c_clo, cqr_cal_hi=c_chi, cqr_test_lo=c_tlo, cqr_test_hi=c_thi,
                        X_scaf_cal=b["X_cal"], X_test=b["X_te"], d_nn=b["d_nn"], alphas=(alpha,),
                    )
                    all_s.extend(s_rows); all_b.extend(b_rows); all_iv.update(iv_dict)

                for r in all_s:
                    r.update({"task": task, "seed": seed, "model": mname, "category": spec.get("category", "Potency")})
                    reg_conf_rows.append(r)
                for r in all_b:
                    r.update({"task": task, "seed": seed, "model": mname})
                    reg_bin_rows.append(r)

                # Top-K virtual screening simulation at alpha=0.10 (choose strongest scaffold-calibrated interval)
                pref_method = "locally_adaptive_ncp" if ("locally_adaptive_ncp", 0.10) in all_iv else (
                    "cqr" if ("cqr", 0.10) in all_iv else "split_scaffold"
                )
                lo10, hi10 = all_iv[(pref_method, 0.10)]
                tk = simulate_topk_screening(
                    y_true=b["y_test"], task_type="regression", favorable=fav,
                    y_pred_point=mp["scaf_test"], conf_lo=lo10, conf_hi=hi10,
                    y_cal_for_threshold=b["y_scaf_cal"], fixed_hit_threshold=hit_thresh,
                )
                for r in tk:
                    r.update({"task": task, "seed": seed, "model": mname, "task_type": "regression", "conformal_method": pref_method})
                    topk_rows.append(r)

                if seed == SEEDS[0] and mname == "ecfp4_rdkit_rf" and task in ("chembl203_egfr", "chembl240_herg"):
                    mpo_records[task] = {
                        "y_test": b["y_test"], "mu": mp["scaf_test"], "lo10": lo10, "hi10": hi10,
                        "te_hashes": b["te_hashes"], "d_nn": b["d_nn"],
                    }
            else:
                s_rows, c_rows, cls_out = evaluate_classification_conformal(
                    y_test=b["y_test"],
                    p_iid_cal=mp["iid_cal"], y_iid_cal=b["y_iid_cal"], p_iid_test=mp["iid_test"],
                    p_scaf_cal=mp["scaf_cal"], y_scaf_cal=b["y_scaf_cal"], p_scaf_test=mp["scaf_test"],
                    sigma_scaf_test=mp.get("sigma_test"),
                    X_scaf_cal=b["X_cal"], X_test=b["X_te"], alphas=ALPHAS,
                )
                for r in s_rows:
                    r.update({"task": task, "seed": seed, "model": mname, "category": spec.get("category", "Tox"), "metric": spec["metric"]})
                    cls_conf_rows.append(r)
                for r in c_rows:
                    r.update({"task": task, "seed": seed, "model": mname})
                    sel_curve_rows.append(r)

                sets10, pvals10 = cls_out[("mondrian_scaffold", 0.10)]
                tk = simulate_topk_screening(
                    y_true=b["y_test"], task_type="classification", favorable=fav,
                    y_pred_point=mp["scaf_test"], cls_sets=sets10, cls_pvals=pvals10,
                )
                for r in tk:
                    r.update({"task": task, "seed": seed, "model": mname, "task_type": "classification", "conformal_method": "mondrian_scaffold"})
                    topk_rows.append(r)

                if seed == SEEDS[0] and mname == "ecfp4_rdkit_rf" and task == "herg":
                    mpo_records["tdc_herg"] = {"p_test": mp["scaf_test"], "sets10": sets10, "pvals10": pvals10, "y_test": b["y_test"]}

reg_df = pd.DataFrame(reg_conf_rows)
reg_bin_df = pd.DataFrame(reg_bin_rows)
cls_df = pd.DataFrame(cls_conf_rows)
sel_df = pd.DataFrame(sel_curve_rows)
topk_df = pd.DataFrame(topk_rows)

reg_df.to_csv(TAB_DIR / "conformal_regression_summary.csv", index=False)
reg_bin_df.to_csv(TAB_DIR / "conformal_tanimoto_bins.csv", index=False)
cls_df.to_csv(TAB_DIR / "conformal_classification_summary.csv", index=False)
sel_df.to_csv(TAB_DIR / "selective_risk_coverage.csv", index=False)
topk_df.to_csv(TAB_DIR / "virtual_screening_topk.csv", index=False)

print(f"[SAVED] conformal_regression_summary.csv ({len(reg_df)}) | conformal_classification_summary.csv ({len(cls_df)}) | virtual_screening_topk.csv ({len(topk_df)})")''')

code('''# Cell 4 — Print key quantitative findings & EGFR+hERG multi-parameter case study [CPU, ~0 CU]
if not reg_df.empty:
    print("=== REGRESSION CONFORMAL (alpha=0.10, mean across tasks/seeds) ===")
    r10 = reg_df[reg_df["alpha"] == 0.10]
    print(r10.groupby(["model", "conformal_method"])[["empirical_coverage", "coverage_gap", "norm_width", "ess"]].mean().round(4))

if not reg_bin_df.empty:
    print("\\n=== TANIMOTO OOD COVERAGE STRATIFICATION (alpha=0.10, RF) ===")
    rb = reg_bin_df[(reg_bin_df["alpha"] == 0.10) & (reg_bin_df["model"] == "ecfp4_rdkit_rf")]
    print(rb.groupby(["conformal_method", "tanimoto_bin"])[["empirical_coverage", "norm_width"]].mean().round(4))

if not cls_df.empty:
    print("\\n=== CLASSIFICATION CONFORMAL (alpha=0.10, minority/positive coverage) ===")
    c10 = cls_df[cls_df["alpha"] == 0.10]
    print(c10.groupby(["model", "conformal_method"])[["empirical_coverage", "coverage_pos", "coverage_neg", "singleton_rate", "singleton_accuracy"]].mean().round(4))

if not topk_df.empty:
    print("\\n=== BUY TOP-K VIRTUAL SCREENING (Greedy vs Conformal) ===")
    print(topk_df.groupby(["K", "task_type"])[["greedy_hit_rate", "conformal_hit_rate", "fp_saved", "relative_fp_reduction_pct"]].mean().round(3))

# Multi-parameter ChEMBL EGFR (potency) + TDC hERG (cardiotoxicity conformal safety filter)
mpo_rows = []
if "chembl203_egfr" in mpo_records:
    from sklearn.ensemble import RandomForestClassifier
    eg = mpo_records["chembl203_egfr"]
    # Score EGFR test compounds on TDC hERG scaffold model to get real conformal safety p-values
    herg_dir = confine_path(RAW_DIR, RAW_DIR / "admet_group" / "herg")
    htv = pd.read_csv(herg_dir / "train_val.csv").dropna(subset=["Drug", "Y"]).reset_index(drop=True)
    htv["mol_hash"] = htv["Drug"].map(canonicalize_smiles).map(sha256_text)
    hsp = create_scaffold_split_pytdc(htv, seed=1, frac=(0.875, 0.125, 0.0), entity="Drug")
    Xh_tr = X_ecfp_global[[hash_to_idx[h] for h in hsp["train"]["mol_hash"]]]
    Xh_va = X_ecfp_global[[hash_to_idx[h] for h in hsp["valid"]["mol_hash"]]]
    Xh_eg = X_ecfp_global[[hash_to_idx[h] for h in eg["te_hashes"]]]
    hclf = RandomForestClassifier(n_estimators=120, max_features="sqrt", class_weight="balanced_subsample", n_jobs=-1, random_state=1).fit(Xh_tr, hsp["train"]["Y"].values.astype(int))
    hp_va = hclf.predict_proba(Xh_va)[:, 1]
    hp_eg = hclf.predict_proba(Xh_eg)[:, 1]
    from conformal_ood import _cls_prediction_sets_mondrian
    h_sets, h_pvals = _cls_prediction_sets_mondrian(hp_va, hsp["valid"]["Y"].values.astype(int), hp_eg, alpha=0.10)
    # Safe = predicted {0} or high p-val for class 0 and low p-val for cardiotoxic class 1
    egfr_hit = eg["y_test"] >= 7.0
    mpo_df = pd.DataFrame({
        "mol_hash": eg["te_hashes"],
        "true_pchembl": eg["y_test"],
        "is_true_hit": egfr_hit,
        "pred_mu": eg["mu"],
        "pred_lcb90": eg["lo10"],
        "herg_prob": hp_eg,
        "herg_pval_safe0": h_pvals[:, 0],
        "herg_cert_safe": (h_sets[:, 0] & ~h_sets[:, 1]),
        "d_nn": eg["d_nn"],
    })
    mpo_df.to_csv(TAB_DIR / "chembl_mpo_case_study.csv", index=False)
    print(f"\\n[SAVED] chembl_mpo_case_study.csv ({len(mpo_df)} EGFR test compounds)")''')

code('''# Cell 5 — Generate 5 publication-grade 300-DPI figures [CPU, ~0 CU]
sns.set_theme(style="whitegrid", context="paper")

def save_fig(fig, name):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    (FIG_DIR / name).write_bytes(buf.getvalue())
    plt.close(fig)
    print(f"[FIG] {name}")

# Fig 1: Scaffold OOD Coverage Collapse + Tanimoto Distance Degradation (alpha=0.10)
if not reg_df.empty:
    r10 = reg_df[reg_df["alpha"] == 0.10]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    m_order = [m for m in ["split_iid", "split_scaffold", "cv_plus_scaffold", "locally_adaptive_ncp", "cqr", "weighted_covariate_shift_cp"] if m in set(r10["conformal_method"])]
    m_labels = {
        "split_iid": "Split IID\\n(Naive)", "split_scaffold": "Split\\nScaffold",
        "cv_plus_scaffold": "CV+\\nScaffold", "locally_adaptive_ncp": "Adaptive\\nNCP",
        "cqr": "CQR\\n(XGB)", "weighted_covariate_shift_cp": "Weighted\\nCov-Shift",
    }
    sns.boxplot(data=r10, x="conformal_method", y="empirical_coverage", order=m_order, ax=axes[0], palette="Set2", showfliers=False)
    sns.stripplot(data=r10, x="conformal_method", y="empirical_coverage", order=m_order, ax=axes[0], color="black", alpha=0.3, size=3)
    axes[0].axhline(0.90, ls="--", color="crimson", lw=1.4, label="Nominal 90% Target")
    axes[0].set_xticklabels([m_labels.get(x, x) for x in m_order], fontsize=8.5)
    axes[0].set_xlabel(""); axes[0].set_ylabel("Empirical Test Coverage")
    axes[0].set_title("A. Scaffold OOD Coverage Collapse & Recovery (90% Target)")
    axes[0].set_ylim(0.55, 1.02); axes[0].legend(loc="lower right")

    if not reg_bin_df.empty:
        rb10 = reg_bin_df[(reg_bin_df["alpha"] == 0.10) & (reg_bin_df["conformal_method"].isin(["split_iid", "split_scaffold", "weighted_covariate_shift_cp"]))]
        b_order = ["Near (<0.45)", "Moderate (0.45-0.65)", "Far (>=0.65)"]
        sns.pointplot(data=rb10, x="tanimoto_bin", y="empirical_coverage", hue="conformal_method", order=b_order, ax=axes[1], capsize=0.08, markers=["o", "s", "^"])
        axes[1].axhline(0.90, ls="--", color="crimson", lw=1.4)
        axes[1].set_xlabel("Nearest-Neighbor ECFP4 Tanimoto Distance to Train")
        axes[1].set_ylabel("Empirical Coverage")
        axes[1].set_title("B. Coverage Decay by Structural OOD Distance")
        axes[1].set_ylim(0.55, 1.02)
        axes[1].legend(title="Method", fontsize=8)
    fig.tight_layout(); save_fig(fig, "fig1_scaffold_ood_coverage_collapse.png")

# Fig 2: Efficiency-Coverage Pareto Frontier (Interval Width vs Coverage)
if not reg_df.empty:
    r10 = reg_df[(reg_df["alpha"] == 0.10) & np.isfinite(reg_df["norm_width"])]
    agg = r10.groupby(["model", "conformal_method"], as_index=False)[["empirical_coverage", "norm_width"]].mean()
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    sns.scatterplot(data=agg, x="norm_width", y="empirical_coverage", hue="model", style="conformal_method", s=110, ax=ax)
    ax.axhline(0.90, ls="--", color="crimson", lw=1.3, label="Nominal 90%")
    ax.set_xlabel("Mean Normalized Interval Width (Width / Train Std)")
    ax.set_ylabel("Empirical Test Coverage")
    ax.set_title("Conformal Interval Efficiency Frontier Across Models & Calibrators (alpha=0.10)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7.5)
    fig.tight_layout(); save_fig(fig, "fig2_interval_efficiency_cqr_vs_split.png")

# Fig 3: Mondrian Calibration on Imbalanced Classification + Risk-Coverage Curve
if not cls_df.empty:
    c10 = cls_df[cls_df["alpha"] == 0.10]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.5))
    cm_order = [m for m in ["icp_iid", "icp_scaffold", "mondrian_scaffold", "weighted_mondrian_cp"] if m in set(c10["conformal_method"])]
    melt_c = c10.melt(id_vars=["task", "model", "conformal_method"], value_vars=["coverage_pos", "coverage_neg"], var_name="class", value_name="coverage")
    melt_c["class"] = melt_c["class"].map({"coverage_pos": "Positive (Y=1)", "coverage_neg": "Negative (Y=0)"})
    sns.barplot(data=melt_c, x="conformal_method", y="coverage", hue="class", order=cm_order, ax=axes[0], palette="Paired")
    axes[0].axhline(0.90, ls="--", color="crimson", lw=1.3, label="90% Target")
    axes[0].set_xlabel(""); axes[0].set_ylabel("Class-Conditional Coverage")
    axes[0].set_title("A. Standard vs Mondrian Coverage on Imbalanced ADMET")
    axes[0].set_ylim(0.50, 1.03); axes[0].tick_params(axis="x", rotation=18)

    if not sel_df.empty:
        sns.lineplot(data=sel_df, x="keep_fraction", y="risk_error_rate", hue="uncertainty_signal", style="model", marker="o", ax=axes[1])
        axes[1].invert_xaxis()
        axes[1].set_xlabel("Retained Test Fraction (Coverage)")
        axes[1].set_ylabel("Classification Error Rate (Risk)")
        axes[1].set_title("B. Selective Prediction Risk-Coverage Curve")
    fig.tight_layout(); save_fig(fig, "fig3_mondrian_imbalanced_classification.png")

# Fig 4: Top-K Virtual Screening Hit Rate & False Positives Saved
if not topk_df.empty:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
    tk_melt = topk_df.melt(
        id_vars=["task", "seed", "model", "K", "task_type"],
        value_vars=["greedy_hit_rate", "conformal_hit_rate"],
        var_name="strategy", value_name="hit_rate",
    )
    tk_melt["strategy"] = tk_melt["strategy"].map({"greedy_hit_rate": "Greedy Point Top-K", "conformal_hit_rate": "Conformal Risk-Controlled Top-K"})
    sns.barplot(data=tk_melt, x="K", y="hit_rate", hue="strategy", ax=axes[0], palette=["#7f8c8d", "#1f77b4"])
    axes[0].set_xlabel("Screening Budget (Top-K Molecules)")
    axes[0].set_ylabel("Hit Rate (Precision@K)")
    axes[0].set_title("A. Virtual Screening Hit Rate: Greedy vs Conformal")

    sns.barplot(data=topk_df, x="K", y="relative_fp_reduction_pct", hue="task_type", ax=axes[1], palette="Set1")
    axes[1].axhline(0.0, color="gray", lw=0.8)
    axes[1].set_xlabel("Screening Budget (Top-K Molecules)")
    axes[1].set_ylabel("Relative False-Positive Reduction (%)")
    axes[1].set_title("B. False Positives Saved via Conformal LCB / Mondrian")
    fig.tight_layout(); save_fig(fig, "fig4_topk_virtual_screening_fp_saved.png")

# Fig 5: ChEMBL EGFR Potency + Conformal hERG Cardiotoxicity MPO Case Study
mpo_csv = TAB_DIR / "chembl_mpo_case_study.csv"
if mpo_csv.exists():
    mdf = pd.read_csv(mpo_csv)
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    hits = mdf[mdf["is_true_hit"]]
    nonh = mdf[~mdf["is_true_hit"]]
    ax.scatter(nonh["pred_lcb90"], nonh["herg_pval_safe0"], s=22, alpha=0.45, color="#95a5a6", label="Non-Hit (pChEMBL < 7.0)")
    ax.scatter(hits["pred_lcb90"], hits["herg_pval_safe0"], s=36, alpha=0.8, color="#2ecc71", edgecolor="k", lw=0.4, label="True EGFR Hit (pChEMBL >= 7.0)")
    ax.axvline(6.2, ls="--", color="#2980b9", lw=1.2, label="EGFR Conformal LCB90 Cutoff")
    ax.axhline(0.10, ls="--", color="#e74c3c", lw=1.2, label="hERG Non-Toxic Conformal p > 0.10")
    ax.set_xlabel("EGFR Conformal 90% Lower Confidence Bound (pChEMBL LCB)")
    ax.set_ylabel("Conformal hERG Non-Toxic p-value (Y=0)")
    ax.set_title("Multi-Parameter Virtual Screening: EGFR Potency LCB vs Conformal hERG Safety")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout(); save_fig(fig, "fig5_egfr_herg_mpo_case_study.png")

print("[DONE] Direction D evaluation and figures complete.")''')

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
out = ROOT / ("d_nb03_pilot_test.ipynb" if TEST else "d_notebook_03_conformal_ood_virtual_screening.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {out} ({len(CELLS)} cells)")
