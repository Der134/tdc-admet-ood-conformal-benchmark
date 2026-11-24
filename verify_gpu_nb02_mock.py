# -*- coding: utf-8 -*-
"""End-to-end Stage 1 -> Stage 2 (Mock CLM + Real CPU GNN) -> Stage 3 compatibility harness."""
import gc
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from admet_core import (
    ALL_TASKS, atomic_save_npz, canonicalize_smiles, compute_nn_tanimoto_distance,
    create_scaffold_split_pytdc, featurize_ecfp_rdkit, get_scaffold_smiles,
    scaffold_group_kfold, sha256_text,
)
from gin_gps_model import smiles_to_graph_dict, train_gnn_ensemble


def _build_diverse_smiles():
    """Generate distinct Bemis-Murcko scaffolds for train_val (>=140) and test (>=40)."""
    tv_cores = [
        "c1ccccc1", "c1ccncc1", "c1cncnc1", "c1ccc2ccccc2c1",
        "c1ccc2[nH]ccc2c1", "c1ccc2occc2c1", "c1ccc2sccc2c1",
        "C1CCCCC1", "C1CCNCC1", "C1CCOCC1", "c1ccc(cc1)c2ccccc2",
        "c1ccc(Cc2ccccc2)cc1", "c1ccc(Oc2ccccc2)cc1", "c1ccc(Nc2ccccc2)cc1",
        "c1ccc2ncccc2c1", "c1ccc2ncncc2c1", "c1cnc2ccccc2n1",
        "C1CC1", "C1CCC1", "C1CCCC1",
    ]
    te_cores = [
        "c1ccoc1", "c1ccsc1", "c1cc[nH]c1", "c1cnccn1",
        "c1ccc(Sc2ccccc2)cc1", "c1ccc(C(=O)c2ccccc2)cc1",
        "C1COCCN1", "C1CNCCN1", "c1ccc2c(c1)OCO2", "c1ccc2c(c1)CC2",
    ]
    subs = ["C", "CC", "CCC", "OC", "OCC", "N", "NC", "F", "Cl", "C(=O)O", "C(=O)N", "C#N"]

    def _expand(cores, target_n):
        out = []
        seen = set()
        for c in cores:
            for s in subs:
                cand = f"{c}{s}" if not c.endswith(")") else f"{s}{c}"
                cs = canonicalize_smiles(cand)
                if cs is None:
                    # Try explicit attachment on first ring atom
                    cs = canonicalize_smiles(f"{s}.{c}")
                if cs is not None and cs not in seen and "." not in cs:
                    seen.add(cs)
                    out.append(cs)
                if len(out) >= target_n:
                    return out
        return out

    # Use guaranteed valid SMILES via explicit substituent chains on rings
    tv_smis, te_smis = [], []
    seen_all = set()
    tv_templates = [
        "c1ccccc1{s}", "c1cc({s})ccc1C", "c1ccncc1{s}", "c1cnc(nc1){s}",
        "c1ccc2ccccc2c1{s}", "c1ccc2[nH]c({s})cc2c1", "c1ccc2oc({s})cc2c1",
        "c1ccc2sc({s})cc2c1", "C1CCCCC1{s}", "C1CCN(CC1){s}", "C1CCOCC1{s}",
        "c1ccc(cc1)c2ccc({s})cc2", "c1ccc(Cc2ccc({s})cc2)cc1",
        "c1ccc(Oc2ccc({s})cc2)cc1", "c1ccc2ncc({s})cc2c1",
    ]
    te_templates = [
        "c1cc({s})oc1", "c1cc({s})sc1", "c1cc({s})[nH]c1", "c1cnc({s})cn1",
        "c1ccc(Sc2ccc({s})cc2)cc1", "c1ccc(C(=O)c2ccc({s})cc2)cc1",
        "C1COCCN1{s}", "c1ccc2c(c1)OCO2",
    ]
    chain_subs = ["C", "CC", "CCC", "CCCC", "OC", "OCC", "N", "NC", "F", "Cl", "Br", "C(=O)O", "C(=O)N", "C#N"]
    for tmpl in tv_templates:
        for s in chain_subs:
            cs = canonicalize_smiles(tmpl.format(s=s))
            if cs and cs not in seen_all:
                seen_all.add(cs)
                tv_smis.append(cs)
    tv_scafs = {get_scaffold_smiles(x) for x in tv_smis}

    for tmpl in te_templates:
        for s in chain_subs:
            raw = tmpl.format(s=s) if "{s}" in tmpl else tmpl
            cs = canonicalize_smiles(raw)
            if cs and cs not in seen_all and get_scaffold_smiles(cs) not in tv_scafs:
                seen_all.add(cs)
                te_smis.append(cs)

    assert len(tv_smis) >= 140 and len(te_smis) >= 40, f"Insufficient diverse SMILES: {len(tv_smis)}, {len(te_smis)}"
    return tv_smis[:140], te_smis[:40]


def main():
    mock_dir = (ROOT / "_mock_gpu_test").resolve()
    if mock_dir.exists():
        shutil.rmtree(mock_dir, ignore_errors=True)
    try:
        mock_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        mock_dir = Path(tempfile.mkdtemp(prefix="direction_d_mock_gpu_")).resolve()

    ns = {}
    try:
        raw_dir = mock_dir / "data" / "raw"
        proc_dir = mock_dir / "data" / "processed"
        feat_dir = mock_dir / "cache" / "features"
        emb_dir = mock_dir / "cache" / "embeddings"
        pred_dir = mock_dir / "cache" / "predictions"
        tab_dir = mock_dir / "outputs" / "tables"
        fig_dir = mock_dir / "outputs" / "figures"
        for d in (raw_dir / "admet_group", proc_dir, feat_dir, emb_dir, pred_dir, tab_dir, fig_dir):
            d.mkdir(parents=True, exist_ok=True)

        tv_smis, te_smis = _build_diverse_smiles()
        rng = np.random.RandomState(2026)

        tasks = ["caco2_wang", "herg"]
        all_records = []
        task_frames = {}

        # 1) Synthesize Stage 1 raw CSVs & manifest
        for task in tasks:
            ttype = ALL_TASKS[task]["task_type"]
            tdir = raw_dir / "admet_group" / task
            tdir.mkdir(parents=True, exist_ok=True)
            if ttype == "regression":
                y_tv = np.array([0.05 * len(s) + rng.normal(0, 0.25) for s in tv_smis], dtype=np.float64)
                y_te = np.array([0.05 * len(s) + rng.normal(0, 0.25) for s in te_smis], dtype=np.float64)
            else:
                # Imbalanced ~22% positive rate to exercise Mondrian ICP
                y_tv = (rng.rand(len(tv_smis)) < 0.22).astype(int)
                y_te = (rng.rand(len(te_smis)) < 0.22).astype(int)
                y_tv[:30] = np.array([1] * 8 + [0] * 22)
                y_te[:10] = np.array([1] * 3 + [0] * 7)

            tv_df = pd.DataFrame({"Drug": tv_smis, "Y": y_tv})
            te_df = pd.DataFrame({"Drug": te_smis, "Y": y_te})
            tv_df.to_csv(tdir / "train_val.csv", index=False)
            te_df.to_csv(tdir / "test.csv", index=False)

            tv_df["canonical_smiles"] = tv_df["Drug"].map(canonicalize_smiles)
            te_df["canonical_smiles"] = te_df["Drug"].map(canonicalize_smiles)
            tv_df["mol_hash"] = tv_df["canonical_smiles"].map(sha256_text)
            te_df["mol_hash"] = te_df["canonical_smiles"].map(sha256_text)
            task_frames[task] = (tv_df, te_df)
            all_records.extend(zip(tv_df["canonical_smiles"], tv_df["mol_hash"]))
            all_records.extend(zip(te_df["canonical_smiles"], te_df["mol_hash"]))

        uniq_df = (
            pd.DataFrame(all_records, columns=["canonical_smiles", "mol_hash"])
            .drop_duplicates(subset=["mol_hash"])
            .sort_values("mol_hash")
            .reset_index(drop=True)
        )
        uniq_sha = sha256_text("".join(uniq_df["mol_hash"].tolist()))
        uniq_df.to_csv(proc_dir / "unique_molecules.csv", index=False)
        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "run_scope": "mock_gpu_test",
            "seeds": [1],
            "n_unique_molecules": int(len(uniq_df)),
            "unique_sha256": uniq_sha,
            "tasks": {
                t: {**ALL_TASKS[t], "n_train_val": int(len(task_frames[t][0])), "n_test": int(len(task_frames[t][1]))}
                for t in tasks
            },
        }
        (proc_dir / "admet_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        # Global ECFP+RDKit feature cache
        X_global, ecfp_global, valid_m = featurize_ecfp_rdkit(uniq_df["canonical_smiles"].tolist())
        assert valid_m.all()
        atomic_save_npz(
            feat_dir / f"ecfp_rdkit__global__{uniq_sha[:12]}.npz",
            X_feat=X_global,
            ecfp_bits=ecfp_global,
            mol_hashes=np.asarray(uniq_df["mol_hash"].tolist(), dtype=np.str_),
        )
        hash_to_idx = {h: i for i, h in enumerate(uniq_df["mol_hash"].tolist())}

        # Stage 1 tabular caches (lightweight RF + XGBoost matching d_notebook_01 schema)
        seed = 1
        for task in tasks:
            tv, te = task_frames[task]
            ttype = ALL_TASKS[task]["task_type"]
            sp = create_scaffold_split_pytdc(tv, seed=seed, frac=(0.875, 0.125, 0.0), entity="Drug")
            tr_df, va_df = sp["train"], sp["valid"]
            tr_idx = [hash_to_idx[h] for h in tr_df["mol_hash"]]
            va_idx = [hash_to_idx[h] for h in va_df["mol_hash"]]
            te_idx = [hash_to_idx[h] for h in te["mol_hash"]]
            tv_idx = [hash_to_idx[h] for h in tv["mol_hash"]]

            X_tr = X_global[tr_idx].copy(); X_va = X_global[va_idx].copy(); X_te = X_global[te_idx].copy()
            sc = StandardScaler().fit(X_tr[:, 2215:])
            X_tr[:, 2215:] = sc.transform(X_tr[:, 2215:])
            X_va[:, 2215:] = sc.transform(X_va[:, 2215:])
            X_te[:, 2215:] = sc.transform(X_te[:, 2215:])
            y_tr = tr_df["Y"].values.astype(np.float64)
            y_va = va_df["Y"].values.astype(np.float64)
            y_te = te["Y"].values.astype(np.float64)

            rng_s = np.random.RandomState(seed * 100 + 7)
            perm = rng_s.permutation(len(X_tr))
            n_iid_cal = max(15, int(round(0.15 * len(X_tr))))
            iid_cal_i, iid_fit_i = perm[:n_iid_cal], perm[n_iid_cal:]

            def _rf(Xfit, yfit, Xev):
                if ttype == "regression":
                    m = RandomForestRegressor(n_estimators=20, max_features="sqrt", random_state=seed).fit(Xfit, yfit)
                    preds = [np.stack([t.predict(Xe) for t in m.estimators_], 0) for Xe in Xev]
                else:
                    m = RandomForestClassifier(n_estimators=20, max_features="sqrt", class_weight="balanced_subsample", random_state=seed).fit(Xfit, yfit.astype(int))
                    preds = [np.stack([t.predict_proba(Xe)[:, 1] for t in m.estimators_], 0) for Xe in Xev]
                return [p.mean(0) for p in preds], [p.std(0) for p in preds]

            def _xgb(Xfit, yfit, Xev):
                if ttype == "regression":
                    m = xgb.XGBRegressor(n_estimators=25, max_depth=3, tree_method="hist", random_state=seed, verbosity=0).fit(Xfit, yfit)
                    return [m.predict(Xe) for Xe in Xev]
                m = xgb.XGBClassifier(n_estimators=25, max_depth=3, tree_method="hist", eval_metric="logloss", random_state=seed, verbosity=0).fit(Xfit, yfit.astype(int))
                return [m.predict_proba(Xe)[:, 1] for Xe in Xev]

            [rf_iid_cal, rf_iid_test], _ = _rf(X_tr[iid_fit_i], y_tr[iid_fit_i], [X_tr[iid_cal_i], X_te])
            xgb_iid_cal, xgb_iid_test = _xgb(X_tr[iid_fit_i], y_tr[iid_fit_i], [X_tr[iid_cal_i], X_te])
            [rf_scaf_cal, rf_scaf_test], [rf_sigma_cal, rf_sigma_test] = _rf(X_tr, y_tr, [X_va, X_te])
            xgb_scaf_cal, xgb_scaf_test = _xgb(X_tr, y_tr, [X_va, X_te])

            X_tv = X_global[tv_idx].copy(); X_te_tv = X_global[te_idx].copy()
            sc_tv = StandardScaler().fit(X_tv[:, 2215:])
            X_tv[:, 2215:] = sc_tv.transform(X_tv[:, 2215:])
            X_te_tv[:, 2215:] = sc_tv.transform(X_te_tv[:, 2215:])
            y_tv = tv["Y"].values.astype(np.float64)
            fold_ids = scaffold_group_kfold(tv["canonical_smiles"].tolist(), n_splits=5, seed=seed)
            rf_cv_oof = np.zeros(len(y_tv)); xgb_cv_oof = np.zeros(len(y_tv))
            rf_cv_test = np.zeros((5, len(y_te))); xgb_cv_test = np.zeros((5, len(y_te)))
            if ttype == "regression":
                for k in range(5):
                    m_va = fold_ids == k; m_tr = ~m_va
                    [ro, rt], _ = _rf(X_tv[m_tr], y_tv[m_tr], [X_tv[m_va], X_te_tv])
                    xo, xt = _xgb(X_tv[m_tr], y_tv[m_tr], [X_tv[m_va], X_te_tv])
                    rf_cv_oof[m_va] = ro; rf_cv_test[k] = rt
                    xgb_cv_oof[m_va] = xo; xgb_cv_test[k] = xt
                q10_l = xgb.XGBRegressor(n_estimators=20, max_depth=3, objective="reg:quantileerror", quantile_alpha=0.05, tree_method="hist", random_state=seed, verbosity=0).fit(X_tr, y_tr)
                q10_h = xgb.XGBRegressor(n_estimators=20, max_depth=3, objective="reg:quantileerror", quantile_alpha=0.95, tree_method="hist", random_state=seed, verbosity=0).fit(X_tr, y_tr)
                q20_l = xgb.XGBRegressor(n_estimators=20, max_depth=3, objective="reg:quantileerror", quantile_alpha=0.10, tree_method="hist", random_state=seed, verbosity=0).fit(X_tr, y_tr)
                q20_h = xgb.XGBRegressor(n_estimators=20, max_depth=3, objective="reg:quantileerror", quantile_alpha=0.90, tree_method="hist", random_state=seed, verbosity=0).fit(X_tr, y_tr)
                c10_clo, c10_chi = q10_l.predict(X_va), q10_h.predict(X_va)
                c10_tlo, c10_thi = q10_l.predict(X_te), q10_h.predict(X_te)
                c20_clo, c20_chi = q20_l.predict(X_va), q20_h.predict(X_va)
                c20_tlo, c20_thi = q20_l.predict(X_te), q20_h.predict(X_te)
            else:
                c10_clo = c10_chi = c10_tlo = c10_thi = np.zeros(0)
                c20_clo = c20_chi = c20_tlo = c20_thi = np.zeros(0)

            atomic_save_npz(
                pred_dir / f"tabular__{task}__seed{seed}.npz",
                y_iid_cal=y_tr[iid_cal_i], y_scaf_cal=y_va, y_test=y_te, y_train_val=y_tv,
                y_train_std=np.array([float(np.std(y_tr))], dtype=np.float64),
                d_nn_test=compute_nn_tanimoto_distance(ecfp_global[tr_idx], ecfp_global[te_idx]),
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
                xgb_cqr10_cal_lo=np.minimum(c10_clo, c10_chi), xgb_cqr10_cal_hi=np.maximum(c10_clo, c10_chi),
                xgb_cqr10_test_lo=np.minimum(c10_tlo, c10_thi), xgb_cqr10_test_hi=np.maximum(c10_tlo, c10_thi),
                xgb_cqr20_cal_lo=np.minimum(c20_clo, c20_chi), xgb_cqr20_cal_hi=np.maximum(c20_clo, c20_chi),
                xgb_cqr20_test_lo=np.minimum(c20_tlo, c20_thi), xgb_cqr20_test_hi=np.maximum(c20_tlo, c20_thi),
            )
        print("[OK] Synthesized miniature 2-task Stage 1 workspace (caco2_wang + herg)")

        # 2) Stage 2 mock frozen CLM caches + real CPU 2-member GIN-VN & GPS-Lite training
        clm_dims = {"chemberta2_mtr": 384, "chemberta2_mlm": 384, "molformer_xl": 768}
        for idx_m, (mkey, dim) in enumerate(clm_dims.items()):
            proj = rng.normal(0, 1.0 / np.sqrt(X_global.shape[1]), size=(X_global.shape[1], dim)).astype(np.float32)
            embs = (X_global @ proj + rng.normal(0, 0.05, size=(len(uniq_df), dim))).astype(np.float32)
            atomic_save_npz(
                emb_dir / f"{mkey}__global__{uniq_sha[:12]}.npz",
                embeddings=embs,
                mol_hashes=np.asarray(uniq_df["mol_hash"].tolist(), dtype=np.str_),
            )
        print("[OK] Generated mock frozen CLM caches (chemberta2_mtr, chemberta2_mlm, molformer_xl)")

        graph_cache = {h: smiles_to_graph_dict(s) for h, s in zip(uniq_df["mol_hash"], uniq_df["canonical_smiles"])}
        cpu_dev = torch.device("cpu")
        for task in tasks:
            tv, te = task_frames[task]
            ttype = ALL_TASKS[task]["task_type"]
            sp = create_scaffold_split_pytdc(tv, seed=seed, frac=(0.875, 0.125, 0.0), entity="Drug")
            tr_df, va_df = sp["train"], sp["valid"]
            g_tr = [graph_cache[h] for h in tr_df["mol_hash"]]
            g_va = [graph_cache[h] for h in va_df["mol_hash"]]
            g_te = [graph_cache[h] for h in te["mol_hash"]]
            y_tr = tr_df["Y"].values.astype(np.float64)
            y_va = va_df["Y"].values.astype(np.float64)
            y_te = te["Y"].values.astype(np.float64)

            rng_g = np.random.RandomState(seed * 100 + 7)
            perm = rng_g.permutation(len(g_tr))
            n_iid = max(15, int(round(0.15 * len(g_tr))))
            iid_cal_i, iid_fit_i = perm[:n_iid], perm[n_iid:]
            g_iid_fit = [g_tr[i] for i in iid_fit_i]
            g_iid_cal = [g_tr[i] for i in iid_cal_i]

            payload = {"y_iid_cal": y_tr[iid_cal_i], "y_scaf_cal": y_va, "y_test": y_te}
            for arch in ("gin_vn", "gps_lite"):
                [iid_mu_c, iid_mu_t], _ = train_gnn_ensemble(
                    g_iid_fit, y_tr[iid_fit_i], g_va, y_va, [g_iid_cal, g_te],
                    task_type=ttype, arch=arch, n_ensemble=2, base_seed=seed,
                    device=cpu_dev, hidden_dim=32, num_layers=2, max_epochs=3, patience=2, batch_size=32,
                )
                [scaf_mu_c, scaf_mu_t], [scaf_sd_c, scaf_sd_t] = train_gnn_ensemble(
                    g_tr, y_tr, g_va, y_va, [g_va, g_te],
                    task_type=ttype, arch=arch, n_ensemble=2, base_seed=seed + 10,
                    device=cpu_dev, hidden_dim=32, num_layers=2, max_epochs=3, patience=2, batch_size=32,
                )
                payload[f"{arch}_iid_cal"] = iid_mu_c
                payload[f"{arch}_iid_test"] = iid_mu_t
                payload[f"{arch}_scaf_cal"] = scaf_mu_c
                payload[f"{arch}_scaf_test"] = scaf_mu_t
                payload[f"{arch}_sigma_cal"] = scaf_sd_c
                payload[f"{arch}_sigma_test"] = scaf_sd_t
            atomic_save_npz(pred_dir / f"gnn__{task}__seed{seed}.npz", **payload)
        print("[OK] Trained real CPU 2-member gin_vn & gps_lite ensembles -> saved gnn__<task>__seed1.npz")

        # 3) Execute actual Stage 3 (d_notebook_03 Cells 1-3) against _mock_gpu_test/
        nb3_path = ROOT / "d_notebook_03_conformal_ood_virtual_screening.ipynb"
        nb3 = json.loads(nb3_path.read_text(encoding="utf-8"))
        code_cells = ["".join(c["source"]) for c in nb3["cells"] if c["cell_type"] == "code"]

        c1_src = code_cells[0].replace(
            "BASE_DIR = resolve_workspace(in_colab=IN_COLAB)",
            f"BASE_DIR = Path({repr(str(mock_dir))}).resolve()",
        )
        exec(compile(c1_src, "d_nb03_cell1", "exec"), ns)
        exec(compile(code_cells[1], "d_nb03_cell2", "exec"), ns)
        exec(compile(code_cells[2], "d_nb03_cell3", "exec"), ns)

        # Verify all 7 models & all conformal tables
        expected_models = {
            "ecfp4_rdkit_rf", "ecfp4_rdkit_xgb",
            "chemberta2_mtr", "chemberta2_mlm", "molformer_xl",
            "gin_vn", "gps_lite",
        }
        clm_df = pd.read_csv(tab_dir / "clm_baseline_metrics.csv")
        reg_df = pd.read_csv(tab_dir / "conformal_regression_summary.csv")
        cls_df = pd.read_csv(tab_dir / "conformal_classification_summary.csv")
        bin_df = pd.read_csv(tab_dir / "conformal_tanimoto_bins.csv")
        sel_df = pd.read_csv(tab_dir / "selective_risk_coverage.csv")
        tk_df = pd.read_csv(tab_dir / "virtual_screening_topk.csv")

        assert len(clm_df) == 6, f"Expected 6 CLM rows, got {len(clm_df)}"
        assert set(reg_df["model"]) == expected_models, f"Missing models in reg_df: {set(reg_df['model'])}"
        assert set(cls_df["model"]) == expected_models, f"Missing models in cls_df: {set(cls_df['model'])}"
        assert {"split_iid", "split_scaffold", "cv_plus_scaffold", "locally_adaptive_ncp", "cqr", "weighted_covariate_shift_cp"} == set(reg_df["conformal_method"])
        assert {"icp_iid", "icp_scaffold", "mondrian_scaffold", "weighted_mondrian_cp"} == set(cls_df["conformal_method"])
        unweighted_reg = reg_df[reg_df["conformal_method"] != "weighted_covariate_shift_cp"]
        assert np.isfinite(reg_df["empirical_coverage"]).all()
        assert (unweighted_reg["finite_frac"] == 1.0).all(), f"Unweighted finite_frac < 1: {unweighted_reg[['model', 'conformal_method', 'finite_frac']]}"
        assert (reg_df["ess"] > 1.0).all()
        assert np.isfinite(cls_df["empirical_coverage"]).all() and (cls_df["ess"] > 1.0).all()
        assert not bin_df.empty and not sel_df.empty and not tk_df.empty
        print("[INFO] Mock reg finite_frac by method:", reg_df.groupby("conformal_method")["finite_frac"].mean().round(3).to_dict())

        for npz_p in list(emb_dir.glob("*.npz")) + list(pred_dir.glob("*.npz")):
            with np.load(npz_p, allow_pickle=False) as d:
                assert len(d.files) > 0

        print(
            f"[OK] Stage 3 D-03 loader verified 7 models ({len(reg_df)} reg rows, {len(cls_df)} cls rows, "
            f"{len(tk_df)} Top-K rows, all allow_pickle=False)"
        )
        print("\n[ALL MOCK-GPU STAGE 2 -> STAGE 3 HANDOFF CHECKS PASSED]")
    finally:
        for obj in list(ns.values()):
            if hasattr(obj, "close"):
                try:
                    obj.close()
                except Exception:
                    pass
        ns.clear()
        gc.collect()
        if mock_dir.exists():
            shutil.rmtree(mock_dir, ignore_errors=True)
        assert not mock_dir.exists(), f"Failed to clean up {mock_dir}"
        print("[OK] Cleaned up temporary mock GPU workspace")


if __name__ == "__main__":
    main()
