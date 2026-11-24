# -*- coding: utf-8 -*-
"""Deterministic CPU unit, security guardrail & statistical protocol verification suite for Direction D."""
import ast
import inspect
import json
import math
import sys
import tempfile
import zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from admet_core import (
    ADMET_TASKS, ALLOWED_HOSTS, CHEMBL_TARGETS, _validate_remote_url,
    atomic_save_npz, atomic_write_bytes, compute_nn_tanimoto_distance,
    confine_path, create_scaffold_split_pytdc, featurize_ecfp_rdkit,
    get_scaffold_smiles, resolve_workspace, scaffold_group_kfold,
)
from conformal_ood import (
    _cls_prediction_sets_marginal, _cls_prediction_sets_mondrian,
    conformal_quantile, estimate_covariate_shift_weights,
    evaluate_classification_conformal, evaluate_regression_conformal,
    simulate_topk_screening, weighted_conformal_quantile,
)
from gin_gps_model import MolecularGNN, collate_graphs, smiles_to_graph_dict

# 1. Task registry & security guardrails
assert len(ADMET_TASKS) == 22, f"Expected 22 TDC tasks, got {len(ADMET_TASKS)}"
assert len(CHEMBL_TARGETS) == 3
assert ALLOWED_HOSTS == {
    "dataverse.harvard.edu",
    "dvn-cloud-iqss.s3.amazonaws.com",
    "dvn-cloud.s3.amazonaws.com",
    "www.ebi.ac.uk",
}

for bad_url in [
    "http://dataverse.harvard.edu/api/access/datafile/4426004",
    "https://127.0.0.1/secret",
    "https://localhost/test",
    "https://169.254.169.254/latest/meta-data/",
    "https://10.0.0.1/internal",
    "https://evil.example.com/payload.zip",
]:
    try:
        _validate_remote_url(bad_url)
        raise AssertionError(f"Failed to block unsafe URL: {bad_url}")
    except (ValueError, RuntimeError):
        pass

with tempfile.TemporaryDirectory() as td:
    base = Path(td)
    for bad_rel in ["../escape.txt", "../../etc/passwd", "/abs/escape.txt"]:
        try:
            confine_path(base, base / bad_rel)
            raise AssertionError(f"confine_path failed to catch traversal: {bad_rel}")
        except ValueError:
            pass

    # Zip-Slip simulation with malicious archive member
    zpath = base / "malicious.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("../../pwned.txt", "bad")
    with zipfile.ZipFile(zpath, "r") as zf:
        for m in zf.infolist():
            try:
                confine_path(base / "extract", (base / "extract").joinpath(*Path(m.filename).parts))
                raise AssertionError("Zip-Slip member was not blocked")
            except ValueError:
                pass

    # Atomic write & NPZ zero-pickle check
    bin_out = base / "atomic.bin"
    atomic_write_bytes(bin_out, b"direction-d-ok")
    assert bin_out.read_bytes() == b"direction-d-ok"
    assert not (base / "atomic.bin.tmp").exists()

    npz_out = base / "safe.npz"
    atomic_save_npz(npz_out, vals=np.array([1.0, 2.0]), labels=np.asarray(["a", "b"], dtype=np.str_))
    assert not (base / "safe.npz.tmp").exists()
    with np.load(npz_out, allow_pickle=False) as loaded:
        assert np.allclose(loaded["vals"], [1.0, 2.0])
        assert loaded["labels"].tolist() == ["a", "b"]

# Static AST / source audit across core & builder files (zero eval(), zero allow_pickle=True, G4 guard, Drive reuse)
audit_files = [
    "admet_core.py", "conformal_ood.py", "gin_gps_model.py",
    "build_d_nb01.py", "build_d_nb02.py", "build_d_nb03.py",
]
for fname in audit_files:
    src = (ROOT / fname).read_text(encoding="utf-8")
    assert "allow_pickle=True" not in src, f"Forbidden allow_pickle=True in {fname}"
    tree = ast.parse(src, filename=fname)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "eval":
            raise AssertionError(f"Forbidden builtin eval() call in {fname}")

nb02_json = json.loads((ROOT / "d_notebook_02_clm_gnn_representation_gpu.ipynb").read_text(encoding="utf-8"))
nb02_code = "\n".join("".join(c["source"]) for c in nb02_json["cells"] if c["cell_type"] == "code")
assert "ALLOW_G4 = False" in nb02_code and "GPU_VRAM_GB >= 70" in nb02_code, "Missing G4 anti-waste guard in D-02"

core_src = inspect.getsource(resolve_workspace)
assert "flip2_benchmark" in core_src and "direction_d_admet" in core_src, "Missing shared Drive auto-reuse in resolve_workspace"
print("[OK] Security checks: HTTPS/allowlist, IP rejection, Zip-Slip, zero eval()/pickle, atomic I/O, G4 guard, Drive reuse")

# 2. Scaffold split & zero shared scaffolds check
smiles_pool = [
    "c1ccccc1C(=O)O", "c1ccccc1CC(=O)O", "c1ccccc1NC(=O)C",
    "c1ccncc1C(=O)N", "c1ccncc1CC(=O)N", "c1ccncc1OC",
    "C1CCCCC1N", "C1CCCCC1CO", "C1CCCCC1C(=O)O",
    "c1ccc2ccccc2c1O", "c1ccc2ccccc2c1N", "c1ccc2ccccc2c1C",
    "c1ccc(cc1)c2ccccc2", "c1ccc(cc1)c2ccccc2O",
    "CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
]
df_sc = pd.DataFrame({"Drug": smiles_pool, "Y": np.linspace(0, 1, len(smiles_pool))})
sp = create_scaffold_split_pytdc(df_sc, seed=1, frac=(0.75, 0.25, 0.0), entity="Drug")
tr_scafs = set(sp["train"]["Drug"].map(get_scaffold_smiles))
va_scafs = set(sp["valid"]["Drug"].map(get_scaffold_smiles))
assert len(tr_scafs & va_scafs) == 0, f"Shared scaffolds across split: {tr_scafs & va_scafs}"
folds = scaffold_group_kfold(smiles_pool, n_splits=3, seed=1)
assert set(folds.tolist()).issubset({0, 1, 2})
print("[OK] Bemis-Murcko scaffold split & scaffold K-fold verified (0 shared scaffolds)")

# 3. Featurization & exact bitwise Tanimoto distance check against RDKit
X_feat, bits, valid = featurize_ecfp_rdkit(smiles_pool)
assert X_feat.shape == (len(smiles_pool), 2231) and valid.all()
d_nn = compute_nn_tanimoto_distance(bits[:8], bits[8:])
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
tr_fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles_pool[:8]]
for j, s in enumerate(smiles_pool[8:]):
    fp = gen.GetFingerprint(Chem.MolFromSmiles(s))
    exp_d = 1.0 - max(DataStructs.TanimotoSimilarity(fp, tfp) for tfp in tr_fps)
    assert abs(d_nn[j] - exp_d) < 1e-5, f"Tanimoto mismatch: {d_nn[j]} vs {exp_d}"
print("[OK] ECFP4+MACCS+RDKit16 featurization & exact Tanimoto distance verified")

# 4. Pure-PyTorch GIN-VN & GPS-Lite forward/backward (including single-atom empty-bond edge case)
graphs = [smiles_to_graph_dict(s) for s in ["C", "c1ccccc1", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]]
batch_g = collate_graphs(graphs, np.array([1.0, 2.0, 3.0]))
for arch in ("gin_vn", "gps_lite"):
    net = MolecularGNN(arch=arch, hidden_dim=32, num_layers=2)
    out = net(batch_g)
    assert out.shape == (3,) and torch.isfinite(out).all()
    out.sum().backward()
print("[OK] Pure-PyTorch GIN-VN & GPS-Lite forward/backward verified (incl. single-atom edge case)")

# 5. Statistical rigor: zero test-label leakage, finite-sample quantile, exact CV+ / CQR / Mondrian / Top-K
sig_cs = inspect.signature(estimate_covariate_shift_weights)
assert list(sig_cs.parameters.keys()) == ["X_cal", "X_test", "clip_min", "clip_max", "seed"], (
    f"Unexpected parameters in estimate_covariate_shift_weights: {list(sig_cs.parameters.keys())}"
)

assert abs(conformal_quantile(np.arange(1.0, 10.0), 0.10) - 9.0) < 1e-8
assert math.isinf(conformal_quantile(np.arange(1.0, 9.0), 0.10))  # ceil(9*0.9)=9 > 8 -> inf
assert weighted_conformal_quantile(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 1.0]), 0.1, 0.10) == 3.0

rng = np.random.RandomState(42)
X_cal_syn = rng.randn(80, 12); X_te_syn = rng.randn(60, 12) + 0.8
w_c, w_t, ess, d_auc = estimate_covariate_shift_weights(X_cal_syn, X_te_syn)
assert len(w_c) == 80 and len(w_t) == 60 and ess > 10.0 and d_auc > 0.6

# Exact Barber et al. (2021) CV+ & Romano et al. (2019) CQR hand-check
cv_oof_y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
cv_oof_mu = np.array([0.8, 2.3, 2.7, 4.4, 4.9, 6.2, 6.5, 8.1, 8.7, 10.5])
cv_fold_ids = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
cv_test_preds = np.array([[1.0, 2.0], [1.2, 2.2], [0.9, 2.1], [1.1, 1.9], [1.3, 2.3]])
cqr_cal_lo = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
cqr_cal_hi = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
y_cqr_cal = np.array([1.2, 1.8, 3.5, 5.4, 4.7, 6.5, 7.2, 8.8, 9.1])  # max err = 0.4 at idx 3 -> q_cqr = 0.4
_, _, exact_ivs = evaluate_regression_conformal(
    y_test=np.array([1.1, 2.0]), y_train_std=1.0,
    mu_iid_cal=None, y_iid_cal=None, mu_iid_test=None,
    mu_scaf_cal=cqr_cal_lo, y_scaf_cal=y_cqr_cal, mu_scaf_test=np.array([1.0, 2.0]),
    cv_oof_mu=cv_oof_mu, cv_oof_y=cv_oof_y, cv_test_preds=cv_test_preds, cv_fold_ids=cv_fold_ids,
    cqr_cal_lo=cqr_cal_lo[:9], cqr_cal_hi=cqr_cal_hi[:9],
    cqr_test_lo=np.array([0.5, 1.5]), cqr_test_hi=np.array([1.5, 2.5]),
    alphas=(0.10,),
)
oof_res = np.abs(cv_oof_y - cv_oof_mu)
mu_t_i = cv_test_preds[cv_fold_ids]
exp_lo = np.sort(mu_t_i - oof_res[:, None], axis=0)[math.floor(11 * 0.10) - 1]
exp_hi = np.sort(mu_t_i + oof_res[:, None], axis=0)[math.ceil(11 * 0.90) - 1]
assert np.allclose(exact_ivs[("cv_plus_scaffold", 0.10)][0], exp_lo)
assert np.allclose(exact_ivs[("cv_plus_scaffold", 0.10)][1], exp_hi)
assert np.allclose(exact_ivs[("cqr", 0.10)][0], [0.1, 1.1])
assert np.allclose(exact_ivs[("cqr", 0.10)][1], [1.9, 2.9])

# Mondrian class-conditional recovery on imbalanced binary task (90% neg, 10% pos)
n_cal_imb = 200
y_cal_imb = np.array([0] * 180 + [1] * 20)
p_cal_imb = np.concatenate([rng.uniform(0.02, 0.25, 180), rng.uniform(0.35, 0.70, 20)])
y_te_imb = np.array([0] * 180 + [1] * 20)
p_te_imb = np.concatenate([rng.uniform(0.02, 0.25, 180), rng.uniform(0.35, 0.70, 20)])
s_marg, _ = _cls_prediction_sets_marginal(p_cal_imb, y_cal_imb, p_te_imb, alpha=0.10)
s_mond, _ = _cls_prediction_sets_mondrian(p_cal_imb, y_cal_imb, p_te_imb, alpha=0.10)
marg_pos_cov = s_marg[y_te_imb == 1, 1].mean()
mond_pos_cov = s_mond[y_te_imb == 1, 1].mean()
assert marg_pos_cov < 0.60 and mond_pos_cov >= 0.80, f"Unexpected Mondrian check: marg={marg_pos_cov}, mond={mond_pos_cov}"

reg_s, reg_b, ivs = evaluate_regression_conformal(
    y_test=rng.randn(60), y_train_std=1.0,
    mu_iid_cal=rng.randn(40), y_iid_cal=rng.randn(40), mu_iid_test=rng.randn(60),
    mu_scaf_cal=rng.randn(80), y_scaf_cal=rng.randn(80), mu_scaf_test=rng.randn(60),
    sigma_scaf_cal=np.abs(rng.randn(80)) + 0.1, sigma_scaf_test=np.abs(rng.randn(60)) + 0.1,
    cv_oof_mu=rng.randn(100), cv_oof_y=rng.randn(100),
    cv_test_preds=rng.randn(5, 60), cv_fold_ids=np.arange(100) % 5,
    cqr_cal_lo=rng.randn(80) - 1.0, cqr_cal_hi=rng.randn(80) + 1.0,
    cqr_test_lo=rng.randn(60) - 1.0, cqr_test_hi=rng.randn(60) + 1.0,
    X_scaf_cal=X_cal_syn, X_test=X_te_syn, d_nn=rng.uniform(0.3, 0.8, 60), alphas=(0.10,),
)
methods_present = {r["conformal_method"] for r in reg_s}
assert methods_present == {"split_iid", "split_scaffold", "cv_plus_scaffold", "locally_adaptive_ncp", "cqr", "weighted_covariate_shift_cp"}

cls_s, cls_c, cls_o = evaluate_classification_conformal(
    y_test=rng.randint(0, 2, 60),
    p_iid_cal=rng.uniform(0.1, 0.9, 40), y_iid_cal=rng.randint(0, 2, 40), p_iid_test=rng.uniform(0.1, 0.9, 60),
    p_scaf_cal=rng.uniform(0.1, 0.9, 80), y_scaf_cal=rng.randint(0, 2, 80), p_scaf_test=rng.uniform(0.1, 0.9, 60),
    sigma_scaf_test=rng.uniform(0.01, 0.2, 60), X_scaf_cal=X_cal_syn, X_test=X_te_syn, alphas=(0.10,),
)
assert {r["conformal_method"] for r in cls_s} == {"icp_iid", "icp_scaffold", "mondrian_scaffold", "weighted_mondrian_cp"}

# Top-K screening FP reduction exactness check
tk_y = np.array([10.0] * 15 + [1.0] * 45)
tk_greedy = np.concatenate([np.linspace(1.0, 5.0, 15), np.linspace(6.0, 9.0, 10), np.zeros(35)])
tk_lo = np.concatenate([np.linspace(6.0, 8.0, 15), np.zeros(45)])
tk_res = simulate_topk_screening(
    y_true=tk_y, task_type="regression", favorable="high",
    y_pred_point=tk_greedy, conf_lo=tk_lo, conf_hi=tk_lo + 1.0, k_values=(10,),
)
assert len(tk_res) == 1
assert tk_res[0]["greedy_fp"] == 10 and tk_res[0]["conformal_fp"] == 0
assert tk_res[0]["fp_saved"] == 10 and abs(tk_res[0]["relative_fp_reduction_pct"] - 100.0) < 1e-8
print("[OK] Conformal OOD suite (CV+, CQR, Weighted-CP, Mondrian imbalanced recovery, Top-K FP math) verified")
print("\n[ALL DIRECTION D UNIT & LOGIC CHECKS PASSED]")
