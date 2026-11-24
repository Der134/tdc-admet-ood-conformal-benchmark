# -*- coding: utf-8 -*-
"""Conformal prediction under scaffold OOD shift + Top-K virtual screening simulator."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from admet_core import tanimoto_bin_label


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample split conformal quantile ceil((n+1)*(1-alpha))/n."""
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n == 0:
        return float("inf")
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        return float("inf")
    rank = max(1, rank)
    sorted_s = np.sort(scores)
    return float(sorted_s[rank - 1])


def weighted_conformal_quantile(
    cal_scores: np.ndarray,
    cal_weights: np.ndarray,
    test_weight: float,
    alpha: float,
) -> float:
    """Tibshirani et al. (2019) weighted split conformal quantile for a test point."""
    cal_scores = np.asarray(cal_scores, dtype=np.float64)
    cal_weights = np.asarray(cal_weights, dtype=np.float64)
    total_w = float(cal_weights.sum() + test_weight)
    if total_w <= 0:
        return float("inf")
    p_cal = cal_weights / total_w
    if float(p_cal.sum()) < 1.0 - alpha:
        return float("inf")
    order = np.argsort(cal_scores, kind="mergesort")
    cum_p = np.cumsum(p_cal[order])
    idx = int(np.searchsorted(cum_p, 1.0 - alpha, side="left"))
    idx = min(idx, len(order) - 1)
    return float(cal_scores[order[idx]])


def estimate_covariate_shift_weights(
    X_cal: np.ndarray,
    X_test: np.ndarray,
    clip_min: float = 0.05,
    clip_max: float = 20.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Estimate unlabeled domain density ratios w(x) = p_test(x)/p_cal(x) via OOF logistic discriminator."""
    n_cal, n_test = len(X_cal), len(X_test)
    if n_cal < 10 or n_test < 10:
        return np.ones(n_cal), np.ones(n_test), float(n_cal), float("nan")

    X_all = np.vstack([X_cal, X_test]).astype(np.float32)
    z_all = np.concatenate([np.zeros(n_cal, dtype=np.int64), np.ones(n_test, dtype=np.int64)])

    # Drop zero-variance columns before scaling
    std_cols = X_all.std(axis=0)
    keep_cols = std_cols > 1e-8
    if not keep_cols.any():
        return np.ones(n_cal), np.ones(n_test), float(n_cal), 0.5
    X_sub = X_all[:, keep_cols]

    n_comp = min(32, X_sub.shape[1], max(2, len(X_all) // 10))
    scaler = StandardScaler().fit(X_sub)
    X_std = scaler.transform(X_sub)
    if n_comp >= 2 and X_sub.shape[1] > n_comp:
        X_proj = PCA(n_components=n_comp, random_state=seed).fit_transform(X_std)
    else:
        X_proj = X_std

    n_splits = min(3, n_cal, n_test)
    oof_probs = np.zeros(len(z_all), dtype=np.float64)
    if n_splits >= 2:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr_idx, va_idx in skf.split(X_proj, z_all):
            clf = LogisticRegression(C=0.25, max_iter=500, random_state=seed)
            clf.fit(X_proj[tr_idx], z_all[tr_idx])
            oof_probs[va_idx] = clf.predict_proba(X_proj[va_idx])[:, 1]
    else:
        clf = LogisticRegression(C=0.25, max_iter=500, random_state=seed).fit(X_proj, z_all)
        oof_probs = clf.predict_proba(X_proj)[:, 1]

    oof_probs = np.clip(oof_probs, 1e-3, 1.0 - 1e-3)
    try:
        domain_auc = float(roc_auc_score(z_all, oof_probs))
    except ValueError:
        domain_auc = float("nan")

    ratio = (oof_probs / (1.0 - oof_probs)) * (n_cal / max(1, n_test))
    weights = np.clip(ratio, clip_min, clip_max)
    w_cal = weights[:n_cal]
    w_test = weights[n_cal:]
    ess = float((w_cal.sum() ** 2) / max(1e-12, (w_cal ** 2).sum()))
    return w_cal, w_test, ess, domain_auc


def _summarize_intervals(
    y_true: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    y_train_std: float,
    alpha: float,
    d_nn: Optional[np.ndarray] = None,
) -> Tuple[dict, List[dict]]:
    y_true = np.asarray(y_true, dtype=np.float64)
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    covered = (y_true >= lo) & (y_true <= hi)
    finite_mask = np.isfinite(lo) & np.isfinite(hi)
    widths = hi - lo
    mean_width = float(np.mean(widths[finite_mask])) if finite_mask.any() else float("inf")
    norm_width = mean_width / max(1e-8, y_train_std)
    cov = float(np.mean(covered))

    summary = {
        "alpha": float(alpha),
        "target_coverage": float(1.0 - alpha),
        "empirical_coverage": cov,
        "coverage_gap": cov - (1.0 - alpha),
        "mean_width": mean_width,
        "norm_width": float(norm_width),
        "finite_frac": float(np.mean(finite_mask)),
        "n_test": int(len(y_true)),
    }

    bin_rows: List[dict] = []
    if d_nn is not None and len(d_nn) == len(y_true):
        bins = tanimoto_bin_label(d_nn)
        for bname in ("Near (<0.45)", "Moderate (0.45-0.65)", "Far (>=0.65)"):
            m = bins == bname
            if m.any():
                b_fin = finite_mask[m]
                b_w = float(np.mean(widths[m][b_fin])) if b_fin.any() else float("inf")
                b_cov = float(np.mean(covered[m]))
                bin_rows.append({
                    "tanimoto_bin": bname,
                    "n_bin": int(m.sum()),
                    "empirical_coverage": b_cov,
                    "coverage_gap": b_cov - (1.0 - alpha),
                    "norm_width": float(b_w / max(1e-8, y_train_std)),
                })
    return summary, bin_rows


def evaluate_regression_conformal(
    y_test: np.ndarray,
    y_train_std: float,
    mu_iid_cal: Optional[np.ndarray],
    y_iid_cal: Optional[np.ndarray],
    mu_iid_test: Optional[np.ndarray],
    mu_scaf_cal: np.ndarray,
    y_scaf_cal: np.ndarray,
    mu_scaf_test: np.ndarray,
    sigma_scaf_cal: Optional[np.ndarray] = None,
    sigma_scaf_test: Optional[np.ndarray] = None,
    cv_oof_mu: Optional[np.ndarray] = None,
    cv_oof_y: Optional[np.ndarray] = None,
    cv_test_preds: Optional[np.ndarray] = None,  # shape (n_train_val, n_test) or (K, n_test) + fold_ids
    cv_fold_ids: Optional[np.ndarray] = None,
    cqr_cal_lo: Optional[np.ndarray] = None,
    cqr_cal_hi: Optional[np.ndarray] = None,
    cqr_test_lo: Optional[np.ndarray] = None,
    cqr_test_hi: Optional[np.ndarray] = None,
    X_scaf_cal: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None,
    d_nn: Optional[np.ndarray] = None,
    alphas: Tuple[float, ...] = (0.10, 0.20),
) -> Tuple[List[dict], List[dict], Dict[Tuple[str, float], Tuple[np.ndarray, np.ndarray]]]:
    """Run full regression conformal suite and return (summary_rows, bin_rows, intervals_by_method_alpha)."""
    summary_rows: List[dict] = []
    bin_rows: List[dict] = []
    intervals: Dict[Tuple[str, float], Tuple[np.ndarray, np.ndarray]] = {}

    # Precompute covariate shift weights once per task/seed
    if X_scaf_cal is not None and X_test is not None:
        w_cal, w_test, ess, dom_auc = estimate_covariate_shift_weights(X_scaf_cal, X_test)
    else:
        w_cal = np.ones(len(y_scaf_cal))
        w_test = np.ones(len(y_test))
        ess, dom_auc = float(len(y_scaf_cal)), float("nan")

    for alpha in alphas:
        method_outputs: Dict[str, Tuple[np.ndarray, np.ndarray, dict]] = {}

        # 1. split_iid
        if mu_iid_cal is not None and y_iid_cal is not None and mu_iid_test is not None:
            q_iid = conformal_quantile(np.abs(y_iid_cal - mu_iid_cal), alpha)
            method_outputs["split_iid"] = (mu_iid_test - q_iid, mu_iid_test + q_iid, {"ess": float(len(y_iid_cal)), "domain_auc": float("nan")})

        # 2. split_scaffold
        res_scaf = np.abs(y_scaf_cal - mu_scaf_cal)
        q_scaf = conformal_quantile(res_scaf, alpha)
        method_outputs["split_scaffold"] = (mu_scaf_test - q_scaf, mu_scaf_test + q_scaf, {"ess": float(len(y_scaf_cal)), "domain_auc": dom_auc})

        # 3. cv_plus_scaffold (Barber et al., 2021)
        if cv_oof_mu is not None and cv_oof_y is not None and cv_test_preds is not None and cv_fold_ids is not None:
            oof_res = np.abs(cv_oof_y - cv_oof_mu)
            # cv_test_preds has shape (n_folds, n_test); map each calibration sample i to its fold's test predictions
            mu_test_per_i = cv_test_preds[cv_fold_ids]  # (n_train_val, n_test)
            lo_mat = mu_test_per_i - oof_res[:, None]
            hi_mat = mu_test_per_i + oof_res[:, None]
            n_tv = len(oof_res)
            q_lo_idx = max(0, math.floor((n_tv + 1) * alpha) - 1)
            q_hi_idx = min(n_tv - 1, math.ceil((n_tv + 1) * (1.0 - alpha)) - 1)
            lo_cv = np.partition(lo_mat, q_lo_idx, axis=0)[q_lo_idx]
            hi_cv = np.partition(hi_mat, q_hi_idx, axis=0)[q_hi_idx]
            method_outputs["cv_plus_scaffold"] = (lo_cv, hi_cv, {"ess": float(n_tv), "domain_auc": dom_auc})

        # 4. locally_adaptive_ncp
        if sigma_scaf_cal is not None and sigma_scaf_test is not None:
            gamma = max(float(np.median(sigma_scaf_cal)), 1e-3)
            norm_res = res_scaf / (sigma_scaf_cal + gamma)
            q_norm = conformal_quantile(norm_res, alpha)
            half_w = q_norm * (sigma_scaf_test + gamma)
            method_outputs["locally_adaptive_ncp"] = (mu_scaf_test - half_w, mu_scaf_test + half_w, {"ess": float(len(y_scaf_cal)), "domain_auc": dom_auc})

        # 5. cqr (Romano et al., 2019)
        if cqr_cal_lo is not None and cqr_cal_hi is not None and cqr_test_lo is not None and cqr_test_hi is not None:
            cqr_scores = np.maximum(cqr_cal_lo - y_scaf_cal, y_scaf_cal - cqr_cal_hi)
            q_cqr = conformal_quantile(cqr_scores, alpha)
            method_outputs["cqr"] = (cqr_test_lo - q_cqr, cqr_test_hi + q_cqr, {"ess": float(len(y_scaf_cal)), "domain_auc": dom_auc})

        # 6. weighted_covariate_shift_cp (Tibshirani et al., 2019)
        if sigma_scaf_cal is not None and sigma_scaf_test is not None:
            gamma = max(float(np.median(sigma_scaf_cal)), 1e-3)
            base_scores = res_scaf / (sigma_scaf_cal + gamma)
            scale_test = sigma_scaf_test + gamma
        else:
            base_scores = res_scaf
            scale_test = np.ones(len(y_test))
        q_w = np.array([weighted_conformal_quantile(base_scores, w_cal, float(wt), alpha) for wt in w_test], dtype=np.float64)
        half_w_w = q_w * scale_test
        method_outputs["weighted_covariate_shift_cp"] = (
            mu_scaf_test - half_w_w,
            mu_scaf_test + half_w_w,
            {"ess": ess, "domain_auc": dom_auc},
        )

        for method_name, (lo, hi, extra) in method_outputs.items():
            intervals[(method_name, alpha)] = (lo, hi)
            srow, brows = _summarize_intervals(y_test, lo, hi, y_train_std, alpha, d_nn=d_nn)
            srow.update({"conformal_method": method_name, **extra})
            summary_rows.append(srow)
            for br in brows:
                br.update({"conformal_method": method_name, "alpha": float(alpha)})
                bin_rows.append(br)

    return summary_rows, bin_rows, intervals


def _cls_prediction_sets_marginal(
    p_cal: np.ndarray,
    y_cal: np.ndarray,
    p_test: np.ndarray,
    alpha: float,
    w_cal: Optional[np.ndarray] = None,
    w_test: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (set_mask [N_test, 2] bool for {0, 1}, conformal_pvals [N_test, 2])."""
    p_cal = np.clip(p_cal, 1e-7, 1.0 - 1e-7)
    p_test = np.clip(p_test, 1e-7, 1.0 - 1e-7)
    y_cal = y_cal.astype(int)
    scores_cal = np.where(y_cal == 1, 1.0 - p_cal, p_cal)
    scores_test = np.column_stack([p_test, 1.0 - p_test])  # s(x, 0)=p1, s(x, 1)=1-p1

    n = len(scores_cal)
    if w_cal is None or w_test is None:
        q = conformal_quantile(scores_cal, alpha)
        sets = scores_test <= q
        pvals = np.column_stack([
            ((scores_cal[:, None] >= scores_test[:, 0][None, :]).sum(axis=0) + 1.0) / (n + 1.0),
            ((scores_cal[:, None] >= scores_test[:, 1][None, :]).sum(axis=0) + 1.0) / (n + 1.0),
        ])
        return sets, pvals

    sets = np.zeros((len(p_test), 2), dtype=bool)
    pvals = np.zeros((len(p_test), 2), dtype=np.float64)
    for j in range(len(p_test)):
        qj = weighted_conformal_quantile(scores_cal, w_cal, float(w_test[j]), alpha)
        sets[j] = scores_test[j] <= qj
        tot = float(w_cal.sum() + w_test[j])
        for c in (0, 1):
            pvals[j, c] = (float(w_cal[scores_cal >= scores_test[j, c]].sum()) + float(w_test[j])) / tot
    return sets, pvals


def _cls_prediction_sets_mondrian(
    p_cal: np.ndarray,
    y_cal: np.ndarray,
    p_test: np.ndarray,
    alpha: float,
    w_cal: Optional[np.ndarray] = None,
    w_test: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Class-conditional Mondrian ICP (marginal coverage guaranteed per class Y=0 and Y=1)."""
    p_cal = np.clip(p_cal, 1e-7, 1.0 - 1e-7)
    p_test = np.clip(p_test, 1e-7, 1.0 - 1e-7)
    y_cal = y_cal.astype(int)
    scores_cal = np.where(y_cal == 1, 1.0 - p_cal, p_cal)
    scores_test = np.column_stack([p_test, 1.0 - p_test])

    idx0 = np.where(y_cal == 0)[0]
    idx1 = np.where(y_cal == 1)[0]
    s0, s1 = scores_cal[idx0], scores_cal[idx1]

    if w_cal is None or w_test is None:
        q0 = conformal_quantile(s0, alpha)
        q1 = conformal_quantile(s1, alpha)
        sets = np.column_stack([scores_test[:, 0] <= q0, scores_test[:, 1] <= q1])
        pv0 = ((s0[:, None] >= scores_test[:, 0][None, :]).sum(axis=0) + 1.0) / (len(s0) + 1.0) if len(s0) else np.ones(len(p_test))
        pv1 = ((s1[:, None] >= scores_test[:, 1][None, :]).sum(axis=0) + 1.0) / (len(s1) + 1.0) if len(s1) else np.ones(len(p_test))
        return sets, np.column_stack([pv0, pv1])

    w0, w1 = w_cal[idx0], w_cal[idx1]
    sets = np.zeros((len(p_test), 2), dtype=bool)
    pvals = np.zeros((len(p_test), 2), dtype=np.float64)
    for j in range(len(p_test)):
        wt = float(w_test[j])
        q0j = weighted_conformal_quantile(s0, w0, wt, alpha) if len(s0) else float("inf")
        q1j = weighted_conformal_quantile(s1, w1, wt, alpha) if len(s1) else float("inf")
        sets[j, 0] = scores_test[j, 0] <= q0j
        sets[j, 1] = scores_test[j, 1] <= q1j
        pvals[j, 0] = (float(w0[s0 >= scores_test[j, 0]].sum()) + wt) / float(w0.sum() + wt) if len(s0) else 1.0
        pvals[j, 1] = (float(w1[s1 >= scores_test[j, 1]].sum()) + wt) / float(w1.sum() + wt) if len(s1) else 1.0
    return sets, pvals


def evaluate_classification_conformal(
    y_test: np.ndarray,
    p_iid_cal: Optional[np.ndarray],
    y_iid_cal: Optional[np.ndarray],
    p_iid_test: Optional[np.ndarray],
    p_scaf_cal: np.ndarray,
    y_scaf_cal: np.ndarray,
    p_scaf_test: np.ndarray,
    sigma_scaf_test: Optional[np.ndarray] = None,
    X_scaf_cal: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None,
    alphas: Tuple[float, ...] = (0.10, 0.20),
) -> Tuple[List[dict], List[dict], Dict[Tuple[str, float], Tuple[np.ndarray, np.ndarray]]]:
    """Run binary classification conformal + selective prediction risk-coverage curves."""
    y_test = y_test.astype(int)
    summary_rows: List[dict] = []
    curve_rows: List[dict] = []
    outputs: Dict[Tuple[str, float], Tuple[np.ndarray, np.ndarray]] = {}

    if X_scaf_cal is not None and X_test is not None:
        w_cal, w_test, ess, dom_auc = estimate_covariate_shift_weights(X_scaf_cal, X_test)
    else:
        w_cal, w_test, ess, dom_auc = np.ones(len(y_scaf_cal)), np.ones(len(y_test)), float(len(y_scaf_cal)), float("nan")

    for alpha in alphas:
        methods: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}
        if p_iid_cal is not None and y_iid_cal is not None and p_iid_test is not None:
            s_iid, pv_iid = _cls_prediction_sets_marginal(p_iid_cal, y_iid_cal, p_iid_test, alpha)
            methods["icp_iid"] = (s_iid, pv_iid, float(len(y_iid_cal)))

        s_scaf, pv_scaf = _cls_prediction_sets_marginal(p_scaf_cal, y_scaf_cal, p_scaf_test, alpha)
        methods["icp_scaffold"] = (s_scaf, pv_scaf, float(len(y_scaf_cal)))

        s_mon, pv_mon = _cls_prediction_sets_mondrian(p_scaf_cal, y_scaf_cal, p_scaf_test, alpha)
        methods["mondrian_scaffold"] = (s_mon, pv_mon, float(len(y_scaf_cal)))

        s_wmon, pv_wmon = _cls_prediction_sets_mondrian(p_scaf_cal, y_scaf_cal, p_scaf_test, alpha, w_cal=w_cal, w_test=w_test)
        methods["weighted_mondrian_cp"] = (s_wmon, pv_wmon, ess)

        for mname, (sets, pvals, m_ess) in methods.items():
            outputs[(mname, alpha)] = (sets, pvals)
            covered = sets[np.arange(len(y_test)), y_test]
            set_sizes = sets.sum(axis=1)
            singleton_mask = set_sizes == 1
            empty_mask = set_sizes == 0
            ambig_mask = set_sizes == 2
            pos_mask = y_test == 1
            neg_mask = y_test == 0

            if singleton_mask.any():
                sing_pred = sets[singleton_mask, 1].astype(int)
                sing_acc = float(np.mean(sing_pred == y_test[singleton_mask]))
            else:
                sing_acc = float("nan")

            summary_rows.append({
                "conformal_method": mname,
                "alpha": float(alpha),
                "target_coverage": float(1.0 - alpha),
                "empirical_coverage": float(np.mean(covered)),
                "coverage_gap": float(np.mean(covered) - (1.0 - alpha)),
                "coverage_pos": float(np.mean(covered[pos_mask])) if pos_mask.any() else float("nan"),
                "coverage_neg": float(np.mean(covered[neg_mask])) if neg_mask.any() else float("nan"),
                "avg_set_size": float(np.mean(set_sizes)),
                "singleton_rate": float(np.mean(singleton_mask)),
                "abstention_rate": float(np.mean(ambig_mask)),
                "empty_rate": float(np.mean(empty_mask)),
                "singleton_accuracy": sing_acc,
                "ess": m_ess,
                "domain_auc": dom_auc,
                "n_test": int(len(y_test)),
            })

    # Selective prediction risk-coverage curves based on confidence margin & epistemic std
    y_hat = (p_scaf_test >= 0.5).astype(int)
    errors = (y_hat != y_test).astype(np.float64)
    conf_margin = np.abs(p_scaf_test - 0.5)
    order_conf = np.argsort(-conf_margin, kind="mergesort")
    for keep in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        nk = max(1, int(round(keep * len(y_test))))
        kept = order_conf[:nk]
        curve_rows.append({
            "uncertainty_signal": "prob_margin",
            "keep_fraction": keep,
            "n_keep": nk,
            "risk_error_rate": float(np.mean(errors[kept])),
            "selective_accuracy": float(1.0 - np.mean(errors[kept])),
        })
    if sigma_scaf_test is not None:
        order_epi = np.argsort(sigma_scaf_test, kind="mergesort")
        for keep in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
            nk = max(1, int(round(keep * len(y_test))))
            kept = order_epi[:nk]
            curve_rows.append({
                "uncertainty_signal": "epistemic_std",
                "keep_fraction": keep,
                "n_keep": nk,
                "risk_error_rate": float(np.mean(errors[kept])),
                "selective_accuracy": float(1.0 - np.mean(errors[kept])),
            })

    return summary_rows, curve_rows, outputs


def simulate_topk_screening(
    y_true: np.ndarray,
    task_type: str,
    favorable: str,
    y_pred_point: np.ndarray,
    conf_lo: Optional[np.ndarray] = None,
    conf_hi: Optional[np.ndarray] = None,
    cls_sets: Optional[np.ndarray] = None,
    cls_pvals: Optional[np.ndarray] = None,
    y_cal_for_threshold: Optional[np.ndarray] = None,
    fixed_hit_threshold: Optional[float] = None,
    k_values: Tuple[int, ...] = (10, 20, 50),
) -> List[dict]:
    """Simulate 'Buy Top-K Molecules' decision-making: Greedy point ranking vs Conformal Risk-Controlled."""
    y_true = np.asarray(y_true, dtype=np.float64)
    n = len(y_true)
    if n < 10:
        return []

    if task_type == "regression":
        if fixed_hit_threshold is not None:
            is_hit = (y_true >= fixed_hit_threshold) if favorable == "high" else (y_true <= fixed_hit_threshold)
        else:
            ref = y_cal_for_threshold if y_cal_for_threshold is not None else y_true
            if favorable == "high":
                thresh = float(np.quantile(ref, 0.75))
                is_hit = y_true >= thresh
            else:
                thresh = float(np.quantile(ref, 0.25))
                is_hit = y_true <= thresh

        greedy_score = y_pred_point if favorable == "high" else -y_pred_point
        if conf_lo is not None and conf_hi is not None:
            # Conformal Lower Confidence Bound for high, negative Upper Confidence Bound for low
            conf_score = conf_lo if favorable == "high" else -conf_hi
            # Replace infinities gracefully if any test point collapsed
            finite = np.isfinite(conf_score)
            if not finite.all():
                min_fin = float(np.min(conf_score[finite])) if finite.any() else -1e9
                conf_score = np.where(finite, conf_score, min_fin - 1.0)
        else:
            conf_score = greedy_score
    else:
        target_cls = 1 if favorable == "high" else 0
        other_cls = 1 - target_cls
        is_hit = y_true.astype(int) == target_cls
        greedy_score = y_pred_point if target_cls == 1 else (1.0 - y_pred_point)
        if cls_sets is not None and cls_pvals is not None:
            is_cert_singleton = (cls_sets.sum(axis=1) == 1) & cls_sets[:, target_cls]
            # Rank certified singletons first, ordered by high target p-val & low opposite-class p-val
            conf_score = is_cert_singleton.astype(np.float64) * 10.0 + (cls_pvals[:, target_cls] - cls_pvals[:, other_cls])
        else:
            conf_score = greedy_score

    base_rate = float(np.mean(is_hit))
    if base_rate <= 0.0 or base_rate >= 1.0:
        return []

    greedy_order = np.argsort(-greedy_score, kind="mergesort")
    conf_order = np.argsort(-conf_score, kind="mergesort")

    rows: List[dict] = []
    for k in k_values:
        if k > n:
            continue
        g_idx = greedy_order[:k]
        c_idx = conf_order[:k]
        g_hits = int(is_hit[g_idx].sum())
        c_hits = int(is_hit[c_idx].sum())
        g_fp = k - g_hits
        c_fp = k - c_hits
        fp_saved = g_fp - c_fp
        rel_fp_red = (fp_saved / max(1, g_fp)) * 100.0 if g_fp > 0 else 0.0

        rows.append({
            "K": int(k),
            "n_test": int(n),
            "base_hit_rate": base_rate,
            "greedy_hits": g_hits,
            "greedy_hit_rate": g_hits / k,
            "greedy_fp": g_fp,
            "greedy_enrichment": (g_hits / k) / base_rate,
            "conformal_hits": c_hits,
            "conformal_hit_rate": c_hits / k,
            "conformal_fp": c_fp,
            "conformal_enrichment": (c_hits / k) / base_rate,
            "fp_saved": fp_saved,
            "relative_fp_reduction_pct": float(rel_fp_red),
        })
    return rows
