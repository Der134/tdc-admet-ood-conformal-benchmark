# -*- coding: utf-8 -*-
"""Regenerate publication-grade Fig 3, Fig 4, and Fig 5 directly from verified 7-model CSV tables."""
from pathlib import Path
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent
TAB_DIR = ROOT / "results" / "tables"
FIG_DIR = ROOT / "results" / "figures"

sns.set_theme(style="whitegrid", context="paper")

MODEL_ORDER = [
    "ecfp4_rdkit_rf",
    "ecfp4_rdkit_xgb",
    "molformer_xl",
    "chemberta2_mlm",
    "chemberta2_mtr",
    "gps_lite",
    "gin_vn",
]
MODEL_LABELS = {
    "ecfp4_rdkit_rf": "RF (ECFP4+RDKit)",
    "ecfp4_rdkit_xgb": "XGB (ECFP4+RDKit)",
    "molformer_xl": "MoLFormer-XL",
    "chemberta2_mlm": "ChemBERTa-2 MLM",
    "chemberta2_mtr": "ChemBERTa-2 MTR",
    "gps_lite": "GPS-Lite (GNN)",
    "gin_vn": "GIN-VN (GNN)",
}
MODEL_COLORS = {
    "ecfp4_rdkit_rf": "#1b9e77",
    "ecfp4_rdkit_xgb": "#d95f02",
    "molformer_xl": "#7570b3",
    "chemberta2_mlm": "#e7298a",
    "chemberta2_mtr": "#e6ab02",
    "gps_lite": "#a6761d",
    "gin_vn": "#386cb0",
}


def save_fig(fig, name: str) -> None:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    (FIG_DIR / name).write_bytes(buf.getvalue())
    plt.close(fig)
    print(f"[FIG] Regenerated {name}")


def plot_fig3() -> None:
    cls_df = pd.read_csv(TAB_DIR / "conformal_classification_summary.csv")
    sel_df = pd.read_csv(TAB_DIR / "selective_risk_coverage.csv")

    c10 = cls_df[cls_df["alpha"] == 0.10]
    cm_order = ["icp_iid", "icp_scaffold", "mondrian_scaffold", "weighted_mondrian_cp"]
    cm_labels = {
        "icp_iid": "ICP IID\n(Marginal)",
        "icp_scaffold": "ICP Scaffold\n(Marginal)",
        "mondrian_scaffold": "Mondrian\nScaffold",
        "weighted_mondrian_cp": "Weighted\nMondrian",
    }

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6), gridspec_kw={"width_ratios": [1.0, 1.22]})

    # Panel A: Bilateral class-conditional coverage
    melt_c = c10.melt(
        id_vars=["task", "seed", "model", "conformal_method"],
        value_vars=["coverage_pos", "coverage_neg"],
        var_name="class",
        value_name="coverage",
    )
    melt_c["class"] = melt_c["class"].map({"coverage_pos": "Positive (Y=1)", "coverage_neg": "Negative (Y=0)"})
    ax0 = axes[0]
    sns.barplot(
        data=melt_c, x="conformal_method", y="coverage", hue="class",
        order=cm_order, ax=ax0, palette=["#6baed6", "#08519c"], errorbar=("ci", 95), err_kws={"linewidth": 1.2},
    )
    ax0.axhline(0.90, ls="--", color="crimson", lw=1.35, label="90% Nominal Target")
    for container in ax0.containers:
        ax0.bar_label(container, fmt="%.3f", fontsize=7.4, padding=7)
    ax0.set_xticks(range(len(cm_order)))
    ax0.set_xticklabels([cm_labels[m] for m in cm_order], fontsize=8.5)
    ax0.set_xlabel("")
    ax0.set_ylabel("Class-Conditional Empirical Coverage")
    ax0.set_title("A. Marginal vs. Mondrian Coverage (13 Tasks x 7 Models)")
    ax0.set_ylim(0.55, 1.06)
    ax0.legend(loc="lower right", fontsize=8, frameon=True)

    # Panel B: Clean Selective Risk-Coverage curves (no muddy overlapping bands, external legend)
    ax1 = axes[1]
    agg_sel = sel_df.groupby(["uncertainty_signal", "model", "keep_fraction"], as_index=False)["risk_error_rate"].mean()
    for m in MODEL_ORDER:
        col = MODEL_COLORS[m]
        sub_pm = agg_sel[(agg_sel["model"] == m) & (agg_sel["uncertainty_signal"] == "prob_margin")].sort_values("keep_fraction", ascending=False)
        if not sub_pm.empty:
            ax1.plot(
                sub_pm["keep_fraction"], sub_pm["risk_error_rate"],
                color=col, ls="-", marker="o", ms=4.5, lw=1.8, label=MODEL_LABELS[m],
            )
        sub_ep = agg_sel[(agg_sel["model"] == m) & (agg_sel["uncertainty_signal"] == "epistemic_std")].sort_values("keep_fraction", ascending=False)
        if not sub_ep.empty:
            ax1.plot(
                sub_ep["keep_fraction"], sub_ep["risk_error_rate"],
                color=col, ls="--", marker="^", ms=4.0, lw=1.2, alpha=0.72,
            )

    # Style key entries for solid vs dashed signal lines
    ax1.plot([], [], color="#333333", ls="-", marker="o", ms=4.5, lw=1.8, label="Solid: Prob. Margin")
    ax1.plot([], [], color="#333333", ls="--", marker="^", ms=4.0, lw=1.2, label="Dashed: Epistemic Std")
    ax1.invert_xaxis()
    ax1.set_xticks([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    ax1.set_xlabel("Retained Test Fraction (Coverage)")
    ax1.set_ylabel("Mean Classification Error Rate (Risk)")
    ax1.set_title("B. Selective Prediction Risk-Coverage Across All 7 Models")
    ax1.set_ylim(0.08, 0.285)
    ax1.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=7.8, frameon=True)

    fig.tight_layout()
    save_fig(fig, "fig3_mondrian_imbalanced_classification.png")


def plot_fig4() -> None:
    topk = pd.read_csv(TAB_DIR / "virtual_screening_topk.csv")
    tk20 = topk[topk["K"] == 20]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.7), gridspec_kw={"width_ratios": [1.05, 1.15]})

    # Panel A: Regression Top-20 Hit Rate by Model (Greedy vs Conformal LCB)
    ax0 = axes[0]
    reg20 = tk20[tk20["task_type"] == "regression"].groupby("model", as_index=False)[["greedy_hit_rate", "conformal_hit_rate"]].mean()
    reg20["order"] = reg20["model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    reg20 = reg20.sort_values("order")

    x = np.arange(len(reg20))
    w = 0.36
    b1 = ax0.bar(x - w / 2, reg20["greedy_hit_rate"], width=w, color="#8c96a0", label="Greedy Point Top-20")
    b2 = ax0.bar(x + w / 2, reg20["conformal_hit_rate"], width=w, color="#2171b5", label="Conformal LCB Top-20")
    short_m = {
        "ecfp4_rdkit_rf": "RF\n(ECFP4)",
        "ecfp4_rdkit_xgb": "XGB\n(ECFP4)",
        "molformer_xl": "MoLFormer\n-XL",
        "chemberta2_mlm": "ChemB-2\nMLM",
        "chemberta2_mtr": "ChemB-2\nMTR",
        "gps_lite": "GPS-Lite\n(GNN)",
        "gin_vn": "GIN-VN\n(GNN)",
    }
    ax0.bar_label(b1, fmt="%.3f", fontsize=6.9, padding=3, rotation=90)
    ax0.bar_label(b2, fmt="%.3f", fontsize=6.9, padding=3, rotation=90)
    ax0.set_xticks(x)
    ax0.set_xticklabels([short_m[m] for m in reg20["model"]], fontsize=7.8)
    ax0.set_ylabel("Mean Hit Rate (Precision@20)")
    ax0.set_ylim(0.50, 0.87)
    ax0.set_title("A. Regression Top-20 Hit Rate by Representation Family")
    ax0.legend(loc="lower left", fontsize=8, frameon=True)

    # Panel B: Per-Endpoint Conformal Top-20 FP Reduction (%) for RF & MoLFormer-XL
    ax1 = axes[1]
    reg_tasks = (
        tk20[(tk20["task_type"] == "regression") & (tk20["model"].isin(["ecfp4_rdkit_rf", "molformer_xl"]))]
        .groupby(["task", "model"], as_index=False)["relative_fp_reduction_pct"]
        .mean()
    )
    # Order endpoints by average gain across RF & MoLFormer-XL
    t_order = (
        reg_tasks.groupby("task")["relative_fp_reduction_pct"]
        .mean()
        .sort_values(ascending=True)
        .index.tolist()
    )
    short_t = {
        "chembl240_herg": "ChEMBL hERG",
        "clearance_microsome_az": "Clearance (Mic.)",
        "solubility_aqsoldb": "Solubility",
        "vdss_lombardo": "VDss (Lombardo)",
        "caco2_wang": "Caco-2",
        "chembl203_egfr": "ChEMBL EGFR",
        "half_life_obach": "Half-Life",
        "ppbr_az": "PPBR",
        "chembl4822_bace1": "ChEMBL BACE1",
        "lipophilicity_astrazeneca": "Lipophilicity",
        "clearance_hepatocyte_az": "Clearance (Hep.)",
        "ld50_zhu": "LD50 (Zhu)",
    }
    piv = reg_tasks.pivot(index="task", columns="model", values="relative_fp_reduction_pct").reindex(t_order)
    y = np.arange(len(t_order))
    h = 0.38
    ax1.barh(y + h / 2, piv["molformer_xl"], height=h, color="#7570b3", label="MoLFormer-XL (Adaptive LCB)")
    ax1.barh(y - h / 2, piv["ecfp4_rdkit_rf"], height=h, color="#1b9e77", label="RF (Tree-Std Adaptive LCB)")
    ax1.axvline(0.0, color="#222222", lw=0.9)
    ax1.set_yticks(y)
    ax1.set_yticklabels([short_t.get(t, t) for t in t_order], fontsize=8.0)
    ax1.set_xlabel("Relative False-Positive Reduction at Top-20 (%)")
    ax1.set_title("B. Where Conformal LCB Helps vs. Backfires (Top-20)")
    ax1.legend(loc="lower right", fontsize=8, frameon=True)

    fig.tight_layout()
    save_fig(fig, "fig4_topk_virtual_screening_fp_saved.png")


def plot_fig5() -> None:
    mdf = pd.read_csv(TAB_DIR / "chembl_mpo_case_study.csv")
    cand = mdf[(mdf["pred_lcb90"] >= 6.2) & (mdf["herg_pval_safe0"] > 0.10)]
    cert = cand[cand["herg_cert_safe"]]

    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    hits = mdf[mdf["is_true_hit"]]
    nonh = mdf[~mdf["is_true_hit"]]

    # Shade MPO pass region
    ax.axvspan(6.2, 7.65, ymin=0.10 / 1.05, ymax=1.0, color="#2ecc71", alpha=0.07, zorder=0)

    ax.scatter(nonh["pred_lcb90"], nonh["herg_pval_safe0"], s=24, alpha=0.45, color="#95a5a6", label="Non-Hit (pChEMBL < 7.0)")
    ax.scatter(hits["pred_lcb90"], hits["herg_pval_safe0"], s=38, alpha=0.85, color="#2ecc71", edgecolor="#1e4620", lw=0.5, label="True EGFR Hit (pChEMBL >= 7.0)")
    ax.axvline(6.2, ls="--", color="#2980b9", lw=1.3, label="EGFR Conformal LCB90 >= 6.2")
    ax.axhline(0.10, ls="--", color="#e74c3c", lw=1.3, label="hERG Non-Toxic Conformal p > 0.10")

    callout = (
        f"MPO Pass Region (Top-Right):\n"
        f"  {int(cand['is_true_hit'].sum())}/{len(cand)} true hits ({100*cand['is_true_hit'].mean():.2f}% prec., 2.31x)\n"
        f"Dual-Certified hERG Safe {{0}}:\n"
        f"  {int(cert['is_true_hit'].sum())}/{len(cert)} true hits ({100*cert['is_true_hit'].mean():.2f}% prec.)"
    )
    ax.text(
        6.26, 0.99, callout, fontsize=7.8, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.38", facecolor="white", edgecolor="#2980b9", alpha=0.92),
    )

    ax.set_xlim(3.3, 7.6)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("EGFR Conformal 90% Lower Confidence Bound (pChEMBL LCB)")
    ax.set_ylabel("Conformal hERG Non-Toxic p-value (Y=0)")
    ax.set_title("Multi-Parameter Virtual Screening: EGFR Potency LCB vs. Conformal hERG Safety")
    ax.legend(loc="lower left", fontsize=8, frameon=True)
    fig.tight_layout()
    save_fig(fig, "fig5_egfr_herg_mpo_case_study.png")


if __name__ == "__main__":
    plot_fig3()
    plot_fig4()
    plot_fig5()
    print("[DONE] Regenerated Fig 3, Fig 4, Fig 5 from verified 7-model CSV tables.")
