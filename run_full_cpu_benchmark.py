# -*- coding: utf-8 -*-
"""Reproducible CLI runner for Direction D full 25-task x 3-seed CPU benchmark."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
WS_DIR = ROOT / "tdc_admet_workspace"
LOCAL_TEST_DIR = ROOT / "_local_test"
RESULTS_TABLES_DIR = ROOT / "results" / "tables"
RESULTS_FIGURES_DIR = ROOT / "results" / "figures"

EXPECTED_TABLES = [
    "scaffold_shift_diagnostics.csv",
    "tabular_baseline_metrics.csv",
    "conformal_regression_summary.csv",
    "conformal_tanimoto_bins.csv",
    "conformal_classification_summary.csv",
    "selective_risk_coverage.csv",
    "virtual_screening_topk.csv",
    "chembl_mpo_case_study.csv",
]

EXPECTED_FIGURES = [
    "fig1_scaffold_ood_coverage_collapse.png",
    "fig2_interval_efficiency_cqr_vs_split.png",
    "fig3_mondrian_imbalanced_classification.png",
    "fig4_topk_virtual_screening_fp_saved.png",
    "fig5_egfr_herg_mpo_case_study.png",
]


def run_notebook(nb_name: str, out_name: str, timeout: int = 2400) -> None:
    """Execute a Jupyter notebook via nbconvert into _local_test/."""
    LOCAL_TEST_DIR.mkdir(parents=True, exist_ok=True)
    nb_path = ROOT / nb_name
    if not nb_path.exists():
        raise FileNotFoundError(f"Notebook not found: {nb_path}")
    exec_path = nb_path
    temp_patched = None
    if nb_name == "d_notebook_03_conformal_ood_virtual_screening.ipynb":
        raw_text = nb_path.read_text(encoding="utf-8")
        # Work around unescaped literal newlines in Cell 5 m_labels if upstream build_d_nb03.py has not yet been patched
        broken_pairs = [
            ('"Split IID\n(Naive)"', '"Split IID\\n(Naive)"'),
            ('"Split\nScaffold"', '"Split\\nScaffold"'),
            ('"CV+\nScaffold"', '"CV+\\nScaffold"'),
            ('"Adaptive\nNCP"', '"Adaptive\\nNCP"'),
            ('"CQR\n(XGB)"', '"CQR\\n(XGB)"'),
            ('"Weighted\nCov-Shift"', '"Weighted\\nCov-Shift"'),
        ]
        nb_obj = json.loads(raw_text)
        changed = False
        for cell in nb_obj.get("cells", []):
            src = cell.get("source", "")
            src_str = "".join(src) if isinstance(src, list) else src
            new_str = src_str
            for old, new in broken_pairs:
                if old in new_str:
                    new_str = new_str.replace(old, new)
                    changed = True
            if new_str != src_str:
                cell["source"] = new_str
        if changed:
            temp_patched = LOCAL_TEST_DIR / "_tmp_d_nb03_patched.ipynb"
            temp_patched.write_text(json.dumps(nb_obj, indent=1, ensure_ascii=False), encoding="utf-8")
            exec_path = temp_patched
            print("[WARN] Detected unescaped newlines in d_notebook_03 Cell 5 m_labels; executing patched copy in _local_test/", flush=True)

    cmd = [
        sys.executable,
        "-m",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        f"--ExecutePreprocessor.timeout={timeout}",
        "--output-dir",
        str(LOCAL_TEST_DIR),
        "--output",
        out_name,
        str(exec_path),
    ]
    print(f"[RUN ] {' '.join(cmd)}", flush=True)
    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    try:
        subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)
    finally:
        if temp_patched is not None and temp_patched.exists():
            temp_patched.unlink()
    print(f"[OK  ] Finished {nb_name} -> _local_test/{out_name} in {time.time() - t0:.1f}s", flush=True)


def cleanup_stale_feature_caches() -> None:
    """Keep only the global ECFP feature cache matching admet_manifest.json."""
    manifest_path = WS_DIR / "data" / "processed" / "admet_manifest.json"
    feat_dir = WS_DIR / "cache" / "features"
    if not manifest_path.exists() or not feat_dir.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_name = f"ecfp_rdkit__global__{manifest['unique_sha256'][:12]}.npz"
    for p in feat_dir.glob("ecfp_rdkit__global__*.npz"):
        if p.name != expected_name:
            print(f"[CLEAN] Removing stale feature cache: {p.name}")
            p.unlink()


def copy_artifacts() -> None:
    """Copy generated CSV tables and 300-DPI PNG figures into results/."""
    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    src_tab = WS_DIR / "outputs" / "tables"
    src_fig = WS_DIR / "outputs" / "figures"

    for fname in EXPECTED_TABLES:
        s = src_tab / fname
        if not s.exists():
            raise FileNotFoundError(f"Missing generated table: {s}")
        shutil.copy2(s, RESULTS_TABLES_DIR / fname)

    for fname in EXPECTED_FIGURES:
        s = src_fig / fname
        if not s.exists():
            raise FileNotFoundError(f"Missing generated figure: {s}")
        shutil.copy2(s, RESULTS_FIGURES_DIR / fname)

    print(f"[COPY] Copied {len(EXPECTED_TABLES)} CSVs -> {RESULTS_TABLES_DIR}")
    print(f"[COPY] Copied {len(EXPECTED_FIGURES)} PNGs -> {RESULTS_FIGURES_DIR}")


def sanity_check() -> None:
    """Validate 25-task x 3-seed completeness, finite interval rates, and ESS health."""
    sys.path.insert(0, str(ROOT))
    from admet_core import ADMET_TASKS, CHEMBL_TARGETS, ALL_TASKS

    manifest = json.loads((WS_DIR / "data" / "processed" / "admet_manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_scope"] == "all", f"Expected run_scope='all', got {manifest['run_scope']!r}"
    assert manifest["seeds"] == [1, 2, 3], f"Expected seeds [1, 2, 3], got {manifest['seeds']}"
    assert set(manifest["tasks"].keys()) == set(ALL_TASKS.keys()), "Manifest tasks mismatch ALL_TASKS"

    uniq_df = pd.read_csv(WS_DIR / "data" / "processed" / "unique_molecules.csv")
    assert len(uniq_df) == manifest["n_unique_molecules"]
    feat_path = WS_DIR / "cache" / "features" / f"ecfp_rdkit__global__{manifest['unique_sha256'][:12]}.npz"
    assert feat_path.exists(), f"Missing global feature cache: {feat_path}"

    diag_df = pd.read_csv(RESULTS_TABLES_DIR / "scaffold_shift_diagnostics.csv")
    assert len(diag_df) == 25, f"Expected 25 tasks in diagnostics, got {len(diag_df)}"
    assert (diag_df["shared_scaffolds"] == 0).all(), "Non-zero shared scaffolds detected!"

    tab_df = pd.read_csv(RESULTS_TABLES_DIR / "tabular_baseline_metrics.csv")
    reg_df = pd.read_csv(RESULTS_TABLES_DIR / "conformal_regression_summary.csv")
    cls_df = pd.read_csv(RESULTS_TABLES_DIR / "conformal_classification_summary.csv")
    topk_df = pd.read_csv(RESULTS_TABLES_DIR / "virtual_screening_topk.csv")
    mpo_df = pd.read_csv(RESULTS_TABLES_DIR / "chembl_mpo_case_study.csv")

    assert tab_df.groupby(["task", "seed"]).ngroups == 75, "Missing task-seed pairs in tabular_baseline_metrics.csv"

    reg_tasks = {t for t, sp in ALL_TASKS.items() if sp["task_type"] == "regression"}
    cls_tasks = {t for t, sp in ALL_TASKS.items() if sp["task_type"] == "classification"}
    assert len(reg_tasks) + len(cls_tasks) == 25, f"Expected 25 tasks, got {len(reg_tasks)}+{len(cls_tasks)}"

    assert set(reg_df["task"].unique()) == reg_tasks, "Regression task coverage mismatch"
    assert reg_df.groupby(["task", "seed"]).ngroups == len(reg_tasks) * 3, "Missing regression task-seed pairs"
    assert set(cls_df["task"].unique()) == cls_tasks, "Classification task coverage mismatch"
    assert cls_df.groupby(["task", "seed"]).ngroups == len(cls_tasks) * 3, "Missing classification task-seed pairs"

    min_finite = float(reg_df["finite_frac"].min())
    mean_finite = float(reg_df["finite_frac"].mean())
    unweighted_min_finite = float(
        reg_df[reg_df["conformal_method"] != "weighted_covariate_shift_cp"]["finite_frac"].min()
    )
    weighted_mean_finite = float(
        reg_df[reg_df["conformal_method"] == "weighted_covariate_shift_cp"]["finite_frac"].mean()
    )
    assert unweighted_min_finite == 1.0, f"Unweighted conformal produced non-finite intervals: {unweighted_min_finite:.4f}"
    assert mean_finite > 0.95 and weighted_mean_finite > 0.95, (
        f"Mean finite interval rate too low: overall={mean_finite:.4f}, weighted={weighted_mean_finite:.4f}"
    )

    reg_w = reg_df[reg_df["conformal_method"] == "weighted_covariate_shift_cp"]
    cls_w = cls_df[cls_df["conformal_method"] == "weighted_mondrian_cp"]
    min_ess_reg = float(reg_w["ess"].min())
    med_ess_reg = float(reg_w["ess"].median())
    mean_ess_reg = float(reg_w["ess"].mean())
    min_ess_cls = float(cls_w["ess"].min())
    med_ess_cls = float(cls_w["ess"].median())
    mean_ess_cls = float(cls_w["ess"].mean())
    assert med_ess_reg >= 20.0 and med_ess_cls >= 20.0 and mean_ess_reg > 100.0 and mean_ess_cls > 100.0, (
        f"Unhealthy ESS: reg_med={med_ess_reg:.1f}, reg_mean={mean_ess_reg:.1f}, "
        f"cls_med={med_ess_cls:.1f}, cls_mean={mean_ess_cls:.1f}"
    )

    for fname in EXPECTED_FIGURES:
        fpath = RESULTS_FIGURES_DIR / fname
        assert fpath.exists() and fpath.stat().st_size > 10_000, f"Invalid figure: {fpath}"

    print("\n=== SANITY CHECK PASSED (25 tasks x 3 seeds) ===")
    print(f"Unique molecules: {len(uniq_df)} | Feature cache: {feat_path.name}")
    print(
        f"Finite interval rate: overall_mean={mean_finite:.4f}, "
        f"weighted_mean={weighted_mean_finite:.4f}, unweighted_min={unweighted_min_finite:.4f}, "
        f"overall_min={min_finite:.4f}"
    )
    print(f"Weighted CP ESS (reg): min={min_ess_reg:.1f}, median={med_ess_reg:.1f}, mean={mean_ess_reg:.1f}")
    print(f"Weighted CP ESS (cls): min={min_ess_cls:.1f}, median={med_ess_cls:.1f}, mean={mean_ess_cls:.1f}")
    print(f"Top-K rows: {len(topk_df)} | MPO EGFR compounds: {len(mpo_df)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full 25-task x 3-seed Direction D CPU benchmark.")
    parser.add_argument("--timeout", type=int, default=2400, help="nbconvert cell execution timeout in seconds")
    parser.add_argument("--skip-nb01", action="store_true", help="Skip D-01 notebook execution")
    parser.add_argument("--skip-nb03", action="store_true", help="Skip D-03 notebook execution")
    args = parser.parse_args()

    if not args.skip_nb01:
        run_notebook("d_notebook_01_data_prep_ecfp_xgboost.ipynb", "d_nb01_full_executed.ipynb", timeout=args.timeout)
    cleanup_stale_feature_caches()
    if not args.skip_nb03:
        run_notebook("d_notebook_03_conformal_ood_virtual_screening.ipynb", "d_nb03_full_executed.ipynb", timeout=args.timeout)
    copy_artifacts()
    sanity_check()


if __name__ == "__main__":
    main()
