# -*- coding: utf-8 -*-
"""Core data engineering, security-hardened fetching, scaffold splits, and molecular featurization for Direction D."""
from __future__ import annotations

import hashlib
import io
import ipaddress
import json
import os
import re
import socket
import urllib.request
import warnings
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from random import Random
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, MACCSkeys, QED, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

RDLogger.DisableLog("rdApp.*")

# Official TDC ADMET_Group (22 tasks)
ADMET_TASKS: Dict[str, dict] = {
    # Absorption (6)
    "caco2_wang": {"category": "Absorption", "task_type": "regression", "metric": "mae", "favorable": "high", "pilot": True},
    "hia_hou": {"category": "Absorption", "task_type": "classification", "metric": "roc-auc", "favorable": "high", "pilot": False},
    "pgp_broccatelli": {"category": "Absorption", "task_type": "classification", "metric": "roc-auc", "favorable": "low", "pilot": False},
    "bioavailability_ma": {"category": "Absorption", "task_type": "classification", "metric": "roc-auc", "favorable": "high", "pilot": False},
    "lipophilicity_astrazeneca": {"category": "Absorption", "task_type": "regression", "metric": "mae", "favorable": "high", "pilot": False},
    "solubility_aqsoldb": {"category": "Absorption", "task_type": "regression", "metric": "mae", "favorable": "high", "pilot": True},
    # Distribution (3)
    "bbb_martins": {"category": "Distribution", "task_type": "classification", "metric": "roc-auc", "favorable": "high", "pilot": True},
    "ppbr_az": {"category": "Distribution", "task_type": "regression", "metric": "mae", "favorable": "low", "pilot": False},
    "vdss_lombardo": {"category": "Distribution", "task_type": "regression", "metric": "spearman", "favorable": "high", "pilot": True},
    # Metabolism (6)
    "cyp2c9_veith": {"category": "Metabolism", "task_type": "classification", "metric": "pr-auc", "favorable": "low", "pilot": False},
    "cyp2d6_veith": {"category": "Metabolism", "task_type": "classification", "metric": "pr-auc", "favorable": "low", "pilot": True},
    "cyp3a4_veith": {"category": "Metabolism", "task_type": "classification", "metric": "pr-auc", "favorable": "low", "pilot": False},
    "cyp2c9_substrate_carbonmangels": {"category": "Metabolism", "task_type": "classification", "metric": "pr-auc", "favorable": "low", "pilot": False},
    "cyp2d6_substrate_carbonmangels": {"category": "Metabolism", "task_type": "classification", "metric": "pr-auc", "favorable": "low", "pilot": False},
    "cyp3a4_substrate_carbonmangels": {"category": "Metabolism", "task_type": "classification", "metric": "roc-auc", "favorable": "low", "pilot": False},
    # Excretion (3)
    "half_life_obach": {"category": "Excretion", "task_type": "regression", "metric": "spearman", "favorable": "high", "pilot": False},
    "clearance_hepatocyte_az": {"category": "Excretion", "task_type": "regression", "metric": "spearman", "favorable": "low", "pilot": False},
    "clearance_microsome_az": {"category": "Excretion", "task_type": "regression", "metric": "spearman", "favorable": "low", "pilot": False},
    # Toxicity (4)
    "ld50_zhu": {"category": "Toxicity", "task_type": "regression", "metric": "mae", "favorable": "high", "pilot": False},
    "herg": {"category": "Toxicity", "task_type": "classification", "metric": "roc-auc", "favorable": "low", "pilot": True},
    "ames": {"category": "Toxicity", "task_type": "classification", "metric": "roc-auc", "favorable": "low", "pilot": False},
    "dili": {"category": "Toxicity", "task_type": "classification", "metric": "roc-auc", "favorable": "low", "pilot": False},
}

CHEMBL_TARGETS: Dict[str, dict] = {
    "chembl203_egfr": {"chembl_id": "CHEMBL203", "name": "EGFR_kinase", "task_type": "regression", "metric": "mae", "favorable": "high", "hit_threshold": 7.0, "pilot": True},
    "chembl4822_bace1": {"chembl_id": "CHEMBL4822", "name": "BACE1", "task_type": "regression", "metric": "mae", "favorable": "high", "hit_threshold": 7.0, "pilot": False},
    "chembl240_herg": {"chembl_id": "CHEMBL240", "name": "hERG_IC50", "task_type": "regression", "metric": "mae", "favorable": "low", "hit_threshold": 6.0, "pilot": True},
}

ALL_TASKS: Dict[str, dict] = {**ADMET_TASKS, **CHEMBL_TARGETS}

TDC_ADMET_DATAVERSE_URL = "https://dataverse.harvard.edu/api/access/datafile/4426004"
ALLOWED_HOSTS = {
    "dataverse.harvard.edu",
    "dvn-cloud-iqss.s3.amazonaws.com",
    "dvn-cloud.s3.amazonaws.com",
    "www.ebi.ac.uk",
}

RDKIT_DESC_NAMES = [
    "MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "RingCount", "NumAromaticRings", "FractionCSP3",
    "qed", "BertzCT", "HeavyAtomCount", "NumHeteroatoms",
    "NumAliphaticRings", "LabuteASA", "BalabanJ",
]


def _validate_remote_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(f"Only https URLs are permitted, got {parsed.scheme!r}")
    host = (parsed.hostname or "").lower()
    if host not in ALLOWED_HOSTS:
        raise ValueError(f"Host {host!r} outside allowlist {sorted(ALLOWED_HOSTS)}")
    try:
        infos = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise RuntimeError(f"DNS resolution failed for {host!r}: {exc}") from exc
    for info in infos:
        ip_str = info[4][0].split("%")[0]
        ip_obj = ipaddress.ip_address(ip_str)
        if (
            ip_obj.is_private
            or ip_obj.is_loopback
            or ip_obj.is_link_local
            or ip_obj.is_reserved
            or ip_obj.is_multicast
            or ip_obj.is_unspecified
        ):
            raise RuntimeError(f"Blocked non-public resolved IP {ip_str} for {host!r}")
    return url


class _AllowlistedRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _validate_remote_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_SAFE_OPENER = urllib.request.build_opener(_AllowlistedRedirectHandler)


def safe_fetch_bytes(url: str, timeout: int = 90, accept: str = "*/*") -> bytes:
    _validate_remote_url(url)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "tdc-admet-conformal-benchmark/1.0", "Accept": accept},
    )
    with _SAFE_OPENER.open(req, timeout=timeout) as resp:
        data = resp.read()
    if not data:
        raise IOError(f"Empty response from {url}")
    return data


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def atomic_save_npz(path: Path, **arrays) -> None:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    atomic_write_bytes(Path(path), buf.getvalue())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def confine_path(base: Path, target: Path) -> Path:
    base_r = Path(base).resolve()
    target_r = Path(target).resolve()
    if not target_r.is_relative_to(base_r):
        raise ValueError(f"Path {target_r} escapes base {base_r}")
    return target_r


def resolve_workspace(in_colab: Optional[bool] = None, local_root: Optional[Path] = None) -> Path:
    """Resolve workspace path, auto-reusing flip2_benchmark shared Drive if present in Colab."""
    import sys
    if in_colab is None:
        in_colab = "google.colab" in sys.modules
    if not in_colab:
        base = (local_root or Path(__file__).resolve().parent / "tdc_admet_workspace").resolve()
        base.mkdir(parents=True, exist_ok=True)
        return base

    from google.colab import drive
    drive.mount("/content/drive", force_remount=True)
    primary = Path("/content/drive/MyDrive/tdc_admet_benchmark")
    primary_sentinel = primary / ".workspace_id"
    if primary_sentinel.exists():
        actual = primary_sentinel.read_text(encoding="utf-8").strip()
        assert actual == "tdc-admet-shared-prod-2026", f"Sentinel mismatch: {actual!r}"
        print(f"[OK] Dedicated Direction D Drive workspace: {primary} -> {os.path.realpath(primary)}")
        return primary

    flip2_dir = Path("/content/drive/MyDrive/flip2_benchmark")
    flip2_sentinel = flip2_dir / ".workspace_id"
    if flip2_sentinel.exists():
        actual = flip2_sentinel.read_text(encoding="utf-8").strip()
        assert actual == "flip2-shared-prod-2026", f"FLIP2 sentinel mismatch: {actual!r}"
        d_dir = flip2_dir / "direction_d_admet"
        d_dir.mkdir(parents=True, exist_ok=True)
        print(f"[OK] Reusing shared Drive workspace: {d_dir} -> {os.path.realpath(d_dir)}")
        return d_dir

    raise RuntimeError(
        "No shared workspace sentinel found. Create /content/drive/MyDrive/tdc_admet_benchmark/.workspace_id "
        "('tdc-admet-shared-prod-2026') or mount shared flip2_benchmark."
    )


def download_tdc_admet_archive(raw_dir: Path) -> Tuple[Path, dict]:
    """Download & safely extract official TDC ADMET_Group zip archive."""
    raw_dir = Path(raw_dir).resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = confine_path(raw_dir, raw_dir / "admet_group.zip")
    group_dir = confine_path(raw_dir, raw_dir / "admet_group")
    retrieved = datetime.now(timezone.utc).isoformat()

    if not (zip_path.exists() and zip_path.stat().st_size > 100_000):
        print(f"[GET ] Downloading TDC admet_group.zip from {TDC_ADMET_DATAVERSE_URL}")
        data = safe_fetch_bytes(TDC_ADMET_DATAVERSE_URL, timeout=120)
        atomic_write_bytes(zip_path, data)
        print(f"[OK  ] Saved {zip_path.name} ({len(data)} bytes)")
    else:
        print(f"[SKIP] Cached {zip_path.name} ({zip_path.stat().st_size} bytes)")

    sample_check = group_dir / "caco2_wang" / "train_val.csv"
    if not sample_check.exists():
        group_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                # Strip optional leading 'admet_group/' prefix so files land cleanly under raw_dir/admet_group/<task>/
                rel_parts = Path(member.filename).parts
                if rel_parts and rel_parts[0].lower() == "admet_group":
                    rel_parts = rel_parts[1:]
                if not rel_parts:
                    continue
                dest = confine_path(group_dir, group_dir.joinpath(*rel_parts))
                dest.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_bytes(dest, zf.read(member))
        print(f"[OK  ] Extracted 22 tasks into {group_dir}")

    return group_dir, {
        "url": TDC_ADMET_DATAVERSE_URL,
        "zip_sha256": sha256_file(zip_path),
        "retrieved_utc": retrieved,
    }


def canonicalize_smiles(smi: str) -> Optional[str]:
    if not isinstance(smi, str) or not smi.strip():
        return None
    mol = Chem.MolFromSmiles(smi.strip())
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)


def get_scaffold_smiles(smi: str) -> str:
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def create_scaffold_split_pytdc(
    df: pd.DataFrame,
    seed: int,
    frac: Tuple[float, float, float] = (0.875, 0.125, 0.0),
    entity: str = "Drug",
) -> Dict[str, pd.DataFrame]:
    """Exact bit-for-bit reproduction of PyTDC tdc/utils/split.py create_scaffold_split."""
    rng = Random(seed)
    smiles_arr = df[entity].values
    scaffolds: Dict[str, set] = defaultdict(set)
    error_smiles = 0

    for i, smi in enumerate(smiles_arr):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            error_smiles += 1
            continue
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        scaffolds[scaf].add(i)

    n_valid = len(df) - error_smiles
    train_size = int(n_valid * frac[0])
    val_size = int(n_valid * frac[1])
    test_size = n_valid - train_size - val_size

    big_sets, small_sets = [], []
    for idx_set in scaffolds.values():
        if len(idx_set) > val_size / 2 or len(idx_set) > test_size / 2:
            big_sets.append(idx_set)
        else:
            small_sets.append(idx_set)

    rng.seed(seed)
    rng.shuffle(big_sets)
    rng.shuffle(small_sets)
    ordered_sets = big_sets + small_sets

    train_idx, val_idx, test_idx = [], [], []
    if frac[2] == 0:
        for idx_set in ordered_sets:
            if len(train_idx) + len(idx_set) <= train_size:
                train_idx += idx_set
            else:
                val_idx += idx_set
    else:
        for idx_set in ordered_sets:
            if len(train_idx) + len(idx_set) <= train_size:
                train_idx += idx_set
            elif len(val_idx) + len(idx_set) <= val_size:
                val_idx += idx_set
            else:
                test_idx += idx_set

    return {
        "train": df.iloc[train_idx].reset_index(drop=True),
        "valid": df.iloc[val_idx].reset_index(drop=True),
        "test": df.iloc[test_idx].reset_index(drop=True),
    }


def scaffold_group_kfold(smiles_list: List[str], n_splits: int = 5, seed: int = 42) -> np.ndarray:
    """Assign Bemis-Murcko scaffold groups into balanced k folds for CV+ conformal."""
    scaf_to_idx: Dict[str, List[int]] = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        scaf = get_scaffold_smiles(smi)
        scaf_to_idx[scaf].append(i)

    groups = list(scaf_to_idx.values())
    rng = Random(seed)
    rng.shuffle(groups)
    # Sort descending by group size after tie-break shuffle for balanced fold sizes
    groups.sort(key=len, reverse=True)

    folds = np.zeros(len(smiles_list), dtype=np.int64)
    fold_counts = np.zeros(n_splits, dtype=np.int64)
    for grp in groups:
        k = int(np.argmin(fold_counts))
        folds[grp] = k
        fold_counts[k] += len(grp)
    return folds


def fetch_chembl_target_ic50(
    target_key: str,
    chembl_id: str,
    raw_dir: Path,
    max_records: int = 2000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch IC50 pChEMBL bioactivities from ChEMBL REST API and apply 70/10/20 scaffold split."""
    if not re.fullmatch(r"CHEMBL[0-9]+", chembl_id):
        raise ValueError(f"Invalid ChEMBL ID: {chembl_id!r}")
    out_dir = confine_path(raw_dir, raw_dir / "chembl" / target_key)
    out_dir.mkdir(parents=True, exist_ok=True)
    tv_path = out_dir / "train_val.csv"
    te_path = out_dir / "test.csv"
    if tv_path.exists() and te_path.exists():
        print(f"[SKIP] Cached ChEMBL target {target_key} ({chembl_id})")
        return pd.read_csv(tv_path), pd.read_csv(te_path)

    rows = []
    limit = 1000
    offset = 0
    while len(rows) < max_records * 2:
        url = (
            f"https://www.ebi.ac.uk/chembl/api/data/activity.json"
            f"?target_chembl_id={chembl_id}&standard_type=IC50"
            f"&pchembl_value__isnull=false&limit={limit}&offset={offset}"
        )
        payload = json.loads(safe_fetch_bytes(url, timeout=60, accept="application/json").decode("utf-8"))
        acts = payload.get("activities", [])
        if not acts:
            break
        for a in acts:
            smi = a.get("canonical_smiles")
            val = a.get("pchembl_value")
            rel = a.get("standard_relation")
            if smi and val is not None and rel in ("=", None):
                c_smi = canonicalize_smiles(smi)
                if c_smi:
                    rows.append({"Drug": c_smi, "Y": float(val), "Drug_ID": a.get("molecule_chembl_id", "")})
        if not payload.get("page_meta", {}).get("next"):
            break
        offset += limit

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No ChEMBL records fetched for {chembl_id}")
    df = (
        df.groupby("Drug", as_index=False)
        .agg({"Y": "median", "Drug_ID": "first"})
        .sort_values("Drug")
        .reset_index(drop=True)
    )
    if len(df) > max_records:
        # Deterministic stratified-by-scaffold subsample up to max_records
        df = df.sample(n=max_records, random_state=42).reset_index(drop=True)

    # Outer 80/20 scaffold split (train_val vs test), matching TDC admet_group structure
    sp = create_scaffold_split_pytdc(df, seed=42, frac=(0.80, 0.0, 0.20), entity="Drug")
    # Note: with frac[2] > 0 and frac[1] == 0, PyTDC puts train_size in train and remaining in test
    tv_df = sp["train"].reset_index(drop=True)
    te_df = pd.concat([sp["valid"], sp["test"]], ignore_index=True)
    tv_df.to_csv(tv_path, index=False)
    te_df.to_csv(te_path, index=False)
    print(f"[OK  ] ChEMBL {chembl_id}: {len(df)} unique mols -> train_val={len(tv_df)}, test={len(te_df)}")
    return tv_df, te_df


_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _compute_rdkit16(mol) -> np.ndarray:
    vals = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.RingCount(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.FractionCSP3(mol),
        QED.qed(mol),
        Descriptors.BertzCT(mol),
        Descriptors.HeavyAtomCount(mol),
        Descriptors.NumHeteroatoms(mol),
        Descriptors.NumAliphaticRings(mol),
        Descriptors.LabuteASA(mol),
        Descriptors.BalabanJ(mol),
    ]
    arr = np.asarray(vals, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def featurize_ecfp_rdkit(smiles_list: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_feat [N, 2231] float32, ecfp_bits [N, 2048] uint8, valid_mask [N] bool)."""
    n = len(smiles_list)
    X_feat = np.zeros((n, 2048 + 167 + 16), dtype=np.float32)
    ecfp_bits = np.zeros((n, 2048), dtype=np.uint8)
    valid = np.zeros(n, dtype=bool)

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        fp = _MORGAN_GEN.GetFingerprintAsNumPy(mol).astype(np.uint8)
        maccs = np.asarray(MACCSkeys.GenMACCSKeys(mol), dtype=np.uint8)  # 167 bits
        desc = _compute_rdkit16(mol)
        ecfp_bits[i] = fp
        X_feat[i, :2048] = fp.astype(np.float32)
        X_feat[i, 2048:2048 + 167] = maccs.astype(np.float32)
        X_feat[i, 2048 + 167:] = desc
        valid[i] = True

    return X_feat, ecfp_bits, valid


def compute_nn_tanimoto_distance(ecfp_train_bits: np.ndarray, ecfp_eval_bits: np.ndarray, chunk_size: int = 512) -> np.ndarray:
    """Exact nearest-neighbor Tanimoto distance d_NN = 1 - max_j Tanimoto(eval_i, train_j)."""
    tr = ecfp_train_bits.astype(np.float32)
    ev = ecfp_eval_bits.astype(np.float32)
    tr_sum = tr.sum(axis=1)[None, :]  # (1, N_tr)
    ev_sum = ev.sum(axis=1)           # (N_ev,)
    n_ev = ev.shape[0]
    nn_dist = np.ones(n_ev, dtype=np.float32)

    for start in range(0, n_ev, chunk_size):
        end = min(start + chunk_size, n_ev)
        ev_blk = ev[start:end]
        inter = ev_blk @ tr.T
        union = ev_sum[start:end, None] + tr_sum - inter
        sim = np.where(union > 0, inter / np.maximum(union, 1e-8), 0.0)
        nn_dist[start:end] = 1.0 - sim.max(axis=1)

    return nn_dist


def tanimoto_bin_label(d: np.ndarray) -> np.ndarray:
    out = np.empty(len(d), dtype=object)
    out[d < 0.45] = "Near (<0.45)"
    out[(d >= 0.45) & (d < 0.65)] = "Moderate (0.45-0.65)"
    out[d >= 0.65] = "Far (>=0.65)"
    return out


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi if hi < 1.0 else y_prob <= hi)
        if mask.any():
            ece += (mask.sum() / n) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def compute_task_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if task_type == "regression":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho = float(spearmanr(y_true, y_pred).correlation) if np.std(y_pred) > 1e-8 else float("nan")
        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)),
            "spearman": rho,
        }
    y_int = y_true.astype(int)
    y_prob = np.clip(y_pred, 1e-7, 1.0 - 1e-7)
    try:
        auroc = float(roc_auc_score(y_int, y_prob))
    except ValueError:
        auroc = float("nan")
    try:
        auprc = float(average_precision_score(y_int, y_prob))
    except ValueError:
        auprc = float("nan")
    return {
        "roc-auc": auroc,
        "pr-auc": auprc,
        "brier": float(brier_score_loss(y_int, y_prob)),
        "ece": compute_ece(y_int, y_prob),
    }
