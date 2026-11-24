# -*- coding: utf-8 -*-
"""Builder for d_notebook_02_clm_gnn_representation_gpu.ipynb (Direction D Stage 2, T4/L4 GPU ~4-8 CU)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CELLS = []
_ID = 0


def _next_id():
    global _ID
    _ID += 1
    return f"d-nb02-{_ID:02d}"


def md(src):
    CELLS.append({"cell_type": "markdown", "id": _next_id(), "metadata": {}, "source": src})


def code(src):
    CELLS.append({"cell_type": "code", "id": _next_id(), "execution_count": None, "metadata": {}, "outputs": [], "source": src})


md("""# D-02 — Frozen Chemical LMs & Pure-PyTorch GNN Ensembles (T4 / L4 GPU)

**Resource tier: T4 (~1.8 CU/h) or L4 (~4.7 CU/h). Zero G4 needed!**
- **Anti-Waste Guardrail**: Refuses to run on 96GB G4 (`10.5 CU/h`) unless `ALLOW_G4 = True`.
- **Arm 2 (CLM Embeddings, ~0.5–1.0 CU)**: Extracts & caches frozen embeddings across globally deduplicated SMILES (`unique_molecules.csv`) for:
  - `DeepChem/ChemBERTa-77M-MTR` (multi-task supervised pretraining, dim=384)
  - `DeepChem/ChemBERTa-77M-MLM` (masked LM pretraining, dim=384)
  - `ibm-research/MoLFormer-XL-both-10pct` (rotary linear attention, dim=768, `.contiguous()` safe)
- **Arm 3 (Pure-PyTorch 2D GNN Ensembles, ~3–6 CU)**: Trains 5-member `gin_vn` (GIN + Virtual Node) and `gps_lite` (local GINE + global SDPA attention) ensembles without `torch-scatter` compilation.
""")

code('''# Cell 1 — Workspace verification, GPU telemetry & G4 anti-waste guard [GPU]
import os, sys, gc, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

os.environ.setdefault("HF_HOME", "/content/hf_cache")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
        "Cannot find admet_core.py! Please upload admet_core.py, conformal_ood.py, gin_gps_model.py "
        "to /content/ or your Google Drive folder, then re-run Cell 1."
    )
if str(_mod_dir) not in sys.path:
    sys.path.insert(0, str(_mod_dir))

try:
    import rdkit  # noqa: F401
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "rdkit", "transformers>=4.46.0", "tqdm"])

from admet_core import (
    ALL_TASKS, resolve_workspace, create_scaffold_split_pytdc,
    compute_task_metrics, atomic_save_npz, canonicalize_smiles, sha256_text, confine_path,
)
from gin_gps_model import smiles_to_graph_dict, train_gnn_ensemble

BASE_DIR = resolve_workspace(in_colab=IN_COLAB)
RAW_DIR  = (BASE_DIR / "data" / "raw").resolve()
PROC_DIR = (BASE_DIR / "data" / "processed").resolve()
EMB_DIR  = (BASE_DIR / "cache" / "embeddings").resolve()
PRED_DIR = (BASE_DIR / "cache" / "predictions").resolve()
TAB_DIR  = (BASE_DIR / "outputs" / "tables").resolve()
for d in (EMB_DIR, PRED_DIR, TAB_DIR):
    d.mkdir(parents=True, exist_ok=True)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU required (use T4 or L4 runtime).")

GPU_NAME = torch.cuda.get_device_name(0)
GPU_VRAM_GB = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
BF16_OK = torch.cuda.is_bf16_supported()
ALLOW_G4 = False
if GPU_VRAM_GB >= 70 and not ALLOW_G4:
    raise RuntimeError(
        f"[Anti-Waste Guard] Detected {GPU_NAME} ({GPU_VRAM_GB:.1f} GB, ~10.5 CU/h). "
        "Direction D needs only T4/L4 (1.8-4.7 CU/h). Switch runtime or set ALLOW_G4=True."
    )
print(f"[GPU] {GPU_NAME} ({GPU_VRAM_GB:.1f} GB) | bf16={BF16_OK}")''')

code('''# Cell 2 — Dependencies check (zero PyG/torch-scatter compilation) [GPU]
try:
    import transformers, rdkit
    from tqdm.auto import tqdm
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.46.0", "rdkit", "tqdm"])
    import transformers, rdkit
    from tqdm.auto import tqdm
print(f"transformers {transformers.__version__} | torch {torch.__version__} | rdkit {rdkit.__version__}")''')

code('''# Cell 3 — Configuration, dual-account sharding & manifest load [GPU]
SMOKE_TEST = False        # Set True for a 20-sec smoke test before full run
EXTRACT_CLMS = True       # Set False if 2b CLM caches are already complete
TRAIN_GNNS = True
OVERWRITE_CACHE = False

# ---- Dual-Account Parallel Sharding (zero file conflicts on shared Drive) ----
# Single account: set NUM_WORKERS = 1, WORKER_ID = 0
# Dual accounts:  Account 1 -> NUM_WORKERS = 2, WORKER_ID = 0
#                 Account 2 -> NUM_WORKERS = 2, WORKER_ID = 1
NUM_WORKERS = 2
WORKER_ID = 0
assert 0 <= WORKER_ID < NUM_WORKERS, f"Invalid WORKER_ID={WORKER_ID} for NUM_WORKERS={NUM_WORKERS}"

CLM_REGISTRY = {
    "chemberta2_mtr": {"hf_id": "DeepChem/ChemBERTa-77M-MTR", "trust_remote": False, "batch_size": 256},
    "chemberta2_mlm": {"hf_id": "DeepChem/ChemBERTa-77M-MLM", "trust_remote": False, "batch_size": 256},
    "molformer_xl":   {"hf_id": "ibm-research/MoLFormer-XL-both-10pct", "trust_remote": True, "batch_size": 128},
}
GNN_ARCHS = ["gin_vn", "gps_lite"]
N_ENSEMBLE = 5
MAX_EPOCHS = 50
PATIENCE = 8
BATCH_SIZE = 256

manifest = json.loads((PROC_DIR / "admet_manifest.json").read_text(encoding="utf-8"))
uniq_df = pd.read_csv(PROC_DIR / "unique_molecules.csv")
uniq_sha = manifest["unique_sha256"]
SEEDS = manifest["seeds"]
ALL_MANIFEST_TASKS = list(manifest["tasks"].keys())

if SMOKE_TEST:
    uniq_df = uniq_df.head(24).reset_index(drop=True)
    uniq_sha = f"smoke24_{uniq_sha[:8]}"
    ACTIVE_TASKS = ALL_MANIFEST_TASKS[:1]
    ACTIVE_CLM_KEYS = list(CLM_REGISTRY.keys())
    SEEDS = [1]
    N_ENSEMBLE = 2
    MAX_EPOCHS = 5
    print("[SMOKE TEST] Running minimal 24-mol CLM + 1-task GNN smoke test.")
else:
    ACTIVE_TASKS = [t for i, t in enumerate(ALL_MANIFEST_TASKS) if i % NUM_WORKERS == WORKER_ID]
    ACTIVE_CLM_KEYS = [k for i, k in enumerate(CLM_REGISTRY.keys()) if i % NUM_WORKERS == WORKER_ID]
    cached_n = sum(
        1 for t in ACTIVE_TASKS for s in SEEDS
        if (PRED_DIR / f"gnn__{t}__seed{s}.npz").exists() and not OVERWRITE_CACHE
    )
    total_n = len(ACTIVE_TASKS) * len(SEEDS)
    rem_tasks = [
        t for t in ACTIVE_TASKS
        if any(not (PRED_DIR / f"gnn__{t}__seed{s}.npz").exists() or OVERWRITE_CACHE for s in SEEDS)
    ]
    rem_samples = sum(manifest["tasks"][t]["n_train_val"] for t in rem_tasks)
    print(f"[WORKER {WORKER_ID}/{NUM_WORKERS}] CLMs={ACTIVE_CLM_KEYS}")
    print(f"[WORKER {WORKER_ID}/{NUM_WORKERS}] Assigned {len(ACTIVE_TASKS)}/{len(ALL_MANIFEST_TASKS)} tasks | seeds={SEEDS} | cached={cached_n}/{total_n} | todo_samples={rem_samples}")
    print(f"[WORKER {WORKER_ID}/{NUM_WORKERS}] Remaining tasks: {rem_tasks}")''')

code('''# Cell 4 — Batched frozen Chemical LM embedding extraction [GPU]
from transformers import AutoModel, AutoTokenizer

@torch.inference_mode()
def extract_clm_embeddings(smiles_list, hf_id, trust_remote, batch_size, device):
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=trust_remote)
    model_kwargs = {"trust_remote_code": trust_remote}
    if not trust_remote:
        model_kwargs["torch_dtype"] = torch.bfloat16 if BF16_OK else torch.float16
    model = AutoModel.from_pretrained(hf_id, **model_kwargs)
    for p in model.parameters():
        if not p.is_contiguous():
            p.data = p.data.contiguous()
    model = model.to(device).eval()

    # Sort by SMILES length descending so peak memory triggers on batch 0
    order = np.argsort([-len(s) for s in smiles_list], kind="mergesort")
    embs = None
    special_ids = torch.tensor(tokenizer.all_special_ids, device=device)

    for start in tqdm(range(0, len(order), batch_size), desc=hf_id.split("/")[-1]):
        idx = order[start:start + batch_size]
        batch_smi = [smiles_list[i] for i in idx]
        enc = tokenizer(batch_smi, padding=True, truncation=True, max_length=256, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        out = model(input_ids=input_ids, attention_mask=mask)
        hidden = out.last_hidden_state
        res_mask = mask.bool() & (~torch.isin(input_ids, special_ids))
        mf = res_mask.unsqueeze(-1).to(torch.float32)
        pooled = (hidden.to(torch.float32) * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1.0)
        pooled_np = pooled.cpu().numpy()
        if embs is None:
            embs = np.empty((len(smiles_list), pooled_np.shape[1]), dtype=np.float32)
        embs[idx] = pooled_np

    del model, tokenizer
    gc.collect(); torch.cuda.empty_cache()
    return embs

if EXTRACT_CLMS:
    device = torch.device("cuda")
    out_dir = (EMB_DIR / "_smoke_test") if SMOKE_TEST else EMB_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for mkey in ACTIVE_CLM_KEYS:
        spec = CLM_REGISTRY[mkey]
        cf = out_dir / f"{mkey}__global__{uniq_sha[:12]}.npz"
        if cf.exists() and not OVERWRITE_CACHE:
            print(f"[SKIP] {cf.name}")
            continue
        try:
            t0 = time.time()
            embs = extract_clm_embeddings(
                uniq_df["canonical_smiles"].tolist(), spec["hf_id"], spec["trust_remote"], spec["batch_size"], device
            )
            atomic_save_npz(cf, embeddings=embs, mol_hashes=np.asarray(uniq_df["mol_hash"].tolist(), dtype=np.str_))
            print(f"[SAVED] {cf.name} | shape={embs.shape} | {time.time()-t0:.1f}s")
        except Exception as exc:
            print(f"[WARN] {mkey} failed ({exc}); continuing remaining models.")''')

code('''# Cell 5 — Pure-PyTorch GIN-VN & GPS-Lite 5x Deep Ensemble training (Vectorized GPU-Resident) [GPU]
if TRAIN_GNNS:
    device = torch.device("cuda")
    graph_cache = {}

    def _ensure_graphs(df_part):
        added = 0
        for h, smi in zip(df_part["mol_hash"], df_part["canonical_smiles"]):
            if h not in graph_cache:
                graph_cache[h] = smiles_to_graph_dict(smi)
                added += 1
        return added

    gnn_rows = []
    t_stage = time.time()
    for task in ACTIVE_TASKS:
        spec = ALL_TASKS[task]
        ttype = spec["task_type"]
        pending_seeds = [
            s for s in SEEDS
            if OVERWRITE_CACHE or not (PRED_DIR / f"{'smoke_' if SMOKE_TEST else ''}gnn__{task}__seed{s}.npz").exists()
        ]
        if not pending_seeds:
            print(f"[SKIP TASK] {task} (all seeds {SEEDS} cached)")
            continue

        if task in ("chembl203_egfr", "chembl4822_bace1", "chembl240_herg"):
            tdir = confine_path(RAW_DIR, RAW_DIR / "chembl" / task)
        else:
            tdir = confine_path(RAW_DIR, RAW_DIR / "admet_group" / task)
        tv = pd.read_csv(tdir / "train_val.csv").dropna(subset=["Drug", "Y"]).reset_index(drop=True)
        te = pd.read_csv(tdir / "test.csv").dropna(subset=["Drug", "Y"]).reset_index(drop=True)
        tv["canonical_smiles"] = tv["Drug"].map(canonicalize_smiles)
        te["canonical_smiles"] = te["Drug"].map(canonicalize_smiles)
        tv = tv.dropna(subset=["canonical_smiles"]).reset_index(drop=True)
        te = te.dropna(subset=["canonical_smiles"]).reset_index(drop=True)
        tv["mol_hash"] = tv["canonical_smiles"].map(sha256_text)
        te["mol_hash"] = te["canonical_smiles"].map(sha256_text)
        if SMOKE_TEST:
            tv = tv.sample(n=min(120, len(tv)), random_state=42).reset_index(drop=True)
            te = te.head(16).reset_index(drop=True)

        _ensure_graphs(tv)
        _ensure_graphs(te)
        g_te = [graph_cache[h] for h in te["mol_hash"]]
        y_te = te["Y"].values.astype(np.float64)

        for seed in pending_seeds:
            t_ts = time.time()
            out_npz = PRED_DIR / f"{'smoke_' if SMOKE_TEST else ''}gnn__{task}__seed{seed}.npz"
            sp = create_scaffold_split_pytdc(tv, seed=seed, frac=(0.875, 0.125, 0.0), entity="Drug")
            tr_df, va_df = sp["train"], sp["valid"]
            if SMOKE_TEST and (len(tr_df) < 16 or len(va_df) < 4):
                cut = max(16, int(round(0.8 * len(tv))))
                tr_df, va_df = tv.iloc[:cut].reset_index(drop=True), tv.iloc[cut:].reset_index(drop=True)
            g_tr = [graph_cache[h] for h in tr_df["mol_hash"]]
            g_va = [graph_cache[h] for h in va_df["mol_hash"]]
            y_tr = tr_df["Y"].values.astype(np.float64)
            y_va = va_df["Y"].values.astype(np.float64)

            rng = np.random.RandomState(seed * 100 + 7)
            perm = rng.permutation(len(g_tr))
            n_iid = min(max(10, int(round(0.15 * len(g_tr)))), max(1, len(g_tr) // 3))
            iid_cal_i, iid_fit_i = perm[:n_iid], perm[n_iid:]
            g_iid_fit = [g_tr[i] for i in iid_fit_i]
            g_iid_cal = [g_tr[i] for i in iid_cal_i]

            payload = {"y_iid_cal": y_tr[iid_cal_i], "y_scaf_cal": y_va, "y_test": y_te}
            for arch in GNN_ARCHS:
                [iid_mu_c, iid_mu_t], _ = train_gnn_ensemble(
                    g_iid_fit, y_tr[iid_fit_i], g_va, y_va, [g_iid_cal, g_te],
                    task_type=ttype, arch=arch, n_ensemble=N_ENSEMBLE, base_seed=seed,
                    device=device, max_epochs=MAX_EPOCHS, patience=PATIENCE, batch_size=BATCH_SIZE,
                )
                [scaf_mu_c, scaf_mu_t], [scaf_sd_c, scaf_sd_t] = train_gnn_ensemble(
                    g_tr, y_tr, g_va, y_va, [g_va, g_te],
                    task_type=ttype, arch=arch, n_ensemble=N_ENSEMBLE, base_seed=seed + 10,
                    device=device, max_epochs=MAX_EPOCHS, patience=PATIENCE, batch_size=BATCH_SIZE,
                )
                payload[f"{arch}_iid_cal"] = iid_mu_c
                payload[f"{arch}_iid_test"] = iid_mu_t
                payload[f"{arch}_scaf_cal"] = scaf_mu_c
                payload[f"{arch}_scaf_test"] = scaf_mu_t
                payload[f"{arch}_sigma_cal"] = scaf_sd_c
                payload[f"{arch}_sigma_test"] = scaf_sd_t

                met = compute_task_metrics(y_te, scaf_mu_t, ttype)
                gnn_rows.append({"task": task, "seed": seed, "model": arch, **met})
                print(f"[W{WORKER_ID} {task:28s} s={seed}] {arch:8s} {spec['metric']}={met[spec['metric']]:.4f}")

            atomic_save_npz(out_npz, **payload)
            torch.cuda.empty_cache()
            print(f"    -> saved {out_npz.name} in {time.time()-t_ts:.1f}s")

    if gnn_rows and not SMOKE_TEST:
        csv_name = f"gnn_baseline_metrics__w{WORKER_ID}of{NUM_WORKERS}.csv" if NUM_WORKERS > 1 else "gnn_baseline_metrics.csv"
        pd.DataFrame(gnn_rows).to_csv(TAB_DIR / csv_name, index=False)
        print(f"[DONE W{WORKER_ID}/{NUM_WORKERS}] Finished in {(time.time()-t_stage)/60:.1f} min -> {csv_name}")''')

code('''# Cell 6 — Flush Drive & unmount cleanly [GPU]
if IN_COLAB:
    from google.colab import drive
    print("[SYNC] Flushing Google Drive buffers...")
    drive.flush_and_unmount()
    print("[DONE] Safe to disconnect runtime and stop CU billing.")
else:
    print("[DONE] Local GPU run complete.")''')

nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = ROOT / "d_notebook_02_clm_gnn_representation_gpu.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {out} ({len(CELLS)} cells)")
