# -*- coding: utf-8 -*-
"""Pure-PyTorch molecular graph featurization and GIN+VN / GPS-Lite GNN ensembles (zero torch-scatter)."""
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem

ATOM_TYPES = [6, 7, 8, 9, 15, 16, 17, 35, 53]  # C, N, O, F, P, S, Cl, Br, I
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

ATOM_FEAT_DIM = len(ATOM_TYPES) + 1 + 6 + 5 + len(HYBRIDIZATIONS) + 1 + 5 + 2  # 35
EDGE_FEAT_DIM = len(BOND_TYPES) + 1 + 3  # 8


def _one_hot(val, choices: list) -> List[float]:
    vec = [0.0] * (len(choices) + 1)
    idx = choices.index(val) if val in choices else len(choices)
    vec[idx] = 1.0
    return vec


def smiles_to_graph_dict(smi: str) -> dict:
    """Convert SMILES to pure-numpy atom/edge graph dict (handles empty-bond single atoms safely)."""
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None or mol.GetNumAtoms() == 0:
        return {
            "x": np.zeros((1, ATOM_FEAT_DIM), dtype=np.float32),
            "edge_index": np.zeros((2, 0), dtype=np.int64),
            "edge_attr": np.zeros((0, EDGE_FEAT_DIM), dtype=np.float32),
        }

    x_list = []
    for atom in mol.GetAtoms():
        deg = min(atom.GetTotalDegree(), 5)
        chg = max(-2, min(2, atom.GetFormalCharge())) + 2
        nhs = min(atom.GetTotalNumHs(), 4)
        feat = (
            _one_hot(atom.GetAtomicNum(), ATOM_TYPES)
            + [1.0 if i == deg else 0.0 for i in range(6)]
            + [1.0 if i == chg else 0.0 for i in range(5)]
            + _one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
            + [1.0 if i == nhs else 0.0 for i in range(5)]
            + [1.0 if atom.GetIsAromatic() else 0.0, 1.0 if atom.IsInRing() else 0.0]
        )
        x_list.append(feat)

    src, dst, e_list = [], [], []
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ef = _one_hot(bond.GetBondType(), BOND_TYPES) + [
            1.0 if bond.GetIsConjugated() else 0.0,
            1.0 if bond.IsInRing() else 0.0,
            1.0 if bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE else 0.0,
        ]
        src.extend([u, v])
        dst.extend([v, u])
        e_list.extend([ef, ef])

    x = np.asarray(x_list, dtype=np.float32)
    if src:
        edge_index = np.asarray([src, dst], dtype=np.int64)
        edge_attr = np.asarray(e_list, dtype=np.float32)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, EDGE_FEAT_DIM), dtype=np.float32)

    return {"x": x, "edge_index": edge_index, "edge_attr": edge_attr}


def collate_graphs(graphs: List[dict], targets: Optional[np.ndarray] = None) -> dict:
    n_graphs = len(graphs)
    node_counts = [int(g["x"].shape[0]) for g in graphs]
    counts_long = torch.tensor(node_counts, dtype=torch.long)
    total_nodes = int(counts_long.sum().item())
    max_n = max(node_counts) if node_counts else 0

    x = torch.from_numpy(np.concatenate([g["x"] for g in graphs], axis=0))
    batch = torch.repeat_interleave(torch.arange(n_graphs, dtype=torch.long), counts_long)

    offsets = np.zeros(n_graphs, dtype=np.int64)
    if n_graphs > 1:
        np.cumsum(node_counts[:-1], out=offsets[1:])
    pos_in_graph = torch.arange(total_nodes, dtype=torch.long) - torch.repeat_interleave(
        torch.from_numpy(offsets), counts_long
    )
    pad_mask = torch.arange(max_n, dtype=torch.long).unsqueeze(0) < counts_long.unsqueeze(1)
    counts_t = counts_long.to(dtype=torch.float32).clamp(min=1.0).unsqueeze(-1)

    edge_indices = [
        g["edge_index"] + int(off)
        for g, off in zip(graphs, offsets)
        if g["edge_index"].shape[1] > 0
    ]
    edge_attrs = [g["edge_attr"] for g in graphs if g["edge_index"].shape[1] > 0]

    out = {
        "x": x,
        "edge_index": torch.from_numpy(np.concatenate(edge_indices, axis=1)) if edge_indices else torch.zeros((2, 0), dtype=torch.long),
        "edge_attr": torch.from_numpy(np.concatenate(edge_attrs, axis=0)) if edge_attrs else torch.zeros((0, EDGE_FEAT_DIM), dtype=torch.float32),
        "batch": batch,
        "node_counts": node_counts,
        "num_graphs": n_graphs,
        "counts_t": counts_t,
        "pos_in_graph": pos_in_graph,
        "max_n": max_n,
        "pad_mask": pad_mask,
    }
    if targets is not None:
        out["y"] = torch.as_tensor(targets, dtype=torch.float32)
    return out


class GINEConvLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.edge_proj = nn.Linear(hidden_dim, hidden_dim)
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_emb: torch.Tensor) -> torch.Tensor:
        agg = torch.zeros_like(h)
        if edge_index.shape[1] > 0:
            src, dst = edge_index[0], edge_index[1]
            msg = F.silu(h[src] + self.edge_proj(edge_emb))
            agg.index_add_(0, dst, msg)
        out = self.mlp((1.0 + self.eps) * h + agg)
        return self.norm(h + out)


class _GlobalNodeAttention(nn.Module):
    """Per-graph padded multi-head self-attention via PyTorch native SDPA (safe on CPU/L4/Blackwell)."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        node_counts: List[int],
        batch: Optional[torch.Tensor] = None,
        pos_in_graph: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None,
        max_n: Optional[int] = None,
    ) -> torch.Tensor:
        bsz = len(node_counts)
        if max_n is None:
            max_n = max(node_counts)
        if batch is None or pos_in_graph is None or pad_mask is None:
            counts_long = torch.tensor(node_counts, dtype=torch.long, device=h.device)
            batch = torch.repeat_interleave(torch.arange(bsz, dtype=torch.long, device=h.device), counts_long)
            offsets = torch.zeros(bsz, dtype=torch.long, device=h.device)
            if bsz > 1:
                offsets[1:] = torch.cumsum(counts_long[:-1], dim=0)
            pos_in_graph = torch.arange(h.shape[0], dtype=torch.long, device=h.device) - torch.repeat_interleave(
                offsets, counts_long
            )
            pad_mask = torch.arange(max_n, dtype=torch.long, device=h.device).unsqueeze(0) < counts_long.unsqueeze(1)

        padded = h.new_zeros((bsz, max_n, h.shape[1]))
        padded[batch, pos_in_graph] = h

        qkv = self.qkv(padded).view(bsz, max_n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, L, D)
        # Key mask: True = attend, False = masked
        attn_mask = pad_mask[:, None, None, :].expand(bsz, self.num_heads, max_n, max_n)
        ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)
        ctx = ctx.permute(0, 2, 1, 3).contiguous().view(bsz, max_n, -1)
        out_padded = self.norm(padded + self.drop(self.proj(ctx)))
        return out_padded[batch, pos_in_graph]


class MolecularGNN(nn.Module):
    """Unified GIN+VirtualNode and GPS-Lite (local GINE + global SDPA attention) backbone."""

    def __init__(self, arch: str = "gin_vn", hidden_dim: int = 128, num_layers: int = 4, dropout: float = 0.15):
        super().__init__()
        self.arch = arch
        self.hidden_dim = hidden_dim
        self.atom_enc = nn.Sequential(nn.Linear(ATOM_FEAT_DIM, hidden_dim), nn.SiLU())
        self.edge_enc = nn.Sequential(nn.Linear(EDGE_FEAT_DIM, hidden_dim), nn.SiLU())

        self.convs = nn.ModuleList([GINEConvLayer(hidden_dim, dropout=dropout) for _ in range(num_layers)])
        self.vn_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))
            for _ in range(num_layers)
        ])
        if arch == "gps_lite":
            self.attns = nn.ModuleList([_GlobalNodeAttention(hidden_dim, num_heads=4, dropout=dropout) for _ in range(num_layers)])
        else:
            self.attns = None

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch_dict: dict) -> torch.Tensor:
        h = self.atom_enc(batch_dict["x"])
        edge_emb = self.edge_enc(batch_dict["edge_attr"]) if batch_dict["edge_attr"].shape[0] > 0 else batch_dict["edge_attr"]
        batch = batch_dict["batch"]
        num_graphs = batch_dict["num_graphs"]
        node_counts = batch_dict["node_counts"]
        pos_in_graph = batch_dict.get("pos_in_graph")
        pad_mask = batch_dict.get("pad_mask")
        max_n = batch_dict.get("max_n")

        vn = h.new_zeros((num_graphs, self.hidden_dim))
        if "counts_t" in batch_dict:
            counts_t = batch_dict["counts_t"].to(dtype=h.dtype)
        else:
            counts_t = torch.tensor(node_counts, dtype=h.dtype, device=h.device).clamp(min=1.0).unsqueeze(-1)

        for layer_idx, conv in enumerate(self.convs):
            h = h + vn[batch]
            h_local = conv(h, batch_dict["edge_index"], edge_emb)
            if self.attns is not None:
                h = 0.5 * (
                    h_local
                    + self.attns[layer_idx](
                        h_local,
                        node_counts,
                        batch=batch,
                        pos_in_graph=pos_in_graph,
                        pad_mask=pad_mask,
                        max_n=max_n,
                    )
                )
            else:
                h = h_local
            pooled_sum = h.new_zeros((num_graphs, self.hidden_dim))
            pooled_sum.index_add_(0, batch, h)
            vn = vn + self.vn_mlps[layer_idx](pooled_sum / counts_t + vn)

        mean_pool = h.new_zeros((num_graphs, self.hidden_dim))
        mean_pool.index_add_(0, batch, h)
        mean_pool = mean_pool / counts_t

        # Graph max-like readout via mean + virtual-node representation
        g_repr = torch.cat([mean_pool, vn], dim=-1)
        return self.head(g_repr).squeeze(-1)


def _to_device(batch_dict: dict, device: torch.device) -> dict:
    return {
        k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
        for k, v in batch_dict.items()
    }


def _build_device_batches(
    graphs: List[dict],
    targets: Optional[np.ndarray],
    batch_size: int,
    device: torch.device,
    indices: Optional[np.ndarray] = None,
    merge_tiny_tail: bool = False,
) -> List[dict]:
    if not graphs:
        return []
    n = len(graphs) if indices is None else len(indices)
    chunks: List[List[int]] = []
    for start in range(0, n, batch_size):
        if indices is None:
            chunks.append(list(range(start, min(n, start + batch_size))))
        else:
            chunks.append(indices[start:start + batch_size].tolist())
    if merge_tiny_tail and len(chunks) >= 2:
        tail_nodes = sum(int(graphs[i]["x"].shape[0]) for i in chunks[-1])
        if tail_nodes <= 1:
            chunks[-2].extend(chunks.pop())
    batches = []
    for ch in chunks:
        g_sub = [graphs[i] for i in ch]
        y_sub = targets[ch] if targets is not None else None
        batches.append(_to_device(collate_graphs(g_sub, y_sub), device))
    return batches


@torch.no_grad()
def _predict_gnn_batches(
    model: nn.Module,
    batches: List[dict],
    task_type: str,
    use_bf16: bool = False,
) -> np.ndarray:
    if not batches:
        return np.zeros(0, dtype=np.float32)
    model.eval()
    preds = []
    for bd in batches:
        if use_bf16:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(bd).float()
        else:
            logits = model(bd).float()
        out = torch.sigmoid(logits) if task_type == "classification" else logits
        preds.append(out.cpu().numpy())
    return np.concatenate(preds, axis=0)


@torch.no_grad()
def predict_gnn(model: nn.Module, graphs: List[dict], device: torch.device, task_type: str, batch_size: int = 512) -> np.ndarray:
    if not graphs:
        return np.zeros(0, dtype=np.float32)
    use_bf16 = (device.type == "cuda") and torch.cuda.is_bf16_supported()
    batches = _build_device_batches(graphs, None, batch_size, device)
    return _predict_gnn_batches(model, batches, task_type, use_bf16=use_bf16)


def train_single_gnn(
    train_graphs: List[dict],
    y_train: np.ndarray,
    val_graphs: List[dict],
    y_val: np.ndarray,
    task_type: str,
    arch: str = "gin_vn",
    seed: int = 0,
    device: Optional[torch.device] = None,
    hidden_dim: int = 128,
    num_layers: int = 4,
    max_epochs: int = 60,
    patience: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    _precomputed_train_schedules: Optional[List[List[dict]]] = None,
    _precomputed_val_batches: Optional[List[dict]] = None,
    _precomputed_y_stats: Optional[Tuple[float, float]] = None,
) -> MolecularGNN:
    if len(train_graphs) == 0:
        raise ValueError("train_graphs cannot be empty.")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    use_bf16 = (device.type == "cuda") and torch.cuda.is_bf16_supported()

    if task_type == "regression":
        if _precomputed_y_stats is not None:
            y_mu, y_sd = _precomputed_y_stats
        else:
            y_mu = float(np.mean(y_train))
            y_sd = max(float(np.std(y_train)), 1e-6)
        y_tr_use = ((y_train - y_mu) / y_sd).astype(np.float32)
        y_va_use = ((y_val - y_mu) / y_sd).astype(np.float32)
        loss_fn = nn.SmoothL1Loss(beta=1.0)
    else:
        y_mu, y_sd = 0.0, 1.0
        y_tr_use = y_train.astype(np.float32)
        y_va_use = y_val.astype(np.float32)
        pos_cnt = float((y_tr_use == 1).sum())
        neg_cnt = float((y_tr_use == 0).sum())
        pos_w = torch.tensor([min(10.0, max(1.0, neg_cnt / max(1.0, pos_cnt)))], device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    if _precomputed_train_schedules is None:
        n_sched = 4 if len(train_graphs) > 4000 else 6
        train_schedules = [
            _build_device_batches(
                train_graphs, y_tr_use, batch_size, device,
                indices=rng.permutation(len(train_graphs)), merge_tiny_tail=True,
            )
            for _ in range(n_sched)
        ]
    else:
        train_schedules = _precomputed_train_schedules

    if _precomputed_val_batches is None:
        val_batches = _build_device_batches(val_graphs, y_va_use, max(512, batch_size), device)
    else:
        val_batches = _precomputed_val_batches

    model = MolecularGNN(arch=arch, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_loss = float("inf")
    best_state = None
    bad_epochs = 0
    n_val = max(1, len(val_graphs))
    n_sched_avail = len(train_schedules)
    sched_offset = int(rng.randint(0, n_sched_avail))

    for epoch in range(max_epochs):
        model.train()
        epoch_batches = train_schedules[(sched_offset + epoch) % n_sched_avail]
        b_order = rng.permutation(len(epoch_batches))
        for bi in b_order:
            bd = epoch_batches[bi]
            opt.zero_grad(set_to_none=True)
            if use_bf16:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(bd).float()
            else:
                logits = model(bd).float()
            loss = loss_fn(logits, bd["y"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

        if val_batches:
            model.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for bd in val_batches:
                    if use_bf16:
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            v_logits = model(bd).float()
                    else:
                        v_logits = model(bd).float()
                    val_loss_sum += float(loss_fn(v_logits, bd["y"]).item()) * int(bd["y"].shape[0])
            val_loss = val_loss_sum / n_val
            if val_loss < best_loss - 1e-5:
                best_loss = val_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    model._y_mu = y_mu
    model._y_sd = y_sd
    model.eval()
    return model


def predict_rescaled(model: MolecularGNN, graphs: List[dict], device: torch.device, task_type: str) -> np.ndarray:
    raw = predict_gnn(model, graphs, device, task_type)
    if task_type == "regression":
        return raw * getattr(model, "_y_sd", 1.0) + getattr(model, "_y_mu", 0.0)
    return raw


def train_gnn_ensemble(
    train_graphs: List[dict],
    y_train: np.ndarray,
    val_graphs: List[dict],
    y_val: np.ndarray,
    eval_graphs_list: List[List[dict]],
    task_type: str,
    arch: str = "gin_vn",
    n_ensemble: int = 5,
    base_seed: int = 0,
    device: Optional[torch.device] = None,
    **train_kwargs,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Train M-member GNN ensemble using GPU-resident pre-collated batches."""
    if len(train_graphs) == 0:
        raise ValueError("train_graphs cannot be empty.")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = int(train_kwargs.get("batch_size", 256))
    eval_bs = max(512, batch_size)
    use_bf16 = (device.type == "cuda") and torch.cuda.is_bf16_supported()

    if task_type == "regression":
        y_mu = float(np.mean(y_train))
        y_sd = max(float(np.std(y_train)), 1e-6)
        y_tr_use = ((y_train - y_mu) / y_sd).astype(np.float32)
        y_va_use = ((y_val - y_mu) / y_sd).astype(np.float32)
    else:
        y_mu, y_sd = 0.0, 1.0
        y_tr_use = y_train.astype(np.float32)
        y_va_use = y_val.astype(np.float32)

    sched_rng = np.random.RandomState(base_seed * 997 + 13)
    n_sched = 6 if len(train_graphs) > 4000 else 8
    train_schedules = [
        _build_device_batches(
            train_graphs, y_tr_use, batch_size, device,
            indices=sched_rng.permutation(len(train_graphs)), merge_tiny_tail=True,
        )
        for _ in range(n_sched)
    ]
    val_batches = _build_device_batches(val_graphs, y_va_use, eval_bs, device)
    eval_batches_list = [_build_device_batches(g_list, None, eval_bs, device) for g_list in eval_graphs_list]

    all_preds = [[] for _ in eval_graphs_list]
    try:
        for m in range(n_ensemble):
            model = train_single_gnn(
                train_graphs, y_train, val_graphs, y_val,
                task_type=task_type, arch=arch, seed=base_seed * 100 + m, device=device,
                _precomputed_train_schedules=train_schedules,
                _precomputed_val_batches=val_batches,
                _precomputed_y_stats=(y_mu, y_sd),
                **train_kwargs,
            )
            for idx, e_batches in enumerate(eval_batches_list):
                raw = _predict_gnn_batches(model, e_batches, task_type, use_bf16=use_bf16)
                rescaled = raw * y_sd + y_mu if task_type == "regression" else raw
                all_preds[idx].append(rescaled)
    finally:
        del train_schedules, val_batches, eval_batches_list

    means, stds = [], []
    for p_list in all_preds:
        P = np.stack(p_list, axis=0)
        means.append(P.mean(axis=0))
        stds.append(P.std(axis=0))
    return means, stds
