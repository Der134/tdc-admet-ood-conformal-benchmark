# 方向 D — 本地与 Google Colab 双账号执行手册（~4–7 CU，全程 0 G4）

> 配套核心文件与 Notebooks：
> - 本地一键全量 CPU 复现脚本：[`run_full_cpu_benchmark.py`](run_full_cpu_benchmark.py)（已完成 **25 任务 × 3 种子**全量运行，生成 [`results/tables/`](results/tables/) 8 张 CSV 与 [`results/figures/`](results/figures/) 5 张 300-DPI 图）
> - 阶段 1 Notebook：[`d_notebook_01_data_prep_ecfp_xgboost.ipynb`](d_notebook_01_data_prep_ecfp_xgboost.ipynb)（**CPU ~0 CU**）
> - 阶段 2 Notebook：[`d_notebook_02_clm_gnn_representation_gpu.ipynb`](d_notebook_02_clm_gnn_representation_gpu.ipynb)（**T4/L4 GPU ~4–7 CU**）
> - 阶段 3 Notebook：[`d_notebook_03_conformal_ood_virtual_screening.ipynb`](d_notebook_03_conformal_ood_virtual_screening.ipynb)（**CPU ~0 CU**）

---

## 0. 零配置共享 Google Drive（直接复用 `flip2_benchmark`）

代码中的 `resolve_workspace()` 会自动按以下优先级挂载共享工作区：
1. 若 `/content/drive/MyDrive/tdc_admet_benchmark/.workspace_id`（内容为 `tdc-admet-shared-prod-2026`）存在，优先使用该独立目录；
2. **否则自动检测你已配置好的 `/content/drive/MyDrive/flip2_benchmark/.workspace_id`**，并在其下自动创建 `direction_d_admet/` 子目录！
   → **无需重新建共享文件夹**，账号 1 与账号 2 直接互通嵌入与模型预测缓存。

---

## 1. 执行顺序与 CU 预算

| 步骤 | 入口 | Runtime | 预估时间 | 预估 CU | 产出资产 |
|---|---|---|---|---|---|
| **0（本地校验）** | `verify_direction_d.py` + `verify_gpu_nb02_mock.py` | **CPU（本地）** | ~15 s | **0** | 全部安全护栏、统计协议与 GPU→Stage 3 模拟管线通过验证 |
| **1（已完成）** | `d_notebook_01` 或 `run_full_cpu_benchmark.py` | **CPU（本地 / Colab）** | ~13 min（本地）/ ~20–24 min（Colab 2 vCPU） | **≈ 0** | `admet_manifest.json`、`unique_molecules.csv`（**51,827** SMILES）、`ecfp_rdkit__global__556783ec8166.npz`、75 份 RF/XGB/CQR 预测缓存 |
| **2a** | `d_notebook_02`（`SMOKE_TEST=True`） | **T4** | ~1 min | **~0.03** | 验证 CLM 下载 + Pure-PyTorch GNN 前向/原子写 |
| **2b** | `d_notebook_02`（`SMOKE_TEST=False, EXTRACT_CLMS=True, TRAIN_GNNS=False`） | **T4 或 L4** | ~8–15 min | **~0.5–1.0** | `chemberta2_mtr`、`chemberta2_mlm`、`molformer_xl` 全局唯一 SMILES 嵌入 `.npz` |
| **2c（双账号并行）** | `d_notebook_02`（`EXTRACT_CLMS=False, TRAIN_GNNS=True, NUM_WORKERS=2`；账号 1 设 `WORKER_ID=0`，账号 2 设 `WORKER_ID=1`） | **双 L4 或 双 T4** | ~1.5 h / 账号（含向量化批处理） | **~4–7 CU** | `gin_vn` 与 `gps_lite` 5 成员集成预测 + 不确定性 `.npz`（向量化 GPU 常驻批处理，零文件冲突） |
| **3** | `d_notebook_03`（含 CLM MLP 探针 + 保形评估 + 出图） | **CPU（Colab 2 vCPU）** | ~25–35 min（Cell 2 占 ~25–30 min；无 CLM 仅 Cell 3–5 约 2–3 min） | **≈ 0** | 8 张 CSV 结果表 + 5 张 300-DPI 论文图 |
| **合计** | **全量 22 TDC + 3 ChEMBL × 3 种子** | **0 小时 G4** | **~1.5 h** | **~4–7 CU** | 远低于 20–30 CU 原定上限 |

---

## 2. 本地一键复现与回归验证命令

```bash
# 1. 静态安全审计 + 保形数学单元测试 + 模拟 GPU Stage 2->3 端到端验证 (0 GPU 消耗)
python verify_direction_d.py
python verify_gpu_nb02_mock.py

# 2. 全量 25 任务 x 3 种子 CPU 基准与保形评估 (约 13-16 分钟, 复用已有 raw 数据)
python run_full_cpu_benchmark.py
```

---

## 3. 防浪费与断点续跑纪律

1. **G4 硬拦截**：`d_notebook_02` 内置 `ALLOW_G4 = False`；误开 96GB G4（10.5 CU/h）会在 Cell 1 直接报错终止，绝不烧冤枉 CU。
2. **全局去重 SMILES 缓存**：25 个任务的 **51,827 条**规范 SMILES 全局去重后只提取一次 CLM 嵌入，已存在的 `.npz` 自动 `[SKIP]`。
3. **分任务原子落盘**：`gnn__{task}__seed{seed}.npz` 每跑完 1 个 task-seed 立即使用 `.tmp` + `os.replace` 原子写入 Drive，断连重跑只补未完成部分；结束务必运行 Cell 6 `drive.flush_and_unmount()`。
4. **砍单优先级（若只想花 <2 CU 快速出跨模态表）**：
   - 仅跑 **2a + 2b（~1 CU）** 提取 3 个冻结 CLM 嵌入，跳过 2c（GNN 训练），直接跑 `d_notebook_03` 即可同时对比 `RF / XGB / ChemBERTa-MTR / ChemBERTa-MLM / MoLFormer-XL` 五大模型族！
