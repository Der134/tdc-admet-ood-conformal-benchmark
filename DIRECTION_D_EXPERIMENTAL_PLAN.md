# 方向 D 完整实施方案：TDC ADMET 分子性质 OOD 泛化与保形预测

> **最终完成状态（2026-11-24，v2.0 全模态完整版）**：全部 **7 大分子表征族（RF / XGB / ChemBERTa-2 MTR / ChemBERTa-2 MLM / MoLFormer-XL / GIN-VN / GPS-Lite）× 25 项任务（22 项 TDC `ADMET_Group` + 3 项 ChEMBL `IC50`）× 3 个随机骨架种子** 已全量实测完成，全量数据归档于 [`results/tables/`](results/tables/)（10 张 CSV，4,344 组保形评测）与 [`results/figures/`](results/figures/)（5 张 300-DPI 出版图件，含 [`regenerate_paper_figures.py`](regenerate_paper_figures.py)）。完整分析见 **[正式英文双栏学术 PDF 报告 (`ADMET_Conformal_Technical_Report.pdf`)](ADMET_Conformal_Technical_Report.pdf)**、[`README.md`](README.md) 与 [`README_zh.md`](README_zh.md)。

**定位**：以极低算力预算（**~4–8 CU，全程 T4/L4，0 小时 G4**）补齐“蛋白质（方向 A）+ 科研 Agent（方向 F）+ 小分子药物发现（方向 D）”三大 AI4Science 跨模态拼图。
**新颖性定位**：TDC ADMET 精度榜已被各类 GNN/大模型刷透，本方案的增量落在 **分布偏移下的不确定性校准失效诊断（Coverage Collapse）+ 协变量偏移重加权/Mondrian/CQR 保形预测 + “买 Top-K 分子”虚拟筛选风险决策增益**。

---

## 1. 核心研究问题（RQ1–RQ4）与全量实测结论

1. **RQ1（Scaffold OOD 下的覆盖率坍缩）**：当把随机切分（IID）校准的保形预测器部署到 TDC 官方 Bemis-Murcko scaffold 切分时，经验覆盖率相对名义 $1-\alpha=90\%$ 下降多少？覆盖率坍缩是否随最近邻 ECFP4 Tanimoto 距离 $d_{\text{NN}}(x)$ 单调恶化？
   - **7 模型全量实测结论**：25 项任务测试集与训练集均满足 **`shared_scaffolds = 0`**，平均 $d_{\text{NN}} = 0.5191$（`Far >= 0.65` 占比均值 `30.34%`）。在 7 模型 × 12 项回归任务 × 3 种子上，`split_iid` 在 $\alpha=0.10$ 下全架构一致失覆盖，均值跌至 **`0.8854`**（$\alpha=0.20$ 跌至 **`0.7727`**），在 `ld50_zhu`（`0.7906`）与 `chembl203_egfr`（`0.8340`）上坍缩最剧烈；且随结构距离单调恶化（`Near 0.9102 -> Moderate 0.8777 -> Far 0.8532`）。
2. **RQ2（跨模态表征的不确定性效率前沿）**：经典树模型特征（ECFP4+MACCS+RDKit 2D）、冻结化学语言模型（ChemBERTa-2 MTR/MLM、MoLFormer-XL）与 2D 分子图网络（GIN-VN、GPS-Lite）之间，哪类模型在点预测、**区间归一化宽度 vs 覆盖率 Pareto 前沿** 与 **不平衡分类少数类覆盖率** 上占优？
   - **7 模型全量实测结论**：三大模态高度互补——13 项分类任务中，`ecfp4_rdkit_rf` 全面领先（ROC-AUC `0.8306`、PR-AUC `0.8080`、ECE `0.0834`），`molformer_xl` 居神经模型之首（`0.8099`）；ChEMBL 3 项同系物 SAR 上 `ecfp4_rdkit_xgb` 最强（MAE `0.6023`、Spearman `0.6809`）；而 9 项广谱 TDC PK 回归中全部 5 个神经模型 MAE 均击败树模型，`molformer_xl` 以 `9.1545`（12 回归均值 `7.0292`）夺冠，`gps_lite` 给出最紧致的 `locally_adaptive_ncp` 保形区间（平均归一化宽度 **`2.1077`**，覆盖 `0.8976`）。
3. **RQ3（OOD 保形校准器对比与双向不平衡修复）**：Scaffold 留出校准（`split_scaffold`）、5 折 Scaffold 分组交叉保形（`cv_plus_scaffold`）、局部自适应归一化得分（`locally_adaptive_ncp`）、分位数回归保形（`cqr`）与 Tibshirani 协变量偏移重加权（`weighted_covariate_shift_cp`）的权衡如何？
   - **7 模型全量实测结论**：五类 OOD 校准器在 $\alpha=0.10$ 下均将平均覆盖率恢复至标称 90% 附近：`split_scaffold`（`0.9009`，`molformer_xl` 最窄 `2.1445`）、`locally_adaptive_ncp`（`0.9038`，`gps_lite` `2.1077` / `RF` `2.1514`）、`cqr`（`0.9148`，宽度 `2.1575`，`Far >= 0.65` 达 **`0.8985`**）、`cv_plus_scaffold`（`0.9183`，平均 ESS `2,358.2`）、`weighted_covariate_shift_cp`（`0.9041`，域判别 AUC `0.7119`，中位 ESS `35.3`；`ld50_zhu` s3 `domain_auc=1.000` 自动拒答 `finite_frac=0.0074`）。
   - **双向 Mondrian 修复**：标准边际 `icp_iid` / `icp_scaffold` 呈现**双向少数类盲区**——在少数阳性 CYP 任务上 $Y=1$ 跌至 `0.8354` / `0.8563`（`cyp2c9_substrate` `0.7293`），在多数阳性吸收/毒性任务上 $Y=0$ 跌至 `0.6227` / `0.6388`（`hia_hou` `0.5432`）；`mondrian_scaffold` 将全局两类拉回 `0.8923 / 0.8950`（CYP $Y=1 \to 0.9040$，吸收/毒性 $Y=0 \to 0.9157$）。选择性预测在神经模型上 `prob_margin` 显著优于 `epistemic_std`，而在 RF 上二者同等将错误率减半（`20.45% -> 9.69% / 9.79%`）。
4. **RQ4（虚拟筛选“买 Top-K 分子”决策边界与 MPO 实战）**：
   - **7 模型全量实测结论**：在 Top-20 回归筛选上，`molformer_xl` 实现全局正收益（平均 FP 削减 **`+4.14%`**，`chembl240_herg` `+66.7%`、`clearance_microsome_az` `+42.9%`），`ecfp4_rdkit_rf` 在高噪声 OOD PK 显著降假阳性（`vdss_lombardo` $K=10$ **`+71.0%`** / $K=20$ `+46.2%`）；而异方差深度集成（如 `gin_vn` `-68.8%`）因不确定度随活性幅值放大，直接 LCB 重排会误杀高活性候选。作为**阈值门控**用于 **EGFR (`CHEMBL203`) × hERG (`TDC`) MPO**（$n=400$，基线 `41.25%`）时，`pred_lcb90 >= 6.2 & herg_pval_safe0 > 0.10` 锁定 **`87`** 个候选（**`83` 个真命中，`95.40%` 精确率**，`2.31x` 富集，均值 `pChEMBL=8.360`），其中 **`35`** 个通过严格 hERG 安全单例 `{0}` 认证（**`34/35 = 97.14%` 命中率**）。

---

## 2. 数据集与严谨 OOD 协议

### 2.1 TDC `ADMET_Group` 全量 22 项任务
从 TDC 官方 Harvard Dataverse 归档直连拉取（`https://dataverse.harvard.edu/api/access/datafile/4426004`），严格复用官方 `train_val.csv` / `test.csv`，并比特级复现 PyTDC `create_scaffold_split(train_val, seed=seed, frac=[0.875, 0.125, 0.0])`（最终比对 **70% train / 10% scaffold-valid / 20% scaffold-test**，3 个随机种子 `seeds=[1, 2, 3]`）：
- **Absorption (6)**：`caco2_wang` (MAE), `hia_hou` (ROC-AUC), `pgp_broccatelli` (ROC-AUC), `bioavailability_ma` (ROC-AUC), `lipophilicity_astrazeneca` (MAE), `solubility_aqsoldb` (MAE)
- **Distribution (3)**：`bbb_martins` (ROC-AUC), `ppbr_az` (MAE), `vdss_lombardo` (Spearman)
- **Metabolism (6)**：`cyp2c9_veith` (PR-AUC), `cyp2d6_veith` (PR-AUC), `cyp3a4_veith` (PR-AUC), `cyp2c9_substrate_carbonmangels` (PR-AUC), `cyp2d6_substrate_carbonmangels` (PR-AUC), `cyp3a4_substrate_carbonmangels` (ROC-AUC)
- **Excretion (3)**：`half_life_obach` (Spearman), `clearance_hepatocyte_az` (Spearman), `clearance_microsome_az` (Spearman)
- **Toxicity (4)**：`ld50_zhu` (MAE), `herg` (ROC-AUC), `ames` (ROC-AUC), `dili` (ROC-AUC)

> **Pilot 子集（6 项代表任务）**：`caco2_wang`、`solubility_aqsoldb`、`vdss_lombardo`、`bbb_martins`、`cyp2d6_veith`、`herg`（覆盖回归 MAE/Spearman + 分类 AUROC/AUPRC + 毒性反靶标）。

### 2.2 ChEMBL 真实药化靶点案例（3 项）
通过 ChEMBL REST API 拉取 `IC50` 且 `pchembl_value` 非空的实测化合物，按规范 SMILES 取中位数去重（每靶点上限 2,000 分子），按 Bemis-Murcko 骨架 80/20 外层切分 + 87.5/12.5 内层切分：
- `CHEMBL203`（EGFR 激酶，抗肿瘤，高活性阈值 `pChEMBL >= 7.0`；`1,600 train_val / 400 test`）
- `CHEMBL4822`（BACE1，神经退行，高活性阈值 `pChEMBL >= 7.0`；`1,600 train_val / 400 test`）
- `CHEMBL240`（hERG 钾通道，心脏毒性反靶标，低风险优先 `pChEMBL <= 6.0`；`1,600 train_val / 400 test`）

---

## 3. 模型矩阵与算力分级

| 臂 | 模型 | 运行硬件 | 实测耗时 | 状态与算力 |
|---|---|---|---|---|
| **Arm 1（表格基线）** | `ecfp4_rdkit_rf`（100/120 棵树 std UQ）+ `ecfp4_rdkit_xgb`（点估计 + `reg:quantileerror` CQR） | **CPU** | ~13 min (本地) / ~20–24 min (Colab) | **已完成全量实测（~0 CU）** |
| **Arm 2（冻结化学 LM）** | `ChemBERTa-77M-MTR`、`ChemBERTa-77M-MLM`、`MoLFormer-XL-both-10pct`（GPU 提嵌入 → CPU 5×MLP Deep Ensemble） | **T4/L4** + CPU | GPU ~10–15 min + CPU ~25–30 min | **已完成全量实测（~0.5–1.0 CU）** |
| **Arm 3（纯 PyTorch 2D GNN）** | `GIN-VN`（GIN + Virtual Node）+ `GPS-Lite`（局部 GINE + 向量化全局 SDPA 注意力），5 成员集成 | **双账号 T4/L4** | ~1.5 h / 账号（`NUM_WORKERS=2`） | **已完成全量实测（~4–7 CU）** |
| **全量保形评估 & 绘图** | 6 种回归保形 + 4 种分类保形 + Top-K 筛选 + EGFR/hERG MPO | **CPU** | ~2–3 min（仅统计与绘图） | **已完成全量实测（~0 CU）** |
| **总计** | **全 7 模型 × 22 TDC + 3 ChEMBL × 3 种子** | **全程 T4/L4（0 G4）** | **Colab 全栈完成** | **10 CSV + 5 PNG + 5 页 PDF** |

---

## 4. 保形预测与 Top-K 决策数学协议

设校准集为 $\mathcal{D}_{\text{cal}} = \{(x_i, y_i)\}_{i=1}^n$，名义失覆盖率 $\alpha \in \{0.10, 0.20\}$。

1. **回归非一致性得分与区间构造**：
   - `split_iid` / `split_scaffold`：$s_i = |y_i - \hat{\mu}(x_i)|$，有限样本分位数 $\hat{q} = \text{Quantile}_{\lceil(n+1)(1-\alpha)\rceil/n}(s_{1:n})$。
   - `cv_plus_scaffold`：5 折 Bemis-Murcko 骨架分组交叉保形，$\hat{C}_{\text{CV+}}(x) = [\text{Quantile}_\alpha(\hat{\mu}_{-k(i)}(x)-r_i), \text{Quantile}_{1-\alpha}(\hat{\mu}_{-k(i)}(x)+r_i)]$。
   - `locally_adaptive_ncp`：$s_i = |y_i - \hat{\mu}(x_i)| / (\hat{\sigma}(x_i) + \gamma)$，预测区间随认识不确定性 $\hat{\sigma}(x)$ 自适应放宽（$\gamma = \max(\text{median}(\hat{\sigma}_{\text{cal}}), 10^{-3})$）。
   - `cqr`：基于 Quantile XGBoost 输出 $[\hat{q}_{\alpha/2}(x), \hat{q}_{1-\alpha/2}(x)]$ 的非对称保形分位数回归，$s_i = \max(\hat{q}_{\alpha/2}(x_i) - y_i, y_i - \hat{q}_{1-\alpha/2}(x_i))$。
   - `weighted_covariate_shift_cp`：仅使用无标签特征 $(X_{\text{cal}}, X_{\text{test}})$ 通过 3 折 OOF PCA-逻辑回归判别器估计裁剪似然比 $w(x) = \text{clip}(\frac{\hat{p}(\text{test}|x)}{1-\hat{p}(\text{test}|x)}\frac{n_{\text{cal}}}{n_{\text{test}}}, 0.05, 20.0)$，对测试点计算加权经验分位数并记录有效样本量 $\text{ESS} = (\sum w_i)^2 / \sum w_i^2$。
2. **分类保形（含不平衡类别条件 Mondrian）**：
   - `icp_iid` / `icp_scaffold`：全局边际非一致性得分 $s(x, y) = 1 - \hat{p}_y(x)$。
   - `mondrian_scaffold` / `weighted_mondrian_cp`：分别在 $Y=0$ 与 $Y=1$ 校准子集上估计类条件分位数 $\hat{q}_0, \hat{q}_1$，保护如 `cyp2d6_veith`、`cyp2c9_substrate_carbonmangels` 等极不平衡任务中稀缺活性分子的召回覆盖率。
3. **“买 Top-K 分子”风险可控筛选**：
   - 贪心策略按点估计 $\hat{\mu}(x)$（或 $\hat{p}_1(x)$）取前 $K$；
   - 保形策略对最大化任务按 **90% 下置信界** $\hat{L}_{0.90}(x)$ 排序（对最小化毒性/清除率任务按 $-\hat{U}_{0.90}(x)$ 或 Mondrian 单例安全集排序），统计 `FP_Saved@K = FP_greedy - FP_conformal` 与相对假阳性削减率。

---

## 5. 与 Google Antigravity Science 生态衔接

- `chembl` / `pubchem` skill：用于对 `CHEMBL203`（EGFR）与 `herg` 多参数交叉筛选出的 35 个双认证高置信低毒分子（`34/35 = 97.14%` 命中率，[`results/figures/fig5_egfr_herg_mpo_case_study.png`](results/figures/fig5_egfr_herg_mpo_case_study.png) 右上象限）进行真实化学结构与先导化合物文献背景核查。
- `openfda` skill：对 TDC `dili` / `ames` 高不确定性分子对照 FDA 已上市药物黑框警告标签做临床安全性归因。
