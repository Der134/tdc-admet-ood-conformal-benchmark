#set page(
  paper: "a4",
  margin: (x: 14.5mm, y: 16mm),
  columns: 2,
  numbering: "1",
  header: context {
    if counter(page).get().first() > 1 [
      #set text(size: 8pt, fill: rgb("#555555"), font: "Calibri")
      #grid(
        columns: (1fr, 1fr),
        align(left)[Conformal Uncertainty & OOD Calibration for ADMET Virtual Screening],
        align(right)[Full 7-Model Cross-Modal Report (v2.0)],
      )
      #v(-4pt)
      #line(length: 100%, stroke: 0.4pt + rgb("#cccccc"))
    ]
  },
)

#set text(font: ("Cambria", "Georgia", "Times New Roman"), size: 8.95pt, fill: rgb("#1a1a1a"))
#set par(justify: true, leading: 0.54em, first-line-indent: 0pt, spacing: 0.58em)
#set heading(numbering: "1.1")

#show heading.where(level: 1): it => {
  v(0.60em, weak: true)
  set text(font: "Calibri", size: 10.4pt, weight: "bold", fill: rgb("#0f2b46"))
  block(below: 0.34em, sticky: true)[#it]
}

#show heading.where(level: 2): it => {
  v(0.42em, weak: true)
  set text(font: "Calibri", size: 9.3pt, weight: "bold", fill: rgb("#1f4e79"))
  block(below: 0.25em, sticky: true)[#it]
}

#show figure.caption: it => {
  set text(size: 7.8pt, fill: rgb("#333333"), font: "Calibri")
  it
}

#show table: set text(size: 7.1pt, font: "Calibri")
#show table: set par(justify: false)
#show raw: set text(font: "Consolas", size: 7.7pt)

#let accent = rgb("#0f2b46")
#let lightbg = rgb("#f4f7fa")

// Full-width Title & Abstract Block
#place(
  top + center,
  scope: "parent",
  float: true,
  [
    #align(center)[
      #text(font: "Calibri", size: 14.8pt, weight: "bold", fill: accent)[
        Conformal Risk Control and Out-of-Distribution Calibration Across Seven Molecular Representation Families for ADMET & Virtual Screening
      ]
      #v(0.40em)
      #text(font: "Calibri", size: 9.2pt, fill: rgb("#333333"))[
        *Der134* #h(1.0em) | #h(1.0em) Benchmark: `direction_d_admet_conformal` (7 Models #sym.times 25 Tasks #sym.times 3 Seeds) #h(1.0em) | #h(1.0em) November 2026
      ]
    ]
    #v(0.45em)
    #block(
      width: 100%,
      fill: lightbg,
      stroke: (left: 2.5pt + accent),
      inset: (x: 10pt, y: 7.5pt),
      radius: 2pt,
    )[
      #text(font: "Calibri", size: 8.6pt, weight: "bold", fill: accent)[Abstract — ]
      #text(size: 8.35pt)[
        Molecular property models in drug discovery are routinely deployed onto novel chemical series, yet benchmarks emphasize point accuracy while neglecting distribution-free uncertainty calibration across representation paradigms. We execute a protocol-faithful, cross-modal conformal evaluation spanning *seven model families*—two fingerprint tree ensembles (`RF`, `XGB`), three frozen Chemical Language Models (`MoLFormer-XL` @ross2022molformer, `ChemBERTa-2 MLM/MTR` @ahmad2022chemberta2), and two 5-member pure-PyTorch 2D Graph Neural Network ensembles (`GIN-VN` @xu2019gin, `GPS-Lite` @rampasek2022gps)—across *25 molecular datasets* (all *22 official TDC `ADMET_Group` assays* @huang2021tdc plus *3 ChEMBL `IC50` medicinal chemistry targets* @zdrazil2024chembl; `51,827` unique canonical SMILES, `70,230` train-validation and `17,577` test records #sym.times 3 scaffold seeds; `4,344` conformal evaluations).
        Every task exhibits *zero shared Bemis-Murcko scaffolds* (`shared_scaffolds = 0`) and substantial OOD shift (mean nearest-neighbor ECFP4 Tanimoto $d_"NN" = 0.5191$, `30.34%` in the Far OOD bin $d_"NN" >= 0.65$).
        Four findings emerge:
        *(1) Cross-Modal Complementarity:* Fingerprint trees still dominate ADMET classification (`RF` ROC-AUC `0.8306`, ECE `0.0834`) and congeneric ChEMBL SAR (`XGB` MAE `0.6023`, $rho=0.6809$), whereas pretrained `MoLFormer-XL` achieves the lowest regression MAE across all 12 tasks (`7.0292`; TDC-9 `9.1545` vs. `RF` `10.2731`) and `GPS-Lite` yields the tightest locally adaptive conformal intervals (`2.1077`).
        *(2) Universal IID Collapse & OOD Recovery:* Naive IID split conformal (`split_iid`) under-covers across every model family at $90\%$ (`0.8854`; `0.7727` at $80\%$), dropping to `0.7906` on `ld50_zhu` and decaying from `0.9102` (`Near`) to `0.8532` (`Far`). Scaffold-aligned calibrators restore validity (`split_scaffold` `0.9009`, `locally_adaptive_ncp` `0.9038`, `weighted_covariate_shift_cp` `0.9041`, `cqr` `0.9148` with `0.8985` Far-OOD coverage, `cv_plus_scaffold` `0.9183`).
        *(3) Bilateral Minority Rescue via Mondrian Calibration:* Across 13 imbalanced ADMET panels, marginal ICP fails *both* minority-positive CYP assays (`Y=1` `0.8354`/`0.8563`; `cyp2c9_substrate` `0.7293`) and majority-positive absorption/toxicity assays (`Y=0` `0.6227`/`0.6388`; `hia_hou` `0.5432`), whereas `mondrian_scaffold` restores both classes (`0.8923`/`0.8950`; CYP `0.9040`, Absorption/Tox `0.9157`).
        *(4) Screening Boundary Condition & EGFR #sym.times hERG MPO:* Naive Lower Confidence Bound (LCB) Top-$K$ reranking improves `MoLFormer-XL` (`+4.14%` mean FP reduction at $K=20$; `chembl240_herg` `+66.7%`, `clearance_microsome_az` `+42.9%`) and `RF` on noisy PK (`vdss_lombardo` `+71.0%` at $K=10$), yet backfires on heteroscedastic deep ensembles (`GIN-VN` `-68.8%`) where predictive variance scales with target magnitude. Deployed as a *threshold gate* on ChEMBL EGFR #sym.times TDC hERG (`n=400`), joint conformal screening isolates `87` leads with `95.40%` hit precision (`83/87`, `2.31x` enrichment; `34/35 = 97.14%` among dual-certified hERG-safe singletons).
      ]
    ]
    #v(0.40em)
  ]
)

= Introduction

Machine learning models for Absorption, Distribution, Metabolism, Excretion, and Toxicity (ADMET) and potency optimization are increasingly evaluated on Bemis-Murcko scaffold splits @bemis1996scaffolds @huang2021tdc to simulate lead-hopping into novel chemical series. Modern molecular ML spans three distinct representation paradigms: hand-engineered topological fingerprints with tree ensembles, pretrained Chemical Language Models (ChemBERTa-2 @ahmad2022chemberta2, MoLFormer-XL @ross2022molformer), and 2D message-passing / graph-transformer neural networks (GIN @xu2019gin, GraphGPS @rampasek2022gps). Yet medicinal chemistry teams rarely act on raw point predictions: synthesis campaigns select a *Top-$K$ slate* under strict multi-parameter safety filters (hERG, Ames, DILI).

Conformal prediction @vovk2005algorithmic provides finite-sample coverage guarantees under data exchangeability, which scaffold splitting explicitly violates. We conduct a unified cross-modal evaluation across all *7 representation models #sym.times 25 molecular datasets #sym.times 3 scaffold seeds*, addressing four research questions:
- *RQ1 (Cross-Modal Shift & Collapse):* How do fingerprint trees, frozen CLMs, and GNN ensembles compare in OOD point accuracy, calibration, and naive IID conformal coverage collapse across nearest-neighbor Tanimoto distance $d_"NN"$?
- *RQ2 (OOD Calibrators & Efficiency Frontier):* How do scaffold-split, 5-fold scaffold CV+ @barber2021jackknife, locally adaptive normalized residuals, Conformalized Quantile Regression (CQR) @romano2019cqr, and covariate-shift weighting @tibshirani2019covariate trade off coverage, interval width, and effective sample size (ESS) across all seven backbones?
- *RQ3 (Bilateral Imbalance & Selective Abstention):* Does marginal ICP conceal minority-class collapse in *both* low-prevalence CYP and high-prevalence absorption/toxicity panels, and how do Mondrian calibration and selective risk-coverage curves behave across representations?
- *RQ4 (Virtual Screening Utility & Failure Modes):* When does conformal LCB / Mondrian screening reduce false positives in "Buy Top-$K$" campaigns and multi-parameter EGFR #sym.times hERG gating, and when does heteroscedastic uncertainty cause LCB reranking to backfire?

= Benchmark Datasets, Models & Protocol

== Zero-Leakage Scaffold Datasets

We fetch the official TDC `ADMET_Group` archive (`4426004`) via HTTPS allowlisted verification and query three curated ChEMBL `IC50` targets (`pchembl_value` median-deduplicated per canonical SMILES, 2,000 compounds each) @zdrazil2024chembl. Across all 25 tasks (@tab-tasks), `70,230` `train_val` and `17,577` `test` records yield `51,827` globally deduplicated canonical SMILES (`shared_scaffolds = 0` on 25/25 tasks). For each seed $s in {1, 2, 3}$, we reproduce PyTDC's exact `create_scaffold_split(train_val, seed=s, frac=[0.875, 0.125, 0.0])` (`70%` train / `10%` scaffold-cal / `20%` test), alongside a `15%` random subset of `train` to expose naive IID calibration collapse (`split_iid` / `icp_iid`).

#figure(
  placement: top,
  scope: "parent",
  table(
    columns: (1.05fr, 0.5fr, 1.85fr, 0.7fr, 0.65fr, 0.85fr),
    stroke: (x: none, y: 0.4pt + rgb("#cccccc")),
    inset: 3.1pt,
    fill: (_, row) => if row == 0 { rgb("#e8eef5") } else if calc.odd(row) { rgb("#fafbfd") } else { white },
    table.header(
      [*Group (Count)*], [*Type*], [*Representative Tasks & Official Metrics*], [*Train/Val*], [*Test*], [*Mean $d_"NN"$ / Far*]
    ),
    [Absorption (6)], [3 Reg, 3 Cls], [`caco2_wang` (MAE), `lipophilicity`, `solubility`, `hia_hou`, `pgp`, `bioavail.`], [14,019], [3,507], [0.533 / 31.6%],
    [Distribution (3)], [2 Reg, 1 Cls], [`bbb_martins` (AUROC), `ppbr_az` (MAE), `vdss_lombardo` (Spearman)], [4,759], [1,191], [0.559 / 41.4%],
    [Metabolism (6)], [6 Cls], [`cyp2c9/2d6/3a4_veith` (PR-AUC), `cyp2c9/2d6/3a4_substrate`], [31,639], [7,917], [0.533 / 28.3%],
    [Excretion (3)], [3 Reg], [`half_life_obach`, `clearance_hepatocyte_az`, `clearance_microsome_az`], [2,383], [599], [0.548 / 35.9%],
    [Toxicity (4)], [1 Reg, 3 Cls], [`ld50_zhu` (MAE), `herg`, `ames`, `dili` (all AUROC)], [12,630], [3,163], [0.555 / 36.2%],
    [ChEMBL `IC50` (3)], [3 Reg], [`chembl203_egfr`, `chembl4822_bace1`, `chembl240_herg` (MAE)], [4,800], [1,200], [0.347 / 7.5%],
    [*All 25 Tasks*], [*12 Reg, 13 Cls*], [*`shared_scaffolds = 0` on 25/25 tasks; 51,827 unique SMILES*], [*70,230*], [*17,577*], [*0.519 / 30.3%*],
  ),
  caption: [Benchmark composition across all 22 TDC `ADMET_Group` and 3 ChEMBL `IC50` targets. `Far` is the test fraction with $d_"NN" >= 0.65$ to `train_val`.],
) <tab-tasks>

== Three-Arm Cross-Modal Suite & Conformal Protocol

We train and cache all three representation arms across all 25 tasks #sym.times 3 seeds (`d_notebook_01`–`03`):
- *Arm 1 — Tabular Fingerprints (2,231-D: `2048` ECFP4 + `167` MACCS + `16` train-scaled RDKit 2D):* `ecfp4_rdkit_rf` (100/120 trees with per-tree epistemic std $hat(sigma)(x)$) and `ecfp4_rdkit_xgb` (250 hist trees + `reg:quantileerror` CQR heads + 5-fold scaffold CV+).
- *Arm 2 — Frozen Chemical Language Models (CLMs):* Globally deduplicated embeddings extracted once on GPU (`51,827` SMILES) for `DeepChem/ChemBERTa-77M-MTR` (384-D), `DeepChem/ChemBERTa-77M-MLM` (384-D) @ahmad2022chemberta2, and `ibm-research/MoLFormer-XL-both-10pct` (768-D) @ross2022molformer, each coupled to a 5-member `128-64-1` SiLU MLP Deep Ensemble probe ($hat(mu), hat(sigma)$).
- *Arm 3 — Pure-PyTorch 2D GNN Deep Ensembles:* 5-member ensembles of `gin_vn` (4-layer GINE + Virtual Node) @xu2019gin and `gps_lite` (local GINE + vectorized per-graph SDPA global node attention) @rampasek2022gps trained with GPU-resident pre-collated batches (`batch_size=256`, `bf16` forward, `fp32` loss) across dual sharded GPU workers.

For $alpha in {0.10, 0.20\}$, we benchmark six regression calibrators (`split_iid`, `split_scaffold`, 5-fold `cv_plus_scaffold`, `locally_adaptive_ncp` on all 6 UQ-capable models, `cqr` on XGB, and `weighted_covariate_shift_cp`) and four classification calibrators (`icp_iid`, `icp_scaffold`, `mondrian_scaffold`, `weighted_mondrian_cp`). Covariate-shift density ratios $w(x) = p_"test"(x)/p_"cal"(x)$ are fit strictly without test labels via 3-fold OOF logistic regression on 32-D standardized PCA of the 2,231-D descriptors (clipped to $[0.05, 20.0]$); weighted quantiles follow Tibshirani et al. @tibshirani2019covariate, returning $+infinity$ abstention when calibration mass falls below $1 - alpha$.

= Cross-Modal Empirical Results

== Scaffold Diagnostics & 7-Model Point Benchmarks

TDC assays induce far steeper structural novelty (mean $d_"NN" = 0.5425$, peaking at `dili` `0.6677` / `65.6%` Far OOD and `vdss_lombardo` `0.6158` / `52.2%` Far OOD) than congeneric ChEMBL SAR (`0.3473`). @tab-baselines summarizes full 25-task #sym.times 3-seed point performance across all seven backbones:
1. *Classification & Calibration:* Fingerprint trees remain decisively strongest (`ecfp4_rdkit_rf` leads all four metrics: ROC-AUC `0.8306`, PR-AUC `0.8080`, Brier `0.1418`, ECE `0.0834`). Among neural backbones, `molformer_xl` leads (`0.8099` / `0.7931`), outperforming `chemberta2_mlm` (`0.8042`), `chemberta2_mtr` (`0.7882`), and small-data 2D GNNs (`gin_vn` `0.7865`, `gps_lite` `0.7833`).
2. *Regression Regime Split:* On broad-scaffold TDC PK regression (9 tasks), *every neural model beats tree ensembles on MAE*, led by `molformer_xl` (`9.1545`, `7.0292` across all 12 reg tasks) and `chemberta2_mlm` (`9.1848`) vs. `RF` (`10.2731`), while `RF` retains the highest Spearman rank correlation (`0.6032`; `0.6189` overall). Conversely, on congeneric ChEMBL `IC50` SAR (3 tasks), `ecfp4_rdkit_xgb` dominates (`MAE 0.6023`, $rho=0.6809$, $R^2=0.5190$). Across all 25 official metrics, `RF` (mean rank `2.64`, 8 wins) and `XGB` (`2.76`, 9 wins) lead overall, followed by `MoLFormer-XL` (`3.84`, 2 wins) and `GPS-Lite` (`4.36`, 1 win).

#figure(
  placement: top,
  scope: "parent",
  table(
    columns: (1.35fr, 0.72fr, 0.72fr, 0.72fr, 0.72fr, 0.75fr, 0.75fr, 0.65fr),
    stroke: (x: none, y: 0.4pt + rgb("#cccccc")),
    inset: 3.0pt,
    fill: (_, row) => if row == 0 { rgb("#e8eef5") } else if calc.odd(row) { rgb("#fafbfd") } else { white },
    table.header(
      [*Model Family*], [*Cls AUC*], [*Cls PR*], [*Cls ECE*], [*TDC9 MAE*], [*ChEMBL MAE*], [*12-Reg $rho$*], [*Rank (W)*]
    ),
    [`ecfp4_rdkit_rf`], [*0.8306*], [*0.8080*], [*0.0834*], [10.2731], [0.6369], [*0.6189*], [*2.64* (8)],
    [`ecfp4_rdkit_xgb`], [0.8263], [0.8044], [0.1008], [10.2826], [*0.6023*], [0.6119], [2.76 (*9*)],
    [`molformer_xl`], [0.8099], [0.7931], [0.1132], [*9.1545*], [0.6532], [0.5884], [3.84 (2)],
    [`chemberta2_mlm`], [0.8042], [0.7899], [0.1129], [9.1848], [0.6810], [0.5713], [4.60 (4)],
    [`chemberta2_mtr`], [0.7882], [0.7728], [0.1106], [9.7423], [0.7125], [0.5683], [5.00 (0)],
    [`gps_lite` (5x GNN)], [0.7833], [0.7760], [0.1250], [9.8053], [0.6637], [0.5624], [4.36 (1)],
    [`gin_vn` (5x GNN)], [0.7865], [0.7717], [0.1154], [9.5195], [0.7319], [0.5651], [4.80 (1)],
  ),
  caption: [Complete 7-model #sym.times 25-task #sym.times 3-seed point-prediction benchmark. `Rank (W)` is mean primary-metric rank (1 = best) and task win count across all 25 datasets.],
) <tab-baselines>

== Regression Coverage Collapse & Cross-Modal Efficiency (`RQ1`–`RQ2`)

#figure(
  placement: top,
  scope: "parent",
  image("results/figures/fig1_scaffold_ood_coverage_collapse.png", width: 97%),
  caption: [*A.* Empirical test coverage across 12 regression tasks #sym.times 3 seeds #sym.times 7 model families at $alpha=0.10$ (`2,160` evaluations). *B.* Monotonic coverage decay by nearest-neighbor ECFP4 Tanimoto distance $d_"NN"$.],
) <fig-collapse>

Pooling all seven representation families (@fig-collapse, @tab-reg-conf) confirms that IID conformal collapse is *architecture-agnostic*:
1. *Universal IID Under-Coverage:* Every model under-covers under `split_iid` at $alpha=0.10$ (7-model mean `0.8854`, range `0.8745` XGB to `0.8913` GIN-VN; `0.7727` at $alpha=0.20$), collapsing hardest on `ld50_zhu` (`0.7906`) and `chembl203_egfr` (`0.8340`) and decaying monotonically with $d_"NN"$ (`Near 0.9102 -> Moderate 0.8777 -> Far 0.8532`).
2. *Scaffold Recovery & Cross-Modal Frontier (@fig-frontier):* Scaffold-aligned calibration restores nominal $90\%$ validity across all 7 models (`split_scaffold` `0.9009`, `locally_adaptive_ncp` `0.9038`, `weighted_covariate_shift_cp` `0.9041`, `cqr` `0.9148`, `cv_plus_scaffold` `0.9183`). Remarkably, *representations differ sharply in interval tightness*:
   - Under `locally_adaptive_ncp`, `gps_lite` achieves the narrowest normalized width of any valid calibrator-model pair (`2.1077`, cov `0.8976`), ahead of `ecfp4_rdkit_rf` (`2.1514`, cov `0.9132`) and `molformer_xl` (`2.2048`, cov `0.8984`), whereas `chemberta2_mtr` inflates to `2.4637`.
   - Under unweighted `split_scaffold`, `molformer_xl` (`2.1445`) and `gps_lite` (`2.1651`) produce tighter intervals than `RF` (`2.2760`) or `GIN-VN` (`2.3177`).
   - `cqr` (`0.8985`) and `cv_plus_scaffold` (`0.8932`, ESS `2,358.2`) achieve the strongest Far-OOD coverage, while `weighted_covariate_shift_cp` (`domain_auc = 0.7119`, median ESS `35.3`) emits principled infinite-interval abstentions (`95.2%` finite; `ld50_zhu` s3 `domain_auc = 1.000 -> finite_frac = 0.0074`).

#figure(
  placement: top,
  scope: "parent",
  image("results/figures/fig2_interval_efficiency_cqr_vs_split.png", width: 84%),
  caption: [Conformal efficiency frontier across all 7 model families and 6 calibrators ($alpha=0.10$). `GPS-Lite` (adaptive NCP), `RF` (adaptive NCP), `XGB` (CQR/CV+), and `MoLFormer-XL` trace the upper-left Pareto envelope.],
) <fig-frontier>

#figure(
  placement: top,
  scope: "parent",
  table(
    columns: (1.55fr, 0.7fr, 0.65fr, 0.65fr, 0.65fr, 0.7fr, 0.65fr, 0.7fr),
    stroke: (x: none, y: 0.4pt + rgb("#cccccc")),
    inset: 3.1pt,
    fill: (_, row) => if row == 0 { rgb("#e8eef5") } else if calc.odd(row) { rgb("#fafbfd") } else { white },
    table.header(
      [*Calibrator ($alpha=0.10$)*], [*Cov.*], [*Near*], [*Mod.*], [*Far*], [*Width*], [*Finite*], [*ESS*]
    ),
    [`split_iid` (7 models)], [0.8854], [0.9102], [0.8777], [0.8532], [2.0964], [100.0%], [304.6],
    [`split_scaffold` (7 models)], [0.9009], [0.9258], [0.8933], [0.8706], [2.2390], [100.0%], [328.4],
    [`locally_adaptive_ncp` (6 UQ)], [0.9038], [0.9232], [0.8975], [0.8868], [2.2456], [100.0%], [328.4],
    [`cqr` (XGB Quantile)], [0.9148], [0.9339], [0.9110], [*0.8985*], [*2.1575*], [100.0%], [328.4],
    [`cv_plus_scaffold` (RF/XGB)], [*0.9183*], [0.9455], [0.9056], [0.8932], [2.1752], [100.0%], [*2,358.2*],
    [`weighted_covariate_shift_cp`], [0.9041], [0.9268], [0.8971], [0.8798], [2.3203], [95.2%], [161.5],
  ),
  caption: [Full 7-model regression conformal summary across 12 tasks #sym.times 3 seeds ($alpha=0.10$). Best per-model adaptive width is `gps_lite` (`2.1077`) followed by `ecfp4_rdkit_rf` (`2.1514`). At $alpha=0.20$, `split_iid` drops to `0.7727` vs. `0.8051–0.8336` for scaffold methods (`100%` finite).],
) <tab-reg-conf>

== Bilateral Minority Rescue & Selective Prediction (`RQ3`)

#figure(
  placement: top,
  scope: "parent",
  image("results/figures/fig3_mondrian_imbalanced_classification.png", width: 98%),
  caption: [*A.* Class-conditional coverage across 13 TDC classification tasks #sym.times 7 models ($alpha=0.10$, `2,184` evaluations). *B.* Selective prediction risk-coverage curves across all 7 models comparing probability margin (solid) vs. ensemble epistemic std (dashed).],
) <fig-cls>

Evaluating all 7 models across the 13 TDC binary tasks (@fig-cls, @tab-cls) uncovers *bilateral* minority-class collapse under marginal ICP:
- *Majority-Positive Absorption/Tox Collapse (`Y=0`):* Across all 13 tasks, `icp_iid` (`0.8964`) and `icp_scaffold` (`0.8942`) appear marginally valid only by letting negative-class coverage fall to `0.8062`/`0.8059`—and crashing to `0.6227`/`0.6388` across the four majority-positive panels (`hia_hou`, `bioavailability_ma`, `bbb_martins`, `herg`; `hia_hou` drops to `0.5432`/`0.6367`).
- *Minority-Positive CYP Collapse (`Y=1`):* On the four imbalanced CYP panels (`pos_rate 19%–34%`), marginal ICP under-covers scarce actives (`0.8354`/`0.8563`; `cyp2c9_substrate` `0.7293`/`0.7920`, `cyp2d6_veith` `0.7971`/`0.8022`).
- *Mondrian Restoration:* `mondrian_scaffold` simultaneously restores both sides (`Y=1` `0.8923`, `Y=0` `0.8950` overall; CYP `Y=1` `0.9040`, Absorption/Tox `Y=0` `0.9157`), and `weighted_mondrian_cp` reaches `0.8968` overall (`0.8936`/`0.8965`) with `0.8037` singleton accuracy (trading `46.0%` `{0, 1}` abstention on ambiguous OOD molecules).
- *Cross-Modal Uncertainty Ranking (@fig-cls B):* Retaining the `50%` most confident molecules cuts error monotonically across every model, yet reveals a key representation gap: on *neural ensembles*, `prob_margin` sharply outperforms `epistemic_std` (`molformer_xl` `0.2416 -> 0.1385` vs. `0.1938`; `gin_vn` `0.2678 -> 0.1807` vs. `0.2438`), whereas on `RF` *both* signals halve error equally (`0.2045 -> 0.0969` margin, `0.0979` tree std).

#figure(
  placement: bottom,
  scope: "parent",
  table(
    columns: (1.65fr, 0.65fr, 0.65fr, 0.65fr, 0.72fr, 0.65fr, 0.72fr),
    stroke: (x: none, y: 0.4pt + rgb("#cccccc")),
    inset: 3.1pt,
    fill: (_, row) => if row == 0 { rgb("#e8eef5") } else if calc.odd(row) { rgb("#fafbfd") } else { white },
    table.header([*Method ($alpha=0.10$)*], [*Cov.*], [*$Y=1$*], [*$Y=0$*], [*Sing. Rate*], [*Abst.*], [*Sing. Acc*]),
    [`icp_iid`], [0.8964], [0.9136], [0.8062], [0.6454], [0.3492], [0.8291],
    [`icp_scaffold`], [0.8942], [0.9148], [0.8059], [0.6571], [0.3356], [*0.8315*],
    [`mondrian_scaffold`], [0.8906], [0.8923], [0.8950], [0.5589], [0.4408], [0.7780],
    [`weighted_mondrian_cp`], [*0.8968*], [*0.8936*], [*0.8965*], [0.5358], [0.4603], [0.8037],
  ),
  caption: [Classification conformal metrics averaged across 13 TDC tasks #sym.times 3 seeds #sym.times 7 model families ($alpha=0.10$).],
) <tab-cls>

== "Buy Top-$K$" Screening & EGFR #sym.times hERG MPO (`RQ4`)

#figure(
  placement: top,
  scope: "parent",
  image("results/figures/fig4_topk_virtual_screening_fp_saved.png", width: 98%),
  caption: [*A.* Regression Top-20 hit rate (Greedy vs. Conformal LCB) disaggregated across all 7 model families. *B.* Per-endpoint relative false-positive reduction (%) at Top-20 for `MoLFormer-XL` and `RF`, contrasting high-gain OOD endpoints against heteroscedastic tail-penalty endpoints.],
) <fig-topk>

Disaggregating "Buy Top-$K$" decision simulation across all seven representations (@fig-topk) uncovers a vital practical boundary condition:
1. *Where Conformal LCB / Mondrian Excels:* `molformer_xl` is the strongest regression reranker, lifting mean Top-20 hit rate from `0.7722` to `0.7792` (`+4.14%` mean FP reduction; `chembl240_herg` `+66.7%`, `clearance_microsome_az` `+42.9%`, `caco2_wang` `+22.9%`). `ecfp4_rdkit_rf` maintains exact global Top-20 parity (`0.7681` vs. `0.7681`) while delivering dramatic gains on noisy OOD PK endpoints (`vdss_lombardo` `+71.0%` at $K=10$ and `+46.2%` at $K=20$; `solubility_aqsoldb` `+33.3%`; `clearance_microsome_az` `+33.3%`). Across all 13 classification tasks at $K=10$, Mondrian ranking sustains `91.28%` hit rate (`+1.14%` FP reduction; `GIN-VN` `+5.13%` at $K=20$).
2. *Why Naive LCB Reranking Backfires on Heteroscedastic Ensembles:* For `gin_vn` (`0.7514 -> 0.6097`, `-68.8%` FP change at $K=20$), `ecfp4_rdkit_xgb` CQR (`-48.9%`), and `chemberta2_mlm/mtr` (`-39.5%`/`-33.6%`), subtracting $q hat(sigma)(x)$ penalizes the highest-potency molecules whenever ensemble spread scales positively with target magnitude (e.g., `lipophilicity`, `clearance_hepatocyte_az`, `ld50_zhu`). Thus, *uncertainty calibration alone does not guarantee Top-$K$ ranking gain unless $hat(sigma)(x)$ is decoupled from target magnitude*.

#figure(
  placement: top,
  scope: "parent",
  image("results/figures/fig5_egfr_herg_mpo_case_study.png", width: 80%),
  caption: [Multi-parameter virtual screening on `400` unseen `CHEMBL203` EGFR compounds (`165` true hits, `41.25%` base rate). Joint conformal gating (`pred_lcb90 >= 6.2`, `herg_pval_safe0 > 0.10`) isolates `87` leads with `95.40%` hit precision (`83/87`; `34/35 = 97.14%` among dual-certified hERG-safe singletons).],
) <fig-mpo>

Conversely, deploying conformal bounds as a *multi-parameter safety gate* (@fig-mpo) avoids magnitude-penalty distortion entirely. On `400` held-out `CHEMBL203` EGFR molecules (`165` hits `pChEMBL >= 7.0`), intersecting `pred_lcb90 >= 6.2` with TDC `herg` Mondrian `herg_pval_safe0 > 0.10` isolates `87` candidates containing `83` true EGFR hits (`95.40%` precision, `2.31x` enrichment, mean `pChEMBL = 8.360`). Tightening to strict certified hERG-safe `{0}` singletons retains `35` dual-certified leads with `34` true hits (`97.14%` precision)—whereas greedy Top-10 and Top-20 EGFR picks contain only `3/10` (`30%`) and `8/20` (`40%`) certified hERG-safe compounds.

= Practical Guidelines & Reproducibility

1. *Match Representation to Task & Never Use IID Intervals Under Scaffold Shift:* Deploy `RF`/`XGB` for ADMET classification and congeneric SAR, `MoLFormer-XL` or `GPS-Lite` for broad-scaffold regression/tight adaptive intervals (`2.1077`–`2.1445`), and always replace `split_iid` with `locally_adaptive_ncp`, `cqr` (`0.8985` Far OOD), or `cv_plus_scaffold` (`0.9183`) while monitoring `weighted_covariate_shift_cp` ESS.
2. *Mandate Class-Conditional Mondrian Calibration on All Imbalanced ADMET Panels:* Mondrian calibration eliminates both minority-positive (`CYP`) and minority-negative (`Absorption/Tox`) coverage failures at zero retraining cost; for selective abstention on neural models, prefer `prob_margin` over raw ensemble `epistemic_std`.
3. *Prefer Conformal Threshold Gating Over Raw LCB Reranking:* Use LCB + Mondrian safety p-values as explicit MPO decision gates (`95.4%–97.1%` EGFR hit precision) and reserve LCB Top-$K$ reranking for well-behaved representations (`MoLFormer-XL`, `RF`) on high-OOD PK targets.

All code, unit/security suites (`verify_direction_d.py`, `verify_gpu_nb02_mock.py`), dual-worker GPU notebooks (`d_notebook_01`–`03`), verified 7-model CSV tables (`results/tables/`), and figure scripts (`regenerate_paper_figures.py`) are archived in `direction_d_admet_conformal/`.

#bibliography("refs_d.bib", style: "ieee")
