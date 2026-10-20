# Experiment Log: Financial Fraud Detection

Record of training runs across Stratified 5-Fold Cross-Validation on the 200,000-row dataset (`Bank_Transaction_Fraud_Detection.csv`).

| Date | Model | Hyperparameters & Configuration | Random Seed | PR-AUC (Mean ± Std) | ROC-AUC (Mean ± Std) | Recall@FPR<=1% | Macro-F1 | Notes |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 2026-10-14 | **Logistic Regression** | `solver=lbfgs`, `max_iter=500`, `class_weight='balanced'`, `C=1.0` | 42 | 0.0503 ± 0.0013 | 0.5019 ± 0.0052 | 0.0073 ± 0.0014 | 0.3693 ± 0.0040 | Linear baseline; fast convergence (8.73s total across 5 folds). |
| 2026-10-14 | **Random Forest** | `n_estimators=100`, `max_depth=10`, `class_weight='balanced_subsample'`, `n_jobs=4` | 42 | 0.0505 ± 0.0012 | 0.4966 ± 0.0094 | 0.0093 ± 0.0012 | 0.4969 ± 0.0034 | Bagging tree ensemble; bounded depth for CPU memory stability (44.79s total). |
| 2026-10-14 | **XGBoost** | `n_estimators=100`, `max_depth=4`, `learning_rate=0.1`, `scale_pos_weight=18.8255`, `tree_method='hist'` | 42 | 0.0519 ± 0.0012 | 0.4997 ± 0.0061 | 0.0132 ± 0.0024 | 0.4161 ± 0.0045 | GBDT histogram boosting; highest Recall@FPR<=1% (14.88s total). |
| 2026-10-14 | **TabNet** | `n_d=8`, `n_a=8`, `n_steps=3`, `gamma=1.3`, `lambda_sparse=1e-3`, `batch_size=2048`, `virtual_batch_size=256`, `weights=1` | 42 | 0.0518 ± 0.0006 | 0.5089 ± 0.0028 | 0.0094 ± 0.0012 | 0.3936 ± 0.0368 | Deep attentive architecture; highest ROC-AUC with lowest variance across folds (449.29s total). |
| 2026-10-14 | **SMOTENC + Logistic Regression** | `SMOTENC(cat_idxs=0..8, 1:1 balance)`, `class_weight=None` | 42 | 0.0506 ± 0.0005 | 0.4994 ± 0.0005 | 0.0106 ± 0.0018 | 0.3888 ± 0.0018 | Phase 4.5 Imbalance Exp: Zero gain in PR-AUC; 11x slower execution time (102.11s). |
| 2026-10-14 | **SMOTENC + Random Forest** | `SMOTENC(cat_idxs=0..8, 1:1 balance)`, `class_weight=None` | 42 | 0.0500 ± 0.0011 | 0.4949 ± 0.0053 | 0.0102 ± 0.0011 | 0.4710 ± 0.0074 | Phase 4.5 Imbalance Exp: PR-AUC bounded at prior (0.0500); 4x slower execution (181.98s). |
| 2026-10-14 | **SMOTENC + XGBoost** | `SMOTENC(cat_idxs=0..8, 1:1 balance)`, `scale_pos_weight=1.0` | 42 | 0.0507 ± 0.0008 | 0.5011 ± 0.0036 | 0.0087 ± 0.0013 | 0.4871 ± 0.0000 | Phase 4.5 Imbalance Exp: Threshold shift caused F1=0 at 0.5; PR-AUC remained prior (105.08s). |
| 2026-10-14 | **SMOTENC + TabNet** | `SMOTENC(cat_idxs=0..8, 1:1 balance)`, `weights=0`, `n_d=8, n_a=8` | 42 | 0.0512 ± 0.0009 | 0.5040 ± 0.0053 | 0.0100 ± 0.0026 | 0.3755 ± 0.0807 | Phase 4.5 Imbalance Exp: Discrete category integrity preserved; PR-AUC at prior (414.19s). |
| 2026-10-15 | **Revised Behavioral + Logistic Regression** | 29 features, `class_weight='balanced'`, `C=1.0` | 42 | 0.0501 ± 0.0012 | 0.4985 ± 0.0076 | 0.0095 ± 0.0026 | 0.3715 ± 0.0065 | Phase 5 Behavioral: PR-AUC 0.0501 bounded by prior (2.52s total across 5 folds). |
| 2026-10-15 | **Revised Behavioral + Random Forest** | 29 features, 100 trees, `depth=12`, `balanced_subsample` | 42 | 0.0504 ± 0.0008 | 0.4935 ± 0.0062 | 0.0103 ± 0.0014 | 0.5001 ± 0.0031 | Phase 5 Behavioral: PR-AUC 0.0504 matches empirical fraud rate 5.044% (51.90s total). |
| 2026-10-15 | **Revised Behavioral + XGBoost** | 29 features, 100 trees, `depth=5`, `scale_pos_weight=18.8255` | 42 | 0.0515 ± 0.0011 | 0.5004 ± 0.0041 | 0.0124 ± 0.0032 | 0.4337 ± 0.0034 | Phase 5 Behavioral: PR-AUC 0.0515, ROC-AUC 0.5004 (5.44s total). |
| 2026-10-15 | **Revised Behavioral + TabNet** | 29 features, `n_d=8, n_a=8, steps=3`, `weights=1`, `batch=2048` | 42 | 0.0518 ± 0.0008 | 0.5082 ± 0.0037 | 0.0099 ± 0.0016 | 0.3690 ± 0.0789 | Phase 5 Behavioral: Highest ROC-AUC (0.5082) & lowest variance; bounded at prior (172.52s). |


