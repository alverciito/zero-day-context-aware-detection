---
name: kitnetzday
description: >-
  Companion code of the paper "Towards Context-Aware Detection: Evaluating
  Multi-Autoencoder Ensembles for Zero-Day Network Intrusion Detection"
  (JISAS-D-26-02300). Regenerates its tables, heat map, training audit and
  paired comparisons from the shipped aggregate and per-job files. Trains
  nothing: datasets, per-record scores and checkpoints are not shipped.
  Research code; CPU only.
provides:
  - kitnetzday.io.Inputs: reads the final aggregate, cells and statistics, and cross-checks them.
  - kitnetzday.figures: ROC-AUC heat map, mean ranks, oracle figure of the submitted regime, cases with limited detection.
  - kitnetzday.tables: main-results, ablation, cost, comparator and supplementary LaTeX tables.
  - kitnetzday.audit.TrainingAudit: per-autoencoder loss against its constant-predictor floor, label-free.
  - kitnetzday.audit.ConstantFloor: that floor, from the stream caches or from stored vectors.
  - kitnetzday.audit.RetrainedPair: paired comparison of the retraining experiment, windowed vs per-record.
  - kitnetzday.audit.UnitSummaries: the same comparison averaged to the unit of analysis (dataset), with exact statistics.
  - kitnetzday.audit.PairedStats: Hodges-Lehmann estimate, exact Wilcoxon interval and p value, in pure Python.
  - kitnetzday.audit.ReconstructionCorrelation: Spearman correlations with within-group permutation p values.
  - bin/run_all.py: runs the six entry points against the shipped data.
  - frozen/: the code that produced the results, without its comments, with the sha256 of every executed file.
  - added_analyses/: index of the analyses added after the analysis plan was fixed; their result files are in data/added_analyses/.
  - frozen/modified_encoders/ and data/modified_encoders/: the post hoc experiment with modified Conv1D and Conv2D encoders, its code and its per-run files.
keywords: [intrusion detection, autoencoder, kitsune, kitnet, anomaly detection,
  zero-day, network traffic, evaluation regime, data leakage, label-free threshold,
  roc-auc, windowdiff, hodges-lehmann, wilcoxon, reproducibility, negative result]
install: none (companion code of a single paper; unpack, `uv sync`, run bin/)
---
