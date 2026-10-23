# kitnetzday - companion code of JISAS-D-26-02300

"Towards Context-Aware Detection: Evaluating Multi-Autoencoder Ensembles
for Zero-Day Network Intrusion Detection" (revised manuscript, *Journal of
Information Security and Applications*).

This package holds the final results of the study, the small per-run files
behind them, the code that turns them into the tables and figures listed
below, and the code that ran the experiments. It does **not** hold the
datasets, the per-record scores or the model checkpoints, so it regenerates
the outputs of the paper that are computed from the final results and the
per-run files, and nothing upstream of them. The two tables below say
exactly which is which. The analyses that were added after the analysis
plan was fixed travel as their result files (`data/added_analyses/`,
indexed in `added_analyses/README.md`); each of the two sections below says
which of them the shipped files reproduce. The post hoc experiment with
modified Conv1D and Conv2D encoders, reported at the end of Section 6.1 of
the manuscript and in Section 35 of the supplement, travels apart: its
per-run files and results in `data/modified_encoders/`, and the code that
ran it in `frozen/modified_encoders/`.

`frozen/` is the code as it was executed, with its comments removed, the
docstrings that held working notes shortened and the text of some messages
reworded. `frozen/README.md` says why, how this was checked against the
tagged files and which part of the check can be repeated from this
package. The working notes that the scripts wrote into the per-job files and
the result files of `data/` were removed, and their paths are given from
the root of the authors' repository (`data/README.md`). No number was
changed by either.

The manuscript ("Data availability", "Code availability") and Section 28
of the supplement name what this package holds. It is here:

| Named in the article | Where it is |
|---|---|
| SHA-256 of the source files of every stream, and the row indices of its training, calibration and evaluation segments | `data/streams.json` |
| The per-run metric files | `data/jobs/*/metrics.json` |
| The per-run metric files of the post hoc experiment with modified encoders | `data/modified_encoders/jobs/*/detection_metrics.json` (the 72 runs of seeds 1234 and 5678; the runs of seed 9012 enter no comparison and have none) and `data/modified_encoders/jobs/*/convergence_report.json` (all 108 runs) |
| The code that builds the streams | `frozen/runner/pipelines.py`, `frozen/streams/` |
| The model code | `frozen/models/`, `frozen/runner/networks_ctx.py`, `frozen/runner/networks_eps.py` |
| The code of the modified encoders, and the record of the runs used to choose their changes | `frozen/modified_encoders/` (`modelos_rep.py` builds the modified encoders; `frozen/README.md` lists every file), and `data/modified_encoders/selection_runs.json` with its summary `selection_summary.json` |
| The code that computes the detectors, thresholds, metrics and statistics | `frozen/runner/metrics_rev.py`, `aggregate.py`, `stats.py` |
| The checksum of the analysis plan | SHA-256 `f65830b5029614758e784fba0bee33b8b1d3cf3c212d53c0b0d883d30ebdaf89`, recorded as `provenance.plan.plan_sha256` in `data/stats.json`, in `data/aggregate/aggregate.json` and in every `data/jobs/*/metrics.json`. The plan itself is not in this package; Section 28 of the supplement states that it is available to the editor and the reviewers on request. |
| The checksum of the amendment to the analysis plan that fixed the post hoc experiment with modified encoders | SHA-256 `fafeb68e300b6f63348980de2ea2e92d55dd5219c3a8f60d459bb86b61a7e5d4`, of the amendment fixed on 7 October 2026, recorded as `provenance.enmienda.sha256` in `data/modified_encoders/comparisons.json`. The amendment is not in this package either; the supplement states that it is available to the editor and the reviewers on request, as the plan is. |

## Running it

CPU only, about one minute. Python 3.12 (`.python-version`).

```bash
uv sync                      # or: pip install -r requirements-analysis.txt && pip install -e .
uv run bin/run_all.py        # the six entry points, in order, into data/out/
uv run pytest -q             # the tests, under a minute
```

Each entry point can also be run on its own (`uv run bin/<name>.py --help`).
A script that does not find its inputs stops with exit code 1 and a message
that says which file it needs and where it comes from; none of them writes
an empty result.

`data/reference/` holds the outputs of `bin/run_all.py` as the authors
generated them. After a run, `diff -rq data/reference data/out` compares
the two. With the pinned environment the only files that differ are
`figures.json`, which records the time of the run and its output folder,
and `SHA256SUMS`, which hashes it; the tables, the result files and the
figures come out byte for byte. With other library versions the last digits
of a float and the rendering of the figures may move. `tests/bin/` makes
the comparison of the tables and the result files automatically, number by
number.

## What is regenerated here

Float and section numbers are those of the revised manuscript and of its
supplementary material as submitted in October 2026. For the floats of the
supplement the LaTeX label is given as well.

| Output of the paper | Script | Output file in `data/out/` | Regenerated here |
|---|---|---|---|
| Table 5: ROC-AUC of the eleven configurations on the nine streams, prevalence, AUC_t | `bin/make_figures.py` | `tab_main_results_compact.tex`, `tab_main_results.tex` | **Yes (values).** Read from `data/aggregate/cells.csv`. The table in the manuscript is the same set of values typeset by hand. |
| Table 5, Isolation Forest row | `bin/make_baseline_tables.py` | `baselines.md`, `baselines.json` | **Yes (values).** One run per stream. |
| Table 4: AUC and EER on the six KNAD streams | `bin/make_baseline_tables.py` | `baselines.md`, `baselines.json` | **Yes (values), except the two MLP columns and AUC_t.** The script writes the reference KitNET, the KitNET-style control and, in a third block that Table 4 does not print, the variant with the highest ROC-AUC on each capture. The AUC of the MLP is its row of Table 5, and AUC_t is the AUC_t row of Table 5 (`tab_main_results_compact.tex`); its EER is not written by any script here and is the `eer` column of `data/runs.csv`. |
| Figure 3: ROC-AUC heat map | `bin/make_figures.py` | `fig_main_auroc_heatmap.pdf` | **Yes.** |
| Supplement, Table S47 (`tab:s_finalloss`), left block (mean of the epoch); its counts are summarised in Section 6.1 of the manuscript | `bin/audit_training.py` | `training_audit.json` | **Yes, from stored floors.** The loss of every autoencoder comes from its `train_log.csv`; the constant-predictor floor comes from `data/floor_vectors.json`, which the authors computed from the preprocessed streams (not shipped). The right block of the table (final weights) needs the per-record errors of every autoencoder, which are not shipped. |
| Table 7: ablation study | `bin/make_figures.py` | `tab_ablations.tex` | **Partly.** The per-block summaries regenerate; the table is assembled by hand from them, from `data/stats.json#/ablations` and, for its last row, from the calibration-contamination series (`T4P-b`) of `data/runs.csv`. |
| Table 9: computational cost per variant | `bin/make_figures.py` | `tab_cost.tex` | **Yes (values).** Twelve reference runs. Table 9 prints six of the twelve columns (variant, weights, training time, records per second, GPU peak and peak RAM); it averages the two MLP rows and adds by hand the weight counts of the two NumPy variants. |
| Manuscript, Section 6.2 ("Retraining experiment"), and supplement, Sections 12 and 32: the paired comparison on detection, and the five-epoch ablation | `bin/analyse_retrained_pair.py` | `retrained_pair.json` | **Yes (detection).** Wins, median, Hodges-Lehmann estimate, exact interval and signed-rank p, per cell and at the unit of the plan (`by_unit`: the dataset, the dataset and paradigm, and each paradigm half). The p value is exact up to 20 non-zero differences, which covers every `by_unit` summary; per cell (36 differences) it is the normal approximation. How many autoencoders of each member are below 0.9 of the constant-predictor error comes from `bin/audit_training.py`. The ratios of validation error between the two members are not regenerated: they need the per-run training records, which are not shipped. |
| Supplement, Section 10: the two exploratory correlations on the cases of Section 6.7 of the manuscript ("Cases with limited detection") | `bin/audit_training.py`, then `bin/analyse_reconstruction.py` | `reconstruction_vs_detection.json` | **Yes.** |
| Supplement, Tables S6 to S8, S10 to S13, S16, S32, S35 to S37, S39, S40 and S42 (`tab:s_f1`, `s_wd`, `s_auprc`, `s_kmd`, `s_mvd`, `s_fpr`, `s_alarm`, `s_loc`, `s_sens`, `s_attack`, `s_attack_cfg_a`, `s_attack_cfg_b`, `s_seeds_a`, `s_seeds_b`, `s_perday`) | `bin/make_supplementary_tables.py` | `supplementary_tables.tex`, `supplementary_tables.json` | **Yes.** Fifteen floats, with the labels of the supplement. |
| Supplement, Table S41 (`tab:s_best`) | `bin/make_figures.py` | `tab_best_per_dataset.tex` | **Yes (values).** |
| Figure A.1 (Appendix A of the manuscript): the values of the figure of the original submission, redrawn in one figure | `bin/make_figures.py` | `fig_appendix_fig7_oracle.pdf` | **Yes (drawing).** The 216 bar values are read from `data/fig7_valores_publicados.csv` and the always-alarm F1 from `data/fig7_trivial_f1.csv`; they are the archived values of the submission, not recomputed here. The script writes a title and caption of its own with it. Appendix A prints the same values under the title and caption of the label-aware regime. |
| Table 6 and the confirmatory statistics (H1-H4, Holm, Hodges-Lehmann intervals, Friedman, sensitivities) | - | - | **No, shipped as data** in `data/stats.json` and `data/tests.csv`. `tests/stats/` recomputes the five comparisons of H1 from `cells.csv` with code that shares nothing with the statistics script of `frozen/`, and they agree. |

`bin/make_figures.py` also writes two outputs that are not floats of the
manuscript or of the supplement: `fig_mean_ranks` (its numbers are the H1
rows of Table 6 of the manuscript) and `fig_failure_cases` (the cases of
Section 6.7 of the manuscript).

The `.tex` files of `bin/make_figures.py` are stand-alone floats with
labels and captions of their own (`tab:main-results`, `tab:ablations`,
`tab:best-per-dataset`, `tab:cost`, `fig:auroc-heatmap`,
`fig:appendix-fig7`), not those of the manuscript and the supplement, which
typeset their own version of each float from the same values. "Yes
(values)" in the table above means exactly that: the numbers regenerate,
the typeset float is the one of the article.

### Added analyses that the shipped files reproduce

Seven analyses computed the tables that the supplement gained after the
analysis plan was fixed; the result file of each is in
`data/added_analyses/`, and `added_analyses/README.md` maps every table to
its section of the supplement, its result file, its status and its inputs.
Their scripts are not part of this package and `bin/run_all.py` runs none
of them. For the tables below every input is in this package:
`tests/added_analyses/` recomputes them from the shipped files with the
routines of `src/` and compares the outcome with the result file.

| Table of the supplement | Status | Result file (`.json`) | Recomputed in the tests from |
|---|---|---|---|
| Table S67 (`tab:s_seven`): the twelve confirmatory comparisons on the seven streams whose calibration window holds no attack records | Exploratory, added after the analysis plan was fixed | `siete_flujos` | `data/stats.json` (the per-stream differences of each comparison): Hodges-Lehmann estimate, number of streams with a positive difference, exact 95.3 % interval. |
| Table S68 (`tab:s_insample_two`): in-sample calibration on the two streams with attack records in the calibration window, by configuration | Pre-specified sensitivity analysis | `siete_flujos` | `data/runs.csv`: change of ROC-AUC, F1, FPR and recall of the fused detector. |
| Table S55 (`tab:s_rawcell`): ROC-AUC of the raw score per cell, over the whole evaluation segment and inside blocks of 10,000 records | Pre-specified sensitivity analysis | `score_bruto` | `data/aggregate/cells.csv`, columns `auroc_RAW_mean` and `auc_loc_RAW_mean`. |
| Table S69 (`tab:s_knad_raw`): ROC-AUC and EER of the raw score on the six KNAD captures | Pre-specified sensitivity analysis | `knad_y_recuentos` | `data/jobs/*/metrics.json` of the 36 runs it averages. |
| Table S70 (`tab:s_trained_main`): the counts of the left block of Table S47 restricted to the main experiment | Exploratory, added after the analysis plan was fixed | `knad_y_recuentos` | `data/reference/training_audit.json`, the output of `bin/audit_training.py`. |
| Table S64 (`tab:s_retrain_streams`), and the fused detector at one epoch in Table S63 (`tab:s_retrain_scores`) | Exploratory, added after the analysis plan was fixed | `contencion` | `data/jobs/T8-*/metrics.json`. The other five scores of Table S63 are read from the same files; the tests do not recompute them. |
| Table S59 (`tab:s_h1_references`), block (a) and block (c) | (a) is the confirmatory test, repeated for comparison; (c) is exploratory, added after the analysis plan was fixed | `h1_referencias` | (a) `data/stats.json`; (c) `data/jobs/T8-EPS__*/metrics.json` and `data/aggregate/cells.csv` (the estimates). |

The post hoc experiment with modified encoders is recomputed in the same
way by `tests/added_analyses/test_modified_encoders.py`, from
`data/modified_encoders/` and `data/aggregate/cells.csv`:

| Output of the paper | Status | Result file in `data/modified_encoders/` | Recomputed in the tests from |
|---|---|---|---|
| Supplement, Table S72 (`tab:s_mod_counts`): autoencoders below 0.90 of the validation ratio, by stream, family and seed, with its summary rows, and the entry rule with seed 1234 | Exploratory, post hoc | `supplement_tables.json`, `entry_rule.json` | `jobs/*/convergence_report.json` of the 108 runs. |
| Supplement, Table S71 (`tab:s_mod_auc`), ROC-AUC columns, and the two comparisons with the KitNET-style control at the end of Section 6.1 of the manuscript (Hodges-Lehmann estimate, exact 96.1 % interval, exact signed-rank p, Holm over the two) | Exploratory, post hoc | `supplement_tables.json`, `comparisons.json` | `jobs/*/detection_metrics.json` of the 72 runs of seeds 1234 and 5678, and `data/aggregate/cells.csv` for the control and the unmodified encoders: the ROC-AUC and blocked columns, the AUC_t column, the per-seed values, the within-block estimates and intervals and the dependence rule. The comparison with the control with its input clipped is read from `comparisons.json`; the tests do not recompute it. |

## What is not regenerated here

| Output of the paper | Why not | What is shipped instead |
|---|---|---|
| The final results themselves: per-run `metrics.json`, `aggregate.json`, `runs.csv`, `stats.json` | They are computed from the per-record scores and labels (10.7 GB and 1.3 GB over 475 runs) by `frozen/runner/metrics_rev.py`, `aggregate.py` and `stats.py`, which also check that the analysis plan is the one fixed on 16 September 2026 (supplement, Section 28). The analysis is described in Section 5.4 of the manuscript. | The files themselves (`data/`). |
| Training and execution of the models | GPU, the four datasets, and 162 GB of results (110 GB of checkpoints, 40 GB of per-autoencoder errors, 12 GB of scores and labels). | The code that was run (`frozen/`), and its environments. |
| Table 1 (construction of the nine streams) and Table 8 (attack records in the calibration window) of the manuscript | They come from the label audit of the streams, which needs the datasets. | Cut indices, row counts and file hashes of every stream in `data/streams.json`; the audit script in `frozen/streams/`. |
| Table 2 of the manuscript (behaviour of the feature map); supplement, Tables S5 (`tab:s_groups`) and S3 (`tab:s_weights`) | No generator in this package. | Their inputs: `fm_stats.json` and `config.json` of every run in `data/jobs/`. |
| Figure 2 of the manuscript; supplement, Figures S1 and S2 (`fig:s_detectors`, `fig:s_dxt`) | They draw per-record arrays of single runs. | The drawing code (`kitnetzday.figures.illustrative`), without an entry point. |
| Supplement, Tables S17, S29, S30, S31 and S33 (`tab:s_auct`, `tab:s_window`, `tab:s_dep`, `tab:s_dep2`, `tab:s_nofilter`) | No generator in this package. | Their values are in `data/stats.json`, `data/aggregate/aggregate.json` and `data/jobs/*/metrics.json`. |
| Supplement, Tables S9, S14, S15, S18 to S21, S23 to S28, S34, S38 and S43 to S50 (`tab:s_eer`, `tab:s_operator_stream`, `tab:s_operator`, `tab:s_fullprelude`, `tab:s_holm_attain`, `tab:s_block_rule`, `tab:s_cleancal`, `tab:s_cic18files`, `tab:s_r14expl`, `tab:s_mlpord`, `tab:s_eps`, `tab:s_scoring`, `tab:s_clip`, `tab:s_sens_calib`, `tab:s_calib_attack`, `tab:s_t4pb`, `tab:s_t4e_t4pb`, `tab:s_robust`, `tab:s_hscore`, `tab:s_finalloss` (right block, final weights), `tab:s_perm`, `tab:s_deviations`, `tab:s_deviations_report`) | No generator in this package. Several of them need the per-record arrays, the checkpoints or the datasets. Table S18 is the reference KitNET trained on the whole attack-free prelude, whose runs kept per-record scores only, and Table S25 comes from 18 runs made after the results were closed, which are not in `data/jobs/`. | For those computed from per-run metrics, the inputs: `data/jobs/*/metrics.json`, `data/runs.csv`, `data/aggregate/cells.csv`, `data/stats.json`. The ROC-AUC of every run of Table S25 is in `data/added_analyses/h1_referencias.json`. |
| Added analysis, exploratory: supplement, Tables S51 to S54 (`tab:s_postprelude_region`, `tab:s_postprelude_fused`, `tab:s_postprelude_raw`, `tab:s_postprelude_h`: ROC-AUC on the records from the first attack record of the evaluated region onwards) | Computed from the per-record scores and labels of the 198 runs of the main experiment. | `data/added_analyses/region_post_preludio.json`, with the value of every run. |
| Added analysis, exploratory: supplement, Tables S56 to S58 (`tab:s_rawop`, `tab:s_stages`, `tab:s_blocklen`: operating point of the raw score, blocked ROC-AUC at each stage of the decision chain, H1 on the raw score for four block lengths) | Computed from the per-record scores and labels of the same 198 runs. | `data/added_analyses/score_bruto.json`, with the value of every run. |
| Added analysis, exploratory: supplement, Table S59 (`tab:s_h1_references`), blocks (b), (d) and their combination, and Table S62 (`tab:s_h4_mlp_versions`) | They execute the control again from its stored weights, or rebuild the score of the per-record MLP from its per-autoencoder errors; both need the preprocessed streams or the per-record arrays. | `data/added_analyses/h1_referencias.json`, with the ROC-AUC of the control per dataset under each reference. |
| Added analysis, exploratory: supplement, Tables S65 and S66 (`tab:s_retrain_weights`, `tab:s_retrain_noise`) | The first reads the stored weights of the windowed members, the second the per-run training records. | `data/added_analyses/contencion.json`, with one row per autoencoder. |
| Added analysis, exploratory: supplement, Tables S60 and S61 (`tab:s_harmonised_auc`, `tab:s_harmonised_h1`: the KitNET-style control trained again with the treatment of the five TensorFlow models) | It trains the control again, on CPU, from the preprocessed streams: eight combinations of three factors on each of the nine streams. Its per-record outputs take 9.9 GB. | `data/added_analyses/control_armonizado.json`: the summary per stream and combination, with the sha256 of the metric file of each, and the H1 comparisons. The model classes it trains are those of `frozen/`. |
| The two descriptive checks of `data/added_analyses/knad_y_recuentos.json` (dates of the analysis plan and of the first metric file; the ratio of Table S47 as one minus a coefficient of determination on one run) | They read the history of the authors' repository, and one stored model with its stream. | The values, in that file and in `knad_y_recuentos_recon.json`. |
| The post hoc experiment with modified encoders: its training, its detection metrics and the control with its input clipped | GPU, the datasets, and the per-record outputs of its 108 runs (2.5 GB of scores and 0.3 GB of labels, 9.9 GB of per-autoencoder errors and 31.0 GB of checkpoints). | The per-run files and the results in `data/modified_encoders/`; the code in `frozen/modified_encoders/`. |
| Batch-1 latencies of Section 6.6 of the manuscript | They are timings of this hardware (`frozen/runner/bench_b.py`). | - |
| Figure 1 and Table 3 of the manuscript; supplement, Tables S1, S2, S4 and S22 (`tab:decision_params`, `tab:hyperparams`, `tab:sota_kitsune`, `tab:s_corpora`) | Diagrams, descriptive tables, and a table quoted from Mirsky et al. | - |
| Any other table, figure or number of the manuscript, the supplement or the response letter | If it is not in the table of the previous section, this package does not regenerate it. | For what is computed from per-run metrics, the inputs: `data/jobs/*/metrics.json`, `data/runs.csv`, `data/stats.json`. |

The modules `kitnetzday.audit.containment`, `containment_report`,
`convergence` and `floor_control` are library code without an entry point.
The first three read the `convergence_report.json` that every run of the
retraining experiment writes (89 files, 1.2 MB) and the fourth the
per-record errors of every autoencoder; neither input is shipped. They are
behind the validation-error ratios of the retraining experiment (Section
6.2 of the manuscript; Sections 12 and 32 of the supplement) and the audit
of the KitNET-style control (Section 18 of the supplement).

## Layout

```
.
├── README.md  VAULT.md  NOTICE  CITATION.cff  SHA256SUMS
├── pyproject.toml  uv.lock  requirements-analysis.txt  .python-version
├── bin/                  entry points; each loads src/config.json and a logger
├── src/
│   ├── config.json       every tunable value: paths, plan constants, style
│   └── kitnetzday/
│       ├── naming.py     dataset and configuration identifiers
│       ├── io/           readers of the results, writers of the outputs
│       ├── figures/      heat map, mean ranks, Figure A.1, cases with limited detection
│       ├── tables/       main results, ablations, cost, comparators, supplement
│       ├── audit/        reconstruction-error diagnostic, retrained pair, correlations
│       └── utils/        logger factory, Loggable base class, formatting
├── added_analyses/       index of the analyses added after the plan was fixed
├── tests/                mirrors src/, plus tests/bin, tests/stats and tests/added_analyses
├── data/                 the final results, the per-run files, the reference outputs,
│                         and modified_encoders/ (the post hoc experiment)
└── frozen/               the code that produced the results, without its comments,
                          and modified_encoders/ (the code of the post hoc experiment)
```

`data/README.md` describes every data file; `frozen/README.md` the code of
that folder, the tags it was checked against, its environments and how to
lay it out to run it.

## Environments

| Code | Python | Pinned in |
|---|---|---|
| Analysis (`src/`, `bin/`, `tests/`) | 3.12 | `uv.lock`, `requirements-analysis.txt` (numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.2) |
| Training and execution (`frozen/models`, `frozen/runner/worker.py`, `run_job.py`) | 3.10.21 | `frozen/requirements-tf215.txt` and `.lock.txt` (TensorFlow 2.15.1, CUDA 12.2, cuDNN 8.9) |
| Metrics, aggregate and statistics (`frozen/runner/metrics_rev.py`, `aggregate.py`, `stats.py`) | 3.12.14 | `frozen/requirements-postproc.txt` and `.lock.txt` (numpy 1.26.4, scipy 1.15.1, scikit-learn 1.6.1) |
| Modified encoders (`frozen/modified_encoders/`): the harness that trains them (`harness_seed_*.py`), their models (`modelos_rep.py`) and the selection runs (`prevuelo.py`) | 3.10.21 | as training and execution above |
| Modified encoders: the driver of the selection runs (`escalera.py`, which starts `prevuelo.py` with the interpreter of the row above), the entry rule (`puerta_fase_b.py`), the comparisons (`e1_metricas_t10.py`, with `comun_fase_b.py`, `mlp_sin_barajar_metricas.py` and `puerta_fase_b.py`) and the tables of the supplement (`supl_fase_b.py`) | 3.12 | as metrics above for the comparisons; the driver, the entry rule and the tables use the standard library only |
| Original post-processing (`frozen/postprocess/`), kept for the record | 3.12 | not pinned: it needs PyTorch on CPU; `frozen/README.md` gives the versions it ran with |

## Reading the labels

The outputs carry the labels of the analysis plan and identifiers of the
study, some in Spanish.

| Label | Meaning |
|---|---|
| `[CONFIRMATORIO plan §x]` | Confirmatory: fixed in the analysis plan (section x) before any metric of the revised regime was computed. |
| `[SENSIBILIDAD plan §x]` | Sensitivity analysis fixed in the plan. |
| `[SECUNDARIA plan §x]`, `[DESCRIPTIVO plan §x]` | A secondary metric and a descriptive quantity of the plan: reported, with no confirmatory claim. |
| `[EXPLORATORIO]` | Exploratory: not in the plan. The files of `data/added_analyses/` carry it unless they are one of the sensitivity analyses of the plan: they were added after the analysis plan was fixed. |
| `[ORACULO: ...]` | Oracle: a choice was made with evaluation labels; never an estimate of what a deployed detector attains. |
| `[IN-SAMPLE]` | Calibration on the tail of the training segment: a sensitivity analysis of the plan, not deployable. |
| `[PARTIAL INPUTS]` | A script writes it on an output when its inputs are not the final aggregate; with the data shipped here no output carries it. |
| `[ENMIENDA-1]`, `[POST HOC]` | The post hoc experiment with modified encoders, whose changes and entry rule were fixed in an amendment to the analysis plan after the results of the main experiment were known (supplement, Section 35); `[POST HOC]` marks the runs of seed 9012, added later still. |
| `job_id` | The identifier of one run in the data files (`data/jobs/<job_id>/`, `data/runs.csv`, `data/modified_encoders/jobs/<job_id>/`). |
| `DONE`, `NOT_RUN`, `NOT_COMPLETED`, `NOT_EXECUTABLE` | State of a planned job: run to the end; not run; started and not completed; not executable on this platform. `data/README.md` gives the counts. |
| `without_value` | In `aggregate.json` and `stats.json`, the planned cells that have no value because their job is not `DONE`. |
| `T1`...`T8`, `T10`, `X2` | Blocks of runs: T1 main experiment (seeds A and B; the third seed, C, was not run, a deviation declared in Section 28 of the supplement), T2 window length, T3 feature grouping, T4 training-segment contamination, T5 comparators, T7 exploratory windowed MLP, T8 retraining experiment, its five-epoch ablation and the control with one stabiliser, X2 regression runs of the submitted pipeline, T10 the post hoc experiment with modified encoders (`data/modified_encoders/`; its runs of seed C, 9012, are the only runs with that seed). `T9` and `T12` name runs that appear only in `data/added_analyses/`. |
| `D1`...`D9`, `C1`...`C11` | Streams and configurations, as `data/aggregate/cells.csv` names them. |
| `C12`...`C22` | Configurations outside the eleven of the main experiment, as `data/runs.csv` names them: C12 and C13 the windowed MLP in its two modes, C14 and C15 a deeper windowed MLP, C16 to C19 the models of the retraining experiment (the windowed model in its two modes and their per-record counterparts), C20 the KitNET-style control with the same stabiliser in both phases, C21 and C22 the two models of the retraining experiment with five epochs. |
| `lost`, `celda perdida (§12.2)` | The plan's name for a planned cell whose job is `NOT_EXECUTABLE` or `NOT_COMPLETED`: seven ablation cells (five and two) and no cell of the main experiment. |
| `R1.n`, `R2.n` | The analysis that answers comment n of Reviewer 1 or of Reviewer 2, as the analysis plan names it. |
| `S` (in `frozen/runner/manifest.csv`) | Timing probes of the sizing block; they computed no metric and are not in `data/jobs/`. |
| `AG-n`, `ST-xn`, `MR-n`, `R-n`, `FIG-n` | Rules of the aggregation script, of the statistics script, of the metrics script and of the analysis plan, and the analysis choices listed by `bin/make_figures.py`, as the data files and the generated files cite them. |

The docstrings and messages of `frozen/`, some notes inside the generated
files and many keys and notes of the files in `data/added_analyses/` are in
Spanish. `added_analyses/README.md` has a glossary of the keys of those
files, and `data/README.md` one of the keys of `data/modified_encoders/`.

## Three checks that apply known techniques

The article and its supplement read their results against three checks.
None of them is a new instrument: each applies a technique that is already
in the literature. This section says what each one is and where it is
implemented.

**The ratio to a constant predictor.** The recorded loss of every
autoencoder is divided by the squared error of predicting the mean of each
feature of its own group; no label is used. The ratio is one minus a
coefficient of determination: with the final weights, one minus the R² of
the reconstruction pooled over the features of the group, and with the mean
loss of the epoch, which is what Section 6.1 of the manuscript and the left
block of Table S47 of the supplement use, the same quantity for the
sequence of models of that epoch. It is implemented in
`kitnetzday.audit.TrainingAudit` and run by `bin/audit_training.py`; the
numerical check of the identity on one run is in
`data/added_analyses/knad_y_recuentos.json`.

**The position reference, AUC_t.** The ROC-AUC obtained by ranking the
records of the evaluation segment by their position in the stream. It is a
trivial baseline in the sense of Arp et al. and of Wu and Keogh, here the
one that a score attains by drifting with time alone. It is stored next to
every ROC-AUC (`primary.auc_t` of each `metrics.json`, `auc_t` in
`cells.csv`); Section 5.2 of the manuscript says how it is read.

**A function-preserving initialisation.** In the retraining experiment
(Section 6.2 of the manuscript) the windowed model starts as the per-record
model it is compared with: the weights on the last record of the window are
copied from it and the weights on the context start at zero, so both
compute the same function before the windowed phase of training. This is a
function-preserving initialisation in the sense of Net2Net (Chen et al.).
It is implemented in `frozen/runner/networks_ctx.py`; the paired evaluation
is `bin/analyse_retrained_pair.py`.

The evaluation rules that these checks accompany (a threshold that is a
percentile of unlabelled calibration data, causal detectors and filters, no
evaluation label in any operating choice) are implemented in
`frozen/runner/metrics_rev.py`.

## Citation and third-party code

`CITATION.cff` gives the reference of the article. `NOTICE` reproduces the
notices of Kitsune-py (MIT), from which two files of `frozen/models/` are
adapted. The datasets are not redistributed. This package does not include
a licence file; the notices in `NOTICE` apply to the files they name.
