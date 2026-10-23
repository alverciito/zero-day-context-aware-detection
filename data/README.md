# data/

Everything the analysis reads. Nothing here is a dataset record or a
per-record score.

## The aggregate

This is the final aggregate of the study, written at 12:06 on 7 October
2026, after the results were closed (7 October, 12:00; supplement,
Section 28). `stats.json` carries `run_type: FINAL`.

| File | What it is |
|---|---|
| `aggregate/aggregate.json` | Output of `frozen/runner/aggregate.py` over the 435 planned jobs of the manifest: 331 `DONE`, 97 `NOT_RUN` (the 90 of the third seed of the main experiment, a deviation declared in Section 28 of the supplement, and seven of the window-length ablation), 5 `NOT_EXECUTABLE` on this platform and 2 `NOT_COMPLETED`, as that section states. `complete: true` means that every `DONE` job had a valid `metrics.json` with no section in error. |
| `aggregate/cells.csv` | One row per stream and configuration (99 cells): the value of each of the two seeds, their mean and the trivial references of the stream. |
| `runs.csv` | One row per job and series (16,840 rows over 556 jobs): the 435 jobs of the manifest plus the 121 of the exploratory windowed MLP and the retraining experiment (blocks T7 and T8), with their state, the metrics of every detector and sensitivity analysis, and the label of each row under the plan (`tag`). Over the 556 jobs: 447 `DONE`, 97 `NOT_RUN`, 9 `NOT_EXECUTABLE` and 3 `NOT_COMPLETED`. |
| `stats.json` | Output of `frozen/runner/stats.py`: the confirmatory tests of the plan (H1-H4), the Hodges-Lehmann estimates with their exact intervals, the omnibus, the robustness rules, the ablations and the sensitivity analyses. |
| `tests.csv` | The confirmatory and descriptive tests of `stats.json`, one per row. |
| `env.json` | The environment `stats.py` ran in. |

A job is `DONE` when it ran to the end, `NOT_RUN` when it was planned and
not run, `NOT_COMPLETED` when it was started and did not complete, and
`NOT_EXECUTABLE` when it cannot be executed on this platform (Section 6.4
of the manuscript gives the causes for the window-length cells). The key
`without_value`, in `aggregate.json` and `stats.json`, lists the planned
cells that have no value for one of the last three reasons. `README.md`, at
the root, explains the labels of the column `tag`.

## The per-job files

`jobs/<job_id>/` holds the small files of 475 finished jobs: the 447 that
`runs.csv` lists as `DONE`, the 21 comparator runs (blocks T5: Isolation
Forest, reference KitNET, reference KitNET on the full prelude) and the 7
regression runs of the submitted pipeline (X2).

| File | What it is |
|---|---|
| `metrics.json` | Every metric of the job, written by `frozen/runner/metrics_rev.py` from its scores. 461 jobs have one; the 14 without are the six runs of the reference KitNET on the full prelude, the seven regression runs and one control run of the five-epoch ablation, on Video Injection, whose windowed counterpart did not complete. |
| `config.json` | The job: its manifest row, the hyperparameters in effect, the row regions of the stream (feature map, training, calibration candidates, evaluation), the sha256 of the source files and of the preprocessed arrays, and the hashes of the code that ran it. |
| `train_log.csv` | One row per autoencoder: its size, its training steps and the mean loss of its epoch. The comparator runs have none. |
| `fm_stats.json` | The feature map: the groups of features and the trajectory of the clustering. The comparator runs have none. |
| `timing.json` | Wall clock, phases, throughput and memory peaks. Twelve of them are reference timings (`timing_ref: 1`). |
| `env.json` | Interpreter, library versions, GPU and driver of the run. |
| `arrays.json` | Shape, dtype and sha256 of the per-record arrays of the job (`scores.npy`, `labels.npy`, `ae_errors.npy`), which are not shipped. |
| `DONE` | The completion stamp. |

No job that finished after the results were closed on 7 October 2026 is
among them.

## What was edited for this package

The files of the two sections above are what the scripts of
`frozen/runner/` wrote, with these edits: the free-text working notes were
removed or reworded; paths are given from the root of the authors'
repository; the job states and a few keys carry the names used above; and
two columns of `cells.csv` that held no value, those of the third seed,
were dropped. No number was changed: every numeric value of the edited
files was compared with the file as written.

`freeze.is_plan_default: true` in `aggregate.json` means that the closing
time is the constant of the runner (`common.FREEZE`, 7 October 2026,
12:00). The analysis plan had fixed 30 September, the date that note AG-11
of the same file still quotes; the move is the deviation declared in the
supplementary material, Section 28 (Table S49).

The files record the sha256 of one another (`metrics_sha256` in
`runs.csv`, the `provenance` blocks of `aggregate.json`, `stats.json` and
each `metrics.json`). Those values are the hashes of the files as the
scripts wrote them, before the edits above: they identify the originals in
the authors' repository, and `sha256sum` on a shipped file gives another
value. Each of the 446 `metrics.json` that `runs.csv` covers was checked
against the sha256 recorded there before it was edited. The hashes of the
shipped files are in `SHA256SUMS`, at the root of the package. The hashes
of code recorded in the same blocks are those of the files as executed,
which `frozen/MANIFEST.sha256` lists.

## Derived inputs

| File | What it is |
|---|---|
| `floor_vectors.json` | For every stream, the variance of each column of the min-max scaled training stretch of the autoencoders, and the same vector for the four gradient-trained jobs of the training-segment contamination runs. The constant-predictor floor of a feature group is the mean of its columns. Computed by the authors from the preprocessed streams with `kitnetzday.audit.ConstantFloor`; `bin/audit_training.py` uses it when the streams are not on disk and says so in its result. |
| `streams.json` | For every stream: the files it was built from, with their size and sha256; the shape and sha256 of the preprocessed arrays; the row regions (feature map, autoencoder training, execution) and the calibration and evaluation rows of each calibration candidate. The primary candidate is `calib_first_10pct_post_train`. The two keys ending in `__orig__v1` are the uncorrected streams of the regression runs (X2). |
| `fig7_valores_publicados.csv` | The values published in the figure of the original submission, one row per bar and metric, as archived at the start of the revision. Its column names are in Spanish (`variante`, `paradigma`, `metrica`, `valor`, `regla`, `fuente`). |
| `fig7_trivial_f1.csv` | The F1 of an always-alarm predictor on the smoothed labels of that figure, per stream. |

## Added analyses

`added_analyses/` holds the result files of the analyses that were added
after the analysis plan was fixed, one per analysis;
`../added_analyses/README.md` says which tables of the supplement come from
each, their status under the plan and what each analysis reads, and has a
glossary of their keys. Many keys and notes are in Spanish, and paths are
given from the root of the authors' repository.

| File | What it is |
|---|---|
| `siete_flujos.json` | The twelve confirmatory comparisons on the seven streams whose calibration window holds no attack records, and the in-sample calibration by configuration on the other two. |
| `region_post_preludio.json` | ROC-AUC of every run, cell and comparison on the records from the first attack record of the evaluated region onwards. |
| `score_bruto.json` | The raw score: per cell, at its own threshold, at each stage of the decision chain and for four block lengths. |
| `h1_referencias.json` | H1 under each reference that the stored results allow, and H4 under four versions of the per-record MLP. |
| `contencion.json` | The retraining experiment: six stored scores, the five-epoch ablation stream by stream, the weights of the windowed encoders and two trainings of the same per-record model. |
| `knad_y_recuentos.json`, `knad_y_recuentos_recon.json` | The raw score on the six KNAD captures, the counts of the left block of Table S47 of the supplement restricted to the main experiment (Table S70), and two descriptive checks. |
| `control_armonizado.json` | The KitNET-style control trained again with the treatment of the five TensorFlow models: the summary per stream and combination of factors, and H1 against each. Its per-record outputs (9.9 GB) are not shipped. |

## The post hoc experiment with modified encoders

`modified_encoders/` holds the experiment reported at the end of Section 6.1
of the manuscript and in Section 35 of the supplement: Conv1D and Conv2D
encoders modified with the changes that the supplement defines, each run in
both modes on the nine streams with seeds 1234 (A), 5678 (B) and 9012 (C),
108 runs in all. It is exploratory and post hoc. Its changes and the rule
that admits a modified model to the comparison were fixed in an amendment
to the analysis plan after the results of the main experiment were known
and before the runs that chose the changes. Earlier runs, which read no
attack label and are not in this package, had trained the encoders with
some of the changes, and Section 35 of the supplement describes them.
Seed C was added after the count of modified Conv1D with seed B was known. None of its runs is in
`jobs/`, `runs.csv`, the aggregate or `stats.json`, and nothing in those
files depends on it. Its code is in `../frozen/modified_encoders/`.

| File | What it is |
|---|---|
| `jobs/<job_id>/` | The small files of each of the 108 runs: `DONE`, `arrays.json`, `config.json` (with the block `t10_rep`: the changes applied, the harness, and the check that the run is the job of the main experiment with the same stream, family, mode and seed except for the changes; for seed C, which the main experiment did not run, that job is the one of seed A), `env.json`, `fm_stats.json`, `timing.json`, `train_log.csv`, and two files of their own, described below. Change P trains each layer through one Adam optimiser per distinct learning rate, so the optimiser that the model object carries takes no step: `optimizer_iterations` in `train_log.csv` is 0 for these runs, and the steps and rates actually used are in `grupos_adam` of `convergence_report.json` and in `t10_rep.fits` of `config.json`. |
| `jobs/<job_id>/convergence_report.json` | The held-out check of every autoencoder of the run, written by the harness when the training ends. |
| `jobs/<job_id>/detection_metrics.json` | The metrics of the run, as the comparison script computed them from its scores: only the 72 runs of seeds A and B. The runs of seed C enter no comparison and their detection was not computed. The file is the entry of the run in `per_job` of `comparisons.json`, copied to the folder of the run; no script wrote it as a file of its own. |
| `comparisons.json` | The comparison file: the ROC-AUC of every run and stream, the two comparisons with the KitNET-style control (`e1.comparisons`, with the three robustness checks under `robustness`), the descriptive comparisons with the unmodified encoders, without the three selection streams and with the per-record MLP (`descriptivos`), and the check that the same script reproduces the metrics and the H1 statistics of the main experiment (`control_positivo`). Its first key, `status`, states what the experiment is. The blocks of the original file that described the work queue, and the wording templates kept for the authors, were left out. Every other block is as computed. |
| `entry_rule.json` | The rule that admits a modified model to the comparison, applied with seed A, and its outcome for each family. Two notes for the authors (`aviso`, `nota_contexto`) were left out. |
| `selection_runs.json`, `selection_summary.json` | The record of the selection runs that chose the changes on three streams, one record per event, and its summary. `tipo` is the kind of event: `identidad`, the check that each modified constructor with its changes switched off equals the original class; `inicio`, a run that starts; `celda`, the held-out ratio of one autoencoder; `proceso`, the timing of a process; and `decision`, a decision of the selection (the cases used, a stage of a step, the unmodified controls, the outcome for each family). `fase` is the stage: `criba`, the first 20,000 records of the autoencoder training segment, and `confirma`, the whole of it. `escalon` is the step of the list of changes. The record was written as a JSON-lines file, one event per line, which is how the code reads it, and it is shipped as a JSON list of the same events. |
| `selection_smoke_test.json` | The check that ran before the selection runs: 2,000 records of the last step of each list and of each unmodified encoder, on the largest autoencoder, run on the GPU only to see that the code runs. Its ratios decided nothing. |
| `supplement_tables.json` | The values of Tables S71 and S72 of the supplement and the comparisons that its Section 35 quotes, and the counts of reliance on the window with seeds A and B together (`contexto_ab`) and in the autoregressive mode (`contexto_ar`), with the sha256 of the comparison file as the comparison script wrote it, before the blocks above were left out. |

A job identifier reads `T10-REP__<stream>__<family>_rep__<mode>__corrected.v1__L800__k4__s<seed>__r1`,
as the harness names it: `arF` is the full-sequence mode and `arT` the
autoregressive one. The keys of `convergence_report.json` are in Spanish:

| Key | Meaning |
|---|---|
| `factores` | The changes applied: `P,I,R` for Conv1D and `P,I,R,A` for Conv2D (supplement, Section 35). |
| `filas_train`, `filas_val` | The number of training records used for fitting (the first 90 %) and of held-out ones (the last 10 %). |
| `grupos_adam`, `lr_por_capa` | The learning rate of each trainable variable (change P). |
| `suelo` | The squared error of predicting the mean of the held-out records. |
| `val_mse`, `razon` | The squared error of the autoencoder on the held-out records, and the validation ratio `val_mse / suelo`. |
| `evaluable` | `suelo` is at least 1e-4. Seven of the 36 full-sequence autoencoders of each family and seed are not. |
| `entrenado` | The ratio is below 0.90, without the evaluable condition. The counts of the supplement, like `summary`, require both. |
| `val_mse_contexto_cambiado`, `cociente_contexto_cambiado`, `depende_contexto` | The held-out error when the 799 earlier records of each window are taken from a distant window, its ratio to `val_mse`, and whether that ratio is at least 1.10. `depende_contexto` is set for every autoencoder; the counts of the supplement, like `summary`, take only evaluable autoencoders below 0.90. |
| `mse_persistencia`, `razon_persistencia` | Autoregressive mode only: the error of repeating the previous record, and `val_mse` divided by it. Descriptive. |
| `summary` | `n_evaluables`, `n_entrenados` (evaluable and below 0.90) and `n_dependen_contexto` (also with a ratio of at least 1.10) of the run. |
| `loss_epoca`, `std_salida`, `muertas` | The mean training loss of the epoch, the spread of the output and the fraction of bottleneck units that do not vary. |

Other Spanish words of these files: `familia` is the encoder, `semilla` the
seed, `factores` the changes, `prevuelo` and `fase_a` the selection runs,
`escalon` a step of the list of changes, `criba` and `confirma` its two
stages, and `arnes` the harness. `T10-REP` and `<family>_rep` are the names
that the harness gives to these runs and to the modified variants.

In `detection_metrics.json`, `metricas` holds the metrics with the names of
the main experiment (`auroc_FUS` is the ROC-AUC of the fused detector), and
`reference_job` the run of the main experiment it is paired with. `ar` is
`TSR` for the full-sequence mode (`arF`) and `AR` for the autoregressive one
(`arT`).

The keys of `supplement_tables.json`:

| Key | Meaning |
|---|---|
| `convergencia["<family>|<seed>"]` | One column of Table S72: `tsr_por_stream` (per stream, [below 0.90, evaluable] in the full-sequence mode), `tsr_bajo` and `tsr_evaluables` (All streams), `streams_con_alguno`, `mediana_razon_tsr` (Median ratio), `contexto` (Window replaced, out of `tsr_bajo`), `ar_bajo` and `ar_evaluables` (Autoregressive), and `contexto_ar` (window replaced in the autoregressive mode). |
| `auc["<family>"]["D1".."D9"]` | One row of Table S71: `control` (dA), `sin_modificar`, `modificado`, `A` and `B` (the modified encoder per seed), `bloque_control` and `bloque_modificado` (within blocks). |
| `comparaciones["<family>"]` | The figures of Section 35: `control` (the comparison with the control), `bloque_hl` and `bloque_ci` ([estimate, lower, upper] within blocks), `control_recortado_hl`, `colapsado` and `sin_knad` ([estimate, units higher, units]), `c_vs_sin_modificar`, `e_fuera_del_prevuelo`, `a_vs_mlp_mse`, `a_vs_mlp_rmse`, `solo_tsr`, `ar_menos_tsr`, `margen_0_01_control` ([higher, lower, within the margin]), `max_dif_semillas` ([value, stream]) and `auc_t` (the AUC_t column). |
| `contexto_ab`, `medianas_tabla_auc` | The window check with seeds A and B together ([relying, below 0.90]), and the last row of Table S71 in column order. | The
files of this folder were edited as those of the sections above: free-text
working notes removed or reworded and paths given from the root of the
authors' repository, with one addition. Where the original files named a script or a result file of this experiment, or the git tag of
one of its three harnesses, which this package ships under another name, they now name the
shipped file (for instance
`frozen/modified_encoders/harness_seed_a.py`); a tag that also froze the
entry rule is named by `frozen/modified_encoders/puerta_fase_b.py` where it
records that script. The code itself still reads and writes these files
under their names in the repository, which `frozen/MANIFEST.sha256` gives.
Every sha256 that these files record of another file is that of the file as
written, before the edits described here, so `sha256sum` on a shipped file
gives another value; the hashes of the shipped files are in `SHA256SUMS`, at
the root of the package. The key `t10_rep.decision` of `config.json` named
an entry of the authors' record of decisions: it was removed for seeds A and
B, and for seed C it holds the reworded label that `harness_seed_c.py`
writes. The check of each run in `frozen/modified_encoders/e1_metricas_t10.py`
(`validar_job_t10`) compares that key and the harness tag with the values
the harness wrote, so it would stop on these files as shipped; it ran on the
files as written. No number was changed.

## Outputs

| Directory | What it is |
|---|---|
| `reference/` | The outputs of `bin/run_all.py` as the authors generated them, for comparison. |
| `out/` | Where the scripts write. It is created on the first run. |

## What is not here, and how large it is

| Not shipped | Size | How to obtain it |
|---|---|---|
| The four datasets | - | From their publishers; `streams.json` lists the files and their sha256. |
| Preprocessed streams (`x.npy`, `y.npy` per stream) | 26.2 GB | `frozen/runner/pipelines.py` builds them from the datasets. |
| Per-record scores and labels of the 475 jobs | 10.7 GB and 1.3 GB | Written by `frozen/runner/run_job.py`; their sha256 is in `jobs/*/arrays.json`. |
| Per-record errors of every autoencoder (detector inputs) | 40.3 GB | Same. |
| Checkpoints | 110.1 GB | Same. |
| Per-record scores, labels, per-autoencoder errors and checkpoints of the 108 runs of `modified_encoders/` | 2.5, 0.3, 9.9 and 31.0 GB | Written by the harness of `frozen/modified_encoders/`. |
