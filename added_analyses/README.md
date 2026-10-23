# added_analyses/

The analyses that were added to the article after the analysis plan was
fixed, and three pre-specified sensitivity analyses that are tabulated in
full. This folder is their index: the result file of each analysis is in
`data/added_analyses/`, and the tables of the supplement listed below are
typeset from those files.

All of them were computed on CPU over results that were already stored.
None changes the confirmatory analysis: the primary metric is the ROC-AUC
of the fused detector, and H1 to H4 are those of `data/stats.json`.

The scripts that computed these analyses are in the authors' repository
and are not part of this package. It is the arrangement this package already has for
the other exploratory analyses of the supplement (`README.md`, "What is not
regenerated here").

The post hoc experiment with modified encoders (supplement, Section 35,
Tables S71 and S72) is not indexed here. It trained new models, so its
code, its result files and its tests are described in `README.md` and in
`data/README.md`, section "The post hoc experiment with modified encoders".

## From table to result file

Sections, table numbers and labels are those of the supplementary material
(Sections 29 to 34). The two tables cited from outside those sections are
Table 4 of the manuscript (AUC and EER on the six KNAD streams) and Table
S47 of the supplement (`tab:s_finalloss`, Section 27), whose left block
holds the counts by the ratio to a constant predictor.

| Table of the supplement (LaTeX label) | Section | Result file in `data/added_analyses/` | Status |
|---|---|---|---|
| Tables S51 to S54 (`tab:s_postprelude_region`, `tab:s_postprelude_fused`, `tab:s_postprelude_raw`, `tab:s_postprelude_h`): ROC-AUC on the records from the first attack record of the evaluated region onwards | 29 | `region_post_preludio.json` | Exploratory, added after the analysis plan was fixed |
| Table S55 (`tab:s_rawcell`): ROC-AUC of the raw score per cell, over the whole evaluation segment and inside blocks of 10,000 records | 30 | `score_bruto.json` | Pre-specified sensitivity analysis |
| Tables S56 to S58 (`tab:s_rawop`, `tab:s_stages`, `tab:s_blocklen`): operating point of the raw score beside the fused detector, blocked ROC-AUC at each stage of the decision chain, and H1 on the raw score for four block lengths | 30 | `score_bruto.json` | Exploratory, added after the analysis plan was fixed (the F1 of the raw score at its own threshold is part of the pre-specified sensitivity analysis) |
| Tables S59 and S62 (`tab:s_h1_references`, `tab:s_h4_mlp_versions`): H1 under each reference that the stored results allow, and H4 with the two differences specific to the per-record MLP removed | 31 | `h1_referencias.json` | Exploratory, added after the analysis plan was fixed. The first block of each table is the confirmatory test, repeated for comparison |
| Tables S60 and S61 (`tab:s_harmonised_auc`, `tab:s_harmonised_h1`): the KitNET-style control trained again with the treatment of the five TensorFlow models (one stabiliser in both phases, input clipping, output layer fitted on errors computed with the final weights), and H1 against it | 31 | `control_armonizado.json` | Exploratory, added after the analysis plan was fixed |
| Tables S63 to S66 (`tab:s_retrain_scores`, `tab:s_retrain_streams`, `tab:s_retrain_weights`, `tab:s_retrain_noise`): the retraining experiment on six stored scores, stream by stream at one and five epochs, the distance of the windowed encoders from their initialisation, and two trainings of the same per-record model | 32 | `contencion.json` | Exploratory, added after the analysis plan was fixed |
| Table S67 (`tab:s_seven`): the twelve confirmatory comparisons on the seven streams whose calibration window holds no attack records | 33 | `siete_flujos.json` | Exploratory, added after the analysis plan was fixed |
| Table S68 (`tab:s_insample_two`): in-sample calibration against the main calibration on ACI-IoT-2023 and CSE-CIC-IDS2018, by configuration | 33 | `siete_flujos.json` | Pre-specified sensitivity analysis |
| Table S69 (`tab:s_knad_raw`): ROC-AUC and EER of the raw score on the six KNAD captures, beside the fused values of Table 4 of the manuscript | 34 | `knad_y_recuentos.json` | Pre-specified sensitivity analysis |
| Table S70 (`tab:s_trained_main`): the counts of the left block of Table S47 restricted to the main experiment | 34 | `knad_y_recuentos.json` | Exploratory, added after the analysis plan was fixed |

`knad_y_recuentos.json` also holds two descriptive checks that have no
table: the dates of the analysis plan and of the first metric file, read
from the clocks of the authors' machine, and a numerical check on one run
that the ratio of Table S47, taken with the final weights, is one minus the
coefficient of determination of the reconstruction;
`knad_y_recuentos_recon.json` holds the sums of squares of that check.
The chronology in `knad_y_recuentos.json` also records that the first run
that finished under the revised regime is a timing probe of the sizing block
(`S-w1`), started about a minute before the commit of the analysis plan. It
computed no metric: the first metric file was written on 19 September.

## What this package recomputes

`tests/added_analyses/` recomputes, from the shipped data and with the
routines of `src/`, the tables whose inputs are all in this package, and
compares the outcome with the result file.

| Result file | Its analysis reads | Recomputed in the tests |
|---|---|---|
| `siete_flujos.json` | `data/stats.json`, `data/runs.csv`, `data/aggregate/aggregate.json` | **Both tables.** `tab:s_seven` from the per-stream differences of `data/stats.json` (estimate, count of positive differences, exact 95.3 % interval) and `tab:s_insample_two` from `data/runs.csv`. |
| `knad_y_recuentos.json` | `metrics.json` of the KNAD runs; `train_log.csv`, `config.json` and `fm_stats.json` of the main experiment; the preprocessed streams; the stored model of one run; the history of the authors' repository | **The two tables.** `tab:s_knad_raw` from `data/jobs/*/metrics.json` and `tab:s_trained_main` from `data/reference/training_audit.json`. The two descriptive checks need the repository, one stored model and its stream, which are not shipped. |
| `score_bruto.json` | Per-record scores and labels of the 198 runs of the main experiment | **`tab:s_rawcell`**: its two panels are the columns `auroc_RAW_mean` and `auc_loc_RAW_mean` of `data/aggregate/cells.csv`. The other three tables need the per-record scores. |
| `contencion.json` | `metrics.json` and the training record of the runs of the retraining experiment; the stored weights of the windowed members | **The detection part.** `tab:s_retrain_streams` and the fused detector of `tab:s_retrain_scores` from `data/jobs/T8-*/metrics.json`. `tab:s_retrain_weights` needs the stored weights and `tab:s_retrain_noise` the per-run training records, which are not shipped. |
| `h1_referencias.json` | Stored weights, per-autoencoder errors and per-record scores of the control and of the per-record MLP; the preprocessed streams | **Two blocks.** The confirmatory block is `data/stats.json`, and the block with one stabiliser follows from `data/jobs/T8-EPS__*/metrics.json` and `cells.csv`. The other blocks execute the control again from its stored weights. |
| `region_post_preludio.json` | Per-record scores and labels of the 198 runs of the main experiment | **Its whole-region values**, which are those of `cells.csv`. The values on the restricted region need the per-record scores. |
| `control_armonizado.json` | The preprocessed streams, the feature map of each control run, and the model classes of `frozen/models/` and `frozen/runner/networks_eps.py` | **Its starting point**: the combination that is the control as it ran gives the control of `cells.csv`, stream by stream. The analysis trains the control again on CPU, eight combinations of the three factors on each of the nine streams; its per-record outputs take 9.9 GB and are not shipped. The file is the summary per stream and combination (ROC-AUC of the fused detector, of each detector and of the raw score, with the sha256 of the metric file of every combination), and the H1 comparisons under each. |

## Names and keys

The names of the result files, and many of their keys and notes, are
Spanish.

| Name | Meaning |
|---|---|
| `siete_flujos` | Seven streams: the nine streams without the two whose calibration window holds attack records. |
| `region_post_preludio` | The region after the prelude: the evaluated records from the first attack record onwards. |
| `score_bruto` | The raw score: the anomaly score of the ensemble before the detectors. |
| `h1_referencias` | The references of H1: the versions of the control that H1 can be tested against. |
| `contencion` | The retraining experiment, in which the windowed model starts as the per-record model. |
| `knad_y_recuentos` | KNAD and counts: the raw score on the six KNAD captures, and the counts of the left block of Table S47. |
| `control_armonizado` | The control with the same treatment as the five TensorFlow models. The file names its block of runs `T12`, after the blocks `T1` to `T8` listed in `README.md`. |

Keys and words that recur in the files:

| Key or word | Meaning |
|---|---|
| `estatus`, `status`, `tag` | Status under the plan, with the labels that `README.md` explains (`[CONFIRMATORIO plan §x]`, `[SENSIBILIDAD plan §x]`, `[EXPLORATORIO]`). |
| `entradas`, `inputs` | Inputs: the files read, with their sha256. |
| `guion`, `script`, `command` | The script that computed the file, and how it was run. |
| `creado`, `created`, `tiempo`, `pared_s`, `cpu_s` | Date of the computation; time, wall-clock seconds and CPU seconds. |
| `regresion`, `regression` | The comparison of recomputed values with values already printed in the article. |
| `cifra`, `que` | The value that is compared. |
| `impreso`, `donde_impresa`, `donde_esta_impreso` | The printed value, and where it is printed (`MS`: manuscript; `Supl.`, `suplemento`: supplement). |
| `calculado`, `recalculado`, `obtenido`, `diferencia` | The value computed here, and its difference from the printed one. |
| `coinciden_a_3_decimales`, `ok` | Whether the two agree to three decimals. |
| `nota`, `matiz`, `definicion`, `regla`, `eleccion` | Note, qualification, definition, rule, and how a case was chosen. |
| `flujo`, `por_flujo`, `por_captura` | Stream; by stream; by capture. |
| `semilla`, `semillas`, `por_semilla` | Seed, seeds; by seed. |
| `familia`, `por_familia`, `por_variante`, `por_paradigma` | Family of comparisons or of models; by family, by variant, by paradigm. |
| `clave`, `etiqueta`, `unidades` | Key of a comparison in `data/stats.json`, its label, and the streams it is computed over. |
| `reducciones` | Reductions of the set of streams: `n9` is the nine datasets, `n7_sin_ACI_ni_CIC2018` the seven streams of `tab:s_seven`, `n8_sin_...` the nine without the one named, `n6_solo_KNAD` the six KNAD captures (`sin`: without; `ni`: nor; `solo`: only). |
| `hl`, `k_pos`, `k_neg`, `k_zero` | Hodges-Lehmann estimate; streams with a positive, a negative and a zero difference. |
| `ic`, `ic_contiene_cero`, `ic_excluye_cero` | Exact interval of the estimate; whether it contains or excludes zero. |
| `alcanzabilidad`, `p_bilateral_minimo`, `umbral_paso_1`, `paso_1_alcanzable` | What the signed-rank test can reach with that many units: the smallest two-sided p value, the first Holm threshold of the family, and whether that threshold can be reached. |
| `holm_estatus`, `holm_rechaza`, `p_holm` | For the nine datasets, the Holm-adjusted p value of the confirmatory test and whether it rejects, as in `data/stats.json`. |
| `regla_de_bloques`, `mismo_signo_que_la_fusion` | The robustness rule of the plan on the blocked ROC-AUC of the raw score, and whether its sign is that of the fused detector. |
| `resumen`, `mediana`, `medias`, `signo`, `rango` | Summary, median, means, sign, range. |
| `principal`, `en_muestra` | The main calibration, and the in-sample calibration. |
| `mas_alto`, `mas_bajo`, `cambian_de_signo` | Higher, lower, and the comparisons whose sign changes. |
| `ejecucion`, `ejecuciones` | Run, runs. |
| `ejecuciones_no_incluidas`, `no_incluido` | The runs that a comparison does not include. The lists of `siete_flujos.json` are empty; `contencion.json` names one pair of the five-epoch ablation, on Video Injection, whose windowed member did not complete, so that eight of the nine streams enter that comparison. |
| `parte1` ... `parte4`, `calculo_1`, `calculo_2` | The parts of a file, in the order of its tables. |
| `recuentos`, `cociente`, `cortes`, `experimento_principal` | Counts, the ratio of Table S47, its cuts (0.9 and 1.15), and the main experiment. |
| `cierre_de_resultados`, `terminadas_despues_del_cierre_y_excluidas` | The closing date of the results, and the runs that finished after it and are not counted. |
| `cronologia`, `fecha`, `primer_metrics_json`, `primera_ejecucion_terminada` | Chronology, date, the first metric file and the first finished run. |
| `repositorio_sin_remoto`, `ramas_con_upstream` | Whether the authors' repository has no remote, and its branches that track one. |
| `par_por_score`, `distancia_a_la_inicializacion`, `pesos` | The pair of the retraining experiment on each score, the distance of the weights from their initialisation, and weights. |
| `ruido_entre_controles`, `test_de_signos` | Two trainings of the same per-record model, and a sign test on counts. |
| `tal como se ejecuto`, `as_run` | As run: the configuration of the main experiment. |
| `recortado`, `clipped`; `estabilizador unico`, `same_stabiliser`; `layer_refit` | The control with input clipping; with one stabiliser in both phases; with its output layer fitted after training. |
| `en orden`, `in_order`, `mlp_sin_barajar` | The per-record MLP fitted on its records in stream order. |
| `FUS`, `RAW`, `KMD_f`, `MVD_f` | Fused detector, raw score, and each detector after its filter. |

The streams are `D1` to `D9` and the configurations `C1` to `C11`, as in
`data/aggregate/cells.csv`.

In `control_armonizado.json` each of the eight combinations has a key such
as `dA__eps1e-16__clip1__capapost`: `eps1e-16` is one stabiliser in both
phases and `eps1e-06` the control as it ran; `clip1` is with input clipping
and `clip0` without; `capapost` is the output layer fitted after training
and `capaonline` fitted during it. The key `cell_factors` of the file gives
the same correspondence, and the columns (1) to (8) of
`tab:s_harmonised_auc` state the three factors in their header.

## How to read the files

Each file holds the results of its analysis and the provenance that the
analysis recorded: the hashes of the inputs, the library versions and the
commit of the repository at the time. Paths are given from the root of the
authors' repository (`results/revision/...`, `revision/experiments/` for
the code that is in `frozen/runner/` here).

Four files record `script_sha256` (`score_bruto`, `h1_referencias`,
`region_post_preludio`, `control_armonizado`): it is the sha256 of the
script that computed the file and identifies that script in the repository.
Where a file records the sha256 of another result file, it is likewise that
of the file in the repository; for the files of `data/` that is the file
before the edits that `data/README.md` describes.

Every file states its status under the plan, and holds a block
(`regresion` or `regression`) in which the values it shares with the
article are compared with the printed ones.

P values stored for an exploratory comparison are unadjusted. For the
reductions of the set of streams the files hold the estimate, the counts
and the exact interval: the plan fixes that a reduction is reported without
p values.
