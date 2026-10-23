For this package the comments of the Python sources were removed, the docstrings that held working notes were shortened, the wording of some messages and notes was edited, and the working notes of two data files (`runner/manifest.csv` and `streams/stream_cuts.json`) were removed; no statement that computes a result changed, and this was checked against the tagged files with the comparison of syntax trees described below, which needs those files: they are in the authors' repository and not in this package.

# frozen/ - the code that produced the results

This directory holds the code that trained the models, computed the
metrics, aggregated them and ran the statistics, taken from the git
repository of the revision, so that every number of the aggregate can be
traced to the lines that computed it. The rest of the package
(`src/kitnetzday`, `bin/`) is the analysis downstream of the aggregate,
rewritten for reading.

## What was edited, and how it was checked

- **Python files: 72.** Every comment was removed, 2579 in all; only
  the interpreter line and the encoding line stay. The comments were
  working notes in Spanish that referred to the authors' internal working
  documents, and they were removed so that the package can be read on its
  own. The docstrings that held working notes, 176 of them, were
  replaced by a one-line description. In 8 command-line scripts the module docstring is the
  text that `--help` prints, so it was kept, and only 25 phrases of
  working notes were cut from it or reworded.
- **Messages and notes: 403 string constants in 34 files.** A string
  constant was edited only when it is text that the code emits and
  nothing else: a log or error message, a help text, or a note or a
  label that the code stores as a value in what it writes. The edits
  take out the identifiers of the authors' working documents and reword
  the labels and notes that described a run made before the results were
  frozen; a run of `stats.py` on partial inputs, for instance, is now
  labelled `[PARTIAL INPUTS]`. The notes that were rewritten whole are in
  English, and the others keep their Spanish. Every format field of an
  edited constant is unchanged. In 5 places a test of the runner looks
  for a piece of a label or message that was reworded: there the message
  and the text the test looks for were changed together (13 constants).
- **What was not touched.** No statement, expression, name or number,
  and no string constant that is behaviour: the ones the code compares,
  the keys of the records it reads and writes, the names of the states
  of the job queue and of the types of run, file names and paths. The
  code names the states of a planned run in its own way; the data files
  of this package name them `NOT_RUN`, `NOT_COMPLETED` and
  `NOT_EXECUTABLE`.
- **The check that was made.** Every shipped file was compared with the
  tagged file it comes from. For each Python file,
  `ast.dump(ast.parse(tagged)) == ast.dump(ast.parse(shipped))` holds as
  it stands when only comments were removed (31 files); it holds
  once the replaced docstrings of both sides are set aside in the files
  where only docstrings changed (7 files); and in the 34 files
  with edited messages it holds once, in addition, the text of each
  edited string constant is set aside on both sides. The constants are
  still there, of the same kind, at the same places and, in a formatted
  string, between the same expressions; only their text differs. The
  same was checked on the token streams, so every other literal keeps
  its spelling, and the comparisons of text that the code and its tests
  make were checked against the edited messages, so that none of them
  changes its result.
- **What can be repeated from this package, and what cannot.** From the
  package alone: `sha256sum -c SHA256SUMS` in this directory confirms
  that the files are the ones listed, the second hash of each line of
  `MANIFEST.sha256` is the sha256 of the shipped file, and
  `python -m compileall .` succeeds with Python 3.12 (one file of
  `postprocess/` uses syntax of that version). It prints three
  SyntaxWarnings, for the escape `\l` in the plot labels of
  `models/kitnet.py` and `postprocess/execute.py`, and they do not stop it. The comparison of syntax
  trees and the first hash of each line of `MANIFEST.sha256` need the
  tagged files, which are in the authors' repository and not in this
  package; the section below says which of those first hashes the data
  files record.
- **`runner/manifest.csv`.** The manifest has 584 rows: the 435 of the
  planned grid that the aggregate counts (blocks T1 to T4), 121 of the
  exploratory blocks T7 and T8, 7 of the reproduction block X2 and 21 of
  the sizing block S. The free-text column `notes` was emptied in the
  rows where it held a working note: 171 of 584. The 20 columns
  that define a job are unchanged.
- **`streams/stream_cuts.json`.** The keys that held working notes were
  removed (11 keys and 4 list entries), a working remark was cut
  from 16 texts, and the file paths are now relative to the repository
  (15 paths). Every cut index, count, percentage, hash and file name
  is unchanged.

`MANIFEST.sha256` lists these files, 74 in all, with two
hashes: the sha256 of the file recorded by its git tag and the sha256 of
the file shipped here, followed by what separates the two and by the path
the file has in the repository.
`SHA256SUMS` lists the shipped files, plus the documentation and the
environment files, in the format `sha256sum -c` reads.

What is left is the code as it ran. The other docstrings and most of the
messages and of the texts the code writes to its outputs are in Spanish.
`runner/make_manifest.py` writes the notes of the manifest, so its
`--check` would report the edited column. The absolute paths of the
workstation the code ran on are configuration and were left as written.
Line numbers cited in docstrings and messages are those of the tagged
files.

## What is here, and against which tag it was checked

| Directory | What it is | Path in the repository | Tag | Commit, date |
|---|---|---|---|---|
| `models/` | The TensorFlow models: the KitNET-style ensemble, the feature map (DBSCAN) and the seven autoencoder variants. This is the code behind the figure of the original submission, and the revision trains with it unmodified. | `src/models/` | `submitted-v1` | `90ffa19d`, 15 Sep 2026 |
| `postprocess/` | The post-processing of the original submission: the two second-order detectors, its threshold rule and its segmentation metrics. Kept for the record; the revision does not use it for any reported number except the appendix figure. | `postprocess/detectors_2025/` | `submitted-v1-postproc` | `921a0774`, 15 Sep 2026 |
| `runner/` | The job runner of the revision (`worker.py`, `run_job.py`, `instrument.py`, `pipelines.py`), the windowed and containment-initialised models (`networks_ctx.py`, `networks_eps.py`), the comparators (`baselines_t5.py`), the benchmark (`bench_b.py`), the label-free metrics (`metrics_rev.py`), the aggregation (`aggregate.py`), the statistics (`stats.py`), the manifest of jobs and its generator, and the runner's own tests (`tests/`). | `revision/experiments/` | `runner-v13` | `84356ba3`, 6 Oct 2026 |
| `modified_encoders/` | The post hoc experiment with modified Conv1D and Conv2D encoders (supplement, Section 35): the modified models, the selection runs that chose the changes, the harness that trained and scored the 108 runs, the entry rule, the comparisons and the tables of the supplement. The section below lists the files. | `revision/analysis/` (four folders and one file) | five tags and one commit, listed in `MANIFEST.sha256` | 7 to 22 Oct 2026 |
| `streams/` | What `runner/pipelines.py` imports to build the streams: the loaders of the submission (`launch_all.py`, `database.py`), the label and duplicate audit (`warmup_audit.py`) and the cut indices it produced for every variant of each stream it examined (`stream_cuts.json`). | `experiments/frames/launch_all.py`, `src/database/test_database/database.py`, `revision/analysis/warmup_audit.py`, `revision/analysis/t0/stream_cuts.json` | `submitted-v1` (first two), `runner-v13` (last two) | as above |

The streams of the article use the variants `mon_first` (CIC-IDS2017),
`days7` (CSE-CIC-IDS2018) and `oct31_train_chrono` (ACI-IoT-2023):
`CORRECTED_VARIANT` in `runner/common.py` names them, and every
`data/jobs/*/config.json` of those three datasets records its variant at
the start of `cache.semantics`. The six KNAD captures have a single
variant, in capture order. The key `variante_por_defecto_D7` of
`streams/stream_cuts.json` is the default of the audit script, not the
variant used: for ACI-IoT-2023 it names a variant that was examined and
not used, and for the other streams it names the one used.

The repository and its tags are not part of this package, so the column
"Tag" cannot be checked from here. What can be checked is that the hashes
of the tagged files, in the first column of `MANIFEST.sha256`, are the ones
the data records:

- every `data/jobs/*/metrics.json` records the sha256 of `metrics_rev.py`,
  `common.py` and `pipelines.py` that computed it
  (`provenance.helpers_sha256`);
- `data/aggregate/aggregate.json` records the sha256 of `aggregate.py` and
  of `manifest.csv`, and `data/stats.json` that of `stats.py`;
- every `data/jobs/*/config.json` records the tag of the runner that
  trained the job and the hashes of its code (`code`, `code_hashes`).

## The post hoc experiment with modified encoders

`modified_encoders/` is the code of the experiment reported at the end of
Section 6.1 of the manuscript and in Section 35 of the supplement. It was
compared with its tags and edited in the same way as the rest of this
directory, and `MANIFEST.sha256` gives the tag and the path of each file.

- `modelos_rep.py` builds the modified encoders: the four changes of the
  supplement, each of which can be switched on or off. It also builds the
  attention encoder with the three steps that were tried on it (N, R and
  A, supplement, Section 35), which did not pass the selection, so the
  attention encoder was not modified.
- `prevuelo.py` and `escalera.py` ran the selection runs that chose the
  changes on three streams. `escalera.py` is the driver: it starts
  `prevuelo.py` once per run, with the interpreter of the training
  environment. Their record is
  `data/modified_encoders/selection_runs.json`.
- `harness_seed_a.py`, `harness_seed_b.py` and `harness_seed_c.py` are the
  harness that trained and scored the 108 runs, one version per seed. In
  the repository the three have the same file name in three folders, so
  they are shipped under these names; `MANIFEST.sha256` gives the path of
  each in the repository, and the layout below restores them. Besides
  the seed and what follows from it, the versions differ in two options
  of their queue. The versions of seeds B and C can
  stop launching runs at a given hour (`--corte`), and only the queue of
  seed B used it. The version of seed C has no reference run of its own
  seed, so it pairs each run with the job of seed A and runs no check of
  the harness against the main experiment. The versions of seeds A and B
  also re-ran the unmodified model on eight jobs of the main experiment,
  twice each, to check that the harness reproduces it. The repeats did not
  give the same numbers bit for bit. In each of the eight checks the
  largest distance to the job of the main experiment was at most 4.1 times
  the distance between the two repeats, below the margin of 10 that the
  harness applies (a margin that was not calibrated), so all eight were
  judged equivalent. The check runs themselves are not in this package.
- `puerta_fase_b.py` applies the entry rule with seed A.
- `e1_metricas_t10.py` computes the detection metrics of every run of seeds
  A and B and the comparisons, importing `comun_fase_b.py`,
  `mlp_sin_barajar_metricas.py` and `puerta_fase_b.py`.
- `supl_fase_b.py` reads the comparison file and the held-out reports and
  writes the values of Tables S71 and S72 of the supplement.

The harness runs each job with the version of the runner that its
reference job of the main experiment ran under. The comparison script
computes the metrics of every run with `mlp_sin_barajar_metricas.py`,
which calls the functions of `runner/metrics_rev.py`, and the block
`control_positivo` of `data/modified_encoders/comparisons.json` records
that the same computation reproduces the metric files and the H1
statistics of the main experiment to within 1e-12. The detectors, the
threshold and the metrics are therefore those of the main experiment.

## Thirteen versions of the runner

`runner/` is the last tagged version. The jobs were trained over three
weeks under earlier tags as well, and each job says which: of the 475 jobs
shipped in `data/jobs/`, 35 ran under `runner-v1`, 138 under `v2`, 160
under `v3`, 4 under `v4`, 72 under `v7`, 9 under `v8`, 6 under `v11`, 19
under `v12` and 11 under `v13`. The 21 comparator runs record no tag: they
were run by `baselines_t5.py`, outside the worker. The metrics of all of
them were recomputed with the single `metrics_rev.py` of the tag
`runner-v13`. `models/` did not change at any point: each of the 454 jobs
trained by the worker records that its model code was identical to
`submitted-v1`.

## The analysis plan

`data/stats.json` and `data/aggregate/aggregate.json` record the commit of
the frozen analysis plan (`ab072b58`, tag `analysis-plan-frozen`, 16 Sep
2026 12:15 +02:00) and its sha256
(`f65830b5029614758e784fba0bee33b8b1d3cf3c212d53c0b0d883d30ebdaf89`), and
that the first `metrics.json` was written after it. The plan itself is not
in this package.

## Environments

| Code | Environment | Files |
|---|---|---|
| `models/`, `runner/worker.py`, `run_job.py`, `instrument.py`, `pipelines.py`, `networks_*.py`, `bench_b.py` | Python 3.10.21, TensorFlow 2.15.1, Keras 2.15.0, numpy 1.26.4, with the CUDA 12.2 and cuDNN 8.9.4.25 wheels | `requirements-tf215.txt`, `requirements-tf215.lock.txt` |
| `runner/metrics_rev.py`, `aggregate.py`, `stats.py`, `baselines_t5.py` | Python 3.12.14, numpy 1.26.4, scipy 1.15.1, scikit-learn 1.6.1, pandas 2.2.3, on CPU | `requirements-postproc.txt`, `requirements-postproc.lock.txt` |
| `modified_encoders/`: `modelos_rep.py`, `prevuelo.py` and the three `harness_seed_*.py` | as the first row | as the first row |
| `modified_encoders/`: `e1_metricas_t10.py`, `comun_fase_b.py`, `mlp_sin_barajar_metricas.py` | as the second row. `escalera.py`, `puerta_fase_b.py` and `supl_fase_b.py` need only the standard library, and `escalera.py` ran with the system interpreter (Python 3.12.3) | as the second row |
| `postprocess/` (the original post-processing; in the revision only the regression of the submitted figure runs it, through `runner/metrics_rev.py --o1`) | PyTorch on CPU, which neither file above pins. It was written with Python 3.12.8, numpy 1.26.4, scipy 1.15.1, scikit-learn 1.6.1 and PyTorch 2.5.1, and re-run in the revision with the system interpreter of the workstation: Python 3.12.3, numpy 2.4.3, scipy 1.17.1, scikit-learn 1.8.0 and PyTorch 2.10.0 (CPU) | none |

Hardware of the runs, as each `data/jobs/*/env.json` records it: one NVIDIA
RTX A6000 (48 GB), driver 580.173.02 or 580.178.04 (535.309.01 in five
jobs), an AMD Ryzen 9 7900X and 62 GB of memory, Linux 7.0 with glibc 2.39.

## Third-party code

`models/cluster/corrclust.py` and `models/networks/original_kitsune.py` are
adapted from Kitsune-py (MIT). `runner/baselines_t5.py` imports Kitsune-py
unmodified, at commit `7c8ea51e89268d9a20399dca4bf32d663e638eb4`, from a
directory `external/Kitsune-py` that is not shipped. The notices are in
`NOTICE`, at the root of the package.

## Laying it out to run it

The code is not runnable from this directory as it stands. It expects the
tree of the repository it was written in:

```
<root>/src/models/                               <- frozen/models/
<root>/src/database/test_database/database.py    <- frozen/streams/database.py
<root>/experiments/frames/launch_all.py          <- frozen/streams/launch_all.py
<root>/postprocess/detectors_2025/execute.py     <- frozen/postprocess/execute.py
<root>/postprocess/detectors_2025/src/detectors/ <- frozen/postprocess/detectors/
<root>/revision/experiments/                     <- frozen/runner/
<root>/revision/analysis/warmup_audit.py         <- frozen/streams/warmup_audit.py
<root>/revision/analysis/t0/stream_cuts.json     <- frozen/streams/stream_cuts.json
<root>/external/Kitsune-py/                      <- github.com/ymirsky/Kitsune-py at the commit above
<root>/revision/analysis/...                     <- frozen/modified_encoders/, each file at the
                                                    path that MANIFEST.sha256 gives for it
```

and, beyond the layout:

- the four datasets, at the paths `runner/pipelines.py` maps them to
  (`--data-root`; `data/streams.json` lists every file with its sha256);
- a GPU for the training jobs, which are launched as

  ```
  python revision/experiments/worker.py --gpu 0 \
      --manifest revision/experiments/manifest.csv \
      --results-root <results> --code-root <root>
  ```

- for `metrics_rev.py`, `aggregate.py` and `stats.py`, a git repository
  with the tag `analysis-plan-frozen` on the commit of the plan: they refuse
  to run without it (rule R-7 of the plan), by design;
- for `modified_encoders/`, the git tags of `MANIFEST.sha256` as well. The
  harness and the comparison script check their own files, and the files
  they import, against those tags before they run, and the harness was
  started by queue scripts and a record of the selection that are not
  shipped. They are here to be read. What they wrote is in
  `data/modified_encoders/`.

The runner guards its own provenance: in production mode `run_job.py`
compares the working tree with the git tags, and the code records the
sha256 of its own files, and in two places of the source text of some
functions, in what it writes. With the files of this directory those
comparisons and fingerprints give other values than the tagged files do,
because a hash covers the comments and the message texts too. The values
the data records are those of the tagged files.

Each job writes the calibration and execution scores of both detectors,
the detector inputs, timings and the environment. Metrics are computed
afterwards from those scores.
