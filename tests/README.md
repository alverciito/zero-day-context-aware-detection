# tests/

Run them with `uv run pytest -q`. They mirror `src/kitnetzday/`, plus two
folders that test the package against the shipped data.

**Unit tests: what computes.** Every main function has at least one case
computed by hand, with the arithmetic written out in the docstring of the
test so that a reader can repeat it on paper, plus the edges: empty input,
one value, NaN, and unexpected types checked through their message.

| File | What it covers |
|---|---|
| `utils/test_formatting.py` | Number and LaTeX formatting; the average ranks behind the Friedman omnibus. |
| `utils/test_logging_tools.py` | The logger factory and the `Loggable` base class. |
| `test_naming.py` | The vocabulary of streams, configurations and plan labels. |
| `audit/test_training_audit.py` | The four verdicts of the training audit and their cuts, 0.90 and 1.15. |
| `audit/test_constant_floor.py` | The constant-predictor floor, from the stream cache and from the stored vectors, and that the two give the same verdicts. |
| `audit/test_retrained_pair.py` | The Hodges-Lehmann estimate, its exact interval by inversion of the Wilcoxon test, the exact p value and the win counting. |
| `audit/test_retrained_units.py` | The same statistics after averaging the paired differences to the unit of the plan: the dataset, the dataset and paradigm, and each paradigm half. |
| `audit/test_reconstruction.py` | The Spearman correlations and the within-group permutation p value. |
| `io/test_legacy.py` | The readers of the archived figure values, of the trivial F1 table and of the reference timing jobs. |
| `io/test_inputs.py` | The reader of the aggregate: the rule that tells the final inputs from any other (`partial_inputs` and its reasons), the pointers of the values and the cross-checks. |
| `io/test_outputs.py` | The register of values, the provenance comments of a table, and the three reports of a run (`numbers_used.md`, `README_figures.md`, `figures.json`), with final and with partial inputs. |
| `tables/test_ablations.py` | The summary of an ablation block (median, range, cells within noise) and the count of planned cells without a value. |
| `figures/test_failure_rule.py` | The selection rule of the failure cases applied to the aggregate: the margin branch, the fallback branch and the rows of values. |
| `figures/test_oracle.py` | The count of bars at or below the trivial F1 in the figure of the submitted values and the counts its caption states, with complete and with short inputs. |

**`stats/test_confirmatory.py`: an independent recomputation.** The five
comparisons of H1 (differences per stream, Hodges-Lehmann estimate, exact
96.1 % interval, exact signed-rank p value and Holm adjustment) are
recomputed from `data/aggregate/cells.csv` with the routines of this
package, which share no code with `frozen/runner/stats.py`, and compared
with `data/stats.json`.

**`bin/test_entry_points.py`: the scripts against the shipped data.** Every
script of `bin/` is run in a temporary folder. The tables and the result
files it writes are compared with `data/reference/`: the supplementary
tables byte for byte, the other tables line by line without their provenance
comments, and the JSON results number by number within a relative tolerance
of 1e-9. Each script is also run without its inputs, and must end with exit
code 1 without writing a result. The figures are only checked for
existence, because their bytes depend on the matplotlib build.
`bin/test_reference_latex.py` reads the LaTeX files of `data/reference/`
and checks that no float cites a label that nothing defines.

**`added_analyses/test_added_analyses.py`: the analyses added after the
analysis plan was fixed.** Their result files are in
`data/added_analyses/` and `added_analyses/README.md` is their index. The
tests check that the eight result files are present and listed there, that
their paths are given from the root of the repository, that the four files
that record the sha256 of their script hold one digest, and that the
reductions of the set of streams hold no p value, as the plan fixes. They
then recompute, from the shipped data and with the
routines of this package, what the shipped data allow: the twelve
comparisons on seven streams (estimate, count of positive differences and
exact 95.3 % interval, from `data/stats.json`), the in-sample calibration
by configuration (`data/runs.csv`), the raw score per cell
(`data/aggregate/cells.csv`), the raw score on the six KNAD captures and
the fused detector of the retraining experiment at one and five epochs
(`data/jobs/*/metrics.json`), the counts of the left block of Table S47 of
the supplement restricted to the main experiment
(`data/reference/training_audit.json`), and the block of H1
against the control with one stabiliser. For the analyses that need the
per-record scores, the stored weights or the streams, they only check that
the values the result file repeats from the shipped results are the
shipped ones.

**`added_analyses/test_modified_encoders.py`: the post hoc experiment with
modified encoders.** It recomputes from `data/modified_encoders/` the counts
of the validation ratio by stream, family and seed (Table S72 of the
supplement), with its summary rows and the counts with the window
replaced, and the entry rule with seed 1234 together with the values of
`entry_rule.json`. It recomputes the columns of Table S71: the ROC-AUC per
stream of the modified encoders, per seed and averaged, of the unmodified
ones and of the KitNET-style control, the within-block ROC-AUC, the
record-index reference AUC_t and the median row (the control, the
unmodified encoders and AUC_t from `data/aggregate/cells.csv`). It
recomputes the two comparisons with the control (Hodges-Lehmann estimate,
exact interval, exact signed-rank p and Holm over the two), their
within-block version, the dependence rule (the six packet captures merged
into one unit, and left out) and the count of streams beyond the margin
of 0.01.

These three folders need `data/`; without it they are skipped, except the
check on the index.

**What has no test.** The modules whose inputs are not shipped (the
convergence reports of the retraining experiment and the per-record
arrays):
`audit/containment.py`,
`audit/containment_report.py`, `audit/convergence.py`,
`audit/floor_control.py`, `figures/illustrative.py`,
`figures/stream_context.py` and `figures/stream_panels.py`. That is why
there is no coverage gate. `uv run pytest --cov=kitnetzday` prints the
figure: 82 % of the 4,703 statements of the package, which is 93 % of
the 3,953 statements of the modules whose inputs are shipped and 24 % of
the 750 statements of the seven modules above (the statements that
importing them executes).
