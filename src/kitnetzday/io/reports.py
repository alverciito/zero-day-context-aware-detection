# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         src/kitnetzday/io/reports.py                              #
#   Version:      2.0.1                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-13                                                #
#   Description:  README, table of the values used and manifest of one      #
#                 run of the figure pipeline.                               #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import argparse
import datetime as dt
import logging
from collections.abc import Sequence
from typing import Any

from kitnetzday.io.inputs import Inputs
from kitnetzday.io.outputs import Out

# Schema and version of the manifest written by this module. They
# identify the format of figures.json for whoever reads it. Schema 2
# names the status of the inputs "partial_inputs", the count of listed
# values "numbers_listed" and the notes "implementation_notes".
SCHEMA = "jisa-figures/2"
FIGURES_VERSION = "2.0.0"

# Analysis choices of the figure pipeline that the analysis plan does
# not fix, quoted verbatim in the README and in the manifest as the
# notes FIG-1 to FIG-7.
IMPLEMENTATION_NOTES: list[str] = [
    "FIG-1 (plan §11, §5.3) The plan asks for a 'mean-rank plot' and "
    "does not fix which interval to draw. The mean rank is drawn with "
    "+-1.96 standard errors across datasets, plus the per-dataset "
    "ranks; it is descriptive, not adjusted for multiplicity, and "
    "replaces no test. No critical-difference diagram is drawn (§5.3 "
    "rules it out with 55 pairs).",
    "FIG-2 (plan §14.1) The label vocabulary (CONFIRMATORIO / "
    "SENSIBILIDAD / EXPLORATORIO) has no label for the secondary "
    "metrics of §3.2 (calibrated F1, WindowDiff) nor for the cost of "
    "§9. '[SENSIBILIDAD plan §3.2 ...]' and '[CONFIRMATORIO plan §9 "
    "...]' are used with an explicit qualifier, always with one of the "
    "three prefixes of §14.1.",
    "FIG-3 (plan §3.6) The smoothed trivial F1 line of the appendix "
    "figure, and the count of bars below it, come from an audit made "
    "before the plan was frozen, so they are labelled [EXPLORATORIO] "
    "even though they accompany figures of the original submission.",
    "FIG-4 (plan §9.5) The plan asks for the records per second of the "
    "post-processing stage (KMD, MVD, filters, fusion, threshold). "
    "timing.json does not separate that stage: 'timing.total_s' of "
    "metrics.json is reported, which includes every sensitivity "
    "analysis, and it is declared in the caption as an upper bound.",
    "FIG-5 The full main table (9 datasets x 11 configurations x 3 "
    "metrics) does not fit on one page: it is generated as a longtable "
    "(requires \\usepackage{longtable}) and also as a compact one-page "
    "version with ROC-AUC only.",
    "FIG-6 (plan §4.5) An output built from inputs other than the "
    "final aggregate and its statistics carries the label [PARTIAL "
    "INPUTS]. The outputs generated from the shipped data read the "
    "final inputs and do not carry it.",
    "FIG-7 (plan §11 R1.10) The 'best-performing configuration per "
    "dataset' table the plan asks for is generated as "
    "`tab_best_per_dataset.tex`, with the literal note 'selected on "
    "evaluation results; descriptive, not an estimate of expected "
    "performance' in the caption and as a LaTeX comment, plus AUC_t "
    "and the number of seeds per row. Selection on the evaluation "
    "segment remains excluded from the abstract, the highlights and "
    "the contributions (§11).",
]


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                AUXILIARY                                  #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def _now_iso() -> str:
    """
    Current local time, to the second, in ISO 8601 form.

    :return: Text like "2026-10-06T18:42:07", without a time zone, as
        the rest of the pipeline writes it.
    :rtype: str
    """
    return dt.datetime.now().isoformat(timespec="seconds")


def _check(inputs: Inputs,
           out: Out,
           method: str,
           logger: logging.Logger | None = None) -> None:
    """
    Check that the two mandatory arguments are of the right class.

    :param inputs: Inputs of the run.
    :type inputs: Inputs
    :param out: Output directory of the run.
    :type out: Out
    :param method: Name of the calling function, for the message.
    :type method: str
    :param logger: Logger created by the calling script, or None.
    :type logger: logging.Logger | None
    :raises TypeError: If inputs is not an Inputs or out is not an Out.
    """
    message = ""
    if not isinstance(inputs, Inputs):
        message = (f"[{method}] Input error: the parameter inputs must "
                   "be a kitnetzday.io.Inputs, not "
                   f"{type(inputs).__name__}.")
    elif not isinstance(out, Out):
        message = (f"[{method}] Input error: the parameter out must be "
                   f"a kitnetzday.io.Out, not {type(out).__name__}.")
    if message:
        if logger is not None:
            logger.error(message)
        raise TypeError(message)


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                              MAIN FUNCTION                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def write_index(inputs: Inputs,
                out: Out,
                timing_rows: Sequence[dict[str, Any]],
                args: argparse.Namespace,
                provenance: Any,
                implementation_notes: Sequence[str] = tuple(
                    IMPLEMENTATION_NOTES),
                logger: logging.Logger | None = None) -> None:
    """
    Write ``README_figures.md``, the human index of the run.

    The README says which outputs exist, which label of the analysis
    plan each one carries, whether the run read the final inputs, the
    caption of each output in academic English, the ids of the values
    behind it, the inputs that were not found, the implementation
    notes of the pipeline, and the command that regenerates
    everything.

    Note: ``timing_rows`` and ``provenance`` are not used in the
    README; they are kept so that the three writers of this module
    take the same arguments.

    :param inputs: Inputs of the run, for the hashes and the status.
    :type inputs: Inputs
    :param out: Output directory, for the records and the command.
    :type out: Out
    :param timing_rows: Reference timing jobs; not used here.
    :type timing_rows: Sequence[dict[str, Any]]
    :param args: Parsed command line; the optional attribute ``note``
        is quoted as a run note.
    :type args: argparse.Namespace
    :param provenance: Provenance of the run; not used here.
    :type provenance: Any
    :param implementation_notes: Analysis choices the plan does not
        fix, one paragraph each.
    :type implementation_notes: Sequence[str]
    :param logger: Logger created by the calling script, or None.
    :type logger: logging.Logger | None
    :raises TypeError: If inputs or out are of the wrong class.
    :raises OSError: If the README cannot be written.
    """
    lines = list()
    _check(inputs, out, "write_index", logger)
    lines = ["# Figures and tables of the revised manuscript — "
             "JISAS-D-26-02300", "",
             f"Generated by `{out.generator}` ({out.script_ref}); the "
             "time of the run is in `figures.json#/created`.", "",
             "- Inputs: " + ", ".join(
                 f"`{k}` (sha256 {v[:12] if v else 'missing'})"
                 for k, v in inputs.sha256.items()), "",
             f"- `aggregate.complete` = {inputs.agg_complete}; "
             f"`stats.json run_type` = {inputs.run_type};"
             f" **{'PARTIAL INPUTS' if inputs.partial_inputs else 'FINAL'}"
             "**."]
    if inputs.partial_inputs_reasons:
        lines.append("  - Reasons: " + "; ".join(
            inputs.partial_inputs_reasons))
    if inputs.consistency:
        lines.append(f"- **{len(inputs.consistency)} mismatches** "
                     "between `cells.csv` and `aggregate.json` "
                     "(see `figures.json#/consistency`).")
    for m in inputs.missing:
        lines.append(f"- Missing input: {m}")
    for nt in (getattr(args, "note", None) or list()):
        lines.append(f"- Run note: {nt}")
    lines += ["", "| ID | Output | Kind | Plan label | Status | Files |",
              "|---|---|---|---|---|---|"]
    for r in out.records:
        lines.append(f"| {r['id']} | `{r['name']}` | {r['kind']} | "
                     f"{r['tag']} | "
                     f"{'partial inputs' if r['partial_inputs'] else 'final'}"
                     " | "
                     + ", ".join(f"`{f}`" for f in r["files"]) + " |")
    lines += ["", "## Captions and traceability", ""]
    for r in out.records:
        lines += [f"### {r['id']} — `{r['name']}` ({r['label']})", "",
                  f"- Plan label: {r['tag']}"
                  + (f"  **{out.partial_tag}**" if r["partial_inputs"]
                     else ""),
                  f"- ids of `numbers_used.md` it uses "
                  f"({len(r['numbers_ids'])}): "
                  + (", ".join(f"`{i}`" for i in r["numbers_ids"][:12])
                     + (" …" if len(r["numbers_ids"]) > 12 else "")
                     if r["numbers_ids"] else "(none)"),
                  ""]
        if r["notes"]:
            lines += ["- Notes: " + " ".join(r["notes"]), ""]
        if r["missing"]:
            lines += [f"- Missing data ({len(r['missing'])}): "
                      + "; ".join(r["missing"][:6])
                      + (" …" if len(r["missing"]) > 6 else ""), ""]
        lines += ["> " + r["caption"], ""]
    lines += ["## Analysis choices the plan does not fix", ""]
    lines += [f"- {d}" for d in implementation_notes]
    lines += ["", "## Using the outputs", "",
              "- Every figure `.tex` is a self-contained `figure*` "
              "that includes `figures/<name>.pdf`. The `.png` files "
              "are raster copies of the same figures.",
              "- The tables need `\\usepackage{booktabs}`; "
              "`tab_main_results.tex` also needs "
              "`\\usepackage{longtable}` (one-page alternative: "
              "`tab_main_results_compact.tex`).",
              "- Every value printed by these outputs is listed in "
              "`numbers_used.md` with the pointer to its source.",
              "", "## Regenerating", "", f"```\n{out.cmd}\n```", ""]
    out.write_text("README_figures.md", "\n".join(lines) + "\n")


def write_numbers(inputs: Inputs,
                  out: Out,
                  logger: logging.Logger | None = None) -> int:
    """
    Write ``numbers_used.md``, the table of the values used.

    Every value printed by a figure or a table of this run is listed
    with its id, the pointer to its source, the script and the command
    that regenerates it. The "Location" column holds a dash: this file
    lists the values by output, and ``README_figures.md`` says which
    output uses each id. When the run read partial inputs the file is
    headed by the partial-inputs label and its reasons.

    The rows are sorted by id, so two runs of the same inputs produce
    the same file.

    :param inputs: Inputs of the run, for the status and the hashes.
    :type inputs: Inputs
    :param out: Output directory, holding the registered values.
    :type out: Out
    :param logger: Logger created by the calling script, or None.
    :type logger: logging.Logger | None
    :return: How many rows were written.
    :rtype: int
    :raises TypeError: If inputs or out are of the wrong class.
    :raises OSError: If the file cannot be written.
    """
    head = list()
    rows = list()
    _check(inputs, out, "write_numbers", logger)
    head = ["# Values used by the figures and tables, with their "
            "source", ""]
    if inputs.partial_inputs:
        head = [f"# {out.partial_tag} Values read from inputs other "
                "than the final aggregate and its statistics: "
                + "; ".join(inputs.partial_inputs_reasons), ""]
    head += ["- Inputs: " + "; ".join(
        f"{k} sha256 {v[:12]}" for k, v in inputs.sha256.items() if v),
        "- Location: not recorded in this file; `README_figures.md` "
        "lists the ids each output uses.", "",
        "| ID | Value | Location | Source (path or runs.csv filter) "
        "| Script | "
        "Command to regenerate |", "|---|---|---|---|---|---|"]
    for idd in sorted(out.numbers):
        v, src = out.numbers[idd]
        val = (repr(float(v))
               if isinstance(v, (int, float))
               and not isinstance(v, bool) else str(v))
        rows.append(f"| {idd} | {val} | - | {src} | "
                    f"{out.script_ref} | `{out.cmd_rows}` |")
    out.write_text("numbers_used.md", "\n".join(head + rows) + "\n")
    return len(rows)


def write_manifest(inputs: Inputs,
                   out: Out,
                   provenance: Any,
                   args: argparse.Namespace,
                   timing_rows: Sequence[dict[str, Any]],
                   schema: str = SCHEMA,
                   figures_version: str = FIGURES_VERSION,
                   implementation_notes: Sequence[str] = tuple(
                       IMPLEMENTATION_NOTES),
                   logger: logging.Logger | None = None,
                   ) -> dict[str, Any]:
    """
    Write ``figures.json`` and ``SHA256SUMS``, the machine manifest.

    The manifest is the machine-readable twin of the README: schema and
    version, creation time, whether the run read partial inputs and
    why, every input with its path and hash, the reference timing jobs,
    one record per output, the cross-check results, the inputs that
    were not found, the implementation notes, the command, the parsed
    arguments and the provenance of the run. ``SHA256SUMS`` then pins
    every file of the directory, including ``figures.json`` itself.

    Note: the hash list is built from whatever is in the output
    directory at this moment, excluding ``SHA256SUMS``, so a file left
    behind by an earlier run of the same directory is listed too.

    :param inputs: Inputs of the run, for the hashes and the status.
    :type inputs: Inputs
    :param out: Output directory, for the records and the files.
    :type out: Out
    :param provenance: Provenance of the run, as the caller built it
        (commit, environment, host); stored as given.
    :type provenance: Any
    :param args: Parsed command line; every attribute is stored as a
        string, and the optional ``note`` list is appended to the notes.
    :type args: argparse.Namespace
    :param timing_rows: Reference timing jobs; only their "job_id" is
        stored.
    :type timing_rows: Sequence[dict[str, Any]]
    :param schema: Schema identifier of the manifest.
    :type schema: str
    :param figures_version: Version of the figure pipeline.
    :type figures_version: str
    :param implementation_notes: Analysis choices the plan does not
        fix, one paragraph each.
    :type implementation_notes: Sequence[str]
    :param logger: Logger created by the calling script, or None.
    :type logger: logging.Logger | None
    :return: The manifest as it was written, before cleaning.
    :rtype: dict[str, Any]
    :raises TypeError: If inputs or out are of the wrong class.
    :raises OSError: If a file cannot be written or read.
    """
    files = list()
    sums = dict()
    _check(inputs, out, "write_manifest", logger)
    files = out.output_files(exclude=("SHA256SUMS",))
    sums = {f: out.sha256(f) for f in files}
    obj = {"schema": schema, "figures_version": figures_version,
           "created": _now_iso(),
           "partial_inputs": inputs.partial_inputs,
           "partial_inputs_reasons": inputs.partial_inputs_reasons,
           "aggregate_complete": inputs.agg_complete,
           "stats_run_type": inputs.run_type,
           "inputs": {k: {"path": out.shown(v),
                          "sha256": inputs.sha256.get(k)}
                      for k, v in inputs.paths.items()},
           "results_root": out.shown(inputs.results_root),
           "timing_reference_jobs": [r["job_id"] for r in timing_rows],
           "outputs": out.records, "numbers_listed": len(out.numbers),
           "consistency": inputs.consistency,
           "cross_stats_aggregate": getattr(inputs, "cross_stats",
                                            None),
           "missing_inputs": inputs.missing,
           "notes": inputs.notes + list(
               getattr(args, "note", None) or list()),
           "implementation_notes": list(implementation_notes),
           "command": out.cmd,
           "args": {k: str(v) for k, v in vars(args).items()},
           "provenance": provenance, "outputs_sha256": sums}
    out.write_json("figures.json", obj)
    sums["figures.json"] = out.sha256("figures.json")
    out.write_text("SHA256SUMS",
                   "".join(f"{v}  {k}\n"
                           for k, v in sorted(sums.items())))
    return obj


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
