# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         src/kitnetzday/tables/cost.py                             #
#   Version:      1.0.3                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Table T3: computational cost of one full run per          #
#                 variant on the reference stream.                          #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
from collections.abc import Mapping, Sequence
from typing import Any

from kitnetzday.io.inputs import Inputs
from kitnetzday.io.outputs import Out
from kitnetzday.utils.formatting import fnum, tex_escape
from kitnetzday.utils.loggable import Loggable

# Variants implemented in numpy: their job id carries no paradigm suffix,
# so the row is labelled with the bare variant name.
NUMPY_VARIANTS: tuple[str, ...] = ("original", "stat")
# Output strings of the paper pipeline, written verbatim into the .tex
# comments and into the generated report.
MISSING_NO_TIMING: str = ("no timing.json with timing_ref = 1 under the "
                          "results root")
EXCLUSIVE_UNKNOWN: str = "not exclusive"
NOTE_NON_EXCLUSIVE: str = ("Timings measured with other jobs running, or "
                           "whose exclusivity is unknown, are EXCLUDED "
                           "from the table (plan §9.2) and listed in the "
                           "comments of the .tex.")
NOTE_MICROBENCHMARKS: str = ("The micro-benchmarks of block B-final (plan "
                             "§9.4) are reported in the manuscript from "
                             "their own run: they are not in this "
                             "table.")
# Header of the Markdown preview, as the submitted pipeline writes it.
MD_HEADER: tuple[str, ...] = (
    "| Variant | Device | AEs | Params | Wall (h) | FM (s) | Train (s) "
    "| Exec (s) | Exec rows/s | GPU peak (MiB) | Peak RSS (GiB) | "
    "Post-proc (s) |",
    "|---|---|---|---|---|---|---|---|---|---|---|---|")
# Figures listed as rows of numbers_used.md, as (id suffix, pointer
# inside timing.json), in the order they are emitted.
TIMING_KEYS: tuple[tuple[str, str], ...] = (
    ("WALLH", "wall_seconds"), ("FM_S", "phases/fm_s"),
    ("TRAIN_S", "phases/train_s"), ("EXEC_S", "phases/exec_s"),
    ("ROWS_PER_S", "exec_rows_per_s"),
    ("GPU_PEAK_MIB", "gpu_peak_bytes"),
    ("RSS_PEAK_GIB", "rss_peak_bytes"))
# Thousands separator of the paper tables: a LaTeX thin space.
THIN_SPACE: str = "\\,"
# Folder of the package that holds one sub-folder per job, as the
# source column of numbers_used.md names it.
JOBS_POINTER_ROOT: str = "data/jobs"


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                MAIN CLASS                                 #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class CostTable(Loggable):
    """
    Output T3: the computational cost (Table 9 of the manuscript).

    Plan section 9.2 asks for the cost of one full run per variant on a
    single reference stream (KNAD Active Wiretap, seed A), measured on
    the jobs whose ``timing.json`` was recorded with ``timing_ref = 1``.
    The table reports the wall time, the device, the number of
    autoencoders and their parameters, the duration of the three phases
    (DBSCAN feature map, training pass, streaming execution), the
    execution throughput, the peak GPU and host memory, and the wall
    time of the whole post-processing stage.

    A timing is only usable when the job states that it had exclusive
    use of the machine. Runs that were concurrent, or whose record does
    not state exclusivity, are dropped from the table, counted in the
    caption and listed by job id in a LaTeX comment of the output file.

    This class contains no predictive-performance figure and computes
    no metric: it formats what ``read_timing_reference`` collected.

    Note: a wall time, a GPU peak or a host peak stored as exactly zero
    is treated as missing and printed as the missing marker, because the
    submitted pipeline guards those three conversions with a
    truthiness test instead of a ``None`` test. The behaviour is kept.

    Note: the "AEs" column reads ``n_ae`` with a default of "--", so a
    record that stores an explicit ``null`` prints "None" instead of
    the missing marker; the Markdown preview of the same column uses a
    truthiness test and prints "--". The two columns are left as they
    are in the submitted pipeline.

    :param inputs: Reader of the aggregated results, used for the
        analysis-plan label and the partial-inputs flag.
    :type inputs: Inputs
    :param out: Writer of the outputs, pointing at the figures
        directory of the run.
    :type out: Out
    :param timing_rows: Reference timings, as
        ``kitnetzday.io.legacy.read_timing_reference`` returns them: one
        mapping per job, with at least the keys "job_id", "variant" and
        "paradigm".
    :type timing_rows: Sequence[Mapping[str, Any]]
    :param logger: Logger created by the calling script, or None to
        stay silent.
    :type logger: logging.Logger | None
    :raises TypeError: If inputs is not an Inputs, if out is not an
        Out, if timing_rows is not a sequence of mappings or if logger
        is neither a logging.Logger nor None.
    """

    def __init__(self,
                 inputs: Inputs,
                 out: Out,
                 timing_rows: Sequence[Mapping[str, Any]] = tuple(),
                 logger: logging.Logger | None = None) -> None:
        """
        Store the reader, the writer and the reference timings.

        :param inputs: Reader of the aggregated results.
        :type inputs: Inputs
        :param out: Writer of the outputs.
        :type out: Out
        :param timing_rows: Reference timings, one mapping per job;
            an empty sequence is allowed and gives an empty table.
        :type timing_rows: Sequence[Mapping[str, Any]]
        :param logger: Logger created by the calling script, or None.
        :type logger: logging.Logger | None
        :raises TypeError: If inputs is not an Inputs, if out is not an
            Out or if timing_rows is not a sequence of mappings.
        """
        super().__init__(logger=logger)
        if not isinstance(inputs, Inputs):
            self._error("[CostTable.__init__] Input error: the "
                        "parameter inputs must be an Inputs, not "
                        f"{type(inputs).__name__}.", TypeError)
        if not isinstance(out, Out):
            self._error("[CostTable.__init__] Input error: the "
                        "parameter out must be an Out, not "
                        f"{type(out).__name__}.", TypeError)
        if isinstance(timing_rows, (str, bytes)) or \
                not isinstance(timing_rows, Sequence):
            self._error("[CostTable.__init__] Input error: the "
                        "parameter timing_rows must be a sequence of "
                        "mappings, not "
                        f"{type(timing_rows).__name__}.", TypeError)
        for row in timing_rows:
            if not isinstance(row, Mapping):
                self._error("[CostTable.__init__] Input error: every "
                            "timing row must be a mapping, not "
                            f"{type(row).__name__}.", TypeError)
        self.inputs = inputs
        self.out = out
        self.timing_rows = list(timing_rows)
        self.ids: list[str] = list()
        self.missing: list[str] = list()
        self.notes: list[str] = list()
        self.devices: set[str] = set()
        self.excluded: list[str] = list()

    def build(self) -> bool:
        """
        Write ``tab_cost.tex`` and its Markdown preview.

        The rows are visited sorted by variant and then by paradigm,
        which is what fixes the row order of the published table. Jobs
        without confirmed exclusive use of the machine are skipped
        before any number of theirs is registered.

        :return: Always True, so the caller can count the outputs it
            asked for; with no usable timing the table still gets an
            explicit "no reference timing available" row.
        :rtype: bool
        :raises KeyError: If a timing row has no "variant", "paradigm"
            or "job_id" key.
        :raises OSError: If an output file cannot be written.
        """
        rows: list[str] = list()
        md: list[str] = list()
        tex: list[str] = list()
        caption = ""
        self.ids = list()
        self.missing = list()
        self.notes = list()
        self.devices = set()
        self.excluded = list()
        if not self.timing_rows:
            self.missing.append(MISSING_NO_TIMING)
        md = list(MD_HEADER)
        for row in sorted(self.timing_rows,
                          key=lambda z: (z["variant"], z["paradigm"])):
            self._emit(row, rows, md)
        if self.excluded:
            self.missing.append(
                f"{len(self.excluded)} reference jobs excluded as not "
                "exclusive or of unknown exclusivity: "
                f"{self.excluded[:3]}")
            self.notes.append(NOTE_NON_EXCLUSIVE)
        caption = self._caption(len(rows))
        tex = [r"% Requires \usepackage{booktabs}.",
               "% Jobs excluded as not exclusive or of unknown "
               "exclusivity (plan §9.2): "
               + ("; ".join(self.excluded) if self.excluded
                  else "none"),
               r"\begin{table*}[t]", r"  \centering", r"  \footnotesize",
               r"  \begin{tabular}{@{}llrrrrrrrrrr@{}}", r"  \toprule",
               r"  Variant & Device & AEs & Params & Wall (h) & FM (s) "
               r"& Train (s) & Exec (s) & Exec rows/s & GPU peak (MiB) "
               r"& Peak RSS (GiB) & Post-proc. (s) \\", r"  \midrule"]
        tex += ["  " + r for r in rows] or [
            r"  \multicolumn{12}{c}{no reference timing available} \\"]
        tex += [r"  \bottomrule", r"  \end{tabular}",
                f"  \\caption{{{caption}}}", r"  \label{tab:cost}",
                r"\end{table*}"]
        self.out.save_table(
            "tab_cost", "\n".join(tex),
            "# Cost (Markdown rendering)\n" + "\n".join(md), caption,
            "tab:cost",
            self.inputs.naming.plan_tag("cost"), sorted(set(self.ids)),
            self.inputs.partial_inputs,
            {"out_id": "T3",
             "partial_inputs_reasons": self.inputs.partial_inputs_reasons,
             "missing": self.missing,
             "notes": self.notes + [NOTE_MICROBENCHMARKS]})
        return True

    # ----- pieces of the table ---------------------------------------- #
    def _emit(self,
              row: Mapping[str, Any],
              rows: list[str],
              md: list[str]) -> None:
        """
        Add one reference timing as a LaTeX row and a preview row.

        A job that does not state exclusive use of the machine is
        recorded in ``excluded`` and contributes nothing else: section
        9.2 forbids using concurrent timings, and printing the row
        anyway would make the caption claim something false.

        :param row: One reference timing, as
            ``read_timing_reference`` returns it.
        :type row: Mapping[str, Any]
        :param rows: LaTeX rows built so far; appended in place.
        :type rows: list[str]
        :param md: Markdown preview rows built so far; appended in
            place.
        :type md: list[str]
        :raises TypeError: If row is not a mapping.
        :raises KeyError: If row has no "variant", "paradigm" or
            "job_id" key.
        """
        variant = ""
        phases: Mapping[str, Any] = dict()
        wall_h: float | None = None
        gpu_mib: float | None = None
        rss_gib: float | None = None
        key = ""
        if not isinstance(row, Mapping):
            self._error("[CostTable._emit] Input error: the parameter "
                        "row must be a mapping, not "
                        f"{type(row).__name__}.", TypeError)
        variant = (f"{row['variant']}-{row['paradigm']}"
                   if row["variant"] not in NUMPY_VARIANTS
                   else row["variant"])
        if row.get("exclusive") is not True:
            self.excluded.append(
                f"{row['job_id']} "
                f"({row.get('exclusive_reason') or EXCLUSIVE_UNKNOWN})")
            return
        self.devices.add(row.get("device") or "?")
        phases = row.get("phases") or dict()
        wall_h = ((row["wall_seconds"] / 3600.0)
                  if row.get("wall_seconds") else None)
        gpu_mib = ((row["gpu_peak_bytes"] / (1024 ** 2))
                   if row.get("gpu_peak_bytes") else None)
        rss_gib = ((row["rss_peak_bytes"] / (1024 ** 3))
                   if row.get("rss_peak_bytes") else None)
        key = variant.replace("-", "").upper()
        self._add_numbers(row, key, phases, wall_h, gpu_mib, rss_gib)
        rows.append(
            r"%s & %s & %s & %s & %s & %s & %s & %s & %s & %s & %s & "
            r"%s \\"
            % (tex_escape(variant), tex_escape(row.get("device") or "--"),
               row.get("n_ae", "--"), self._grouped(row, "params_total"),
               fnum(wall_h, 2), fnum(phases.get("fm_s"), 1),
               fnum(phases.get("train_s"), 1),
               fnum(phases.get("exec_s"), 1),
               self._grouped(row, "exec_rows_per_s"), fnum(gpu_mib, 0),
               fnum(rss_gib, 2), fnum(row.get("postproc_total_s"), 1)))
        md.append(
            f"| {variant} | {row.get('device') or '--'} | "
            f"{row.get('n_ae') or '--'} | "
            f"{row.get('params_total') or '--'} | {fnum(wall_h, 2)} | "
            f"{fnum(phases.get('fm_s'), 1)} | "
            f"{fnum(phases.get('train_s'), 1)} | "
            f"{fnum(phases.get('exec_s'), 1)} | "
            f"{fnum(row.get('exec_rows_per_s'), 0)} | "
            f"{fnum(gpu_mib, 0)} | {fnum(rss_gib, 2)} | "
            f"{fnum(row.get('postproc_total_s'), 1)} |")

    def _add_numbers(self,
                     row: Mapping[str, Any],
                     key: str,
                     phases: Mapping[str, Any],
                     wall_h: float | None,
                     gpu_mib: float | None,
                     rss_gib: float | None) -> None:
        """
        Register the eight figures of one run as rows of values.

        Seven of them point inside the ``timing.json`` of the job.
        The parameter count does not: it is the sum of
        ``count_params`` over the ensemble autoencoders, which lives in
        the ``config.json`` of the job, and its pointer says so.

        :param row: One reference timing, already known to be
            exclusive.
        :type row: Mapping[str, Any]
        :param key: Variant label without separators and in upper
            case, used inside the ids of the rows.
        :type key: str
        :param phases: Phase durations of the job, in seconds.
        :type phases: Mapping[str, Any]
        :param wall_h: Wall time of the run in hours, or None.
        :type wall_h: float | None
        :param gpu_mib: Peak GPU memory in MiB, or None.
        :type gpu_mib: float | None
        :param rss_gib: Peak resident host memory in GiB, or None.
        :type rss_gib: float | None
        :raises KeyError: If row has no "job_id" key.
        """
        values: dict[str, Any] = dict()
        tag = ""
        values = {"WALLH": wall_h, "FM_S": phases.get("fm_s"),
                  "TRAIN_S": phases.get("train_s"),
                  "EXEC_S": phases.get("exec_s"),
                  "ROWS_PER_S": row.get("exec_rows_per_s"),
                  "GPU_PEAK_MIB": gpu_mib, "RSS_PEAK_GIB": rss_gib}
        tag = self.inputs.naming.plan_tag("cost")
        for idk, pointer in TIMING_KEYS:
            self.ids.append(self.out.add_number(
                f"R.COST.{key}.{idk}", values[idk],
                f"{JOBS_POINTER_ROOT}/{row['job_id']}/timing.json#/"
                f"{pointer}", tag, self.inputs.partial_inputs))
        self.ids.append(self.out.add_number(
            f"R.COST.{key}.PARAMS", row.get("params_total"),
            f"{JOBS_POINTER_ROOT}/{row['job_id']}/config.json#/"
            "hyperparameters_effective/autoencoders[*]/count_params",
            tag, self.inputs.partial_inputs))

    @staticmethod
    def _grouped(row: Mapping[str, Any], field: str) -> str:
        """
        Write an integer count with LaTeX thin spaces, or the marker.

        ``params_total`` is written as a plain integer and
        ``exec_rows_per_s`` is rounded to the unit; both get a thin
        space every three digits so that the column stays readable.

        Note: a value of exactly zero is treated as missing, because
        the submitted pipeline guards both cells with a truthiness
        test. The behaviour is kept.

        :param row: One reference timing.
        :type row: Mapping[str, Any]
        :param field: Field to write, "params_total" (a count) or
            "exec_rows_per_s" (rows per second).
        :type field: str
        :return: Text like "1\\,234\\,567", or "--" when the field is
            absent, null or zero.
        :rtype: str
        """
        if not row.get(field):
            return "--"
        if field == "exec_rows_per_s":
            return f"{row[field]:,.0f}".replace(",", THIN_SPACE)
        return f"{row[field]:,}".replace(",", THIN_SPACE)

    def _caption(self, n_rows: int = 0) -> str:
        """
        Build the caption of the table in academic English.

        The caption names the reference stream, states how many runs
        are in the table and on which device, how many were excluded
        and why, what the three phases are, why the throughput is an
        amortised figure and not a per-packet decision cost, where the
        parameter counts come from, and that the last column is an
        upper bound on the cost of the post-processing.

        :param n_rows: Number of runs actually printed in the table,
            which is not the number of reference timings read.
        :type n_rows: int
        :return: The caption, with its LaTeX commands written out,
            ready for the .tex float and for the README.
        :rtype: str
        """
        excluded_txt = ""
        devices_txt = ""
        excluded_txt = (" " + str(len(self.excluded)) + " further "
                        "reference run(s) were excluded because they "
                        "were not executed with exclusive use of the "
                        "machine, or because their record does not "
                        "state it; their job identifiers are listed in "
                        "the source file of this table."
                        if self.excluded else "")
        devices_txt = tex_escape(", ".join(sorted(self.devices)) or "n/a")
        return ("Computational cost of one full run per variant on the "
                "reference stream (KNAD Active Wiretap, seed A). All "
                + str(n_rows) + " runs in the table were executed with "
                "exclusive use of the machine, on device "
                + devices_txt + "." + excluded_txt
                + " Phases are the DBSCAN feature map, the autoencoder "
                  "training pass and the streaming execution; the "
                  "execution throughput is measured as implemented, "
                  "with the per-row Python loop included, at the batch "
                  "size of the implementation, and is therefore an "
                  "amortised throughput and not a per-packet decision "
                  "cost: the batch-1 latency of one window, which is "
                  "what an operator waits for, is reported separately "
                  "in Section~6.6 of the manuscript. Parameters are "
                  "the sum of "
                  r"\texttt{count\_params} over the ensemble "
                  "autoencoders; the two numpy variants "
                  r"(\texttt{original} and \texttt{stat}) do not "
                  "publish that attribute and are shown as "
                  r"\texttt{--} here, with their explicit weight "
                  "counts given in "
                  "Table~9 of the manuscript. The last "
                  "column is the wall time of the whole "
                  "post-processing stage on CPU, which also computes "
                  "every pre-specified sensitivity and is therefore "
                  "an upper bound on the cost of the detectors, the "
                  "fusion and the threshold alone. This table contains "
                  "no predictive-performance figure.")

    def __repr__(self) -> str:
        """
        Show what the last build produced.

        :return: Text like "CostTable(rows=12, excluded=2, ids=96)".
        :rtype: str
        """
        return (f"CostTable(rows={len(self.timing_rows)}, "
                f"excluded={len(self.excluded)}, "
                f"ids={len(self.ids)})")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
