# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         src/kitnetzday/tables/ablations.py                        #
#   Version:      2.0.2                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Table T2: ablation blocks with the median and range of    #
#                 their ROC-AUC deltas against the reference cell.          #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from kitnetzday.io.inputs import Inputs
from kitnetzday.io.outputs import Out
from kitnetzday.utils.formatting import fnum, get, tex_escape
from kitnetzday.utils.loggable import Loggable

# Pointers written into the rows of numbers_used.md. They name the input
# document the summary was taken from, so the two branches below stay
# distinguishable in the table of the values used.
SRC_STATS: str = "stats.json#/ablations/blocks"
SRC_AGGREGATE: str = "aggregate.json#/ablations"
# Output strings of the paper pipeline, written verbatim into the .tex
# comments and into the generated report.
NOTE_NO_STATS: str = ("No stats.json: the summary per dataset and value "
                      "is built from aggregate.json.")
MISSING_NO_CELLS: str = "no ablation cells in the inputs"
# Note of the table when some planned cells have no value. The state of
# each of those cells is in data/runs.csv: NOT_RUN, NOT_COMPLETED or
# NOT_EXECUTABLE.
NOTE_WITHOUT_VALUE: str = ("{n} planned cells without a value (not run, "
                           "not completed or not executable)")
# Header of the Markdown preview.
MD_HEADER: tuple[str, ...] = (
    "| Block | Factor | Dataset | Value | cells | with Delta | median | "
    "min | max | within noise | no value |",
    "|---|---|---|---|---|---|---|---|---|---|---|")
# Keys of the three summary figures listed in numbers_used.md, in the
# order they are emitted.
SUMMARY_KEYS: tuple[tuple[str, str], ...] = (
    ("MED", "delta_auroc_median"), ("MIN", "delta_auroc_min"),
    ("MAX", "delta_auroc_max"))


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                MAIN CLASS                                 #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class AblationsTable(Loggable):
    """
    Output T2: the ablation blocks behind Table 7 of the manuscript.

    Plan section 7 pre-specifies a set of ablations (window length,
    number of clusters, epochs and so on) that are run with a single
    seed, on one or two streams except the group count of the
    context-free configurations, which covers the nine. With so few
    streams no confidence interval is defensible, so the table is
    deliberately descriptive: per block,
    factor level and dataset it reports how many cells exist, how many
    have a ROC-AUC delta against their ``L=800``, ``k=4`` reference
    cell, the median and the range of that delta, how many deltas fall
    inside the run-to-run noise reference
    ``max(sqrt(2) SD_pool, 0.01)`` of the dataset, and how many planned
    cells have no value (not run, not completed or not executable).

    The per-dataset-and-level summary is taken from ``stats.json`` when
    that document is among the inputs. Otherwise it is rebuilt here
    from the raw ablation cells of ``aggregate.json``, and the table
    says so in its notes.

    This class does not compute any metric: it only reads what
    ``metrics_rev.py``, ``aggregate.py`` and ``stats.py`` produced,
    writes the LaTeX table and the Markdown preview through ``Out``,
    and registers the rows of ``numbers_used.md``.

    Note: ``n_configs``, ``n_with_delta`` and ``n_within_noise`` are
    printed with the ``%d`` conversion, so a summary that stores an
    explicit ``null`` in one of them raises ``TypeError`` instead of
    printing the missing marker.

    :param inputs: Reader of the aggregated results, already built and
        cross-checked.
    :type inputs: Inputs
    :param out: Writer of the outputs, pointing at the figures
        directory of the run.
    :type out: Out
    :param logger: Logger created by the calling script, or None to stay
        silent.
    :type logger: logging.Logger | None
    :raises TypeError: If inputs is not an Inputs, if out is not an Out
        or if logger is neither a logging.Logger nor None.
    """

    def __init__(self,
                 inputs: Inputs,
                 out: Out,
                 logger: logging.Logger | None = None) -> None:
        """
        Store the reader and the writer; ``build`` does the work.

        :param inputs: Reader of the aggregated results.
        :type inputs: Inputs
        :param out: Writer of the outputs.
        :type out: Out
        :param logger: Logger created by the calling script, or None.
        :type logger: logging.Logger | None
        :raises TypeError: If inputs is not an Inputs or out is not an
            Out.
        """
        super().__init__(logger=logger)
        if not isinstance(inputs, Inputs):
            self._error("[AblationsTable.__init__] Input error: the "
                        "parameter inputs must be an Inputs, not "
                        f"{type(inputs).__name__}.", TypeError)
        if not isinstance(out, Out):
            self._error("[AblationsTable.__init__] Input error: the "
                        "parameter out must be an Out, not "
                        f"{type(out).__name__}.", TypeError)
        self.inputs = inputs
        self.out = out
        self.ids: list[str] = list()
        self.missing: list[str] = list()
        self.notes: list[str] = list()
        self.n_without_value = 0

    def build(self) -> bool:
        """
        Write ``tab_ablations.tex`` and its Markdown preview.

        The blocks are visited in ``sorted`` order and, inside a block,
        the summary entries in the order the input document stores them
        (or, when rebuilt here, sorted by dataset id and by the string
        form of the factor level). That order is what fixes the row
        order of the published table.

        :return: Always True, so the caller can count the outputs it
            asked for; a block with no cells still produces a table
            with an explicit "no ablation cells in the inputs" row.
        :rtype: bool
        :raises KeyError: If a raw ablation cell of ``aggregate.json``
            has no "block", "dataset_id", "value" or "job_id" key.
        :raises TypeError: If a summary stores an explicit null in one
            of the three count columns.
        :raises OSError: If an output file cannot be written.
        """
        blocks: dict[str, Any] = dict()
        src = ""
        rows: list[str] = list()
        md: list[str] = list()
        tex: list[str] = list()
        summary: Any = None
        cells: list[Any] = list()
        factor: Any = "?"
        caption = ""
        self.ids = list()
        self.missing = list()
        self.notes = list()
        self.n_without_value = 0
        blocks, src = self._resolve_blocks()
        md = list(MD_HEADER)
        for block in sorted(blocks):
            summary = blocks[block].get("summary_by_dataset_and_value")
            cells = blocks[block].get("cells") or list()
            factor = (cells[0].get("factor") if cells else "?")
            if not summary:
                summary = self._summary_from_cells(cells)
            for entry in summary:
                self._emit(block, factor, entry, src, rows, md)
        if self.n_without_value:
            self.notes.append(
                NOTE_WITHOUT_VALUE.format(n=self.n_without_value))
        caption = self._caption(self._noise_text())
        tex = [r"% Requires \usepackage{booktabs}.",
               r"\begin{table*}[t]", r"  \centering", r"  \footnotesize",
               r"  \begin{tabular}{@{}llllrrrrrrr@{}}", r"  \toprule",
               r"  Block & Factor & Data & Level & Cells & With $\Delta$"
               r" & Median $\Delta$ & Min & Max & Within noise & "
               r"No value \\", r"  \midrule"]
        tex += ["  " + r for r in rows] or [
            r"  \multicolumn{11}{c}{no ablation cells in the inputs} \\"]
        tex += [r"  \bottomrule", r"  \end{tabular}",
                f"  \\caption{{{caption}}}", r"  \label{tab:ablations}",
                r"\end{table*}"]
        self.out.save_table(
            "tab_ablations", "\n".join(tex),
            "# Ablations (Markdown rendering)\n" + "\n".join(md), caption,
            "tab:ablations", self.inputs.naming.plan_tag("abl"),
            sorted(set(self.ids)), self.inputs.partial_inputs,
            {"out_id": "T2",
             "partial_inputs_reasons": list(
                 self.inputs.partial_inputs_reasons),
             "missing": self.missing, "notes": self.notes})
        return True

    # ----- pieces of the table ---------------------------------------- #
    def _resolve_blocks(self) -> tuple[dict[str, Any], str]:
        """
        Find the ablation blocks, preferring the statistics document.

        ``stats.json`` holds the blocks summarised by dataset and
        factor level. When it is absent, or holds no block, the raw
        cells of ``aggregate.json`` are grouped by their "block" field
        and the summary is rebuilt from them; that fallback is recorded
        as a note of the table.

        :return: The blocks, keyed by block name, and the pointer of
            the document they came from.
        :rtype: tuple[dict[str, Any], str]
        :raises KeyError: If a raw ablation cell has no "block" key.
        """
        blocks: Any = None
        src = SRC_STATS
        blocks = get(self.inputs.stats or dict(), "ablations", "blocks",
                     default=None)
        if not blocks:
            blocks = dict()
            src = SRC_AGGREGATE
            for cell in self.inputs.agg.get("ablations") or list():
                blocks.setdefault(
                    cell["block"],
                    {"cells": list(),
                     "summary_by_dataset_and_value": list()},
                )["cells"].append(cell)
            self.notes.append(NOTE_NO_STATS)
        if not blocks:
            self.missing.append(MISSING_NO_CELLS)
        return blocks, src

    def _summary_from_cells(
            self,
            cells: Sequence[Mapping[str, Any]] = tuple(),
    ) -> list[dict[str, Any]]:
        """
        Summarise raw ablation cells by dataset and factor level.

        The cells of one block are grouped by their dataset id and by
        the string form of their factor level, and each group gives the
        median, the minimum and the maximum of the available deltas,
        how many cells fall inside the noise reference, and the job ids
        of the planned cells that have no delta.

        Note: the level is grouped and reported as ``str(value)``, so
        the levels ``2`` and ``"2"`` land in the same group, and the
        level printed in the table is the string form. The median goes
        through ``float(np.median(...))`` while the minimum and the
        maximum are the stored values untouched.

        :param cells: Raw ablation cells of one block, each with the
            keys "dataset_id", "value", "job_id" and, when the cell has
            a value, "delta_auroc" (ROC-AUC points) and
            "within_noise".
        :type cells: Sequence[Mapping[str, Any]]
        :return: One summary dictionary per (dataset, level) pair,
            sorted by dataset id and then by the string level, with the
            keys "dataset_id", "value", "n_configs", "n_with_delta",
            "delta_auroc_median", "delta_auroc_min",
            "delta_auroc_max", "n_within_noise" and "without_value".
        :rtype: list[dict[str, Any]]
        :raises TypeError: If cells is not a sequence of mappings.
        :raises KeyError: If a cell has no "dataset_id", "value" or
            "job_id" key.
        """
        grouped: dict[tuple[Any, str], list[Any]] = dict()
        summary: list[dict[str, Any]] = list()
        deltas: list[Any] = list()
        if isinstance(cells, (str, bytes)) or \
                not isinstance(cells, Sequence):
            self._error("[AblationsTable._summary_from_cells] Input "
                        "error: the parameter cells must be a sequence "
                        f"of mappings, not {type(cells).__name__}.",
                        TypeError)
        for cell in cells:
            if not isinstance(cell, Mapping):
                self._error("[AblationsTable._summary_from_cells] Input "
                            "error: every ablation cell must be a "
                            "mapping, not "
                            f"{type(cell).__name__}.", TypeError)
            grouped.setdefault(
                (cell["dataset_id"], str(cell["value"])),
                list()).append(cell)
        for (dataset, level), group in sorted(grouped.items()):
            deltas = [c.get("delta_auroc") for c in group
                      if c.get("delta_auroc") is not None]
            summary.append(
                {"dataset_id": dataset, "value": level,
                 "n_configs": len(group), "n_with_delta": len(deltas),
                 "delta_auroc_median": (float(np.median(deltas))
                                        if deltas else None),
                 "delta_auroc_min": (min(deltas) if deltas else None),
                 "delta_auroc_max": (max(deltas) if deltas else None),
                 "n_within_noise": sum(1 for c in group
                                       if c.get("within_noise")),
                 "without_value": [c["job_id"] for c in group
                                   if c.get("delta_auroc") is None]})
        return summary

    def _emit(self,
              block: str,
              factor: Any,
              entry: Mapping[str, Any],
              src: str,
              rows: list[str],
              md: list[str]) -> None:
        """
        Add one summary entry as a LaTeX row and a preview row.

        The three summary figures of the entry are also registered as
        rows of ``numbers_used.md``, with one pointer per (block,
        dataset, level) triple, and the count of planned cells without
        a value is accumulated in ``n_without_value``.

        Note: the dataset short label and the counts are written into
        the LaTeX row without escaping, and an unknown dataset id is
        printed as it arrives. The factor level, the block and the
        factor are escaped.

        :param block: Name of the ablation block, as the input stores
            it.
        :type block: str
        :param factor: Factor varied by the block, or "?" when the
            block has no cell to read it from.
        :type factor: Any
        :param entry: Summary of one (dataset, level) pair, as
            ``_summary_from_cells`` describes it.
        :type entry: Mapping[str, Any]
        :param src: Pointer of the document the summary came from.
        :type src: str
        :param rows: LaTeX rows built so far; appended in place.
        :type rows: list[str]
        :param md: Markdown preview rows built so far; appended in
            place.
        :type md: list[str]
        :raises TypeError: If entry is not a mapping, or if one of the
            three count columns holds an explicit null.
        """
        dataset: Any = None
        level: Any = None
        without = 0
        label = ""
        if not isinstance(entry, Mapping):
            self._error("[AblationsTable._emit] Input error: the "
                        "parameter entry must be a mapping, not "
                        f"{type(entry).__name__}.", TypeError)
        dataset = entry.get("dataset_id")
        level = entry.get("value")
        without = len(entry.get("without_value") or list())
        self.n_without_value += without
        for key, field in SUMMARY_KEYS:
            self.ids.append(self.out.add_number(
                f"R.ABL.{block}.{dataset}."
                f"{str(level).replace('.', 'p')}.{key}",
                entry.get(field),
                f"{src}/{block}/summary_by_dataset_and_value "
                f"(dataset {dataset}, value {level})",
                self.inputs.naming.plan_tag("abl"),
                self.inputs.partial_inputs))
        label = self.inputs.naming.dataset_shorts.get(dataset, dataset)
        rows.append(
            r"%s & %s & %s & %s & %d & %d & %s & %s & %s & %d & %d \\"
            % (tex_escape(block), tex_escape(factor), label,
               tex_escape(level), entry.get("n_configs", 0),
               entry.get("n_with_delta", 0),
               fnum(entry.get("delta_auroc_median")),
               fnum(entry.get("delta_auroc_min")),
               fnum(entry.get("delta_auroc_max")),
               entry.get("n_within_noise", 0), without))
        md.append(
            f"| {block} | {factor} | {label} | {level} | "
            f"{entry.get('n_configs', 0)} | "
            f"{entry.get('n_with_delta', 0)} | "
            f"{fnum(entry.get('delta_auroc_median'))} | "
            f"{fnum(entry.get('delta_auroc_min'))} | "
            f"{fnum(entry.get('delta_auroc_max'))} | "
            f"{entry.get('n_within_noise', 0)} | {without} |")

    def _noise_text(self) -> str:
        """
        List the per-dataset noise reference for the caption.

        Note: the threshold is read straight from the aggregate rather
        than through ``Inputs.noise_threshold``, so a value stored as
        something other than a number prints as the missing marker
        instead of raising.

        :return: Text like "AW 0.012, FUZZ 0.010", with one item per
            dataset that has a threshold, in the order of the dataset
            list; empty when no dataset has one.
        :rtype: str
        :raises KeyError: If a dataset of the list has no short label.
        """
        thresholds: dict[str, Any] = dict()
        shorts: Mapping[str, str] = dict()
        shorts = self.inputs.naming.dataset_shorts
        thresholds = {x: get(self.inputs.agg, "datasets", x, "sd_pool",
                             "threshold")
                      for x in self.inputs.naming.datasets}
        return ", ".join(f"{shorts[x]} {fnum(thresholds[x])}"
                         for x in self.inputs.naming.datasets
                         if thresholds.get(x) is not None)

    @staticmethod
    def _caption(noise_text: str = "") -> str:
        """
        Build the caption of the table in academic English.

        The caption states that the block is descriptive, names the
        reference cell of the deltas, gives the noise reference of
        every dataset and says explicitly that a delta below that
        reference is not read as an effect.

        :param noise_text: Per-dataset noise reference, as
            ``_noise_text`` writes it.
        :type noise_text: str
        :return: The caption, with LaTeX maths and ready for the .tex
            float and for the README.
        :rtype: str
        """
        return ("Ablations (descriptive, single seed unless stated). "
                "For every block, factor level and dataset: number of "
                "ablated cells, how many have a "
                r"$\Delta$ROC-AUC against their L=800, $k=4$ "
                r"reference cell, the median and range of $\Delta$, "
                "how many fall inside the run-to-run noise reference "
                r"$\max(\sqrt{2}\,\mathrm{SD}_{\mathrm{pool}}, 0.01)$ "
                "and how many planned cells have no value (not run, "
                "not completed or not executable). Noise reference per "
                "dataset: " + noise_text + ". Except block "
                "T3-min-a, which covers the nine streams, every block "
                "uses one or two streams, which cannot characterise "
                "the range of traffic structures, so these results are "
                "descriptive and no confidence interval is given. A "
                r"$|\Delta|$ below the noise reference is not "
                "interpreted as an effect.")

    def __repr__(self) -> str:
        """
        Show what the last build produced.

        :return: Text like "AblationsTable(ids=12, without_value=0,
            missing=0)".
        :rtype: str
        """
        return (f"AblationsTable(ids={len(self.ids)}, "
                f"without_value={self.n_without_value}, "
                f"missing={len(self.missing)})")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
