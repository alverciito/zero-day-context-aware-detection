# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         src/kitnetzday/tables/supplementary.py                    #
#   Version:      1.0.2                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-13                                                #
#   Description:  The fifteen LaTeX tables of the supplementary material    #
#                 of the revised manuscript that this package writes.       #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from kitnetzday.tables.supplementary_inputs import SupplementaryInputs
from kitnetzday.tables.supplementary_latex import (
                                                ATTACK_CONFIG_BEGIN,
                                                ATTACK_CONFIG_GROUPS,
                                                CAPTION_ATTACK,
                                                CAPTION_ATTACK_CONFIG,
                                                CAPTION_PERDAY,
                                                CAPTION_SEEDS,
                                                CAPTION_SENS,
                                                GRID_TABLES,
                                                SEED_GROUPS,
                                                caption_below,
                                                escape_cell,
)
from kitnetzday.tables.supplementary_style import SupplementaryStyle
from kitnetzday.utils.loggable import Loggable


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                MAIN CLASS                                 #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class SupplementaryTables(Loggable):
    """
    Fifteen tables of the supplementary material, as LaTeX.

    Nothing is computed here: every number comes from
    ``SupplementaryInputs``, which read the per-job ``metrics.json``
    files, the ``cells.csv`` of the aggregated pass and its
    ``stats.json``. The fifteen floats are emitted in this order:
    ``tab:s_f1`` (F1 of the fused detector at the calibrated
    threshold), ``tab:s_wd`` (its WindowDiff), ``tab:s_loc`` (its
    blocked ROC-AUC, the drift control), ``tab:s_auprc`` (its area under
    the precision--recall curve), ``tab:s_kmd`` and ``tab:s_mvd`` (the
    ROC-AUC of each detector alone), ``tab:s_fpr`` and ``tab:s_alarm``
    (the false-positive rate and the alarm rate at the calibrated
    threshold), ``tab:s_attack`` (recall per attack type, with the
    record counts of the stream), ``tab:s_attack_cfg_a`` and
    ``tab:s_attack_cfg_b`` (the same recall with one column per
    configuration), ``tab:s_seeds_a`` and ``tab:s_seeds_b`` (ROC-AUC
    per seed, split so that the rows fit the page width),
    ``tab:s_sens`` (the pre-specified sensitivities) and
    ``tab:s_perday`` (the per-day results of the two CIC corpora). The
    supplement holds other tables that are typeset by hand or come from
    scripts outside this package. Two registers come out: ``lines``,
    the LaTeX line by line, and ``tables``, the value of every cell
    keyed by the label of its float, which is what the JSON document of
    the supplement publishes. ``tex_text`` closes the job by moving the
    caption of every float below its tabular.

    :param data: Inputs of the supplement, already read.
    :type data: SupplementaryInputs
    :param style: House style of the tables, or None for the style of
        the submitted supplement.
    :type style: SupplementaryStyle | None
    :param reference_level: Calibration level of the analysis T4-P-c that
        the other levels are compared against; "10".
    :type reference_level: str
    :param levels: The two calibration levels the sensitivity table has
        columns for, in column order; ("5", "20").
    :type levels: Sequence[str]
    :param per_day_datasets: Streams of the per-day table; the two CIC
        corpora, because KNAD is one continuous capture and ACI has a
        single evaluation day.
    :type per_day_datasets: Sequence[str]
    :param logger: Logger created by the calling script, or None to stay
        silent.
    :type logger: logging.Logger | None
    :raises TypeError: If data, style or logger are of the wrong type.
    :raises ValueError: If reference_level is not a non-empty string or
        levels does not hold exactly two levels.
    """

    def __init__(self,
                 data: SupplementaryInputs,
                 style: SupplementaryStyle | None = None,
                 reference_level: str = "10",
                 levels: Sequence[str] = ("5", "20"),
                 per_day_datasets: Sequence[str] = ("D8", "D9"),
                 logger: logging.Logger | None = None) -> None:
        """
        Check the settings and build the fifteen tables right away.

        :param data: Inputs of the supplement, already read.
        :type data: SupplementaryInputs
        :param style: House style of the tables, or None.
        :type style: SupplementaryStyle | None
        :param reference_level: Reference calibration level.
        :type reference_level: str
        :param levels: The two calibration levels reported.
        :type levels: Sequence[str]
        :param per_day_datasets: Streams of the per-day table.
        :type per_day_datasets: Sequence[str]
        :param logger: Logger created by the calling script, or None.
        :type logger: logging.Logger | None
        :raises TypeError: If data, style or logger are of the wrong
            type.
        :raises ValueError: If reference_level is not a non-empty
            string or levels does not hold exactly two levels.
        """
        super().__init__(logger=logger)
        if not isinstance(data, SupplementaryInputs):
            self._error("[SupplementaryTables.__init__] Input error: "
                        "the parameter data must be a "
                        "SupplementaryInputs, not "
                        f"{type(data).__name__}.", TypeError)
        if style is not None and \
                not isinstance(style, SupplementaryStyle):
            self._error("[SupplementaryTables.__init__] Input error: "
                        "the parameter style must be a "
                        "SupplementaryStyle or None, not "
                        f"{type(style).__name__}.", TypeError)
        if not isinstance(reference_level, str) or not reference_level:
            self._error("[SupplementaryTables.__init__] Input error: "
                        "the parameter reference_level must be a "
                        f"non-empty string, not {reference_level!r}.")
        if len(levels) != 2:
            self._error("[SupplementaryTables.__init__] Input error: "
                        "the parameter levels must hold exactly the two "
                        "calibration levels the sensitivity table has "
                        f"columns for, not {len(levels)}.")
        self.data = data
        self.style = style or SupplementaryStyle(logger=logger)
        self.reference_level = str(reference_level)
        self.levels = [str(x) for x in levels]
        self.per_day_datasets = [str(x) for x in per_day_datasets]
        self.lines = list()
        self.tables = dict()
        self.build()

    # ----- the eight metric grids ------------------------------------ #
    def _loc_blocks_text(self) -> str:
        """
        Spell out, per stream, how many blocks hold both classes.

        Reads what :class:`SupplementaryInputs` collected and writes it
        in the short stream names of the column headers, as
        "AW 98/187, ... and CIC18 364/642", so the caption of the
        blocked ROC-AUC says what the metric rests on. A capture with a
        single late attack episode leaves most of its blocks pure, and
        on this grid the count ranges from 21 of 134 to 34 of 43.

        :return: The counts, already in the order of the columns.
        :rtype: str
        """
        parts = [f"{self.style.dataset_shorts[d]} "
                 f"{self.data.loc_blocks[d][1]}/"
                 f"{self.data.loc_blocks[d][0]}"
                 for d in self.data.naming.datasets
                 if d in self.data.loc_blocks]
        if len(parts) < 2:
            return "".join(parts)
        return ", ".join(parts[:-1]) + " and " + parts[-1]

    def _grid_table(self,
                    title: str,
                    label: str,
                    field: str,
                    note: str = "") -> None:
        """
        Emit one configuration-by-stream grid of a single metric.

        One row per configuration and one column per stream, both in
        the order of the vocabulary; an empty cell prints the missing
        marker.

        :param title: Caption of the float, already valid LaTeX.
        :type title: str
        :param label: Label of the float, such as "tab:s_f1".
        :type label: str
        :param field: Metric key of the averaged records, such as "wd".
        :type field: str
        :param note: Extra line appended after the tabular, or "" for
            none; no table of the supplement uses it.
        :type note: str
        """
        values = list()
        datasets = self.data.naming.datasets
        configs = self.data.naming.configurations
        self.lines.append(self.style.float_begin())
        self.lines.append(r"\caption{" + title + r"}\label{" + label
                          + "}")
        self.lines.append(r"\begin{tabular}{l" + "r" * len(datasets)
                          + "}")
        self.lines.append(r"\hline")
        self.lines.append(self.style.header("Configuration",
                                            datasets))
        self.lines.append(r"\hline")
        for config in configs:
            values = [self.data.metrics_of(d, config).get(field)
                      for d in datasets]
            self.lines.append(
                self.style.row(self.data.config_label(config)) + " & "
                + " & ".join(self.style.number(v) for v in values)
                + r"\\")
        self.lines.append(r"\hline")
        self.lines.append(r"\end{tabular}")
        if note:
            self.lines.append(note)
        self.lines.append(self.style.float_end())
        self.lines.append("")
        self.tables[label] = {
            d: {c: self.data.metrics_of(d, c).get(field)
                for c in configs} for d in datasets}

    # ----- recall per attack type ------------------------------------ #
    def _attack_table(self) -> None:
        """
        Emit the recall per attack type, one row per type and stream.

        The rows of a stream are sorted by how many observations were
        averaged, descending, which is the order of the submitted
        table; the record count is the one of the stream and
        prints as "n/a" when the job did not store it.
        """
        recalls = dict()
        rows = dict()
        value = None
        n_rows = None
        self.lines.append(self.style.float_begin())
        self.lines.append(r"\caption{" + CAPTION_ATTACK
                          + r"}\label{tab:s_attack}")
        self.lines.append(r"\begin{tabular}{llrr}")
        self.lines.append(r"\hline")
        self.lines.append(r"\textbf{Stream} & \textbf{Attack type} & "
                          r"\textbf{Records} & \textbf{Recall}\\")
        self.lines.append(r"\hline")
        self.tables["tab:s_attack"] = dict()
        for dataset in self.data.naming.datasets:
            recalls = self.data.attack_recalls.get(dataset, dict())
            rows = self.data.attack_rows.get(dataset, dict())
            for name, observations in sorted(
                    recalls.items(), key=lambda kv: -len(kv[1])):
                value = self.data.mean(observations)
                n_rows = rows.get(name)
                self.lines.append("%s & %s & %s & %s\\\\" % (
                    self.style.stream(dataset), escape_cell(name),
                    ("%d" % n_rows) if n_rows is not None else "n/a",
                    self.style.number(value)))
                self.tables["tab:s_attack"].setdefault(
                    dataset, dict())[name] = value
        self.lines.append(r"\hline")
        self.lines.append(r"\end{tabular}")
        self.lines.append(self.style.float_end())
        self.lines.append("")

    # ----- recall per attack type and configuration ------------------ #
    def _attack_config_tables(self) -> None:
        """
        Emit the recall per attack type with a column per configuration.

        ``tab:s_attack`` averages the recall over the eleven
        configurations; these two floats open that average into its
        eleven columns, averaged over the two seeds only. The rows of a
        stream are sorted by the number of records of the attack type,
        descending.
        """
        configs = self.data.naming.configurations
        heads = " & ".join(r"\textbf{%s}" % c for c in configs)
        legend = ", ".join(
            "%s~%s" % (c, self.style.row(self.data.config_label(c)))
            for c in configs)
        rows = dict()
        values = list()
        self.tables["tab:s_attack_cfg"] = dict()
        for suffix, start, stop, group in ATTACK_CONFIG_GROUPS:
            self.lines.append(ATTACK_CONFIG_BEGIN)
            self.lines.append(
                r"\caption{" + CAPTION_ATTACK_CONFIG % (group, legend)
                + r"}\label{tab:s_attack_cfg_%s}" % suffix)
            self.lines.append(r"\begin{tabular}{ll" + "r" * len(configs)
                              + "}")
            self.lines.append(r"\hline")
            self.lines.append(r"\textbf{Stream} & \textbf{Attack type} "
                              "& " + heads + r"\\")
            self.lines.append(r"\hline")
            for dataset in self.data.naming.datasets[start:stop]:
                rows = self.data.attack_rows.get(dataset, dict())
                for name in sorted(
                        self.data.attack_recalls.get(dataset, dict()),
                        key=lambda t: -(rows.get(t) or 0)):
                    values = [self.data.mean(
                        self.data.attack_recalls_by_cell.get(
                            (dataset, c), dict()).get(name, list()))
                        for c in configs]
                    self.lines.append("%s & %s & %s\\\\" % (
                        self.style.stream(dataset), escape_cell(name),
                        " & ".join(self.style.number(v)
                                   for v in values)))
                    self.tables["tab:s_attack_cfg"].setdefault(
                        dataset, dict())[name] = dict(zip(configs,
                                                          values))
            self.lines.append(r"\hline")
            self.lines.append(r"\end{tabular}")
            self.lines.append(self.style.float_end())
            self.lines.append("")

    # ----- ROC-AUC per seed ------------------------------------------ #
    def _seed_tables(self) -> None:
        """
        Emit the two per-seed ROC-AUC tables, A and B side by side.

        The two seeds are read from ``cells.csv`` and not averaged. The
        seed-noise reference of the paper is pooled over the stochastic
        configurations of the stream, as the caption says; the spread of
        one cell is not that reference.
        """
        cells = list()
        datasets = list()
        self.tables["tab:s_seeds"] = dict()
        for suffix, start, stop, group, closing in SEED_GROUPS:
            datasets = self.data.naming.datasets[start:stop]
            self.lines.append(self.style.float_begin(wide=False))
            self.lines.append(
                r"\caption{" + CAPTION_SEEDS % (group, closing)
                + r"}\label{tab:s_seeds_%s}" % suffix)
            self.lines.append(r"\begin{tabular}{l"
                              + "c" * len(datasets) + "}")
            self.lines.append(r"\hline")
            self.lines.append(self.style.header("Configuration",
                                            datasets))
            self.lines.append(r"\hline")
            for config in self.data.naming.configurations:
                cells = [self._seed_cell(d, config) for d in datasets]
                self.lines.append(
                    self.style.row(self.data.config_label(config))
                    + " & " + " & ".join(cells) + r"\\")
            self.lines.append(r"\hline")
            self.lines.append(r"\end{tabular}")
            self.lines.append(self.style.float_end(wide=False))
            self.lines.append("")

    def _seed_cell(self, dataset: str, config: str) -> str:
        """
        One cell of the per-seed tables, and its entry in the register.

        :param dataset: Dataset id of the column, such as "D8".
        :type dataset: str
        :param config: Configuration id of the row, such as "C7".
        :type config: str
        :return: The two seeds as "0.996/0.994", or the missing marker
            when either of them does not parse as a number, in which
            case the cell is left out of the register as well.
        :rtype: str
        """
        cell = ""
        row = self.data.cell(dataset, config)
        seed_a, seed_b = row.get("auroc_FUS_A"), row.get("auroc_FUS_B")
        try:
            cell = "%.3f/%.3f" % (float(seed_a), float(seed_b))
        except (TypeError, ValueError):
            return self.style.missing
        self.tables["tab:s_seeds"].setdefault(dataset, dict())[config] \
            = [float(seed_a), float(seed_b)]
        return cell

    # ----- pre-specified sensitivities ------------------------------- #
    def _sensitivity_table(self) -> None:
        """
        Emit the pre-specified sensitivity analyses, a row per stream.

        Six columns: the change in ROC-AUC and in F1 at each of the two
        calibration levels, compared against the reference level inside
        the same analysis, and the change in F1 at each of the two threshold
        percentiles, compared against the calibrated threshold of the
        main protocol. Every cell is a median over the configurations.
        """
        values = list()
        levels = self.data.warmup_rows()
        self.lines.append(self.style.float_begin())
        self.lines.append(r"\caption{" + CAPTION_SENS
                          + r"}\label{tab:s_sens}")
        self.lines.append(r"\begin{tabular}{l rr rr rr}")
        self.lines.append(r"\hline")
        self.lines.append(
            r" & \multicolumn{2}{c}{\textbf{Calibration 5\%}} & "
            r"\multicolumn{2}{c}{\textbf{Calibration 20\%}} & "
            r"\multicolumn{2}{c}{\textbf{Threshold P95 / P99.5}}\\")
        self.lines.append(
            r"\textbf{Stream} & $\Delta$AUC & $\Delta$F1 & $\Delta$AUC "
            r"& $\Delta$F1 & $\Delta$F1 & $\Delta$F1\\")
        self.lines.append(r"\hline")
        self.tables["tab:s_sens"] = dict()
        for dataset in self.data.naming.datasets:
            values = self._sensitivity_row(dataset, levels)
            self.lines.append(
                self.style.stream(dataset) + " & "
                + " & ".join(self.style.signed(x) for x in values)
                + r"\\")
            self.tables["tab:s_sens"][dataset] = values
        self.lines.append(r"\hline")
        self.lines.append(r"\end{tabular}")
        self.lines.append(self.style.float_end())
        self.lines.append("")

    def _sensitivity_row(
            self,
            dataset: str,
            levels: Mapping[tuple[str, str], dict[str, Any]]
    ) -> list[Any]:
        """
        The six medians of one row of the sensitivity table.

        :param dataset: Dataset id of the row, such as "D5".
        :type dataset: str
        :param levels: (dataset, configuration) -> level -> row of the
            analysis T4-P-c, as ``SupplementaryInputs.warmup_rows`` builds
            it.
        :type levels: Mapping[tuple[str, str], dict[str, Any]]
        :return: Median change in ROC-AUC and in F1 at the first level,
            the same two at the second level, and the median change in
            F1 at each threshold percentile; None where no
            configuration contributed.
        :rtype: list[Any]
        """
        collected = defaultdict(list)
        by_level = dict()
        metrics = dict()
        keys = list()
        for config in self.data.naming.configurations:
            by_level = levels.get((dataset, config), dict())
            reference = by_level.get(self.reference_level)
            if reference:
                for level in self.levels:
                    row = by_level.get(level)
                    if not row:
                        continue
                    for key, field in (("_auc", "FUS_auroc"),
                                       ("_f1", "FUS_f1")):
                        if row.get(field) is not None \
                                and reference.get(field) is not None:
                            collected[level + key].append(
                                row[field] - reference[field])
            metrics = self.data.metrics_of(dataset, config)
            for prefix in ("q95", "q995"):
                if metrics.get("f1") is not None \
                        and metrics.get(prefix + "_f1") is not None:
                    collected[prefix].append(metrics[prefix + "_f1"]
                                             - metrics["f1"])
        keys = [self.levels[0] + "_auc", self.levels[0] + "_f1",
                self.levels[1] + "_auc", self.levels[1] + "_f1",
                "q95", "q995"]
        return [self.style.median(collected.get(k)) for k in keys]

    # ----- results per day ------------------------------------------- #
    def _per_day_table(self) -> None:
        """
        Emit the per-day results of the two CIC corpora.

        The records, the attack records and the false-positive rate are
        averaged over every cell of the day; the ROC-AUC and the
        temporal reference AUC only over the cells the plan declares
        eligible. KNAD is one continuous capture and ACI has a single
        evaluation day, so neither appears here.
        """
        collected = defaultdict(list)
        self.lines.append(self.style.float_begin())
        self.lines.append(r"\caption{" + CAPTION_PERDAY
                          + r"}\label{tab:s_perday}")
        self.lines.append(r"\begin{tabular}{llrrrrr}")
        self.lines.append(r"\hline")
        self.lines.append(
            r"\textbf{Corpus} & \textbf{Day} & \textbf{Records} & "
            r"\textbf{Attack} & \textbf{ROC-AUC} & \textbf{AUC$_t$} & "
            r"\textbf{FPR}\\")
        self.lines.append(r"\hline")
        self.tables["tab:s_perday"] = dict()
        for dataset in self.per_day_datasets:
            collected = defaultdict(list)
            for config in self.data.naming.configurations:
                for day, records in self.data.per_day.get(
                        (dataset, config), dict()).items():
                    collected[day].extend(records)
            if not collected:
                continue
            for day in self.style.days(collected):
                self._per_day_row(dataset, day, collected[day])
        self.lines.append(r"\hline")
        self.lines.append(r"\end{tabular}")
        self.lines.append(self.style.float_end())
        self.lines.append("")

    def _per_day_row(self,
                     dataset: str,
                     day: str,
                     records: Sequence[Mapping[str, Any]]) -> None:
        """
        Emit one row of the per-day table and its register entry.

        :param dataset: Dataset id of the corpus, "D8" or "D9".
        :type dataset: str
        :param day: Day as the per-day block names it, such as "Wed".
        :type day: str
        :param records: Per-cell records of that day, one per
            configuration and seed.
        :type records: Sequence[Mapping[str, Any]]
        """
        eligible = [x for x in records if x.get("eligible")]
        auroc = (self.data.mean([x["auroc_fus"] for x in eligible])
                 if eligible else None)
        auc_t = (self.data.mean([x["auc_t"] for x in eligible])
                 if eligible else None)
        fpr = self.data.mean([x["fpr"] for x in records])
        self.lines.append("%s & %s & %d & %d & %s & %s & %s\\\\" % (
            self.style.stream(dataset), escape_cell(day),
            int(self.data.mean([x["rows_E"] for x in records]) or 0),
            int(self.data.mean([x["attack_rows"] for x in records])
                or 0),
            self.style.number(auroc), self.style.number(auc_t),
            self.style.number(fpr)))
        self.tables["tab:s_perday"].setdefault(dataset, dict())[day] = {
            "auroc_fus": auroc, "auc_t": auc_t, "fpr": fpr,
            "n_eligible_cells": len(eligible), "n_cells": len(records)}

    # ----- assembly -------------------------------------------------- #
    def build(self) -> list[str]:
        """
        Build the fifteen tables, in the order of the supplement.

        The two registers are reset first, so calling it twice gives
        the same result.

        :return: The LaTeX lines, before the caption pass.
        :rtype: list[str]
        """
        self.lines = list()
        self.tables = dict()
        for title, label, field in GRID_TABLES:
            if "{loc_blocks}" in title:
                title = title.replace("{loc_blocks}",
                                      self._loc_blocks_text())
            self._grid_table(title, label, field)
        self._attack_table()
        self._attack_config_tables()
        self._seed_tables()
        self._sensitivity_table()
        self._per_day_table()
        return self.lines

    def tex_text(self) -> str:
        """
        The whole LaTeX of the supplement, captions already moved down.

        :return: The nine floats, separated by blank lines and with the
            caption of each one below its own tabular.
        :rtype: str
        """
        return caption_below("\n".join(self.lines), logger=self.logger)

    def __repr__(self) -> str:
        """
        Show how much was built and from where.

        :return: Text like "SupplementaryTables(tables=8, lines=214,
            aggregate='aggregate-prov-2102')".
        :rtype: str
        """
        return (f"SupplementaryTables(tables={len(self.tables)}, "
                f"lines={len(self.lines)}, "
                f"aggregate={self.data.aggregate_dir.name!r})")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
