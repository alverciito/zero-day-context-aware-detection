# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         src/kitnetzday/tables/supplementary_summaries.py          #
#   Version:      1.0.1                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Summaries the body of the paper quotes: best cell,        #
#                 threshold and calibration sensitivity, clean calibration. #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
import statistics as st
from collections.abc import Mapping
from typing import Any

from kitnetzday.tables.supplementary_inputs import (
                                                 SUPPL_DATASET_SHORT,
                                                 SupplementaryInputs,
)
from kitnetzday.utils.formatting import get
from kitnetzday.utils.loggable import Loggable

# Why the calibration fraction is compared against its own reference
# level; written verbatim into the JSON document of the supplement.
CALIBRATION_NOTE: str = (
    "the 10 % level of T4-P-c is the reference inside T4-P-c itself "
    "(the main protocol calibrates on the first 10 % of the execution "
    "stream after deduplication, which is not the same window)")
# Direction of the differences of the clean-calibration analysis, as
# frozen/runner/metrics_rev.py stores them (diff_natural_minus_clean).
CLEAN_NOTE: str = (
    "every difference is natural minus clean: a positive value means "
    "the metric is HIGHER with the natural calibration window than with "
    "the window cleaned of attack records; k_pos counts those")
# Pairs compared in the complementarity summary of the failure cases: each
# context-free control against the autoregressive recurrent model.
COMPLEMENTARITY_PAIRS: tuple[tuple[str, str], ...] = (("C11", "C5"),
                                                      ("C9", "C5"))
# Fewest records an attack type needs to enter that summary, and the
# recall from which a configuration counts as detecting it.
COMPLEMENTARITY_MIN_ROWS: int = 1000
COMPLEMENTARITY_RECALL: float = 0.5
# Detectors of the clean-calibration counterfactual, in report order.
DETECTORS: tuple[str, ...] = ("FUS", "KMD_f", "MVD_f")
# Threshold-percentile sensitivities: label of the summary -> prefix of
# the metric keys collected by SupplementaryInputs.
_PERCENTILES: tuple[tuple[str, str], ...] = (("Q95", "q95"),
                                             ("Q995", "q995"))


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                MAIN CLASS                                 #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class SupplementarySummaries(Loggable):
    """
    The numbers of the supplement that the body of the paper quotes,
    and one summary that it does not quote.

    Five summaries are computed, in this order, from what
    ``SupplementaryInputs`` read, and nothing else:

    - ``best``: the configuration with the highest mean ROC-AUC of each
      stream. It is not an oracle claim and is declared as descriptive.
    - ``threshold``: how much the calibrated F1 and the WindowDiff move
      when the threshold percentile goes from P99 to P95 or P99.5. The
      percentile cannot move the ROC-AUC, which is
      threshold-independent, so ROC-AUC is not reported.
    - ``calibration``: how much ROC-AUC and F1 move when the calibration
      fraction goes from 10% to 5% or 20%. T4-P-c is compared against
      itself, because the main protocol calibrates on the first 10% of
      the execution stream after deduplication, which is not the same
      window as the one T4-P-c builds; mixing the two would mix two
      definitions.
    - ``clean``: the clean-calibration counterfactual T4-P-a, per
      detector. The differences are natural minus clean, as the frozen
      metric code stores them.
    - ``complementarity``: per stream, which attack types each of two
      configurations detects (recall of at least 0.5) and the other
      does not, over the per-configuration recalls of
      ``tab:s_attack_cfg``.

    :param data: Inputs of the supplement, already read.
    :type data: SupplementaryInputs
    :param dataset_short: Dataset id -> short label, for the report
        lines; the supplement labels by default.
    :type dataset_short: Mapping[str, str]
    :param reference_level: Calibration level, as a string, that the
        other levels of T4-P-c are compared against; "10" in the plan.
    :type reference_level: str
    :param fallback_config: Configuration whose jobs are read when the
        block T4-P-a does not carry its two counts; "C9" by default.
    :type fallback_config: str
    :param logger: Logger created by the calling script, or None to stay
        silent.
    :type logger: logging.Logger | None
    :raises TypeError: If data or logger are of the wrong type.
    :raises ValueError: If reference_level or fallback_config is not a
        non-empty string.
    """

    def __init__(self,
                 data: SupplementaryInputs,
                 dataset_short: Mapping[str, str] = SUPPL_DATASET_SHORT,
                 reference_level: str = "10",
                 fallback_config: str = "C9",
                 logger: logging.Logger | None = None) -> None:
        """
        Compute the four summaries, in the order the original did.

        :param data: Inputs of the supplement, already read.
        :type data: SupplementaryInputs
        :param dataset_short: Dataset id -> short label.
        :type dataset_short: Mapping[str, str]
        :param reference_level: Reference calibration level, as a string.
        :type reference_level: str
        :param fallback_config: Configuration read when the block
            lacks its counts.
        :type fallback_config: str
        :param logger: Logger created by the calling script, or None.
        :type logger: logging.Logger | None
        :raises TypeError: If data or logger are of the wrong type.
        :raises ValueError: If reference_level or fallback_config is not
            a non-empty string.
        """
        super().__init__(logger=logger)
        if not isinstance(data, SupplementaryInputs):
            self._error("[SupplementarySummaries.__init__] Input error: "
                        "the parameter data must be a "
                        "SupplementaryInputs, not "
                        f"{type(data).__name__}.", TypeError)
        for name, value in (("reference_level", reference_level),
                            ("fallback_config", fallback_config)):
            if not isinstance(value, str) or not value:
                self._error("[SupplementarySummaries.__init__] Input "
                            f"error: the parameter {name} must be a "
                            f"non-empty string, not {value!r}.")
        self.data = data
        self.dataset_shorts = dict(dataset_short)
        self.reference_level = str(reference_level)
        self.fallback_config = str(fallback_config)
        self.best = self._best_per_dataset()
        self.threshold = self._threshold_percentile()
        self.calibration = self._calibration_fraction()
        self.clean = self._clean_calibration()
        self.complementarity = self._complementarity()

    # ----- the five summaries ---------------------------------------- #
    def _best_per_dataset(self) -> dict[str, dict[str, Any]]:
        """
        Configuration with the highest mean ROC-AUC of each stream.

        Note: ties are broken by comparing the configuration ids as
        strings, because the original took the maximum of the
        (value, configuration) pairs; with two equal ROC-AUC values the
        larger id wins ("C9" over "C11", since the comparison is textual
        and not numeric).

        :return: Dataset id -> {"config", "name", "auroc"}; a stream
            with no ROC-AUC at all is left out.
        :rtype: dict[str, dict[str, Any]]
        """
        best = dict()
        candidates = list()
        for dataset in self.data.naming.datasets:
            candidates = [
                (self.data.metrics_of(dataset, config).get("auroc"),
                 config)
                for config in self.data.naming.configurations
                if self.data.metrics_of(dataset,
                                        config).get("auroc") is not None]
            if candidates:
                value, config = max(candidates)
                best[dataset] = {
                    "config": config,
                    "name": self.data.config_label(config),
                    "auroc": value}
        return best

    def _threshold_percentile(self) -> dict[str, dict[str, Any]]:
        """
        Effect of the threshold percentile on the F1 and the WindowDiff.

        Every cell of the analysis set contributes one difference per
        percentile: the sensitivity value minus the value at the
        calibrated threshold of the main protocol.

        :return: "Q95" and "Q995" -> {"d_f1", "d_wd"}, each a block with
            the count, the median, the minimum and the maximum.
        :rtype: dict[str, dict[str, Any]]
        """
        sens = {"Q95": {"d_f1": list(), "d_wd": list()},
                "Q995": {"d_f1": list(), "d_wd": list()}}
        base_f1 = None
        base_wd = None
        for value in self.data.metrics.values():
            base_f1, base_wd = value.get("f1"), value.get("wd")
            for label, prefix in _PERCENTILES:
                if base_f1 is not None \
                        and value.get(prefix + "_f1") is not None:
                    sens[label]["d_f1"].append(
                        value[prefix + "_f1"] - base_f1)
                if base_wd is not None \
                        and value.get(prefix + "_wd") is not None:
                    sens[label]["d_wd"].append(
                        value[prefix + "_wd"] - base_wd)
        for value in sens.values():
            for key in ("d_f1", "d_wd"):
                value[key] = self._spread(value[key])
        return sens

    def _calibration_fraction(self) -> dict[str, dict[str, Any]]:
        """
        Effect of the calibration fraction, within T4-P-c.

        Every (stream, configuration) of it that has the reference
        level contributes one difference per other level. The number of
        levels that got worse is counted as well, because that is the
        claim the body makes.

        :return: Level as a string -> {"d_auroc", "d_f1"}, each a block
            with the count, the median, the minimum, the maximum and
            how many differences are negative.
        :rtype: dict[str, dict[str, Any]]
        """
        warm = dict()
        block = dict()
        for rows in self.data.warmup_rows().values():
            reference = rows.get(self.reference_level)
            if not reference:
                continue
            for level, row in rows.items():
                if level == self.reference_level:
                    continue
                block = warm.setdefault(
                    level, {"d_auroc": list(), "d_f1": list()})
                for key, field in (("d_auroc", "FUS_auroc"),
                                   ("d_f1", "FUS_f1")):
                    if row.get(field) is not None \
                            and reference.get(field) is not None:
                        block[key].append(row[field]
                                          - reference[field])
        for value in warm.values():
            for key in ("d_auroc", "d_f1"):
                value[key] = self._spread(value[key], negatives=True)
        return warm

    def _clean_calibration(self) -> dict[str, dict[str, Any]]:
        """
        Clean-calibration counterfactual T4-P-a, detector by detector.

        The block reports, per stream, how many records of the
        calibration window were clean and how contaminated that window
        naturally was. When it does not carry those two counts, they are
        recovered from the ``t4p.a`` block of the jobs of the fallback
        configuration.

        Note: the recovery loop keeps the first value it finds for each
        of the two counts, and it only runs when ``n_clean`` is missing,
        so a block with ``n_clean`` present and the contamination
        missing keeps the contamination missing.

        :return: Dataset id -> {"n_clean",
            "natural_calib_contamination"} plus one block per detector
            with the count, the median change in ROC-AUC, its minimum
            and maximum, the median change in F1 and how many
            configurations improved.
        :rtype: dict[str, dict[str, Any]]
        """
        summary = dict()
        configs = dict()
        for dataset, block in \
                self.data.clean_calibration_block().items():
            configs = get(block, "configs", default=dict())
            n_clean = block.get("n_clean")
            contamination = block.get("natural_calib_contamination")
            if n_clean is None:
                n_clean, contamination = self._recover_counts(
                    dataset, n_clean, contamination)
            summary[dataset] = {
                "n_clean": n_clean,
                "natural_calib_contamination": contamination}
            for detector in DETECTORS:
                summary[dataset][detector] = self._detector_block(
                    configs, detector)
        return summary

    def _recover_counts(self,
                        dataset: str,
                        n_clean: Any,
                        contamination: Any) -> tuple[Any, Any]:
        """
        Read the two counts of T4-P-a from the jobs of one stream.

        :param dataset: Dataset id of the record, such as "D9".
        :type dataset: str
        :param n_clean: Count read from the block, None when missing.
        :type n_clean: Any
        :param contamination: Natural contamination read from the
            block, None when missing.
        :type contamination: Any
        :return: The two counts, each replaced by the value of the first
            job that carries it.
        :rtype: tuple[Any, Any]
        """
        record = dict()
        for job_id in self.data.job_ids(dataset, self.fallback_config):
            record = get(self.data.job_metrics(job_id), "t4p", "a",
                         default=dict())
            n_clean = (n_clean if n_clean is not None
                       else record.get("n_clean"))
            contamination = (
                contamination if contamination is not None
                else record.get("natural_calib_contamination"))
        return n_clean, contamination

    def _detector_block(self,
                        configs: Mapping[str, Any],
                        detector: str) -> dict[str, Any]:
        """
        Spread of one detector over the configurations of T4-P-a.

        :param configs: Configuration -> its record, each with
            the change in ROC-AUC and in F1 of every detector.
        :type configs: Mapping[str, Any]
        :param detector: Detector name, "FUS", "KMD_f" or "MVD_f".
        :type detector: str
        :return: Block with the count of ROC-AUC changes, their median,
            minimum and maximum, the median change in F1 and how many
            ROC-AUC changes are strictly positive.
        :rtype: dict[str, Any]
        """
        aurocs = [c.get(detector + "_auroc") for c in configs.values()
                  if c.get(detector + "_auroc") is not None]
        f1s = [c.get(detector + "_f1") for c in configs.values()
               if c.get(detector + "_f1") is not None]
        return {"n": len(aurocs),
                "median_d_auroc": st.median(aurocs) if aurocs else None,
                "min": min(aurocs) if aurocs else None,
                "max": max(aurocs) if aurocs else None,
                "median_d_f1": st.median(f1s) if f1s else None,
                "k_pos": sum(1 for x in aurocs if x > 0)}

    def _spread(self,
                values: list[Any],
                negatives: bool = False) -> dict[str, Any]:
        """
        Count, median, minimum and maximum of a list of differences.

        :param values: Differences collected for one summary; an empty
            list gives None in every statistic.
        :type values: list[Any]
        :param negatives: Whether to add ``k_neg``, how many
            differences are strictly negative; the calibration fraction
            reports it, the threshold percentile does not.
        :type negatives: bool
        :return: The block as the JSON summary stores it.
        :rtype: dict[str, Any]
        """
        block = {"n": len(values),
                 "median": st.median(values) if values else None,
                 "min": min(values) if values else None,
                 "max": max(values) if values else None}
        if negatives:
            block["k_neg"] = sum(1 for x in values if x < 0)
        return block

    def _complementarity(self) -> dict[str, dict[str, Any]]:
        """
        Which attack types one configuration detects and the other not.

        For every stream and every pair of ``COMPLEMENTARITY_PAIRS``
        the attack types with at least ``COMPLEMENTARITY_MIN_ROWS``
        records and a recall in both configurations are split in four:
        detected by the first only, by the second only, by both, by
        neither. A stream without such a type is left out.

        :return: "<dataset>_<a>_vs_<b>" -> {"n_types", "only_<a>",
            "only_<b>", "both", "neither", "max_abs_diff",
            "median_diff", "detail"}, the differences being recall of
            the first minus recall of the second.
        :rtype: dict[str, dict[str, Any]]
        """
        out = dict()
        cut = COMPLEMENTARITY_RECALL
        recall_a = dict()
        recall_b = dict()
        common = list()
        for dataset in self.data.naming.datasets:
            types = self.data.attack_recalls.get(dataset, dict())
            rows = self.data.attack_rows.get(dataset, dict())
            for first, second in COMPLEMENTARITY_PAIRS:
                recall_a = self._recalls(dataset, first, types)
                recall_b = self._recalls(dataset, second, types)
                common = [t for t in recall_a
                          if recall_a[t] is not None
                          and recall_b[t] is not None
                          and (rows.get(t) or 0)
                          >= COMPLEMENTARITY_MIN_ROWS]
                if not common:
                    continue
                out[f"{dataset}_{first}_vs_{second}"] = {
                    "n_types": len(common),
                    f"only_{first}": sorted(
                        t for t in common
                        if recall_a[t] >= cut and recall_b[t] < cut),
                    f"only_{second}": sorted(
                        t for t in common
                        if recall_b[t] >= cut and recall_a[t] < cut),
                    "both": sorted(
                        t for t in common
                        if recall_a[t] >= cut and recall_b[t] >= cut),
                    "neither": sorted(
                        t for t in common
                        if recall_a[t] < cut and recall_b[t] < cut),
                    "max_abs_diff": max(abs(recall_a[t] - recall_b[t])
                                        for t in common),
                    "median_diff": st.median(
                        [recall_a[t] - recall_b[t] for t in common]),
                    "detail": {t: {first: recall_a[t],
                                   second: recall_b[t]}
                               for t in common}}
        return out

    def _recalls(self,
                 dataset: str,
                 config: str,
                 types: Mapping[str, Any]) -> dict[str, Any]:
        """
        Recall of one configuration on every attack type of a stream.

        :param dataset: Dataset id, such as "D8".
        :type dataset: str
        :param config: Configuration id, such as "C5".
        :type config: str
        :param types: Attack types of the stream, in reading order.
        :type types: Mapping[str, Any]
        :return: Attack type -> mean recall over the seeds, None when
            the configuration has no value for it.
        :rtype: dict[str, Any]
        """
        cell = self.data.attack_recalls_by_cell.get((dataset, config),
                                                    dict())
        return {t: self.data.mean(cell.get(t, list())) for t in types}

    # ----- the JSON document and the console report ------------------ #
    def document(self,
                 tables: Mapping[str, Any],
                 source: str) -> dict[str, Any]:
        """
        Assemble the JSON document of the supplement.

        The key order is the order the original script of the
        revision inserted the keys in.

        :param tables: Label of each table -> its values, in the order
            the tables were emitted.
        :type tables: Mapping[str, Any]
        :param source: Directory of the aggregated pass the numbers come
            from, written as the provenance of the document.
        :type source: str
        :return: The whole document, ready to be dumped with indent 1.
        :rtype: dict[str, Any]
        :raises TypeError: If tables is not a mapping or source is not a
            string.
        """
        if not isinstance(tables, Mapping):
            self._error("[SupplementarySummaries.document] Input error: "
                        "the parameter tables must be a mapping, not "
                        f"{type(tables).__name__}.", TypeError)
        if not isinstance(source, str):
            self._error("[SupplementarySummaries.document] Input error: "
                        "the parameter source must be a string, not "
                        f"{type(source).__name__}.", TypeError)
        return {"source": source,
                "tables": dict(tables),
                "best_per_dataset": self.best,
                "sensitivity_threshold_percentile": self.threshold,
                "sensitivity_calibration_fraction": self.calibration,
                "sensitivity_calibration_fraction_note":
                    CALIBRATION_NOTE,
                "T4P_a_clean_calibration": self.clean,
                "T4P_a_clean_calibration_note": CLEAN_NOTE,
                "complementarity_6_7": self.complementarity}

    def report_lines(self) -> list[str]:
        """
        Console report of the first four summaries, line by line.

        The script in ``bin/`` is the one that prints them, because
        library code does not print.

        Note: a summary whose median is None raises a TypeError here,
        because without that median the report line cannot be written.

        :return: The report lines, blank lines included, in the order
            the original printed them.
        :rtype: list[str]
        :raises TypeError: If a summary that is quoted has no median,
            that is, when no cell contributed to it.
        """
        lines = list()
        lines.append("")
        lines.append("### threshold percentile (delta against P99; "
                     "ROC-AUC does not depend on the threshold)")
        for label, value in self.threshold.items():
            lines.append(
                f"  {label}: F1 median {value['d_f1']['median']:+.4f} "
                f"[{value['d_f1']['min']:+.4f},"
                f"{value['d_f1']['max']:+.4f}]"
                f" n={value['d_f1']['n']} | WD median "
                f"{value['d_wd']['median']:+.4f} "
                f"[{value['d_wd']['min']:+.4f},"
                f"{value['d_wd']['max']:+.4f}]")
        lines.append("")
        lines.append("### calibration fraction (delta against the "
                     "10 % level of T4-P-c itself)")
        for level, value in sorted(self.calibration.items(),
                                   key=lambda kv: int(kv[0])):
            lines.append(
                f"  {level}%: ROC-AUC median "
                f"{value['d_auroc']['median']:+.4f} "
                f"[{value['d_auroc']['min']:+.4f},"
                f"{value['d_auroc']['max']:+.4f}] "
                f"n={value['d_auroc']['n']} | F1 median "
                f"{value['d_f1']['median']:+.4f} "
                f"[{value['d_f1']['min']:+.4f},"
                f"{value['d_f1']['max']:+.4f}] "
                f"n={value['d_f1']['n']}")
        lines.append("")
        lines.append("### T4-P-a clean calibration (differences "
                     "are natural minus clean)")
        lines.extend(self._clean_lines())
        lines.append("")
        lines.append("### best configuration per dataset "
                     "(descriptive)")
        for dataset, value in self.best.items():
            lines.append(
                f"  {self.dataset_shorts.get(dataset, dataset):6s} "
                f"{value['name']:20s} {value['auroc']:.4f}")
        return lines

    def _clean_lines(self) -> list[str]:
        """
        Console report of the clean-calibration counterfactual.

        :return: One line per stream and one per detector under it.
        :rtype: list[str]
        :raises TypeError: If a detector of a stream has no median.
        """
        lines = list()
        contamination = None
        shown = None
        for dataset, value in self.clean.items():
            contamination = value["natural_calib_contamination"]
            shown = (contamination if contamination is None
                     else round(contamination, 4))
            lines.append(f"  {dataset}: n_clean={value['n_clean']} "
                         f"natural contamination {shown}")
            for detector in DETECTORS:
                block = value[detector]
                lines.append(
                    f"    {detector}: median dROC "
                    f"{block['median_d_auroc']:+.4f} "
                    f"[{block['min']:+.4f},{block['max']:+.4f}] "
                    f"positive in {block['k_pos']}/{block['n']} · "
                    f"median dF1 {block['median_d_f1']:+.4f}")
        return lines

    def __repr__(self) -> str:
        """
        Show how many streams each summary covers.

        :return: Text like "SupplementarySummaries(best=9,
            percentiles=2, levels=2, clean=1)".
        :rtype: str
        """
        return (f"SupplementarySummaries(best={len(self.best)}, "
                f"percentiles={len(self.threshold)}, "
                f"levels={len(self.calibration)}, "
                f"clean={len(self.clean)}, "
                f"complementarity={len(self.complementarity)})")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
