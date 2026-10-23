# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         src/kitnetzday/audit/retrained_pair.py                    #
#   Version:      1.1.2                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Paired comparison of the windowed model against the       #
#                 per-record one over the retraining experiment.            #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from kitnetzday.audit.paired_stats import EXACT_MAX_N, HL_ALPHA, PairedStats
from kitnetzday.audit.retrained_units import UnitSummaries
from kitnetzday.utils.loggable import Loggable

# Variant pairs that are read: (windowed variant, per-record control).
# The first pair is the retraining experiment of the paper (one epoch)
# and the second one is its five-epoch ablation. They are reported side
# by side but never pooled: the epoch budget differs.
PAIRS: tuple[tuple[str, str], ...] = (
    ("mlp_ctx_fix", "mlp_fix"), ("mlp_ctx_fix_e5", "mlp_fix_e5"))
# Dataset directory name -> label printed in the report. Note: these
# strings are deliberately not routed through Naming.dataset_label,
# which spells the first one "ACI-IoT-2023" instead of "ACI-IOT-2023";
# the tables of the experiment were produced with the spellings below.
DATASET_NAMES: dict[str, str] = {
    "aci-iot-2023": "ACI-IOT-2023", "cic2017": "CIC-IDS2017",
    "cic2018": "CSE-CIC-IDS2018"}
# Prefix of the KNAD dataset directories, whose label is built from the
# directory name instead of being looked up.
KNAD_PREFIX: str = "knad-"
# Primary metric of the frozen plan, written verbatim into the JSON:
# the ROC-AUC of the fused detector on the evaluation segment E.
PRIMARY_METRIC: str = ("primary.value (ROC-AUC of FUS on E, plan "
                       "section 3.1)")
# Metrics files of the experiment, relative to the results root.
BLOCK_GLOB: str = "T8-*/metrics.json"
# Paradigm halves reported separately: label -> value of the "ar" flag.
PARADIGMS: tuple[tuple[str, bool], ...] = (("TSR", False), ("AR", True))


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                MAIN CLASS                                 #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class RetrainedPair(Loggable):
    """
    Paired report of the retraining experiment: windowed vs per-record.

    The experiment answers whether giving the autoencoder a window of
    records helps, once both models are retrained under the same budget.
    The comparison has to be paired, because the two variants of a cell
    share the dataset, the paradigm, the seed and the repetition, and
    those factors move the metric far more than the window does. This
    class reads the ``metrics.json`` written by the jobs of the
    ``T8-*`` blocks, pairs the two variants cell by cell, and summarizes
    the differences with the exact statistics of ``PairedStats``: wins,
    median, Hodges-Lehmann estimate with its inverted Wilcoxon interval,
    the signed rank p value, the two paradigm halves, the cells whose
    sign flips with the seed, and the per (dataset, paradigm) averages.

    It only reads the files found under the results root. It never
    trains, never touches the GPU and never writes inside that root:
    the JSON goes wherever the calling script asks for it. The printed
    text of the report lives in ``RetrainedPairReport``, so that this
    class holds no formatting.

    The primary metric is the one of the frozen plan, ``primary.value``
    in each ``metrics.json`` (the ROC-AUC of the fused detector on the
    evaluation window), and no other field of those files is read, so a
    job whose metric is missing is skipped instead of imputed.

    :param results_root: Directory holding the job directories of the
        experiment, ``data/jobs`` in this package.
    :type results_root: str | Path
    :param repo_root: Repository root, used to write the pointer of
        every cell as a path relative to it.
    :type repo_root: str | Path
    :param pairs: Variant pairs to report, each one as (windowed
        variant, per-record control); non-empty.
    :type pairs: Sequence[tuple[str, str]]
    :param dataset_names: Dataset directory name -> printed label, for
        the datasets that are not KNAD.
    :type dataset_names: Mapping[str, str]
    :param alpha: Two-sided alpha of the Hodges-Lehmann interval, in
        (0, 1).
    :type alpha: float
    :param exact_max_n: Largest sample size for which the signed rank p
        value is enumerated exactly; >= 0.
    :type exact_max_n: int
    :param metric_label: Text written in the ``metric`` field of the
        report, naming the primary metric.
    :type metric_label: str
    :param block_glob: Glob of the metrics files, relative to the
        results root.
    :type block_glob: str
    :param logger: Logger created by the calling script, or None to stay
        silent.
    :type logger: logging.Logger | None
    :raises TypeError: If logger is not a logging.Logger or None, or if
        the paths, the labels or the globs are of the wrong type.
    :raises ValueError: If pairs is empty or holds something that is not
        a pair of two non-empty variant names.
    """

    def __init__(self,
                 results_root: str | Path,
                 repo_root: str | Path,
                 pairs: Sequence[tuple[str, str]] = PAIRS,
                 dataset_names: Mapping[str, str] = DATASET_NAMES,
                 alpha: float = HL_ALPHA,
                 exact_max_n: int = EXACT_MAX_N,
                 metric_label: str = PRIMARY_METRIC,
                 block_glob: str = BLOCK_GLOB,
                 logger: logging.Logger | None = None) -> None:
        """
        Check the settings, store them and build the statistics helper.

        :param results_root: Directory holding the job directories.
        :type results_root: str | Path
        :param repo_root: Repository root for the relative pointers.
        :type repo_root: str | Path
        :param pairs: Variant pairs (windowed, per-record); non-empty.
        :type pairs: Sequence[tuple[str, str]]
        :param dataset_names: Directory name -> printed label.
        :type dataset_names: Mapping[str, str]
        :param alpha: Two-sided alpha of the interval, in (0, 1).
        :type alpha: float
        :param exact_max_n: Threshold of the exact enumeration, >= 0.
        :type exact_max_n: int
        :param metric_label: Text of the ``metric`` field.
        :type metric_label: str
        :param block_glob: Glob of the metrics files.
        :type block_glob: str
        :param logger: Logger created by the calling script, or None.
        :type logger: logging.Logger | None
        :raises TypeError: If a parameter is of the wrong type.
        :raises ValueError: If pairs is empty or malformed.
        """
        super().__init__(logger=logger)
        for name, value in (("results_root", results_root),
                            ("repo_root", repo_root)):
            if not isinstance(value, (str, Path)) or not str(value):
                self._error("[RetrainedPair.__init__] Input error: the "
                            f"parameter {name} must be a non-empty str "
                            f"or Path, not {value!r}.", TypeError)
        if not isinstance(dataset_names, Mapping):
            self._error("[RetrainedPair.__init__] Input error: the "
                        "parameter dataset_names must be a mapping, not "
                        f"{type(dataset_names).__name__}.", TypeError)
        if not isinstance(metric_label, str) or \
                not isinstance(block_glob, str) or not block_glob:
            self._error("[RetrainedPair.__init__] Input error: the "
                        "parameters metric_label and block_glob must be "
                        "strings and block_glob must not be empty, not "
                        f"{metric_label!r} and {block_glob!r}.",
                        TypeError)
        self.pairs = self._as_pairs(pairs)
        self.results_root = Path(results_root)
        self.repo_root = Path(repo_root)
        self.dataset_names = dict(dataset_names)
        self.metric_label = metric_label
        self.block_glob = block_glob
        self.stats = PairedStats(alpha=alpha, exact_max_n=exact_max_n,
                                 logger=logger)
        self.cells: dict[tuple[Any, ...], dict[str, Any]] = dict()

    def dataset_label(self, dataset_key: str) -> str:
        """
        Printed label of a dataset directory name.

        The six KNAD subsets are named after their directory, so their
        label is built by capitalizing every underscore-separated word
        and prefixing "KNAD:"; the three flow datasets come from the
        lookup table. An unknown name is returned unchanged, so a new
        dataset shows up in the report instead of disappearing.

        :param dataset_key: Directory name of the dataset, for example
            "cic2017" or "knad-active_wiretap".
        :type dataset_key: str
        :return: The label, for example "CIC-IDS2017" or
            "KNAD:Active_Wiretap".
        :rtype: str
        :raises TypeError: If dataset_key is not a string.
        """
        if not isinstance(dataset_key, str):
            self._error("[RetrainedPair.dataset_label] Input error: the "
                        "parameter dataset_key must be a string, not "
                        f"{type(dataset_key).__name__}.", TypeError)
        if dataset_key.startswith(KNAD_PREFIX):
            return "KNAD:" + "_".join(
                w.capitalize()
                for w in dataset_key[len(KNAD_PREFIX):].split("_"))
        return self.dataset_names.get(dataset_key, dataset_key)

    def load_cells(self) -> dict[tuple[Any, ...], dict[str, Any]]:
        """
        Read the primary metric of every job found, cell by cell.

        A cell is everything that the two variants of a pair share:
        (dataset, paradigm, seed, repetition). The files are visited in
        sorted order, so a cell that somehow holds two jobs of the same
        variant keeps the last one read.

        A file that cannot be parsed, or whose ``primary.value`` is
        missing, is skipped: one unreadable file does not stop the
        report, and no value is imputed for it.

        :return: (dataset, ar, seed, rep) -> variant -> {"job_id",
            "auc", "metric", "path"}, with the path relative to the
            repository root.
        :rtype: dict[tuple[Any, ...], dict[str, Any]]
        :raises ValueError: If a metrics file lies outside the
            repository root, so its relative pointer cannot be written.
        """
        out: dict[tuple[Any, ...], dict[str, Any]] = dict()
        metrics: Any = None
        job: Any = None
        value: Any = None
        key: tuple[Any, ...] = tuple()
        if not self.results_root.is_dir():
            self._warning("[RetrainedPair.load_cells] Input warning: "
                          "the parameter results_root must be an "
                          f"existing directory, and {self.results_root}"
                          " is not. No cell is loaded.")
            self.cells = out
            return out
        for f in sorted(self.results_root.glob(self.block_glob)):
            try:
                metrics = json.loads(f.read_text())
            except Exception as exc:  # noqa: BLE001
                self._warning("[RetrainedPair.load_cells] Input "
                              "warning: every metrics file must be "
                              f"readable JSON, and {f} is not "
                              f"({type(exc).__name__}: {exc}). The file "
                              "is skipped.")
                continue
            job = metrics.get("job", dict())
            value = (metrics.get("primary") or dict()).get("value")
            if value is None:
                continue
            key = (job.get("dataset"), bool(job.get("ar")),
                   job.get("seed"), job.get("rep"))
            out.setdefault(key, dict())[job.get("variant")] = {
                "job_id": job.get("job_id"), "auc": float(value),
                "metric": (metrics.get("primary") or dict()).get("name"),
                "path": str(f.relative_to(self.repo_root))}
        self.cells = out
        return out

    def paired_rows(self,
                    cells: Mapping[tuple[Any, ...], dict[str, Any]],
                    windowed: str,
                    control: str) -> list[dict[str, Any]]:
        """
        One row per cell in which both variants of a pair are present.

        Cells holding only one of the two variants are left out
        silently: a difference needs both members of the pair.

        :param cells: Output of ``load_cells``.
        :type cells: Mapping[tuple[Any, ...], dict[str, Any]]
        :param windowed: Name of the windowed variant, for example
            "mlp_ctx_fix".
        :type windowed: str
        :param control: Name of the per-record control, for example
            "mlp_fix".
        :type control: str
        :return: Rows sorted by cell key, each one with the two ROC-AUC
            values, their difference (windowed minus control) and the
            two job ids.
        :rtype: list[dict[str, Any]]
        :raises TypeError: If cells is not a mapping or the variant
            names are not strings.
        """
        rows: list[dict[str, Any]] = list()
        if not isinstance(cells, Mapping):
            self._error("[RetrainedPair.paired_rows] Input error: the "
                        "parameter cells must be a mapping, not "
                        f"{type(cells).__name__}.", TypeError)
        if not isinstance(windowed, str) or not isinstance(control, str):
            self._error("[RetrainedPair.paired_rows] Input error: the "
                        "parameters windowed and control must be "
                        f"strings, not {windowed!r} and {control!r}.",
                        TypeError)
        for key in sorted(cells):
            got = cells[key]
            if windowed in got and control in got:
                rows.append({
                    "dataset": self.dataset_label(key[0]),
                    "dataset_key": key[0], "ar": key[1],
                    "seed": key[2], "rep": key[3],
                    "auc_ctx": got[windowed]["auc"],
                    "auc_base": got[control]["auc"],
                    "delta": (got[windowed]["auc"]
                              - got[control]["auc"]),
                    "job_ctx": got[windowed]["job_id"],
                    "job_base": got[control]["job_id"]})
        return rows

    def report(self,
               cells: Mapping[tuple[Any, ...], dict[str, Any]] | None
               = None) -> dict[str, Any]:
        """
        Full paired report of every variant pair of the experiment.

        :param cells: Cells to use instead of reading them, as returned
            by ``load_cells``, or None to read them now.
        :type cells: Mapping[tuple[Any, ...], dict[str, Any]] | None
        :return: {"metric": label, "pairs": {windowed variant -> block
            of statistics}}, where a pair without paired cells gives
            just {"n_cells": 0}.
        :rtype: dict[str, Any]
        :raises TypeError: If cells is neither a mapping nor None.
        """
        out: dict[str, Any] = {"metric": self.metric_label,
                               "pairs": dict()}
        data: Mapping[tuple[Any, ...], dict[str, Any]] = dict()
        rows: list[dict[str, Any]] = list()
        data = self.load_cells() if cells is None else cells
        for windowed, control in self.pairs:
            rows = self.paired_rows(data, windowed, control)
            if not rows:
                out["pairs"][windowed] = {"n_cells": 0}
                continue
            out["pairs"][windowed] = self._block(rows)
        return out

    def dump_json(self,
                  report: Mapping[str, Any],
                  path: str | Path) -> Path:
        """
        Write the report as JSON, with one space of indentation.

        :param report: Report returned by ``report``.
        :type report: Mapping[str, Any]
        :param path: File to write; the caller chooses it, and it is
            never inside the results root.
        :type path: str | Path
        :return: The path written.
        :rtype: Path
        :raises TypeError: If report is not a mapping or path is not a
            str or Path.
        :raises OSError: If the file cannot be written.
        """
        if not isinstance(report, Mapping):
            self._error("[RetrainedPair.dump_json] Input error: the "
                        "parameter report must be a mapping, not "
                        f"{type(report).__name__}.", TypeError)
        if not isinstance(path, (str, Path)) or not str(path):
            self._error("[RetrainedPair.dump_json] Input error: the "
                        "parameter path must be a non-empty str or "
                        f"Path, not {path!r}.", TypeError)
        Path(path).write_text(json.dumps(report, indent=1),
                             encoding="utf-8")
        return Path(path)

    def __repr__(self) -> str:
        """
        Show the roots, the pairs and the number of cells loaded.

        :return: Text like "RetrainedPair(results_root='...',
            pairs=2, cells=48)".
        :rtype: str
        """
        return (f"RetrainedPair(results_root={str(self.results_root)!r}, "
                f"repo_root={str(self.repo_root)!r}, "
                f"pairs={len(self.pairs)}, cells={len(self.cells)})")

    def _as_pairs(self,
                  pairs: Sequence[tuple[str, str]],
                  ) -> tuple[tuple[str, str], ...]:
        """
        Validate the variant pairs and freeze them into a tuple.

        :param pairs: Pairs (windowed variant, per-record control).
        :type pairs: Sequence[tuple[str, str]]
        :return: The same pairs as a tuple of two-string tuples.
        :rtype: tuple[tuple[str, str], ...]
        :raises TypeError: If pairs is not a sequence.
        :raises ValueError: If pairs is empty, or a pair does not hold
            exactly two non-empty variant names.
        """
        out: list[tuple[str, str]] = list()
        if isinstance(pairs, (str, bytes)) or \
                not isinstance(pairs, Sequence):
            self._error("[RetrainedPair.__init__] Input error: the "
                        "parameter pairs must be a sequence of (windowed"
                        f", control) pairs, not {type(pairs).__name__}.",
                        TypeError)
        if not pairs:
            self._error("[RetrainedPair.__init__] Input error: the "
                        "parameter pairs must hold at least one pair, "
                        "and it is empty.")
        for pair in pairs:
            if not isinstance(pair, Sequence) or len(pair) != 2 or \
                    not all(isinstance(v, str) and v for v in pair):
                self._error("[RetrainedPair.__init__] Input error: "
                            "every element of pairs must be two "
                            f"non-empty variant names, not {pair!r}.")
            out.append((pair[0], pair[1]))
        return tuple(out)

    def _block(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Statistics of one variant pair over its paired cells.

        The block is built in the order the manuscript reads it: how
        often the window wins, the median difference, the range, the
        Hodges-Lehmann estimate with its exact interval, the signed rank
        test, the same for each paradigm half, the cells whose sign
        flips with the seed, the average difference per (dataset,
        paradigm) with how many of those averages favour the window,
        and the same statistics at the unit of the plan (``by_unit``).

        Note: ``wins_ctx`` counts strictly positive differences and
        ``ties`` counts exact zeros, so a block with ties has
        ``wins_ctx + ties`` below ``n_cells`` and the losses are the
        remainder; the report prints the wins only, as submitted.

        :param rows: Non-empty rows of one pair, from ``paired_rows``.
        :type rows: list[dict[str, Any]]
        :return: The block of statistics, with the rows under "rows".
        :rtype: dict[str, Any]
        """
        deltas: list[float] = list()
        blk: dict[str, Any] = dict()
        sub: list[float] = list()
        sign: dict[tuple[Any, Any], list[tuple[Any, float]]] = dict()
        flip: list[dict[str, Any]] = list()
        avg: dict[tuple[Any, Any], list[float]] = dict()
        lo = 0.0
        hi = 0.0
        cov = 0.0
        wp: float | None = None
        p: float | None = None
        n_eff = 0
        deltas = [x["delta"] for x in rows]
        blk = {"n_cells": len(rows),
               "wins_ctx": sum(1 for x in deltas if x > 0),
               "ties": sum(1 for x in deltas if x == 0),
               "median": self.stats.median(deltas),
               "min": min(deltas), "max": max(deltas),
               "hl": self.stats.hodges_lehmann(deltas), "rows": rows}
        lo, hi, cov = self.stats.hl_interval(deltas)
        blk["hl_ci"] = {"lower": lo, "upper": hi, "coverage": cov}
        wp, p, n_eff = self.stats.wilcoxon_exact(deltas)
        blk["wilcoxon"] = {"W": wp, "p": p, "n_eff": n_eff}
        for label, want_ar in PARADIGMS:
            sub = [x["delta"] for x in rows if bool(x["ar"]) == want_ar]
            if sub:
                blk[label] = {
                    "n": len(sub),
                    "wins_ctx": sum(1 for v in sub if v > 0),
                    "median": self.stats.median(sub),
                    "hl": self.stats.hodges_lehmann(sub),
                    "min": min(sub), "max": max(sub)}
                lo, hi, cov = self.stats.hl_interval(sub)
                blk[label]["hl_ci"] = {"lower": lo, "upper": hi,
                                       "coverage": cov}
        for x in rows:
            sign.setdefault((x["dataset"], x["ar"]), list()).append(
                (x["seed"], x["delta"]))
        for key, seen in sorted(sign.items()):
            if len(seen) >= 2 and len({d > 0 for _, d in seen}) > 1:
                flip.append({"dataset": key[0], "ar": key[1],
                             "deltas": seen})
        blk["sign_flips"] = flip
        blk["n_groups_two_seeds"] = sum(
            1 for seen in sign.values() if len(seen) >= 2)
        for x in rows:
            avg.setdefault((x["dataset"], x["ar"]), list()).append(
                x["delta"])
        blk["per_dataset_ar_mean"] = [
            {"dataset": key[0], "ar": key[1], "n_seeds": len(vals),
             "mean_delta": sum(vals) / len(vals)}
            for key, vals in sorted(avg.items())]
        blk["wins_ctx_dataset_ar_mean"] = sum(
            1 for x in blk["per_dataset_ar_mean"] if x["mean_delta"] > 0)
        blk["by_unit"] = UnitSummaries(
            self.stats, logger=self.logger).summarize(rows)
        return blk


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
