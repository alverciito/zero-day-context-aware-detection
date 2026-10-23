# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         src/kitnetzday/tables/supplementary_inputs.py             #
#   Version:      2.0.1                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Reads the cells, the runs and the per-job metrics that    #
#                 the supplementary tables are built from.                  #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import csv
import json
import logging
import statistics as st
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from kitnetzday.naming import Naming
from kitnetzday.utils.formatting import get
from kitnetzday.utils.loggable import Loggable

# Short stream labels of the supplementary tables. Note: they are NOT
# kitnetzday.naming.DATASET_SHORT, which writes "FUZZ" and "VI": the
# supplement was typeset with "Fuzz" and "Video", and changing a label
# would change the submitted artefact.
SUPPL_DATASET_SHORT: dict[str, str] = {
    "D1": "AW", "D2": "Fuzz", "D3": "OS", "D4": "SSDP", "D5": "SSL",
    "D6": "Video", "D7": "ACI", "D8": "CIC17", "D9": "CIC18"}
# Configuration labels of the aggregate that are rewritten for the
# supplement, keyed by the long name stored in cells.csv.
SUPPL_CONFIG_RENAME: dict[str, str] = {
    "original (KitNET-style)": "KitNET-style dA",
    "MLP": "MLP (per record)"}
# Values of in_S_star that count as true in runs.csv.
_TRUE_VALUES: tuple[str, ...] = ("True", "true", "1")
# Keys of the per-seed accumulator, in the order the original appended
# them, and the pointer inside metrics.json each one is read from.
_METRIC_SOURCES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("f1", ("E", "FUS", "f1")),
    ("wd", ("E", "FUS", "windowdiff", "wd")),
    ("eer", ("E", "FUS", "eer")),
    ("auprc", ("E", "FUS", "auprc")),
    ("auroc", ("E", "FUS", "auroc")),
    ("q95_f1", ("sensitivities", "S-Q95", "FUS", "f1")),
    ("q95_wd", ("sensitivities", "S-Q95", "FUS", "windowdiff", "wd")),
    ("q995_f1", ("sensitivities", "S-Q995", "FUS", "f1")),
    ("q995_wd", ("sensitivities", "S-Q995", "FUS", "windowdiff", "wd")),
    ("auc_loc_fus", ("sensitivities", "S-LOC", "FUS", "auc_loc")),
    ("auroc_kmd", ("E", "KMD_f", "auroc")),
    ("auroc_mvd", ("E", "MVD_f", "auroc")),
    ("fpr", ("E", "FUS", "fpr")),
    ("alarm_rate", ("E", "FUS", "alarm_rate")))
# Means that the body of the paper publishes: they are taken from
# cells.csv and overwrite the mean computed here, so that the supplement
# and the body never disagree by a rounding step.
_CELL_OVERRIDES: tuple[tuple[str, str], ...] = (
    ("auroc", "auroc_FUS_mean"), ("f1", "f1_FUS_mean"),
    ("auprc", "auprc_FUS_mean"))


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                MAIN CLASS                                 #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class SupplementaryInputs(Loggable):
    """
    Everything the supplementary tables read, averaged over the seeds.

    No metric is computed here. The class reads the fields that
    ``metrics_rev.py`` already wrote for every job (``E.FUS.f1``,
    ``E.FUS.windowdiff.wd``, ``E.FUS.eer``, ``E.FUS.auprc``,
    ``E.FUS.auroc``, ``E.FUS.fpr``, ``E.FUS.alarm_rate``,
    ``E.KMD_f.auroc``, ``E.MVD_f.auroc``, ``per_attack_type``,
    ``per_day``, the ``S-Q95``, ``S-Q995`` and ``S-LOC`` sensitivities)
    and averages them over the
    two seeds of each cell, exactly as ``aggregate.py`` does. Which jobs
    belong to a cell is decided by ``runs.csv``: only the rows with
    ``in_S_star`` true, which leaves the ablations, the T4-E block and
    the exploratory blocks out.

    The means of ROC-AUC, F1 and AUPRC are then overwritten with the
    values of ``cells.csv``, because those are the ones the body of the
    paper publishes and they must not be recomputed here.

    Three directories matter: ``<aggregate>/aggregate/cells.csv``,
    ``<aggregate>/runs.csv`` and ``<aggregate>/stats.json`` for the
    aggregated pass, and ``<results_root>/<job_id>/metrics.json`` for
    the per-job records.

    :param aggregate_dir: Directory of the aggregated pass, the one that
        holds ``runs.csv``, ``stats.json`` and ``aggregate/cells.csv``.
    :type aggregate_dir: str | Path
    :param results_root: Directory with one sub-directory per job id,
        each with its ``metrics.json``.
    :type results_root: str | Path
    :param naming: Vocabulary that fixes the dataset and configuration
        order, or None for the default one of the study.
    :type naming: Naming | None
    :param logger: Logger created by the calling script, or None to stay
        silent.
    :type logger: logging.Logger | None
    :raises TypeError: If naming or logger are of the wrong type.
    :raises FileNotFoundError: If ``cells.csv``, ``runs.csv`` or
        ``stats.json`` of the aggregated pass does not exist.
    :raises json.JSONDecodeError: If ``stats.json`` is not valid JSON.
    """

    def __init__(self,
                 aggregate_dir: str | Path,
                 results_root: str | Path,
                 naming: Naming | None = None,
                 logger: logging.Logger | None = None) -> None:
        """
        Read the three aggregated files and every per-job metrics file.

        :param aggregate_dir: Directory of the aggregated pass.
        :type aggregate_dir: str | Path
        :param results_root: Directory with one sub-directory per job.
        :type results_root: str | Path
        :param naming: Vocabulary of the study, or None.
        :type naming: Naming | None
        :param logger: Logger created by the calling script, or None.
        :type logger: logging.Logger | None
        :raises TypeError: If naming or logger are of the wrong type.
        :raises FileNotFoundError: If an input file does not exist.
        """
        super().__init__(logger=logger)
        if naming is not None and not isinstance(naming, Naming):
            self._error("[SupplementaryInputs.__init__] Input error: the "
                        "parameter naming must be a Naming or None, not "
                        f"{type(naming).__name__}.", TypeError)
        self.naming = naming or Naming(logger=logger)
        self.aggregate_dir = Path(aggregate_dir)
        self.results_root = Path(results_root)
        self.paths = {
            "cells.csv": self.aggregate_dir / "aggregate" / "cells.csv",
            "runs.csv": self.aggregate_dir / "runs.csv",
            "stats.json": self.aggregate_dir / "stats.json"}
        self.cells = dict()
        self.config_names = dict()
        self.dataset_names = dict()
        self.jobs = dict()
        self.metrics = dict()
        self.attack_recalls = dict()
        self.attack_recalls_by_cell = dict()
        self.attack_rows = dict()
        self.per_day = dict()
        self.loc_blocks = dict()
        self.stats = dict()
        self._load_cells()
        self._load_jobs()
        self._collect()
        self.stats = json.loads(
            self.paths["stats.json"].read_text(encoding="utf-8"))

    # ----- aggregated pass ------------------------------------------- #
    def _load_cells(self) -> None:
        """
        Read ``cells.csv`` into the cell, configuration and stream maps.

        Note: when two rows share a configuration id with different
        ``config_name``, the last row read wins, as in the original.

        :raises FileNotFoundError: If ``cells.csv`` does not exist.
        """
        rows = list()
        with open(self.paths["cells.csv"], newline="",
                  encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            self.cells[(row["dataset_id"], row["config_id"])] = row
            self.config_names[row["config_id"]] = row["config_name"]
            self.dataset_names[row["dataset_id"]] = row["dataset"]

    def _load_jobs(self) -> None:
        """
        Read from ``runs.csv`` which job ids entered each cell.

        Only the rows whose ``in_S_star`` is true are kept: the analysis
        set of the plan, without the ablations, the T4-E block and the
        exploratory blocks. A row with no dataset, configuration or job
        id is skipped.

        :raises FileNotFoundError: If ``runs.csv`` does not exist.
        """
        found = defaultdict(set)
        with open(self.paths["runs.csv"], newline="",
                  encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("in_S_star") not in _TRUE_VALUES:
                    continue
                if row.get("dataset_id") and row.get("config_id") \
                        and row.get("job_id"):
                    found[(row["dataset_id"],
                           row["config_id"])].add(row["job_id"])
        self.jobs = {k: sorted(v) for k, v in found.items()}

    # ----- per-job records ------------------------------------------- #
    def _collect(self) -> None:
        """
        Average every per-job record over the seeds of its cell.

        A job without a readable ``metrics.json`` is skipped, so the
        tables are produced with the cells that have one.
        The attack-type recalls and the per-day records are collected per
        stream and per cell respectively, without averaging, because
        their tables average over the configurations as well.
        """
        accumulated = dict()
        for key, job_ids in self.jobs.items():
            accumulated = defaultdict(list)
            for job_id in job_ids:
                path = self.results_root / job_id / "metrics.json"
                if not path.exists():
                    continue
                self._read_job(key, path, accumulated)
            self.metrics[key] = {k: self.mean(v)
                                 for k, v in accumulated.items()}
            self._apply_cell_overrides(key)

    def _read_job(self,
                  key: tuple[str, str],
                  path: Path,
                  accumulated: dict[str, list[Any]]) -> None:
        """
        Read one ``metrics.json`` into the accumulators of its cell.

        :param key: Cell the job belongs to, as (dataset id, config id).
        :type key: tuple[str, str]
        :param path: Path of the ``metrics.json`` of the job; it exists.
        :type path: Path
        :param accumulated: Metric name -> values of the seeds read so
            far; it is extended in place, missing fields included as
            None so that the mean skips them.
        :type accumulated: dict[str, list[Any]]
        :raises json.JSONDecodeError: If the file is not valid JSON.
        """
        record = json.loads(path.read_text(encoding="utf-8"))
        for name, pointer in _METRIC_SOURCES:
            accumulated[name].append(get(record, *pointer))
        self._read_loc_blocks(key[0], record)
        if get(record, "per_attack_type", "available"):
            self._read_attacks(key, get(record, "per_attack_type",
                                        "types", default=dict()))
        if get(record, "per_day", "available"):
            self._read_days(key, get(record, "per_day", "days",
                                     default=dict()))

    def _read_loc_blocks(self, dataset: str, record: dict[str, Any]) \
            -> None:
        """
        Collect how many blocks of the stream hold records of both
        classes.

        The blocked ROC-AUC averages only over the blocks that contain
        an attack record and a benign one, so that count is what the
        metric rests on. It depends on the labels of the stream alone,
        which is why it is the same in every configuration and in every
        detector, and why a disagreement between two jobs of the same
        stream is a bug rather than a seed effect.

        :param dataset: Dataset id the job belongs to, such as "D8".
        :type dataset: str
        :param record: The whole ``metrics.json`` of the job.
        :type record: dict[str, Any]
        :raises ValueError: If two jobs of one stream disagree.
        """
        current = (get(record, "sensitivities", "S-LOC", "RAW",
                       "blocks"),
                   get(record, "sensitivities", "S-LOC", "RAW",
                       "blocks_both_classes"))
        if current[0] is None:
            return
        previous = self.loc_blocks.get(dataset)
        if previous is not None and previous != current:
            self._error("[SupplementaryInputs._read_loc_blocks] "
                        "Inconsistent S-LOC block counts for "
                        f"{dataset}: {previous!r} against {current!r}")
            raise ValueError("inconsistent S-LOC block counts")
        self.loc_blocks[dataset] = current

    def _read_attacks(self,
                      key: tuple[str, str],
                      types: dict[str, Any]) -> None:
        """
        Collect the recall and the record count of every attack type.

        The recalls are kept twice: per stream, for the table that
        averages over the eleven configurations, and per cell, for the
        two tables that give one column per configuration.

        The "Records" column prints records, not how many
        observations were averaged (eleven configurations times two
        seeds, that is 22). The count is
        ``per_attack_type/types/<type>/n_rows``, a property of the
        stream that is the same in every configuration, so it is kept
        apart from the recalls instead of being averaged.

        :param key: Cell the job belongs to, as (dataset id, config id).
        :type key: tuple[str, str]
        :param types: Attack type -> its block of ``per_attack_type``.
        :type types: dict[str, Any]
        """
        previous = None
        dataset = key[0]
        rows = self.attack_rows.setdefault(dataset, dict())
        recalls = self.attack_recalls.setdefault(dataset, dict())
        by_cell = self.attack_recalls_by_cell.setdefault(key, dict())
        for name, block in types.items():
            recalls.setdefault(name, list()).append(
                get(block, "recall_FUS"))
            by_cell.setdefault(name, list()).append(
                get(block, "recall_FUS"))
            n_rows = get(block, "n_rows")
            if n_rows is None:
                continue
            previous = rows.get(name)
            if previous is not None and previous != n_rows:
                self._warning(
                    "[SupplementaryInputs._read_attacks] Data warning: "
                    "the record count of an attack type must be the same "
                    f"in every job of {dataset}, and {name} has "
                    f"{previous} and {n_rows}. The last value read is "
                    "kept.")
            rows[name] = n_rows

    def _read_days(self,
                   key: tuple[str, str],
                   days: dict[str, Any]) -> None:
        """
        Collect the per-day record of one job, day by day.

        A day whose block is not a dictionary is skipped. Only the seven
        fields the per-day table prints are kept.

        :param key: Cell the job belongs to, as (dataset id, config id).
        :type key: tuple[str, str]
        :param days: Day name -> its block of ``per_day``.
        :type days: dict[str, Any]
        """
        per_cell = self.per_day.setdefault(key, dict())
        for day, block in days.items():
            if not isinstance(block, dict):
                continue
            per_cell.setdefault(day, list()).append(
                {"auroc_fus": block.get("auroc_fus"),
                 "auc_t": block.get("auc_t"),
                 "rows_E": block.get("rows_E"),
                 "attack_rows": block.get("attack_rows"),
                 "fpr": block.get("fpr"),
                 "eligible": block.get("eligible_auc_per_day"),
                 "wd": get(block, "windowdiff_fus", "wd")})

    def _apply_cell_overrides(self, key: tuple[str, str]) -> None:
        """
        Replace three means with the values published by the body.

        ROC-AUC, F1 and AUPRC are taken from ``cells.csv``; a cell with
        no row, no column or an unparsable value keeps the mean computed
        from the per-job records.

        :param key: Cell to overwrite, as (dataset id, config id).
        :type key: tuple[str, str]
        """
        row = self.cells.get(key, dict())
        for name, column in _CELL_OVERRIDES:
            try:
                self.metrics[key][name] = float(row[column])
            except (KeyError, TypeError, ValueError):
                pass

    # ----- read-only views ------------------------------------------- #
    def mean(self, values: Sequence[Any] = tuple()) -> float | None:
        """
        Mean of the values that are not None, or None when there is none.

        This is the average over seeds, and over configurations in the
        attack-type and per-day tables. A missing seed does not drag the
        cell down: it is dropped and the rest are averaged.

        Note: it is ``statistics.mean``, which returns an int when every
        value is an int and the division is exact, and which propagates
        a NaN. The caller keeps the behaviour of the original.

        :param values: Values to average, None entries included.
        :type values: Sequence[Any]
        :return: The mean, or None when no value survives.
        :rtype: float | None
        """
        kept = [v for v in values if v is not None]
        return st.mean(kept) if kept else None

    def metrics_of(self, dataset: str, config: str) -> dict[str, Any]:
        """
        Averaged metrics of one cell.

        :param dataset: Dataset id such as "D8".
        :type dataset: str
        :param config: Configuration id such as "C7".
        :type config: str
        :return: Metric name -> mean over the seeds, or an empty
            dictionary when the cell is not in the analysis set.
        :rtype: dict[str, Any]
        :raises ValueError: If dataset or config is not a non-empty
            string.
        """
        self._require_ids("metrics_of", dataset=dataset, config=config)
        return self.metrics.get((dataset, config), dict())

    def cell(self, dataset: str, config: str) -> dict[str, Any]:
        """
        Row of ``cells.csv`` of one cell, for the per-seed columns.

        :param dataset: Dataset id such as "D8".
        :type dataset: str
        :param config: Configuration id such as "C7".
        :type config: str
        :return: The row as read, or an empty dictionary.
        :rtype: dict[str, Any]
        :raises ValueError: If dataset or config is not a non-empty
            string.
        """
        self._require_ids("cell", dataset=dataset, config=config)
        return self.cells.get((dataset, config), dict())

    def config_label(self, config: str) -> str:
        """
        Long label of a configuration, as ``cells.csv`` writes it.

        :param config: Configuration id such as "C10".
        :type config: str
        :return: The label of the aggregate, or the id itself.
        :rtype: str
        :raises ValueError: If config is not a non-empty string.
        """
        self._require_ids("config_label", config=config)
        return self.config_names.get(config, config)

    def job_ids(self, dataset: str, config: str) -> list[str]:
        """
        Job ids that entered one cell, sorted.

        :param dataset: Dataset id such as "D9".
        :type dataset: str
        :param config: Configuration id such as "C9".
        :type config: str
        :return: The sorted job ids, empty when the cell has none.
        :rtype: list[str]
        :raises ValueError: If dataset or config is not a non-empty
            string.
        """
        self._require_ids("job_ids", dataset=dataset, config=config)
        return list(self.jobs.get((dataset, config), list()))

    def job_metrics(self, job_id: str) -> dict[str, Any]:
        """
        Whole ``metrics.json`` of one job, read again from disk.

        It is used by the T4-P-a summary to recover the two counts that
        the statistics file may not carry.

        :param job_id: Job id, the name of its directory under the
            results root.
        :type job_id: str
        :return: The parsed record, or an empty dictionary when the file
            is not there.
        :rtype: dict[str, Any]
        :raises ValueError: If job_id is not a non-empty string.
        :raises json.JSONDecodeError: If the file is not valid JSON.
        """
        self._require_ids("job_metrics", job_id=job_id)
        path = self.results_root / job_id / "metrics.json"
        if not path.exists():
            return dict()
        return json.loads(path.read_text(encoding="utf-8"))

    def warmup_rows(self) -> dict[tuple[str, str], dict[str, Any]]:
        """
        Rows of the calibration-fraction analysis, by cell and level.

        The block T4-P-c of ``stats.json`` holds, per stream, one row per
        configuration and calibration level. They are indexed here as
        (dataset id, configuration) -> level as a string -> row.

        :return: The index of those rows; empty when the block is not
            in the statistics file.
        :rtype: dict[tuple[str, str], dict[str, Any]]
        """
        levels = defaultdict(dict)
        for dataset, block in get(self.stats, "T4P", "c",
                                  default=dict()).items():
            for row in get(block, "rows", default=list()):
                levels[(dataset, row.get("config"))][
                    str(row.get("level"))] = row
        return dict(levels)

    def clean_calibration_block(self) -> dict[str, Any]:
        """
        Block T4-P-a of ``stats.json``: the clean-calibration analysis.

        :return: Dataset id -> its record, as stored; empty when the
            block is not in the statistics file.
        :rtype: dict[str, Any]
        """
        return get(self.stats, "T4P", "a", default=dict())

    def _require_ids(self, method: str, **ids: Any) -> None:
        """
        Check that every identifier given is a non-empty string.

        :param method: Calling method, for the message.
        :type method: str
        :param ids: Parameter name -> value received by the caller.
        :type ids: Any
        :raises ValueError: If a value is not a non-empty string.
        """
        for name, value in ids.items():
            if not isinstance(value, str) or not value:
                self._error(f"[SupplementaryInputs.{method}] Input "
                            f"error: the parameter {name} must be a "
                            f"non-empty string, not {value!r}.")

    def __repr__(self) -> str:
        """
        Show the pass read and how much of it came back.

        :return: Text like "SupplementaryInputs(aggregate='agg-2102',
            cells=99, jobs=198, streams=9)".
        :rtype: str
        """
        return (f"SupplementaryInputs("
                f"aggregate={self.aggregate_dir.name!r}, "
                f"cells={len(self.cells)}, "
                f"jobs={sum(len(v) for v in self.jobs.values())}, "
                f"streams={len(self.dataset_names)})")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
