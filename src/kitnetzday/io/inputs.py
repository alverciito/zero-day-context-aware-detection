# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         src/kitnetzday/io/inputs.py                               #
#   Version:      2.0.0                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Single read-only door to the aggregate, the cells, the    #
#                 statistics and their provenance.                          #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import csv
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from kitnetzday.naming import Naming
from kitnetzday.utils.formatting import InputError, get
from kitnetzday.utils.loggable import Loggable

_HASH_CHUNK = 1 << 22


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                MAIN CLASS                                 #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class Inputs(Loggable):
    """
    Everything the figures and tables read, with provenance and status.

    The figures recompute nothing: they read what the metrics,
    aggregate and statistics steps produced. ``aggregate.json`` is
    mandatory, ``cells.csv`` repeats it in flat form as a cross-check,
    ``stats.json`` is optional (heat map, hypothesis, mean ranks) and
    without it every output is labelled as built from partial inputs,
    and the two Fig. 7 artefacts are only hashed so that their sha256
    reaches the manifest. The class also gives the pointer every value
    was read from, for the "Source" column of ``numbers_used.md``, and
    cross-checks ``cells.csv`` and ``stats.json`` against the aggregate
    into ``consistency``: otherwise a ``stats.json`` from another pass
    would overwrite the heat map while the dispersion column still
    came from the aggregate, and one table would mix two passes.

    :param agg_path: Path of ``aggregate.json``; it must exist.
    :type agg_path: str | Path
    :param stats_path: Path of ``stats.json``; optional.
    :type stats_path: str | Path
    :param cells_path: Path of ``cells.csv``; optional.
    :type cells_path: str | Path
    :param results_root: Directory with one sub-directory per job.
    :type results_root: str | Path
    :param fig7_csv: Published Fig. 7 values; hashed, not parsed.
    :type fig7_csv: str | Path
    :param fig7_md: Smoothed trivial F1 of Fig. 7; hashed, not parsed.
    :type fig7_md: str | Path
    :param naming: Vocabulary of the labels, or None for the default.
    :type naming: Naming | None
    :param tolerance: Tolerance of the cross-checks, in ROC-AUC
        points; 1e-9 in the paper, strictly positive.
    :type tolerance: float
    :param logger: Logger of the calling script, or None.
    :type logger: logging.Logger | None
    :raises TypeError: If logger or naming are of the wrong type.
    :raises ValueError: If tolerance is not strictly positive.
    :raises InputError: If ``aggregate.json`` does not exist.
    """

    def __init__(self,
                 agg_path: str | Path,
                 stats_path: str | Path,
                 cells_path: str | Path,
                 results_root: str | Path,
                 fig7_csv: str | Path,
                 fig7_md: str | Path,
                 naming: Naming | None = None,
                 tolerance: float = 1e-9,
                 logger: logging.Logger | None = None) -> None:
        """
        Read the inputs, hash them and decide whether they are final.

        The inputs are final when ``stats.json`` is there, its run type
        is FINAL and the aggregate is complete. Any other case sets
        ``partial_inputs`` and records why in
        ``partial_inputs_reasons``.

        :param agg_path: Path of ``aggregate.json``; it must exist.
        :type agg_path: str | Path
        :param stats_path: Path of ``stats.json``; optional.
        :type stats_path: str | Path
        :param cells_path: Path of ``cells.csv``; optional.
        :type cells_path: str | Path
        :param results_root: Directory with one sub-directory per job.
        :type results_root: str | Path
        :param fig7_csv: Published Fig. 7 values; hashed only.
        :type fig7_csv: str | Path
        :param fig7_md: Smoothed trivial F1 of Fig. 7; hashed only.
        :type fig7_md: str | Path
        :param naming: Vocabulary, or None for the default one.
        :type naming: Naming | None
        :param tolerance: Absolute tolerance of the cross-checks.
        :type tolerance: float
        :param logger: Logger created by the calling script, or None.
        :type logger: logging.Logger | None
        :raises TypeError: If logger or naming are of the wrong type.
        :raises ValueError: If tolerance is not strictly positive.
        :raises InputError: If ``aggregate.json`` does not exist.
        """
        super().__init__(logger=logger)
        why = list()
        if naming is not None and not isinstance(naming, Naming):
            self._error("[Inputs.__init__] Input error: naming must "
                        f"be a Naming or None, not "
                        f"{type(naming).__name__}.", TypeError)
        if not float(tolerance) > 0.0:
            self._error("[Inputs.__init__] Input error: tolerance "
                        f"must be strictly positive (>0), not "
                        f"{tolerance}.")
        self.naming = naming or Naming(logger=logger)
        self.tolerance = float(tolerance)
        self.paths = {"aggregate.json": Path(agg_path),
                      "cells.csv": Path(cells_path),
                      "stats.json": Path(stats_path),
                      "fig7_valores_publicados.csv": Path(fig7_csv),
                      "fig7_trivial_f1.csv": Path(fig7_md)}
        self.results_root = Path(results_root)
        self.missing = list()
        self.notes = list()
        self.consistency = list()
        if not self.paths["aggregate.json"].exists():
            self._error("[Inputs.__init__] Input error: the "
                        "aggregate must exist, and it does not: "
                        f"{self.paths['aggregate.json']}. Run "
                        "aggregate.py first.", InputError)
        self.agg = json.loads(
            self.paths["aggregate.json"].read_text(encoding="utf-8"))
        self.stats = None
        if self.paths["stats.json"].exists():
            self.stats = json.loads(
                self.paths["stats.json"].read_text(encoding="utf-8"))
        else:
            self.missing.append(
                "stats.json (missing): no H1 and no ranks of stats.py; "
                "F2 and F4 are derived from aggregate.json with the "
                "rules of the plan and carry [PARTIAL INPUTS]")
        self.cells_csv = list()
        if self.paths["cells.csv"].exists():
            with open(self.paths["cells.csv"], newline="",
                      encoding="utf-8") as fh:
                self.cells_csv = list(csv.DictReader(fh))
        else:
            self.missing.append(
                "cells.csv (missing): no cross-check of "
                "aggregate.json")
        self.sha256 = {k: (self._sha256_file(p) if p.exists()
                           else None)
                       for k, p in self.paths.items()}
        self.run_type = get(self.stats or dict(), "run_type",
                            default=None)
        self.agg_complete = bool(self.agg.get("complete"))
        self.partial_inputs = ((self.stats is None)
                               or (not self.agg_complete)
                               or (self.run_type != "FINAL"))
        if self.stats is None:
            why.append("no stats.json")
        elif self.run_type != "FINAL":
            why.append(f"stats.json run_type = {self.run_type}: "
                       + "; ".join(get(self.stats, "run_type_reasons",
                                       default=list()) or list()))
        if not self.agg_complete and not any(
                "aggregate.complete" in w for w in why):
            why.append("aggregate.complete = false")
        self.partial_inputs_reasons = why
        self._check_cross()

    def config_name(self, c: str) -> str:
        """
        Long label of a configuration, as the aggregate writes it.

        Note: the label is read from one reference dataset only,
        "D1" in the original, so a run without it falls back here.

        :param c: Configuration id such as "C7".
        :type c: str
        :return: The label of the aggregate, or that of the vocabulary.
        :rtype: str
        :raises ValueError: If c is not a non-empty string.
        """
        self._require_ids("config_name", c=c)
        ref = self.naming.datasets[0] if self.naming.datasets else "D1"
        return get(self.agg, "cells", ref, c, "name",
                   default=self.naming.config_label(c))

    def config_short(self, c: str) -> str:
        """
        Short label of a configuration, for the axes of the figures.

        :param c: Configuration id such as "C10".
        :type c: str
        :return: The short label, or the long one when there is none.
        :rtype: str
        :raises ValueError: If c is not a non-empty string.
        """
        self._require_ids("config_short", c=c)
        return self.naming.config_shorts.get(c, self.config_name(c))

    def auroc_src(self, x: str, c: str) -> tuple[float | None, str]:
        """
        Primary metric of one cell and the exact pointer it comes from.

        :param x: Dataset id such as "D8".
        :type x: str
        :param c: Configuration id such as "C7".
        :type c: str
        :return: The value, or None when the cell is empty, and the
            JSON pointer it was read from: ``stats.json`` when that
            document is there, the aggregate otherwise. Both must
            agree, which the cross-check enforces.
        :rtype: tuple[float | None, str]
        :raises ValueError: If x or c is not a non-empty string.
        """
        v = None
        self._require_ids("auroc_src", x=x, c=c)
        if self.stats is not None:
            v = get(self.stats, "R110", "heatmap_auroc_fus", x, c)
            if v is not None:
                return (float(v),
                        f"stats.json#/R110/heatmap_auroc_fus/{x}/{c}")
        v = get(self.agg, "value_sets", "main", x, c, "auroc_FUS")
        return ((None if v is None else float(v)),
                f"aggregate.json#/value_sets/main/{x}/{c}/auroc_FUS")

    def auroc(self, x: str, c: str) -> float | None:
        """
        Primary metric of one cell.

        :param x: Dataset id such as "D8".
        :type x: str
        :param c: Configuration id such as "C7".
        :type c: str
        :return: ROC-AUC of the fused detector, or None.
        :rtype: float | None
        :raises ValueError: If x or c is not a non-empty string.
        """
        return self.auroc_src(x, c)[0]

    def auroc_ptr(self, x: str, c: str) -> str:
        """
        JSON pointer the primary metric of one cell was read from.

        :param x: Dataset id such as "D8".
        :type x: str
        :param c: Configuration id such as "C7".
        :type c: str
        :return: The pointer, for the "Source" column of the table
            of the values used.
        :rtype: str
        :raises ValueError: If x or c is not a non-empty string.
        """
        return self.auroc_src(x, c)[1]

    def imputed(self, x: str, c: str) -> bool:
        """
        Whether the cell carries a diverged seed imputed by the plan.

        :param x: Dataset id such as "D8".
        :type x: str
        :param c: Configuration id such as "C7".
        :type c: str
        :return: True when the cell lists at least one diverged seed.
        :rtype: bool
        :raises ValueError: If x or c is not a non-empty string.
        """
        self._require_ids("imputed", x=x, c=c)
        return bool(get(self.agg, "cells", x, c, "divergent_seeds",
                        default=list()) or list())

    def sqrt2_sd_pool(self, x: str) -> float | None:
        """
        Square root of two times the pooled seed SD of one dataset.

        :param x: Dataset id such as "D8".
        :type x: str
        :return: The value as the aggregate stored it, or None.
        :rtype: float | None
        :raises ValueError: If x is not a non-empty string.
        """
        self._require_ids("sqrt2_sd_pool", x=x)
        v = get(self.agg, "datasets", x, "sd_pool", "sqrt2_sd_pool")
        return None if v is None else float(v)

    def noise_threshold(self, x: str) -> float | None:
        """
        Noise threshold of one dataset, as fixed by the aggregate.

        :param x: Dataset id such as "D8".
        :type x: str
        :return: The threshold in ROC-AUC points, or None.
        :rtype: float | None
        :raises ValueError: If x is not a non-empty string.
        """
        self._require_ids("noise_threshold", x=x)
        v = get(self.agg, "datasets", x, "sd_pool", "threshold")
        return None if v is None else float(v)

    def cell(self, x: str, c: str) -> dict[str, Any]:
        """
        Whole cell record of the aggregate, for the rarer fields.

        :param x: Dataset id such as "D8".
        :type x: str
        :param c: Configuration id such as "C7".
        :type c: str
        :return: The cell as stored, or an empty dictionary.
        :rtype: dict[str, Any]
        :raises ValueError: If x or c is not a non-empty string.
        """
        self._require_ids("cell", x=x, c=c)
        return get(self.agg, "cells", x, c, default=dict())

    def cell_mean(self, x: str, c: str, key: str) -> float | None:
        """
        Across-seed mean of one metric of one cell.

        :param x: Dataset id such as "D8".
        :type x: str
        :param c: Configuration id such as "C7".
        :type c: str
        :param key: Metric name as the aggregate spells it.
        :type key: str
        :return: The mean, or None when the metric is not there.
        :rtype: float | None
        :raises ValueError: If x, c or key is not a non-empty string.
        """
        self._require_ids("cell_mean", x=x, c=c, key=key)
        v = get(self.agg, "cells", x, c, "mean", key)
        return None if v is None else float(v)

    def trivial(self, x: str, key: str) -> float | None:
        """
        Trivial reference of one dataset, computed from labels only.

        :param x: Dataset id such as "D8".
        :type x: str
        :param key: Name of the reference as the aggregate spells it.
        :type key: str
        :return: The reference value, or None.
        :rtype: float | None
        :raises ValueError: If x or key is not a non-empty string.
        """
        self._require_ids("trivial", x=x, key=key)
        v = get(self.agg, "datasets", x, "trivial", key)
        return None if v is None else float(v)

    def n_seeds(self, x: str) -> int:
        """
        How many seeds entered the analysis set of one dataset.

        :param x: Dataset id such as "D8".
        :type x: str
        :return: Size of the analysis set S*, 0 when there is none.
        :rtype: int
        :raises ValueError: If x is not a non-empty string.
        """
        self._require_ids("n_seeds", x=x)
        return len(get(self.agg, "datasets", x, "S_star",
                       default=list()) or list())

    def dataset_label(self, x: str) -> str:
        """
        Long name of a dataset, for captions and table rows.

        :param x: Dataset id such as "D8".
        :type x: str
        :return: The vocabulary label, the stored name, or the id.
        :rtype: str
        :raises ValueError: If x is not a non-empty string.
        """
        self._require_ids("dataset_label", x=x)
        return self.naming.dataset_longs.get(
            x, get(self.agg, "datasets", x, "dataset", default=x))

    def _check_cross(self) -> None:
        """
        Check ``cells.csv`` against the aggregate, cell by cell.

        Nothing is recomputed; every difference between the two flat
        views of the same pass is appended to ``consistency``.
        """
        a = None
        b = None
        for row in self.cells_csv:
            x, c = row.get("dataset_id"), row.get("config_id")
            if not x or not c:
                continue
            try:
                a = (float(row["auroc_FUS_mean"])
                     if row.get("auroc_FUS_mean") not in ("", None)
                     else None)
            except ValueError:
                a = None
            b = self.cell_mean(x, c, "auroc_FUS")
            if a is None or b is None:
                if a is not None or b is not None:
                    self.consistency.append(
                        f"{x}/{c}: auroc_FUS_mean present in one "
                        "source and absent from the other")
                continue
            if abs(a - b) > self.tolerance:
                self.consistency.append(
                    f"{x}/{c}: cells.csv {a!r} != aggregate.json {b!r}")
        self._check_cross_stats(self.tolerance)
        if self.consistency:
            self.notes.append(
                f"{len(self.consistency)} mismatches between the "
                "inputs (see figures.json)")

    def _check_cross_stats(self, tol: float = 1e-9) -> None:
        """
        Check the heat map of ``stats.json`` against the aggregate.

        Otherwise a ``stats.json`` from another pass would overwrite
        the heat map. Any difference is an input error (code 1).

        :param tol: Absolute tolerance in ROC-AUC points.
        :type tol: float
        """
        hm = None
        b = None
        self.cross_stats = {"checked": 0, "mismatch": list()}
        if self.stats is None:
            return
        hm = get(self.stats, "R110", "heatmap_auroc_fus", default=None)
        if not hm:
            self.missing.append(
                "stats.json without R110/heatmap_auroc_fus: no "
                "cross-check against the aggregate")
            return
        for x, row in hm.items():
            for c, v in (row or dict()).items():
                b = get(self.agg, "value_sets", "main", x, c,
                        "auroc_FUS")
                if v is None and b is None:
                    continue
                self.cross_stats["checked"] += 1
                if v is None or b is None or \
                        abs(float(v) - float(b)) > tol:
                    self.cross_stats["mismatch"].append(
                        f"{x}/{c}: stats.json {v!r} != "
                        f"aggregate.json {b!r}")
        if self.cross_stats["mismatch"]:
            self.consistency.extend(self.cross_stats["mismatch"])

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
                self._error(f"[Inputs.{method}] Input error: the "
                            f"parameter {name} must be a non-empty "
                            f"string, not {value!r}.")

    def _sha256_file(self, path: Path) -> str:
        """
        Hash a file, reading it in 4 MiB chunks.

        :param path: Path of an existing, readable file.
        :type path: Path
        :return: The sha256 digest, 64 hexadecimal characters.
        :rtype: str
        :raises OSError: If the file cannot be read.
        """
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for b in iter(lambda: fh.read(_HASH_CHUNK), b""):
                h.update(b)
        return h.hexdigest()

    def __repr__(self) -> str:
        """
        Show the aggregate read, the run type and the status.

        :return: Text like "Inputs(run_type='FINAL',
            partial_inputs=False)".
        :rtype: str
        """
        return (f"Inputs(aggregate={self.paths['aggregate.json'].name!r}"
                f", run_type={self.run_type!r}, "
                f"cells={len(self.cells_csv)}, "
                f"partial_inputs={self.partial_inputs})")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
