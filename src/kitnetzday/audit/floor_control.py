# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         src/kitnetzday/audit/floor_control.py                     #
#   Version:      1.0.2                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Control of the KitNET-style denoising autoencoder against #
#                 the constant-predictor floor.                             #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import json
import logging
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from kitnetzday.audit.constant_floor import SCALING_EPSILON, ConstantFloor
from kitnetzday.utils.loggable import Loggable

# Defaults of the analysis plan (section 5.2, floor block). They also
# live in src/config.json under "analysis_plan"; the scripts in bin/
# read that file and hand the values to the constructor, library code
# never reads it.
TRAINED_BELOW: float = 0.90
AT_FLOOR_ABOVE: float = 1.15
# A floor below this value means that the features of the group barely
# move, so the ratio measures the floor and not the model.
DEGENERATE_FLOOR: float = 1e-3
# Jobs of the control: the KitNET-style denoising autoencoder.
CONFIG_GLOB: str = "T1-*__original__*/config.json"
# Suffix of the stream cache key for the jobs whose config does not
# record it.
STREAM_KEY_SUFFIX: str = "__corrected__v1"
# Caveats written verbatim into the result file, in Spanish as the
# authors wrote them.
CAVEATS: tuple[str, ...] = (
    "RMSE guardado elevado al cuadrado para comparar MSE con MSE",
    "normalizacion online con +1e-6 frente al min-max del tramo con "
    "+1e-16: cociente aproximado",
    "error medido durante la unica pasada online: sesgado hacia "
    "arriba")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                MAIN CLASS                                 #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class FloorControl(Loggable):
    """
    Does the KitNET-style control beat the constant-predictor floor?

    This is the one hole the retrospective training audit left open.
    The control variant (``original``) does not log ``loss_epoch_1``,
    because it does stochastic gradient descent in numpy row by row,
    so the audit skips it. What it does leave behind is
    ``ae_errors.npy``, the per-row error of every autoencoder over the
    whole stream, and inside the training span the rows carry the
    error *during* the online pass. This class divides that in-sample
    error by the floor a constant predictor would reach on the same
    span, group by group, exactly as the audit does for the Keras
    variants.

    Three caveats travel with the number and are written into the
    result:

    1. the control stores RMSE, so it is squared to compare MSE with
       MSE;
    2. the control normalises with an online min-max and a guard of
       1e-6, not with the min-max of the span and a guard of 1e-16
       like the Keras variants, so its scale is not exactly that of
       the offline floor and the ratio is approximate;
    3. the error is measured during the single training pass, so it
       drags the initial untrained rows along and the ratio is biased
       *upwards*, just like ``loss_epoch_1`` in the audit.

    Note: the summary keys ``trained_lt_0.90`` and ``worse_ge_1.15``
    quote the paper thresholds inside their own names, and are kept
    verbatim even when other thresholds are passed to the constructor,
    because the manuscript refers to those thresholds; the caveats are
    in Spanish, as the authors wrote them.

    :param results_root: Directory with one sub-directory per job.
    :type results_root: str | Path
    :param cache_root: Directory of the pre-processed stream cache,
        with one sub-directory per stream key holding ``x.npy``.
    :type cache_root: str | Path
    :param trained_below: Ratio under which an autoencoder counts as
        trained; 0.90 in the paper, strictly positive.
    :type trained_below: float
    :param at_floor_above: Ratio from which an autoencoder counts as
        worse than the floor; 1.15 in the paper, greater than
        trained_below.
    :type at_floor_above: float
    :param degenerate_floor: Floor under which the group is called
        degenerate; 1e-3 in the paper, strictly positive.
    :type degenerate_floor: float
    :param range_epsilon: Guard added to the min-max range of the
        floor, handed to ``ConstantFloor``; 1e-16 in the paper.
    :type range_epsilon: float
    :param config_glob: Pattern of the ``config.json`` files of the
        control jobs under the results root.
    :type config_glob: str
    :param caveats: Caveats copied into the summary.
    :type caveats: Sequence[str]
    :param logger: Logger created by the calling script, or None to
        stay silent.
    :type logger: logging.Logger | None
    :raises TypeError: If logger is not a logging.Logger or None, or
        if caveats is not a sequence of strings.
    :raises ValueError: If any threshold is out of range or if
        config_glob is empty.
    """

    def __init__(self,
                 results_root: str | Path,
                 cache_root: str | Path,
                 trained_below: float = TRAINED_BELOW,
                 at_floor_above: float = AT_FLOOR_ABOVE,
                 degenerate_floor: float = DEGENERATE_FLOOR,
                 range_epsilon: float = SCALING_EPSILON,
                 config_glob: str = CONFIG_GLOB,
                 caveats: Sequence[str] = CAVEATS,
                 logger: logging.Logger | None = None) -> None:
        """
        Check the settings and remember where the inputs are.

        Nothing is read here: the jobs and the cache are opened by
        ``run`` and by ``floors``, so that a missing input fails where
        it is used.

        :param results_root: Directory with one sub-directory per job.
        :type results_root: str | Path
        :param cache_root: Directory of the stream cache.
        :type cache_root: str | Path
        :param trained_below: Ratio under which an AE is trained.
        :type trained_below: float
        :param at_floor_above: Ratio from which an AE is worse than
            the floor.
        :type at_floor_above: float
        :param degenerate_floor: Floor under which a group is called
            degenerate.
        :type degenerate_floor: float
        :param range_epsilon: Guard of the min-max range.
        :type range_epsilon: float
        :param config_glob: Pattern of the control ``config.json``.
        :type config_glob: str
        :param caveats: Caveats copied into the summary.
        :type caveats: Sequence[str]
        :param logger: Logger created by the calling script, or None.
        :type logger: logging.Logger | None
        :raises TypeError: If logger or caveats are of the wrong type.
        :raises ValueError: If any threshold is out of range or if
            config_glob is empty.
        """
        super().__init__(logger=logger)
        if isinstance(caveats, str) or not isinstance(caveats, Sequence) \
                or not all(isinstance(c, str) for c in caveats):
            self._error("[FloorControl.__init__] Input error: caveats "
                        "must be a sequence of strings, not "
                        f"{type(caveats).__name__}.", TypeError)
        if not float(trained_below) > 0.0:
            self._error("[FloorControl.__init__] Input error: "
                        "trained_below must be strictly positive "
                        f"(>0), not {trained_below}.")
        if not float(at_floor_above) > float(trained_below):
            self._error("[FloorControl.__init__] Input error: "
                        "at_floor_above must be greater than "
                        f"trained_below ({trained_below}), not "
                        f"{at_floor_above}.")
        if not float(degenerate_floor) > 0.0:
            self._error("[FloorControl.__init__] Input error: "
                        "degenerate_floor must be strictly positive "
                        f"(>0), not {degenerate_floor}.")
        if not float(range_epsilon) > 0.0:
            self._error("[FloorControl.__init__] Input error: "
                        "range_epsilon must be strictly positive "
                        f"(>0), not {range_epsilon}.")
        if not isinstance(config_glob, str) or not config_glob:
            self._error("[FloorControl.__init__] Input error: "
                        "config_glob must be a non-empty string, not "
                        f"{config_glob!r}.")
        self.results_root = Path(results_root)
        self.cache_root = Path(cache_root)
        self.trained_below = float(trained_below)
        self.at_floor_above = float(at_floor_above)
        self.degenerate_floor = float(degenerate_floor)
        self.range_epsilon = float(range_epsilon)
        self.config_glob = config_glob
        self.caveats = tuple(caveats)
        self.floor = ConstantFloor(self.cache_root, self.range_epsilon,
                                   logger=logger)

    def floors(self,
               stream_key: str,
               start: int,
               stop: int,
               feature_map: Sequence[Sequence[int]]) -> list[float]:
        """
        Constant-predictor floor of every group of the feature map.

        The floor of a feature is its variance over the span divided by
        the square of its min-max range on that same span, which is
        the error a predictor that always answers the mean would make
        once the feature is min-max scaled; the floor of a group is the
        mean over its features. The work is delegated to
        ``ConstantFloor``, which is the one place of the package that
        knows the formula and caches it per span, so the ratios of the
        control and of the Keras variants are comparable by
        construction.

        Note: the original did not check its arguments; a span with
        ``stop <= start`` used to give NaN floors and now raises.

        :param stream_key: Sub-directory of the cache that holds
            ``x.npy``, non-empty.
        :type stream_key: str
        :param start: First row of the training span, zero-based.
        :type start: int
        :param stop: Row after the last one of the span, above start.
        :type stop: int
        :param feature_map: One sequence of column indices per group,
            as ``fm_stats.json`` writes it; at least one group.
        :type feature_map: Sequence[Sequence[int]]
        :return: One floor per group, in the order of the feature map,
            in squared scaled units.
        :rtype: list[float]
        :raises TypeError: If stream_key is not a string.
        :raises ValueError: If stream_key is empty, if the span is
            empty or negative, or if feature_map is empty.
        :raises FileNotFoundError: If the cached stream is missing.
        """
        variances = np.empty(0)
        if not isinstance(stream_key, str) or not stream_key:
            self._error("[FloorControl.floors] Input error: "
                        "stream_key must be a non-empty string, not "
                        f"{stream_key!r}.", TypeError)
        if not len(feature_map):
            self._error("[FloorControl.floors] Input error: "
                        "feature_map must hold at least one group, "
                        "and it is empty.")
        variances = self.floor.vector(stream_key, start, stop)
        return [self.floor.group_floor(variances, g)
                for g in feature_map]

    def rows(self) -> list[dict[str, Any]]:
        """
        One record per autoencoder of every finished control job.

        A job is used when its directory holds ``DONE`` and
        ``ae_errors.npy`` and when its ``config.json`` records the
        training span. The in-sample error of an autoencoder is the
        mean of its squared per-row errors over that span, squared
        because the control stores RMSE while the floor is an MSE.

        The stream key is taken from ``cache.key``, then from
        ``spec.stream_key``, and finally built from the dataset in the
        job directory name plus the corrected-stream suffix.

        Note: a job whose ``fm_stats.json`` is missing or unreadable
        raises, as in the original, instead of being skipped: a
        control job without its feature map is a broken job, not an
        absent one.

        :return: One record per autoencoder, in the alphabetical order
            of the job directories and then by autoencoder index, with
            the keys "dataset", "job", "ae", "n_visible", "floor_mse",
            "in_sample_mse", "ratio" and "floor_degenerate".
        :rtype: list[dict[str, Any]]
        :raises OSError: If a job file cannot be read.
        :raises ValueError: If a job file is not valid JSON.
        :raises FileNotFoundError: If a cached stream is missing.
        """
        rows: list[dict[str, Any]] = list()
        config: dict[str, Any] = dict()
        floor_list: list[float] = list()
        span: np.ndarray = np.empty(0)
        for path in sorted(self.results_root.glob(self.config_glob)):
            job = path.parent
            if not (job / "DONE").exists() \
                    or not (job / "ae_errors.npy").exists():
                continue
            config = json.loads(path.read_text(encoding="utf-8"))
            regions = config.get("regions") or dict()
            train = regions.get("ae_train")
            if not train:
                continue
            feature_map = json.loads(
                (job / "fm_stats.json").read_text(
                    encoding="utf-8"))["feature_map"]
            stream_key = ((config.get("cache") or dict()).get("key")
                          or (config.get("spec")
                              or dict()).get("stream_key"))
            if stream_key is None:
                stream_key = (job.name.split("__")[1]
                              + STREAM_KEY_SUFFIX)
            floor_list = self.floors(stream_key, train[0], train[1],
                                     feature_map)
            errors = np.load(job / "ae_errors.npy", mmap_mode="r")
            span = np.asarray(errors[train[0]:train[1]],
                              dtype=np.float64)
            dataset = job.name.split("__")[1]
            for i, floor in enumerate(floor_list):
                mse = float(np.mean(span[:, i] ** 2))
                degenerate = floor < self.degenerate_floor
                rows.append({
                    "dataset": dataset, "job": job.name, "ae": i,
                    "n_visible": len(feature_map[i]),
                    "floor_mse": floor, "in_sample_mse": mse,
                    "ratio": (mse / floor) if floor > 0 else None,
                    "floor_degenerate": degenerate})
        return rows

    def run(self) -> dict[str, Any]:
        """
        Audit the control jobs and summarise the ratios.

        The summary counts how many autoencoders fall below the floor
        (trained), sit on it, or do worse, and repeats the median over
        the non-degenerate groups only, which is the number that
        Section 18 of the supplement quotes. The per-dataset medians
        keep every group, degenerate ones included.

        Note: an autoencoder whose floor is not strictly positive gets
        a ratio of None and is left out of every count, but still
        counts in ``n_autoencoders``.

        :return: The result, with the keys of the original file in the
            original order: ``rows``, ``summary`` and
            ``per_dataset_median``.
        :rtype: dict[str, Any]
        :raises OSError: If a job file cannot be read.
        :raises ValueError: If a job file is not valid JSON.
        :raises FileNotFoundError: If a cached stream is missing.
        """
        result: dict[str, Any] = dict()
        per_dataset: dict[str, list[float]] = dict()
        rows = self.rows()
        result["rows"] = rows
        ratios = [r["ratio"] for r in rows if r["ratio"] is not None]
        sound = [r["ratio"] for r in rows if r["ratio"] is not None
                 and not r["floor_degenerate"]]
        result["summary"] = {
            "n_autoencoders": len(rows),
            "n_jobs": len({r["job"] for r in rows}),
            "n_datasets": len({r["dataset"] for r in rows}),
            "median_ratio": (statistics.median(ratios) if ratios
                             else None),
            "min": min(ratios) if ratios else None,
            "max": max(ratios) if ratios else None,
            "trained_lt_0.90": sum(1 for v in ratios
                                   if v < self.trained_below),
            "at_floor": sum(1 for v in ratios
                            if self.trained_below <= v
                            < self.at_floor_above),
            "worse_ge_1.15": sum(1 for v in ratios
                                 if v >= self.at_floor_above),
            "nondegenerate": {
                "n": len(sound),
                "median_ratio": (statistics.median(sound) if sound
                                 else None),
                "trained_lt_0.90": sum(1 for v in sound
                                       if v < self.trained_below)},
            "caveats": list(self.caveats)}
        for row in rows:
            if row["ratio"] is not None:
                per_dataset.setdefault(row["dataset"],
                                       list()).append(row["ratio"])
        result["per_dataset_median"] = {
            k: statistics.median(v)
            for k, v in sorted(per_dataset.items())}
        return result

    def summary_lines(self, result: Mapping[str, Any]) -> list[str]:
        """
        The console summary of the original script, line by line.

        Library code does not print, so the lines are returned and the
        script in ``bin/`` is the one that logs or prints them. The
        wording, the four decimals and the per-dataset column width
        are the ones the original used.

        Note: a run without any usable ratio leaves the median, the
        minimum and the maximum at None and this method then raises a
        TypeError from the format, exactly as the original did.

        :param result: A result as ``run`` returns it; it must carry
            the keys "summary" and "per_dataset_median".
        :type result: Mapping[str, Any]
        :return: One string per printed line, without line breaks.
        :rtype: list[str]
        :raises TypeError: If result is not a mapping, or if the
            summary holds None where a number is formatted.
        :raises ValueError: If result does not carry the summary or
            the per-dataset medians.
        """
        lines: list[str] = list()
        summary: Mapping[str, Any] = dict()
        if not isinstance(result, Mapping):
            self._error("[FloorControl.summary_lines] Input error: "
                        "result must be a mapping, not "
                        f"{type(result).__name__}.", TypeError)
        for name in ("summary", "per_dataset_median"):
            if name not in result:
                self._error("[FloorControl.summary_lines] Input "
                            f"error: result must hold {name!r}, and "
                            f"it holds {sorted(result)}.")
        summary = result["summary"]
        lines.append(f"{summary['n_autoencoders']} AE del control en "
                     f"{summary['n_jobs']} jobs y "
                     f"{summary['n_datasets']} datasets")
        lines.append("  cociente in-sample/suelo: mediana "
                     f"{summary['median_ratio']:.4f}  min "
                     f"{summary['min']:.4f}  max "
                     f"{summary['max']:.4f}")
        lines.append("  por debajo del suelo (<0,90): "
                     f"{summary['trained_lt_0.90']} · en el "
                     f"suelo: {summary['at_floor']} · peor: "
                     f"{summary['worse_ge_1.15']}")
        lines.append("  solo grupos no degenerados: n="
                     f"{summary['nondegenerate']['n']} mediana "
                     f"{summary['nondegenerate']['median_ratio']:.4f}"
                     " entrenados "
                     f"{summary['nondegenerate']['trained_lt_0.90']}")
        for name, value in result["per_dataset_median"].items():
            lines.append(f"    {name:26s} {value:.4f}")
        return lines

    def __repr__(self) -> str:
        """
        Show the inputs and the thresholds.

        :return: Text like "FloorControl(results_root=...,
            cache_root=..., trained_below=0.9, at_floor_above=1.15,
            degenerate_floor=0.001)".
        :rtype: str
        """
        return (f"FloorControl(results_root="
                f"{str(self.results_root)!r}, cache_root="
                f"{str(self.cache_root)!r}, trained_below="
                f"{self.trained_below}, at_floor_above="
                f"{self.at_floor_above}, degenerate_floor="
                f"{self.degenerate_floor})")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
