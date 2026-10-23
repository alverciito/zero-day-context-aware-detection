# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         src/kitnetzday/figures/stream_context.py                  #
#   Version:      1.0.1                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-13                                                #
#   Description:  One job of the revised runner, prepared exactly as the    #
#                 metrics step prepares it: regions, fit and series.        #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import json
import logging
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from kitnetzday.utils.loggable import Loggable

# Files the revised runner leaves in every job directory.
CONFIG_FILE: str = "config.json"
SCORES_FILE: str = "scores.npy"
LABELS_FILE: str = "labels.npy"
# Attributes the injected metrics module must provide. That module is
# frozen/runner/metrics_rev.py, which lives outside the kitnetzday
# package and is handed in by the calling script, so that the
# illustrative figures use the very same causal detectors, the same
# label-free fit and the same calibrated thresholds as the tables.
METRICS_API: tuple[str, ...] = (
    "PRIMARY_CANDIDATE", "fit", "fit_set_mask", "get_regions",
    "rows_of", "sanitize_scores", "transform")
# Quantile key of the calibrated threshold the figures draw; the
# thresholds of the fit are stored under the string of the quantile.
MAIN_QUANTILE: str = "0.99"


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                AUXILIARY                                  #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def _silent(*args: Any, **kwargs: Any) -> None:
    """
    Swallow the provenance messages of the metrics module.

    ``metrics_rev.fit_set_mask`` reports where the duplicate mask comes
    from through a ``log`` callable that defaults to ``print``. The
    original script handed it a lambda that drops everything, and
    library code must not print, so the same silent callable is used.

    :param args: Positional parts of the message; ignored.
    :type args: Any
    :param kwargs: Keyword parts of the message; ignored.
    :type kwargs: Any
    """


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                MAIN CLASS                                 #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class StreamContext(Loggable):
    """
    One job of the revised runner, ready to be drawn.

    The illustrative figures of the revision must show what the tables
    measure, so they may not recompute anything of their own: this
    class replays, step by step, the context the metrics step builds
    for a single job. It reads the job's ``config.json``, its per-record
    scores and its labels, asks the metrics module for the calibration
    and evaluation regions of the primary candidate, drops the exact
    duplicates of the training segment from the fit set, sanitizes the
    non-finite scores, fits the detectors on the calibration segment
    *without labels* and transforms the stream from the end of training
    onwards. Labels are only used afterwards, to shade the attack
    episodes and to draw the ROC curves.

    The metrics module is injected instead of imported: it is frozen
    code of the submission that lives outside the package, and only the
    script in ``bin/`` knows where it is.

    Everything the figures need is left in public attributes:

    - ``job_dir``: directory of the job inside the results root.
    - ``config``: parsed ``config.json`` of the job.
    - ``scores``: sanitized raw scores of the whole stream, shape (N,),
      in log10(1 + s) units, as the plots show them.
    - ``labels``: per-record labels of the whole stream, shape (N,).
    - ``n_records``: length N of the stream.
    - ``train_end``: first record after the training segment.
    - ``calibration``: the calibration interval [a, b).
    - ``evaluation``: the evaluation region, an interval or segments.
    - ``eval_rows``: the evaluation rows, as stream indices.
    - ``fit_mask``: the fit set F_cal over the whole stream.
    - ``params``: the label-free fit (centroids, scales, thresholds).
    - ``series``: detector series from ``train_end`` onwards, with the
      keys of the metrics module ("MVD_raw", "MVD_f", "KMD_raw",
      "KMD_f", "FUS", ...) plus "RAW".

    :param job_id: Directory name of the job inside results_root, for
        example "T1-A-ACI__aci-iot-2023__conv1d__arT__...__r1";
        non-empty.
    :type job_id: str
    :param results_root: Directory that holds one sub-directory per
        job; it must exist and hold job_id.
    :type results_root: str | Path
    :param metrics: The frozen metrics module, with the attributes of
        ``METRICS_API``.
    :type metrics: ModuleType
    :param candidate: Detector protocol candidate of the job's regions;
        None uses the primary candidate of the metrics module, which is
        the one the tables of the revision use.
    :type candidate: str | None
    :param logger: Logger created by the calling script, or None to
        stay silent.
    :type logger: logging.Logger | None
    :raises TypeError: If logger is not a logging.Logger or None, or if
        metrics is not a module.
    :raises ValueError: If job_id is empty or metrics lacks part of the
        API.
    :raises FileNotFoundError: If the job directory or one of its three
        files is missing.
    """

    def __init__(self,
                 job_id: str,
                 results_root: str | Path,
                 metrics: ModuleType,
                 candidate: str | None = None,
                 logger: logging.Logger | None = None) -> None:
        """
        Read the job and replay the preparation of the metrics step.

        :param job_id: Directory name of the job; non-empty.
        :type job_id: str
        :param results_root: Directory with one sub-directory per job.
        :type results_root: str | Path
        :param metrics: The frozen metrics module.
        :type metrics: ModuleType
        :param candidate: Protocol candidate, or None for the primary.
        :type candidate: str | None
        :param logger: Logger created by the calling script, or None.
        :type logger: logging.Logger | None
        :raises TypeError: If logger or metrics have the wrong type.
        :raises ValueError: If job_id is empty or the API is incomplete.
        :raises FileNotFoundError: If an input file is missing.
        """
        super().__init__(logger=logger)
        missing = list()
        if not isinstance(job_id, str) or not job_id:
            self._error("[StreamContext.__init__] Input error: the "
                        "parameter job_id must be a non-empty string, "
                        f"not {job_id!r}.")
        if not isinstance(metrics, ModuleType):
            self._error("[StreamContext.__init__] Input error: the "
                        "parameter metrics must be the frozen metrics "
                        f"module, not {type(metrics).__name__}.",
                        TypeError)
        missing = [a for a in METRICS_API if not hasattr(metrics, a)]
        if missing:
            self._error("[StreamContext.__init__] Input error: the "
                        "parameter metrics must provide "
                        f"{list(METRICS_API)}, and it is missing "
                        f"{missing}.")
        self.job_id = job_id
        self.metrics = metrics
        self.job_dir = Path(results_root) / job_id
        self.candidate = (str(candidate) if candidate
                          else str(metrics.PRIMARY_CANDIDATE))
        if not self.job_dir.is_dir():
            self._error("[StreamContext.__init__] Input error: the job "
                        "directory must exist, and it does not: "
                        f"{self.job_dir}.", FileNotFoundError)
        for name in (CONFIG_FILE, SCORES_FILE, LABELS_FILE):
            if not (self.job_dir / name).is_file():
                self._error("[StreamContext.__init__] Input error: the "
                            f"job must hold {name}, and it does not: "
                            f"{self.job_dir / name}.",
                            FileNotFoundError)
        self._prepare()

    def _prepare(self) -> None:
        """
        Build regions, fit set, label-free fit and detector series.

        This is the body of ``prepara`` of the original script, in the
        same order and with the same calls: the regions come from the
        protocol candidate, the fit set is the calibration interval
        minus the exact duplicates of the training segment, the scores
        are sanitized from the end of training onwards, the fit sees
        only the calibration prefix, and the transformed series start
        at ``train_end``. The "RAW" series is overwritten with the raw
        sanitization afterwards, as the original did, because the two
        sanitizations differ for non-finite scores.

        :raises KeyError: If the job's config.json has no "regions"
            section, or no "train" interval inside it.
        :raises OSError: If one of the three files cannot be read.
        """
        regions: dict[str, Any] = dict()
        calib: Any = None
        evaluation: Any = None
        scores = np.empty(0)
        detector = np.empty(0)
        raw = np.empty(0)
        duplicates = np.empty(0, dtype=bool)
        text = (self.job_dir / CONFIG_FILE).read_text(encoding="utf-8")
        self.config = json.loads(text)
        regions = self.config["regions"]
        scores = np.load(self.job_dir / SCORES_FILE)
        self.labels = np.load(self.job_dir / LABELS_FILE).astype(bool)
        self.n_records = len(scores)
        self.train_end = int(regions["train"][1])
        if self.labels.shape != scores.shape:
            self._warning("[StreamContext._prepare] Input warning: the "
                          "labels must have the shape of the scores "
                          f"{scores.shape}, not {self.labels.shape}. "
                          "They are used as they are.")
        calib, evaluation, _ = self.metrics.get_regions(regions,
                                                        self.candidate)
        duplicates, _ = self.metrics.fit_set_mask(
            self.config, self.n_records, self.train_end, log=_silent)
        self.fit_mask = np.zeros(self.n_records, bool)
        self.fit_mask[calib[0]:calib[1]] = True
        self.fit_mask &= ~duplicates
        detector, raw, _ = self.metrics.sanitize_scores(scores,
                                                        self.train_end)
        self.params = self.metrics.fit(
            detector[self.train_end:calib[1]],
            self.fit_mask[self.train_end:calib[1]], self.train_end)
        self.series = self.metrics.transform(self.params,
                                             detector[self.train_end:])
        self.series["RAW"] = raw[self.train_end:]
        self.scores = raw
        self.calibration = calib
        self.evaluation = evaluation
        self.eval_rows = self.metrics.rows_of(evaluation)

    def attack_spans(self) -> list[tuple[int, int]]:
        """
        Half-open spans of consecutive records labelled as attack.

        The figures shade every attack episode of the stream. The spans
        are read off the labels with a difference of the padded label
        vector, so a span that reaches the last record is closed at N.

        :return: Spans [a, b) of attack records, in stream order;
            empty when no record is labelled as attack.
        :rtype: list[tuple[int, int]]
        """
        padded = np.concatenate(([0], self.labels.view(np.int8), [0]))
        edges = np.diff(padded)
        return [(int(a), int(b))
                for a, b in zip(np.flatnonzero(edges == 1),
                                np.flatnonzero(edges == -1))]

    def eval_series(self, name: str) -> np.ndarray:
        """
        One detector series restricted to the evaluation rows.

        The series start at ``train_end``, so the stream indices of the
        evaluation are shifted by that offset, exactly as the metrics
        step does before computing a curve.

        :param name: Key of the series, for example "MVD_f", "KMD_f",
            "FUS" or "RAW".
        :type name: str
        :return: The values of that series on the evaluation rows,
            shape (len(eval_rows),).
        :rtype: np.ndarray
        :raises TypeError: If name is not a string.
        :raises KeyError: If the series does not exist.
        """
        if not isinstance(name, str):
            self._error("[StreamContext.eval_series] Input error: the "
                        "parameter name must be a string, not "
                        f"{type(name).__name__}.", TypeError)
        if name not in self.series:
            self._error("[StreamContext.eval_series] Input error: the "
                        "parameter name must be one of "
                        f"{sorted(self.series)}, not {name!r}.",
                        KeyError)
        return self.series[name][self.eval_rows - self.train_end]

    def threshold(self,
                  name: str,
                  quantile: str = MAIN_QUANTILE) -> float:
        """
        Calibrated threshold of one series, fitted without labels.

        :param name: Key of the series, for example "MVD_f" or "KMD_f".
        :type name: str
        :param quantile: Quantile of the fit, as the string the fit
            stores: "0.95", "0.99" or "0.995".
        :type quantile: str
        :return: The threshold; a record raises an alarm when its value
            is strictly above it.
        :rtype: float
        :raises TypeError: If name or quantile is not a string.
        :raises KeyError: If that quantile or that series has no
            threshold.
        """
        table: dict[str, Any] = dict()
        if not isinstance(name, str) or not isinstance(quantile, str):
            self._error("[StreamContext.threshold] Input error: the "
                        "parameters name and quantile must be strings, "
                        f"not {type(name).__name__} and "
                        f"{type(quantile).__name__}.", TypeError)
        table = self.params["thresholds"].get(quantile, dict())
        if name not in table:
            self._error("[StreamContext.threshold] Input error: the "
                        f"fit must hold a threshold for {name!r} at "
                        f"quantile {quantile!r}, and it holds "
                        f"{sorted(table)}.", KeyError)
        return float(table[name])

    def __repr__(self) -> str:
        """
        Show the job, the length of the stream and the regions.

        :return: Text like "StreamContext(job='T1-A-ACI__...',
            n_records=554116, train_end=72264, calibration=(72264,
            120449))".
        :rtype: str
        """
        return (f"StreamContext(job={self.job_id!r}, "
                f"n_records={self.n_records}, "
                f"train_end={self.train_end}, "
                f"calibration={tuple(self.calibration)})")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
