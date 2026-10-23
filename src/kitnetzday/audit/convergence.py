# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         src/kitnetzday/audit/convergence.py                       #
#   Version:      1.0.2                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Convergence reports of the retraining experiment and      #
#                 the health verdict (CG-1) of each autoencoder.            #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import json
import logging
import numbers
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from kitnetzday.utils.loggable import Loggable

# Floor of the constant predictor below which the GROUP has no variance
# left to learn, so no gate can tell a trained autoencoder from an
# untrained one.
FLOOR_MIN: float = 1.0e-4
# Health gate CG-1: a validation loss below this fraction of the
# constant floor means the autoencoder learned something, above the
# second one means it is degenerate, and in between the verdict is
# undecided. Both values are the ones of the frozen plan (config.json).
TRAINED_BELOW: float = 0.9
DEGENERATE_ABOVE: float = 1.15
# Convergence reports of the experiment, relative to the results root.
REPORT_GLOB: str = "T8-*/convergence_report.json"
# Verdicts of the health gate. They are output strings of the runner
# and stay in Spanish, as the convergence reports hold them.
VERDICT_NO_DATA: str = "sin datos"
VERDICT_CONSTANT: str = "datos casi constantes (no evaluable)"
VERDICT_TRAINED: str = "entrenado"
VERDICT_DEGENERATE: str = "degenerado"
VERDICT_UNDECIDED: str = "indeciso"
# Verdicts that count towards the denominator of the health summary: the
# autoencoders whose group still had variance to learn.
EVALUABLE_VERDICTS: tuple[str, ...] = (
    VERDICT_TRAINED, VERDICT_DEGENERATE, VERDICT_UNDECIDED)


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                MAIN CLASS                                 #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class ConvergenceReports(Loggable):
    """
    Convergence reports of the retraining experiment, and their gate.

    Every job of it writes one ``convergence_report.json`` holding,
    per autoencoder, the gates that were measured while it trained. This
    class reads those files and turns the CG-1 gate into a verdict.

    The verdict keeps a degenerate MODEL apart from degenerate DATA,
    which matters: in one dataset a group has seven of its eight
    features constant over the training stretch and a constant-predictor
    floor of 3.5e-5, so that predictor is already almost perfect and no
    model can improve on it, while its output is constant by
    definition. Reported against the floor alone such an autoencoder
    looks untrained, which is false: there was nothing to learn. Below
    ``floor_min`` the autoencoder is therefore reported as not
    evaluable, and it is left out of the denominator of the summary.

    It only reads the files found under the results root, and a file
    that cannot be parsed is skipped with a warning, so one unreadable
    report does not stop the audit.

    :param results_root: Directory holding the job directories of the
        experiment, ``data/jobs`` in this package.
    :type results_root: str | Path
    :param floor_min: Smallest constant-predictor floor that still makes
        an autoencoder evaluable; must be > 0.
    :type floor_min: float
    :param trained_below: Ratio of validation loss to floor below which
        the autoencoder counts as trained; must be > 0.
    :type trained_below: float
    :param degenerate_above: Ratio above which it counts as degenerate;
        must be >= trained_below.
    :type degenerate_above: float
    :param report_glob: Glob of the convergence reports, relative to the
        results root; non-empty.
    :type report_glob: str
    :param logger: Logger created by the calling script, or None to stay
        silent.
    :type logger: logging.Logger | None
    :raises TypeError: If logger is not a logging.Logger or None, or if
        a parameter is of the wrong type.
    :raises ValueError: If a threshold is out of range.
    """

    def __init__(self,
                 results_root: str | Path,
                 floor_min: float = FLOOR_MIN,
                 trained_below: float = TRAINED_BELOW,
                 degenerate_above: float = DEGENERATE_ABOVE,
                 report_glob: str = REPORT_GLOB,
                 logger: logging.Logger | None = None) -> None:
        """
        Check the settings and store them; ``load`` reads the files.

        :param results_root: Directory holding the job directories.
        :type results_root: str | Path
        :param floor_min: Smallest evaluable floor, > 0.
        :type floor_min: float
        :param trained_below: Ratio below which a model is trained, > 0.
        :type trained_below: float
        :param degenerate_above: Ratio above which it is degenerate,
            >= trained_below.
        :type degenerate_above: float
        :param report_glob: Glob of the convergence reports; non-empty.
        :type report_glob: str
        :param logger: Logger created by the calling script, or None.
        :type logger: logging.Logger | None
        :raises TypeError: If a parameter is of the wrong type.
        :raises ValueError: If a threshold is out of range.
        """
        super().__init__(logger=logger)
        if not isinstance(results_root, (str, Path)) or \
                not str(results_root):
            self._error("[ConvergenceReports.__init__] Input error: the "
                        "parameter results_root must be a non-empty str "
                        f"or Path, not {results_root!r}.", TypeError)
        self._check_positive("floor_min", floor_min)
        self._check_positive("trained_below", trained_below)
        self._check_positive("degenerate_above", degenerate_above)
        if float(degenerate_above) < float(trained_below):
            self._error("[ConvergenceReports.__init__] Input error: the "
                        "parameter degenerate_above must be >= "
                        f"trained_below ({trained_below}), not "
                        f"{degenerate_above}.")
        if not isinstance(report_glob, str) or not report_glob:
            self._error("[ConvergenceReports.__init__] Input error: the "
                        "parameter report_glob must be a non-empty "
                        f"string, not {report_glob!r}.", TypeError)
        self.results_root = Path(results_root)
        self.floor_min = float(floor_min)
        self.trained_below = float(trained_below)
        self.degenerate_above = float(degenerate_above)
        self.report_glob = report_glob
        self.reports: dict[str, Any] = dict()

    def load(self) -> dict[str, Any]:
        """
        Read every convergence report found, keyed by job id.

        The files are visited in sorted order and kept on the instance,
        so that a caller can tell an empty results root from an audit
        with nothing to report.

        :return: Job id -> parsed convergence report. The job id is the
            one inside the report, or the directory name when the report
            does not carry it.
        :rtype: dict[str, Any]
        """
        out: dict[str, Any] = dict()
        rep: Any = None
        if not self.results_root.is_dir():
            self._warning("[ConvergenceReports.load] Input warning: the "
                          "parameter results_root must be an existing "
                          f"directory, and {self.results_root} is not. "
                          "No report is loaded.")
            self.reports = out
            return out
        for f in sorted(self.results_root.glob(self.report_glob)):
            rep = self.read_one(f)
            if rep is None:
                continue
            out[rep.get("job_id") or f.parent.name] = rep
        self.reports = out
        return out

    def read_one(self, path: str | Path) -> Any:
        """
        Parse one convergence report, or warn and give None.

        Note: the submitted script wrote this warning on standard error;
        here it goes to the logger, which changes no computed value.

        :param path: Path of the ``convergence_report.json``.
        :type path: str | Path
        :return: The parsed document, or None when it cannot be read.
        :rtype: Any
        :raises TypeError: If path is not a str or Path.
        """
        if not isinstance(path, (str, Path)):
            self._error("[ConvergenceReports.read_one] Input error: the "
                        "parameter path must be a str or Path, not "
                        f"{type(path).__name__}.", TypeError)
        try:
            return json.loads(Path(path).read_text())
        except Exception as exc:  # noqa: BLE001
            self._warning("[ConvergenceReports.read_one] Input warning: "
                          "every convergence report must be readable "
                          f"JSON, and {path} is not "
                          f"({type(exc).__name__}: {exc}). The file is "
                          "skipped.")
            return None

    def health_verdict(self, gates: Mapping[str, Any]) -> str:
        """
        Classify one autoencoder from the gates already stored.

        :param gates: The ``gates`` section of one autoencoder of a
            convergence report, with ``CG1_constant_floor`` (the loss of
            the constant predictor) and ``CG1_ratio_val`` (the
            validation loss divided by that floor).
        :type gates: Mapping[str, Any]
        :return: One of the five verdicts of the module, in Spanish as
            the report prints them.
        :rtype: str
        :raises TypeError: If gates is not a mapping.
        """
        floor: Any = None
        ratio: Any = None
        if not isinstance(gates, Mapping):
            self._error("[ConvergenceReports.health_verdict] Input "
                        "error: the parameter gates must be a mapping, "
                        f"not {type(gates).__name__}.", TypeError)
        floor = gates.get("CG1_constant_floor")
        ratio = gates.get("CG1_ratio_val")
        if floor is None or ratio is None:
            return VERDICT_NO_DATA
        if floor < self.floor_min:
            return VERDICT_CONSTANT
        if ratio < self.trained_below:
            return VERDICT_TRAINED
        if ratio > self.degenerate_above:
            return VERDICT_DEGENERATE
        return VERDICT_UNDECIDED

    def health_tally(self,
                     reports: Mapping[str, Any] | None = None,
                     ) -> dict[str, int]:
        """
        Count the verdicts of every autoencoder of every report.

        :param reports: Reports to count instead of the ones held by the
            instance, or None to use those.
        :type reports: Mapping[str, Any] | None
        :return: Verdict -> how many autoencoders got it.
        :rtype: dict[str, int]
        :raises TypeError: If reports is neither a mapping nor None.
        """
        tally: dict[str, int] = dict()
        data: Mapping[str, Any] = dict()
        verdict = ""
        if reports is not None and not isinstance(reports, Mapping):
            self._error("[ConvergenceReports.health_tally] Input error: "
                        "the parameter reports must be a mapping or "
                        f"None, not {type(reports).__name__}.",
                        TypeError)
        data = self.reports if reports is None else reports
        for _, rep in sorted(data.items()):
            for ae in rep.get("autoencoders", list()):
                verdict = self.health_verdict(ae.get("gates", dict()))
                tally[verdict] = tally.get(verdict, 0) + 1
        return tally

    def cell_key(self, job_id: str) -> tuple[str | None, str | None]:
        """
        Split a job id into its cell and its variant.

        A cell is everything but the variant, so that the two variants
        of the same stretch, paradigm, seed and repetition are paired.
        The block is dropped on purpose: it is what tells T8-BASE
        (per-record) from T8-CTX (windowed), which is the very
        difference being measured.

        :param job_id: Job id in the form
            BLOCK__dataset__variant__arX__pipeline__L__k__sSEED__rN.
        :type job_id: str
        :return: (cell key, variant), or (None, None) when the job id
            has fewer than four fields.
        :rtype: tuple[str | None, str | None]
        :raises TypeError: If job_id is not a string.
        """
        parts: list[str] = list()
        if not isinstance(job_id, str):
            self._error("[ConvergenceReports.cell_key] Input error: the "
                        "parameter job_id must be a string, not "
                        f"{type(job_id).__name__}.", TypeError)
        parts = job_id.split("__")
        if len(parts) < 4:
            return None, None
        return "__".join([parts[1]] + parts[3:]), parts[2]

    def __repr__(self) -> str:
        """
        Show the results root, the gate thresholds and what is loaded.

        :return: Text like "ConvergenceReports(results_root='...',
            floor_min=0.0001, reports=24)".
        :rtype: str
        """
        return (f"ConvergenceReports("
                f"results_root={str(self.results_root)!r}, "
                f"floor_min={self.floor_min}, "
                f"trained_below={self.trained_below}, "
                f"degenerate_above={self.degenerate_above}, "
                f"reports={len(self.reports)})")

    def _check_positive(self, name: str, value: Any) -> None:
        """
        Check that a threshold is a strictly positive real number.

        :param name: Name of the parameter, for the message.
        :type name: str
        :param value: Value given by the caller.
        :type value: Any
        :raises TypeError: If value is not a real number.
        :raises ValueError: If value is not strictly positive.
        """
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            self._error("[ConvergenceReports.__init__] Input error: the "
                        f"parameter {name} must be a real number, not "
                        f"{type(value).__name__}.", TypeError)
        if float(value) <= 0.0:
            self._error("[ConvergenceReports.__init__] Input error: the "
                        f"parameter {name} must be strictly positive "
                        f"(>0), not {value}.")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
