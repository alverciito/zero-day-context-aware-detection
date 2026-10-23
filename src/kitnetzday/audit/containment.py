# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         src/kitnetzday/audit/containment.py                       #
#   Version:      2.0.0                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Containment audit (CG-2) of the retraining experiment:    #
#                 the windowed autoencoder must not reconstruct worse.      #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import json
import logging
import numbers
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from kitnetzday.audit.convergence import FLOOR_MIN, ConvergenceReports
from kitnetzday.audit.paired_stats import PairedStats
from kitnetzday.utils.loggable import Loggable

# Relative margin: a windowed autoencoder up to 2 % worse than the
# per-record one counts as a tie, which is the noise of one epoch with
# batch size 1.
TOLERANCE: float = 0.02
# Blocks whose variant carries a window: the one-epoch block and the
# five-epoch ablation. Both enter the test.
WINDOWED_GLOBS: tuple[str, ...] = (
    "T8-CTX__*/convergence_report.json",
    "T8-EP5__*/convergence_report.json")
# Prefix of the five-epoch ablation, used to label each row with its
# training budget.
EP5_PREFIX: str = "T8-EP5"
# Variants paired cell by cell: per-record control and windowed model.
PER_RECORD_VARIANT: str = "mlp_fix"
WINDOWED_VARIANT: str = "mlp_ctx_fix"
# Training budgets reported separately, in this order, with the label
# printed for each one. The manuscript cites them apart, so their rows
# are never pooled.
BUDGET_ORDER: tuple[str, ...] = ("base", "EP5")
BUDGET_LABELS: dict[str, str] = {
    "base": "1 epoca (T8-CTX)",
    "EP5": "ablacion de 5 epocas (T8-EP5)"}
# Number of visible units above which an autoencoder counts as wide, in
# the split of the ratios by width.
WIDE_VISIBLE: int = 50


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                MAIN CLASS                                 #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class ContainmentAudit(Loggable):
    """
    Containment audit (CG-2) of the retraining experiment.

    The claim behind the experiment is one of containment, not of
    detection performance: the per-record model is contained in the
    windowed one, so the windowed one should match it in the worst
    case. In TSR the
    input of the per-record model is the last row of the window and the
    target is that same row, so a weight set of the windowed model
    reproduces it exactly by zeroing the other L - 1 rows; in AR the
    input is the previous row, which the window also contains, so
    containment still holds exactly. If the windowed model validates
    worse, the cause is the optimizer, not the capacity.

    The claim therefore has to be checked per autoencoder, and it can be
    checked neither inside a single job (an autoencoder is not compared
    with itself) nor by training a reference here (that would double the
    cost of the experiment). Two tests are produced:

    - inside the job, which is the correct one: every windowed
      autoencoder against the per-record reference its own job trained,
      with the same stretch, normalization, validation cut, number of
      steps and initialization, because the windowed model starts as a
      copy of that reference;
    - pairing jobs, which is weaker and reported second: the two
      per-record models being compared start at different points of the
      random stream, so their difference also holds initialization
      variance. Pairing by autoencoder index is nevertheless legitimate,
      because the Feature Map is identical between the two variants of a
      cell: it depends only on the dataset and on the first 50k records.

    It only reads the files found under the results root. It never
    trains, never touches the GPU and never writes inside the results
    root. The printed text lives in ``ContainmentReport``.

    :param reports: Reader of the convergence reports of the
        experiment, which also holds the results root and the CG-1
        thresholds.
    :type reports: ConvergenceReports
    :param tolerance: Relative margin above 1 within which a windowed
        autoencoder still complies, as a fraction (0.02 is 2 %); must be
        >= 0.
    :type tolerance: float
    :param floor_min: Smallest constant-predictor floor that still makes
        a row evaluable; must be > 0 and should be the floor of reports.
    :type floor_min: float
    :param wide_visible: Number of visible units above which an
        autoencoder counts as wide in the split by width; must be >= 0.
    :type wide_visible: int
    :param logger: Logger created by the calling script, or None to stay
        silent.
    :type logger: logging.Logger | None
    :raises TypeError: If logger is not a logging.Logger or None, or if
        a parameter is of the wrong type.
    :raises ValueError: If a threshold is out of range.
    """

    def __init__(self,
                 reports: ConvergenceReports,
                 tolerance: float = TOLERANCE,
                 floor_min: float = FLOOR_MIN,
                 wide_visible: int = WIDE_VISIBLE,
                 logger: logging.Logger | None = None) -> None:
        """
        Check the settings and store them.

        :param reports: Reader of the convergence reports.
        :type reports: ConvergenceReports
        :param tolerance: Relative margin of the tie, >= 0.
        :type tolerance: float
        :param floor_min: Smallest evaluable floor, > 0.
        :type floor_min: float
        :param wide_visible: Visible units of a wide autoencoder, >= 0.
        :type wide_visible: int
        :param logger: Logger created by the calling script, or None.
        :type logger: logging.Logger | None
        :raises TypeError: If a parameter is of the wrong type.
        :raises ValueError: If a threshold is out of range.
        """
        super().__init__(logger=logger)
        if not isinstance(reports, ConvergenceReports):
            self._error("[ContainmentAudit.__init__] Input error: the "
                        "parameter reports must be a "
                        "ConvergenceReports, not "
                        f"{type(reports).__name__}.", TypeError)
        if isinstance(tolerance, bool) or \
                not isinstance(tolerance, numbers.Real):
            self._error("[ContainmentAudit.__init__] Input error: the "
                        "parameter tolerance must be a real number, not "
                        f"{type(tolerance).__name__}.", TypeError)
        if float(tolerance) < 0.0:
            self._error("[ContainmentAudit.__init__] Input error: the "
                        f"parameter tolerance must be >= 0, not "
                        f"{tolerance}.")
        if isinstance(floor_min, bool) or \
                not isinstance(floor_min, numbers.Real) or \
                float(floor_min) <= 0.0:
            self._error("[ContainmentAudit.__init__] Input error: the "
                        "parameter floor_min must be a real number "
                        f"strictly positive (>0), not {floor_min!r}.",
                        TypeError)
        if isinstance(wide_visible, bool) or \
                not isinstance(wide_visible, int) or wide_visible < 0:
            self._error("[ContainmentAudit.__init__] Input error: the "
                        "parameter wide_visible must be an int >= 0, "
                        f"not {wide_visible!r}.", TypeError)
        self.reports = reports
        self.tolerance = float(tolerance)
        self.floor_min = float(floor_min)
        self.wide_visible = int(wide_visible)
        self.stats = PairedStats(logger=logger)

    def within_job(self) -> list[dict[str, Any]]:
        """
        CG-2 inside the job: one row per windowed autoencoder.

        This is the correct containment test, because the reference it
        compares against was trained by the very same job from the very
        same initialization. The budget of each row is read from the
        directory name and the number of epochs from the report itself,
        so the one-epoch runs and the five-epoch ablation stay apart
        when the rows are summarized.

        :return: Rows with the job, the cell, the shape of the
            autoencoder, the constant floor, the CG-2 ratio and gate,
            whether the windowed model started as an exact copy of the
            reference, and the CG-1 verdict.
        :rtype: list[dict[str, Any]]
        """
        out: list[dict[str, Any]] = list()
        found: set[Path] = set()
        rep: Any = None
        gates: Any = None
        init: Any = None
        for pattern in WINDOWED_GLOBS:
            found |= set(self.reports.results_root.glob(pattern))
        for f in sorted(found):
            rep = self.reports.read_one(f)
            if rep is None:
                continue
            for ae in rep.get("autoencoders", list()):
                gates = ae.get("gates", dict())
                init = ae.get("containment_init") or dict()
                out.append({
                    "job_id": rep.get("job_id"),
                    "variant": rep.get("variant"),
                    "epochs": rep.get("epochs") or rep.get("haste"),
                    "budget": ("EP5"
                               if f.parent.name.startswith(EP5_PREFIX)
                               else "base"),
                    "dataset": rep.get("dataset"),
                    "seed": rep.get("seed"), "ar": rep.get("ar"),
                    "ae": ae.get("ae_index"),
                    "n_visible": ae.get("n_visible"),
                    "n_hidden": ae.get("n_hidden"),
                    "floor": gates.get("CG1_constant_floor"),
                    "ratio": gates.get(
                        "CG2_ratio_windowed_over_per_record"),
                    "pass": gates.get("CG2_pass"),
                    "copy_exact": init.get("copy_exact"),
                    "verdict": self.reports.health_verdict(gates)})
        return out

    def pair_rows(self,
                  reports: Mapping[str, Any] | None = None,
                  ) -> tuple[list[dict[str, Any]], int, int]:
        """
        CG-2 across jobs: pair the two variants of each cell by index.

        A cell missing one of the two variants, and a pair without a
        usable validation loss, are reported as such instead of being
        dropped, so the audit says what it could not check.

        Note: ``same_shape`` reads the pairing key with
        ``gates.get("CG2_pairing_key", dict())``, so a report storing an
        explicit ``null`` there raises AttributeError instead of
        reporting a mismatch; no report of the experiment stores that
        null.

        :param reports: Reports to pair instead of the ones held by the
            reader, or None to use those.
        :type reports: Mapping[str, Any] | None
        :return: The rows, how many pairs were compared and how many of
            them violate containment.
        :rtype: tuple[list[dict[str, Any]], int, int]
        :raises TypeError: If reports is neither a mapping nor None.
        """
        data: Mapping[str, Any] = dict()
        cells: dict[Any, dict[Any, tuple[str, Any]]] = dict()
        rows: list[dict[str, Any]] = list()
        pairs = 0
        violations = 0
        ratio = 0.0
        if reports is not None and not isinstance(reports, Mapping):
            self._error("[ContainmentAudit.pair_rows] Input error: the "
                        "parameter reports must be a mapping or None, "
                        f"not {type(reports).__name__}.", TypeError)
        data = self.reports.reports if reports is None else reports
        for job_id, rep in data.items():
            key, variant = self.reports.cell_key(job_id)
            if key is None:
                continue
            cells.setdefault(key, dict())[variant] = (job_id, rep)
        for key in sorted(cells):
            got = cells[key]
            if PER_RECORD_VARIANT not in got or \
                    WINDOWED_VARIANT not in got:
                rows.append({"cell": key, "status": "incompleta",
                             "present": sorted(got),
                             "missing": sorted(
                                 {PER_RECORD_VARIANT, WINDOWED_VARIANT}
                                 - set(got))})
                continue
            jid_b, rep_b = got[PER_RECORD_VARIANT]
            jid_c, rep_c = got[WINDOWED_VARIANT]
            by_index = {r["ae_index"]: r
                        for r in rep_b.get("autoencoders", list())}
            for rc in rep_c.get("autoencoders", list()):
                rb = by_index.get(rc["ae_index"])
                if rb is None:
                    continue
                gb, gc = rb.get("gates", dict()), rc.get("gates", dict())
                mb, mc = gb.get("CG2_val_mse"), gc.get("CG2_val_mse")
                same_shape = (
                    rb.get("n_visible") == rc.get("n_visible")
                    and gb.get("CG2_pairing_key", dict()).get(
                        "n_val_windows")
                    == gc.get("CG2_pairing_key", dict()).get(
                        "n_val_windows"))
                if mb is None or mc is None or not mb > 0:
                    rows.append({"cell": key, "ae": rc["ae_index"],
                                 "status": "sin datos",
                                 "mse_per_record": mb,
                                 "mse_windowed": mc,
                                 "same_shape": same_shape})
                    continue
                ratio = mc / mb
                pairs += 1
                violations += 0 if ratio <= 1.0 + self.tolerance else 1
                rows.append({
                    "cell": key, "ae": rc["ae_index"],
                    "n_visible": rc.get("n_visible"),
                    "n_hidden": rc.get("n_hidden"),
                    "same_shape": same_shape, "mse_per_record": mb,
                    "mse_windowed": mc,
                    "ratio_windowed_over_per_record": ratio,
                    "CG1_per_record": gb.get("CG1_pass"),
                    "CG1_windowed": gc.get("CG1_pass"),
                    "status": ("OK" if ratio <= 1.0 + self.tolerance
                               else "VIOLA LA CONTENCION"),
                    "jobs": {"per_record": jid_b, "windowed": jid_c}})
        return rows, pairs, violations

    def evaluable(self,
                  rows: list[dict[str, Any]],
                  budget: str) -> tuple[list[dict[str, Any]],
                                        list[dict[str, Any]]]:
        """
        Rows of one budget, and the evaluable subset of those rows.

        A row is evaluable when it carries a CG-2 ratio and its constant
        floor reaches ``floor_min``: below that floor there is no
        variance to learn and the ratio says nothing.

        :param rows: Rows returned by ``within_job``, every budget
            mixed.
        :type rows: list[dict[str, Any]]
        :param budget: Budget to keep, "base" or "EP5".
        :type budget: str
        :return: (rows of the budget, evaluable rows of the budget).
        :rtype: tuple[list[dict[str, Any]], list[dict[str, Any]]]
        :raises TypeError: If rows is not a list or budget is not a
            string.
        """
        sub: list[dict[str, Any]] = list()
        if not isinstance(rows, list) or not isinstance(budget, str):
            self._error("[ContainmentAudit.evaluable] Input error: the "
                        "parameters rows and budget must be a list and "
                        f"a string, not {type(rows).__name__} and "
                        f"{type(budget).__name__}.", TypeError)
        sub = [r for r in rows if r.get("budget") == budget]
        return sub, [r for r in sub
                     if r["ratio"] is not None and r["floor"]
                     and r["floor"] >= self.floor_min]

    def complying(self,
                  rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Rows whose ratio stays within the tolerance.

        :param rows: Evaluable rows of one budget.
        :type rows: list[dict[str, Any]]
        :return: The rows that comply with containment.
        :rtype: list[dict[str, Any]]
        """
        return [r for r in rows if r["ratio"] <= 1.0 + self.tolerance]

    def improving(self,
                  rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Rows in which the window improves beyond the tolerance.

        :param rows: Evaluable rows of one budget.
        :type rows: list[dict[str, Any]]
        :return: The rows in which the windowed model reconstructs
            better than the per-record reference by more than the
            tolerance.
        :rtype: list[dict[str, Any]]
        """
        return [r for r in rows if r["ratio"] < 1.0 - self.tolerance]

    def within_job_summary(self,
                           rows: list[dict[str, Any]],
                           ) -> dict[str, Any]:
        """
        Summarize the CG-2 rows inside the job, one block per budget.

        :param rows: Rows returned by ``within_job``.
        :type rows: list[dict[str, Any]]
        :return: Budget -> block with how many windowed autoencoders there
            are, how many started as an exact copy, how many are
            evaluable, comply, violate or improve, and the ratio
            statistics, also split by width.
        :rtype: dict[str, Any]
        """
        out: dict[str, Any] = dict()
        blk: dict[str, Any] = dict()
        sub: list[dict[str, Any]] = list()
        ev: list[dict[str, Any]] = list()
        ok: list[dict[str, Any]] = list()
        ratios: list[float] = list()
        wide: list[float] = list()
        narrow: list[float] = list()
        if not rows:
            return out
        for budget in BUDGET_ORDER:
            sub, ev = self.evaluable(rows, budget)
            if not sub:
                continue
            ok = self.complying(ev)
            blk = {"n_windowed": len(sub),
                   "copy_exact": sum(1 for r in sub if r["copy_exact"]),
                   "n_evaluable": len(ev), "n_comply": len(ok),
                   "n_violate": len(ev) - len(ok),
                   "n_improve": len(self.improving(ev))}
            if ev:
                ratios = sorted(r["ratio"] for r in ev)
                wide = [r["ratio"] for r in ev if r["n_visible"]
                        and r["n_visible"] > self.wide_visible]
                narrow = [r["ratio"] for r in ev if r["n_visible"]
                          and r["n_visible"] <= self.wide_visible]
                blk.update({"median_ratio": self.stats.median(ratios),
                            "min": ratios[0], "max": ratios[-1],
                            "median_ratio_nv_gt_50":
                                self.stats.median(wide),
                            "n_nv_gt_50": len(wide),
                            "median_ratio_nv_le_50":
                                self.stats.median(narrow),
                            "n_nv_le_50": len(narrow)})
            out[budget] = blk
        return out

    def report(self) -> dict[str, Any]:
        """
        Full audit: health tally, CG-2 inside the job and across jobs.

        The convergence reports are read first and kept on the reader,
        so that ``ContainmentReport`` can tell an empty results root
        from an audit with nothing to flag.

        :return: {"tolerance", "floor_min", "n_pairs", "n_violations",
            "health_tally", "within_job_summary", "within_job",
            "rows"}, in the order the submitted JSON used.
        :rtype: dict[str, Any]
        """
        rows: list[dict[str, Any]] = list()
        windowed: list[dict[str, Any]] = list()
        tally: dict[str, int] = dict()
        pairs = 0
        violations = 0
        self.reports.load()
        rows, pairs, violations = self.pair_rows()
        tally = self.reports.health_tally()
        windowed = self.within_job()
        return {"tolerance": self.tolerance, "floor_min": self.floor_min,
                "n_pairs": pairs, "n_violations": violations,
                "health_tally": tally,
                "within_job_summary": self.within_job_summary(windowed),
                "within_job": windowed, "rows": rows}

    def dump_json(self,
                  report: Mapping[str, Any],
                  path: str | Path) -> Path:
        """
        Write the audit as JSON, with one space of indentation.

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
            self._error("[ContainmentAudit.dump_json] Input error: the "
                        "parameter report must be a mapping, not "
                        f"{type(report).__name__}.", TypeError)
        if not isinstance(path, (str, Path)) or not str(path):
            self._error("[ContainmentAudit.dump_json] Input error: the "
                        "parameter path must be a non-empty str or "
                        f"Path, not {path!r}.", TypeError)
        Path(path).write_text(json.dumps(report, indent=1),
                             encoding="utf-8")
        return Path(path)

    def __repr__(self) -> str:
        """
        Show the reader and the thresholds of the audit.

        :return: Text like "ContainmentAudit(results_root='...',
            tolerance=0.02, floor_min=0.0001, reports=24)".
        :rtype: str
        """
        return (f"ContainmentAudit(results_root="
                f"{str(self.reports.results_root)!r}, "
                f"tolerance={self.tolerance}, "
                f"floor_min={self.floor_min}, "
                f"reports={len(self.reports.reports)})")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
