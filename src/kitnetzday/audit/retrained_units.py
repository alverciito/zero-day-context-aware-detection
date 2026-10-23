# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         src/kitnetzday/audit/retrained_units.py                   #
#   Version:      1.0.1                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-07                                                #
#   Modified:     2026-10-13                                                #
#   Description:  The paired differences of the retraining experiment       #
#                 averaged to the unit of analysis of the plan, and         #
#                 their exact statistics.                                   #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
from collections.abc import Mapping, Sequence
from typing import Any

from kitnetzday.audit.paired_stats import PairedStats
from kitnetzday.utils.loggable import Loggable

# Nominal two-sided alpha of the unit-level intervals. The exact interval
# is the narrowest one whose coverage is at least 1 - alpha, and the
# coverage it attains is reported next to it: 96.1 % with nine units and
# 95.2 % with eighteen.
UNIT_ALPHA: float = 0.05
# Units reported: name -> (row keys that are averaged within, value the
# "ar" flag of a row must have, or None to take both paradigms).
UNITS: tuple[tuple[str, tuple[str, ...], bool | None], ...] = (
    ("dataset", ("dataset",), None),
    ("dataset_paradigm", ("dataset", "ar"), None),
    ("dataset_TSR", ("dataset",), False),
    ("dataset_AR", ("dataset",), True))


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                MAIN CLASS                                 #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class UnitSummaries(Loggable):
    """
    The paired differences at the unit of analysis of the plan.

    A cell of the retraining experiment is one (dataset, paradigm,
    seed), so the cells of a dataset are not independent of each other:
    they share the stream. The analysis plan fixes the dataset as the
    unit of analysis, and this class restores it by averaging the
    paired differences within each unit before any statistic is
    computed. Four units are
    reported: the dataset (both seeds and both paradigms averaged), the
    (dataset, paradigm) combination (seeds averaged), and the dataset
    within each paradigm half.

    For every unit it gives the number of units, how many have a
    positive mean difference, the median, the Hodges-Lehmann estimate
    with its exact interval and realised coverage, and the exact
    Wilcoxon signed rank test, all from ``PairedStats``.

    :param stats: Helper that computes the exact paired statistics.
    :type stats: PairedStats
    :param alpha: Nominal two-sided alpha of the intervals, in (0, 1).
    :type alpha: float
    :param units: Units to report, each one as (name, row keys averaged
        within, required value of the "ar" flag or None); non-empty.
    :type units: Sequence[tuple[str, tuple[str, ...], bool | None]]
    :param logger: Logger created by the calling script, or None to stay
        silent.
    :type logger: logging.Logger | None
    :raises TypeError: If stats is not a PairedStats, alpha is not a
        number, units is not a sequence or logger is of the wrong type.
    :raises ValueError: If alpha is not in (0, 1) or units is empty.
    """

    def __init__(self,
                 stats: PairedStats,
                 alpha: float = UNIT_ALPHA,
                 units: Sequence[tuple[str, tuple[str, ...], bool | None]]
                 = UNITS,
                 logger: logging.Logger | None = None) -> None:
        """
        Check the settings and store them.

        :param stats: Helper that computes the exact paired statistics.
        :type stats: PairedStats
        :param alpha: Nominal two-sided alpha, in (0, 1).
        :type alpha: float
        :param units: Units to report; non-empty.
        :type units: Sequence[tuple[str, tuple[str, ...], bool | None]]
        :param logger: Logger created by the calling script, or None.
        :type logger: logging.Logger | None
        :raises TypeError: If a parameter is of the wrong type.
        :raises ValueError: If alpha is not in (0, 1) or units is empty.
        """
        super().__init__(logger=logger)
        if not isinstance(stats, PairedStats):
            self._error("[UnitSummaries.__init__] Input error: the "
                        "parameter stats must be a PairedStats, not "
                        f"{type(stats).__name__}.", TypeError)
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
            self._error("[UnitSummaries.__init__] Input error: the "
                        "parameter alpha must be a number, not "
                        f"{type(alpha).__name__}.", TypeError)
        if not 0 < alpha < 1:
            self._error("[UnitSummaries.__init__] Input error: the "
                        "parameter alpha must lie in (0, 1), and it is "
                        f"{alpha}.")
        if isinstance(units, (str, bytes)) or \
                not isinstance(units, Sequence):
            self._error("[UnitSummaries.__init__] Input error: the "
                        "parameter units must be a sequence, not "
                        f"{type(units).__name__}.", TypeError)
        if not units:
            self._error("[UnitSummaries.__init__] Input error: the "
                        "parameter units must hold at least one unit, "
                        "and it is empty.")
        self.stats = stats
        self.alpha = float(alpha)
        self.units = tuple(units)

    def means(self,
              rows: Sequence[Mapping[str, Any]],
              keys: Sequence[str],
              want_ar: bool | None = None) -> list[dict[str, Any]]:
        """
        Mean paired difference of every unit, in sorted order.

        :param rows: Paired rows of one variant pair, each one with the
            fields named in ``keys``, an "ar" flag and a "delta".
        :type rows: Sequence[Mapping[str, Any]]
        :param keys: Fields that define a unit, for example
            ("dataset",) or ("dataset", "ar").
        :type keys: Sequence[str]
        :param want_ar: Value the "ar" flag of a row must have to be
            used, or None to use every row.
        :type want_ar: bool | None
        :return: One {"unit", "n_cells", "mean_delta"} per unit, where
            "unit" is the list of its key values.
        :rtype: list[dict[str, Any]]
        :raises TypeError: If rows is not a sequence of mappings.
        :raises KeyError: If a row lacks one of the fields.
        """
        groups: dict[tuple[Any, ...], list[float]] = dict()
        if isinstance(rows, (str, bytes)) or \
                not isinstance(rows, Sequence) or \
                not all(isinstance(row, Mapping) for row in rows):
            self._error("[UnitSummaries.means] Input error: the "
                        "parameter rows must be a sequence of mappings, "
                        f"not {type(rows).__name__}.", TypeError)
        for row in rows:
            if want_ar is not None and bool(row["ar"]) != want_ar:
                continue
            groups.setdefault(tuple(row[k] for k in keys), list()).append(
                float(row["delta"]))
        return [{"unit": list(key), "n_cells": len(values),
                 "mean_delta": sum(values) / len(values)}
                for key, values in sorted(groups.items())]

    def summary(self, values: Sequence[float]) -> dict[str, Any]:
        """
        Exact paired statistics of the mean differences of one unit.

        :param values: Mean paired differences, one per unit; non-empty.
        :type values: Sequence[float]
        :return: {"n", "wins_ctx", "median", "min", "max", "hl",
            "hl_ci": {"lower", "upper", "coverage"}, "wilcoxon": {"W",
            "p", "n_eff"}}.
        :rtype: dict[str, Any]
        :raises TypeError: If values is not an iterable of real numbers.
        :raises ValueError: If values is empty.
        """
        deltas = [float(v) for v in values]
        lower, upper, coverage = self.stats.hl_interval(deltas,
                                                        alpha=self.alpha)
        w_plus, p_value, n_eff = self.stats.wilcoxon_exact(deltas)
        return {"n": len(deltas),
                "wins_ctx": sum(1 for v in deltas if v > 0),
                "median": self.stats.median(deltas),
                "min": min(deltas), "max": max(deltas),
                "hl": self.stats.hodges_lehmann(deltas),
                "hl_ci": {"lower": lower, "upper": upper,
                          "coverage": coverage},
                "wilcoxon": {"W": w_plus, "p": p_value, "n_eff": n_eff}}

    def summarize(self,
                  rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """
        Statistics of one variant pair at every unit that has rows.

        :param rows: Paired rows of one variant pair.
        :type rows: Sequence[Mapping[str, Any]]
        :return: Unit name -> the output of ``summary`` plus "units",
            the mean difference of each unit. A unit without rows, such
            as the autoregressive half of a pair run in one paradigm, is
            left out.
        :rtype: dict[str, Any]
        :raises TypeError: If rows is not a sequence of mappings.
        :raises KeyError: If a row lacks a field a unit needs.
        """
        out: dict[str, Any] = dict()
        for name, keys, want_ar in self.units:
            units = self.means(rows, keys, want_ar)
            if not units:
                continue
            out[name] = self.summary([u["mean_delta"] for u in units])
            out[name]["units"] = units
        return out

    def __repr__(self) -> str:
        """
        Show the nominal alpha and the units that are reported.

        :return: Text like "UnitSummaries(alpha=0.05,
            units=('dataset', 'dataset_paradigm'))".
        :rtype: str
        """
        return (f"UnitSummaries(alpha={self.alpha}, "
                f"units={tuple(name for name, _, _ in self.units)})")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
