# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         tests/audit/test_retrained_units.py                       #
#   Version:      1.0.1                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-07                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Unit tests for the paired differences of the              #
#                 retraining experiment at the unit of the plan.            #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
from typing import Any

import pytest

from kitnetzday.audit.paired_stats import PairedStats
from kitnetzday.audit.retrained_units import UNIT_ALPHA, UnitSummaries

# The toy case: two datasets, two paradigms, two seeds. The differences
# are chosen so that every mean is a round number; the statistics are
# worked out in the docstrings of the tests.
#   dataset A: TSR +0.1, +0.3 (mean +0.2); AR -0.2, 0.0 (mean -0.1)
#   dataset B: TSR +0.2, +0.4 (mean +0.3); AR +0.1, +0.5 (mean +0.3)
DELTAS: tuple[tuple[str, bool, int, float], ...] = (
    ("A", False, 1, 0.1), ("A", False, 2, 0.3),
    ("A", True, 1, -0.2), ("A", True, 2, 0.0),
    ("B", False, 1, 0.2), ("B", False, 2, 0.4),
    ("B", True, 1, 0.1), ("B", True, 2, 0.5))


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                AUXILIARY                                  #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def _rows(deltas: tuple[tuple[str, bool, int, float], ...] = DELTAS,
          ) -> list[dict[str, Any]]:
    """
    Paired rows as ``RetrainedPair.paired_rows`` builds them.

    :param deltas: (dataset, ar, seed, paired difference) of every cell.
    :type deltas: tuple[tuple[str, bool, int, float], ...]
    :return: One row per cell, with the fields the summaries read.
    :rtype: list[dict[str, Any]]
    """
    return [{"dataset": dataset, "ar": ar, "seed": seed, "delta": delta}
            for dataset, ar, seed, delta in deltas]


def _units() -> UnitSummaries:
    """
    The summaries with the settings of the package.

    :return: A ``UnitSummaries`` over a silent ``PairedStats``.
    :rtype: UnitSummaries
    """
    return UnitSummaries(PairedStats())


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                  TESTS                                    #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class TestMeans:
    """
    ``UnitSummaries.means``: the average within each unit.
    """

    def test_dataset_means(self) -> None:
        """
        The four cells of a dataset are averaged into one value.

        A: (0.1 + 0.3 - 0.2 + 0.0) / 4 = 0.05.
        B: (0.2 + 0.4 + 0.1 + 0.5) / 4 = 0.30.
        """
        got = _units().means(_rows(), ("dataset",))
        assert [u["unit"] for u in got] == [["A"], ["B"]]
        assert [u["n_cells"] for u in got] == [4, 4]
        assert [u["mean_delta"] for u in got] == pytest.approx(
            [0.05, 0.30])

    def test_dataset_paradigm_means(self) -> None:
        """
        The two seeds of a (dataset, paradigm) are averaged.

        Sorted by (dataset, ar), with False before True: A-TSR 0.2,
        A-AR -0.1, B-TSR 0.3, B-AR 0.3.
        """
        got = _units().means(_rows(), ("dataset", "ar"))
        assert [tuple(u["unit"]) for u in got] == [
            ("A", False), ("A", True), ("B", False), ("B", True)]
        assert [u["mean_delta"] for u in got] == pytest.approx(
            [0.2, -0.1, 0.3, 0.3])

    def test_paradigm_filter(self) -> None:
        """
        Only the rows of the requested paradigm are averaged.

        Autoregressive half: A (-0.2 + 0.0) / 2 = -0.1, B (0.1 + 0.5)
        / 2 = 0.3.
        """
        got = _units().means(_rows(), ("dataset",), want_ar=True)
        assert [u["n_cells"] for u in got] == [2, 2]
        assert [u["mean_delta"] for u in got] == pytest.approx(
            [-0.1, 0.3])

    def test_no_rows(self) -> None:
        """
        No row gives no unit, not an error.
        """
        assert _units().means(list(), ("dataset",)) == list()

    def test_rows_must_be_mappings(self) -> None:
        """
        A sequence of something else is refused with its type named.
        """
        with pytest.raises(TypeError, match="sequence of mappings"):
            _units().means("rows", ("dataset",))


class TestSummary:
    """
    ``UnitSummaries.summary`` and ``summarize``: the exact statistics.
    """

    def test_dataset_unit_by_hand(self) -> None:
        """
        Two dataset means, 0.05 and 0.30.

        Walsh averages: 0.05, (0.05 + 0.30) / 2 = 0.175, 0.30; their
        median, the Hodges-Lehmann estimate, is 0.175. With n = 2 the
        null distribution of W+ puts 1/4 on each of 0, 1, 2 and 3, so
        not even P(W+ <= 0) = 0.25 is below alpha / 2 = 0.025: the
        interval is the whole range of the Walsh averages, with
        coverage 1. Both differences are positive, so W+ = 1 + 2 = 3
        and the two-sided p value counts the patterns with a sum >= 3
        or <= 0, which are 2 of 4: p = 0.5.
        """
        got = _units().summarize(_rows())["dataset"]
        assert got["n"] == 2 and got["wins_ctx"] == 2
        assert got["hl"] == pytest.approx(0.175)
        assert got["median"] == pytest.approx(0.175)
        assert got["hl_ci"]["lower"] == pytest.approx(0.05)
        assert got["hl_ci"]["upper"] == pytest.approx(0.30)
        assert got["hl_ci"]["coverage"] == pytest.approx(1.0)
        assert got["wilcoxon"]["W"] == pytest.approx(3.0)
        assert got["wilcoxon"]["p"] == pytest.approx(0.5)
        assert got["wilcoxon"]["n_eff"] == 2

    def test_dataset_paradigm_unit_by_hand(self) -> None:
        """
        Four (dataset, paradigm) means: 0.2, -0.1, 0.3 and 0.3.

        The ten Walsh averages, sorted: -0.1, 0.05, 0.1, 0.1, 0.2,
        0.25, 0.25, 0.3, 0.3, 0.3; their median is (0.2 + 0.25) / 2 =
        0.225. Ranks of the absolute values: 0.1 -> 1, 0.2 -> 2 and
        the two 0.3 share (3 + 4) / 2 = 3.5, so W+ = 2 + 3.5 + 3.5 = 9
        out of a rank sum of 10. Of the 16 sign patterns, two have a
        sum >= 9 (9 and 10) and two a sum <= 1 (0 and 1): p = 4 / 16.
        Three of the four means are positive.
        """
        got = _units().summarize(_rows())["dataset_paradigm"]
        assert got["n"] == 4 and got["wins_ctx"] == 3
        assert got["hl"] == pytest.approx(0.225)
        assert got["min"] == pytest.approx(-0.1)
        assert got["max"] == pytest.approx(0.3)
        assert got["wilcoxon"]["W"] == pytest.approx(9.0)
        assert got["wilcoxon"]["p"] == pytest.approx(0.25)

    def test_paradigm_halves_by_hand(self) -> None:
        """
        Each half at the dataset level.

        Full-sequence means 0.2 and 0.3: Walsh averages 0.2, 0.25, 0.3,
        estimate 0.25. Autoregressive means -0.1 and 0.3: Walsh
        averages -0.1, 0.1, 0.3, estimate 0.1, one positive mean.
        """
        got = _units().summarize(_rows())
        assert got["dataset_TSR"]["hl"] == pytest.approx(0.25)
        assert got["dataset_AR"]["hl"] == pytest.approx(0.1)
        assert got["dataset_AR"]["wins_ctx"] == 1
        assert [u["mean_delta"] for u in got["dataset_AR"]["units"]] \
            == pytest.approx([-0.1, 0.3])

    def test_unit_without_rows_is_left_out(self) -> None:
        """
        A pair run in one paradigm has no autoregressive half.
        """
        rows = [row for row in _rows() if not row["ar"]]
        got = _units().summarize(rows)
        assert sorted(got) == ["dataset", "dataset_TSR",
                               "dataset_paradigm"]

    def test_nine_units_reach_the_level_of_the_plan(self) -> None:
        """
        With nine units the exact interval has coverage 0.9609375.

        For n = 9 there are 2 ** 9 = 512 sign patterns, and the number
        with W+ <= 5 is 1 + 1 + 1 + 2 + 2 + 3 = 10 (the ways to write
        0, 1, 2, 3, 4 and 5 as a sum of distinct ranks). 10 / 512 =
        0.0195 is the largest tail below alpha / 2 = 0.025, so the
        interval drops five Walsh averages on each side and its
        coverage is 1 - 20 / 512 = 0.9609375, the 96.1 % of the plan.
        With the differences 1, 2, ..., 9 the 45 sorted Walsh averages
        start 1, 1.5, 2, 2, 2.5, 2.5 (1 with 4 and 2 with 3 both give
        2.5), so the lower bound is the sixth one, 2.5, and by
        symmetry around 5 the upper bound is 7.5.
        """
        got = _units().summary([float(v) for v in range(1, 10)])
        assert got["hl_ci"]["coverage"] == pytest.approx(0.9609375)
        assert got["hl_ci"]["lower"] == pytest.approx(2.5)
        assert got["hl_ci"]["upper"] == pytest.approx(7.5)
        assert got["hl"] == pytest.approx(5.0)


class TestSettings:
    """
    Constructor checks and the printed form.
    """

    def test_default_alpha(self) -> None:
        """
        The nominal level of the unit intervals is 95 %.
        """
        assert UNIT_ALPHA == pytest.approx(0.05)
        assert _units().alpha == pytest.approx(0.05)

    def test_repr_names_the_units(self) -> None:
        """
        The printed form shows the alpha and the unit names.
        """
        text = repr(_units())
        assert text.startswith("UnitSummaries(alpha=0.05")
        assert "'dataset_paradigm'" in text

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 2.0])
    def test_alpha_out_of_range(self, alpha: float) -> None:
        """
        An alpha outside (0, 1) is refused.

        :param alpha: Value that must be rejected.
        :type alpha: float
        """
        with pytest.raises(ValueError, match="must lie in"):
            UnitSummaries(PairedStats(), alpha=alpha)

    def test_wrong_types(self) -> None:
        """
        Each parameter of the wrong type is refused by name.
        """
        with pytest.raises(TypeError, match="stats must be"):
            UnitSummaries("stats")
        with pytest.raises(TypeError, match="alpha must be a number"):
            UnitSummaries(PairedStats(), alpha="0.05")
        with pytest.raises(TypeError, match="units must be a sequence"):
            UnitSummaries(PairedStats(), units="dataset")
        with pytest.raises(ValueError, match="at least one unit"):
            UnitSummaries(PairedStats(), units=tuple())


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
