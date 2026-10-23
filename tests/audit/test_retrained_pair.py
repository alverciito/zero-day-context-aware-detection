# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         tests/audit/test_retrained_pair.py                        #
#   Version:      1.0.1                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Unit tests for the paired statistics of the retraining    #
#                 experiment and for its win counting.                      #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import json
import logging
from pathlib import Path
from typing import Any

import pytest

from kitnetzday.audit.paired_stats import PairedStats
from kitnetzday.audit.retrained_pair import PAIRS, RetrainedPair

# The four paired cells every test of the win counting uses. Every
# ROC-AUC is an exact binary fraction, so the differences, the median
# and the Walsh averages are exact in float64 and can be checked by
# hand. Cell -> variant -> ROC-AUC.
CELLS: dict[tuple[Any, ...], dict[str, float]] = {
    ("cic2017", False, "A", 1): {"mlp_ctx_fix": 0.75, "mlp_fix": 0.50},
    ("cic2017", False, "B", 1): {"mlp_ctx_fix": 0.25, "mlp_fix": 0.50},
    ("cic2017", True, "A", 1): {"mlp_ctx_fix": 0.50, "mlp_fix": 0.50},
    ("knad-os_scan", False, "A", 1): {"mlp_ctx_fix": 0.875,
                                      "mlp_fix": 0.375}}


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                AUXILIARY                                  #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def _cells() -> dict[tuple[Any, ...], dict[str, Any]]:
    """
    Build the cell map in the shape ``load_cells`` returns it in.

    :return: (dataset, ar, seed, rep) -> variant -> its record.
    :rtype: dict[tuple[Any, ...], dict[str, Any]]
    """
    return {key: {variant: {"job_id": f"T8-{variant}", "auc": value,
                            "metric": "auroc_FUS", "path": "x.json"}
                  for variant, value in got.items()}
            for key, got in CELLS.items()}


def _write_metrics(results: Path,
                   job_id: str,
                   dataset: str,
                   variant: str,
                   value: float | None,
                   ar: bool = False,
                   seed: str = "A") -> None:
    """
    Write one ``T8-*/metrics.json`` that ``RetrainedPair`` can read.

    :param results: Results root the job directory is created in.
    :type results: Path
    :param job_id: Name of the job directory, which must match the glob.
    :type job_id: str
    :param dataset: Dataset directory name of the job.
    :type dataset: str
    :param variant: Variant of the job.
    :type variant: str
    :param value: Primary metric of the job, or None to leave it out,
        which is how a job without a primary metric looks.
    :type value: float | None
    :param ar: Value of the "ar" flag of the job.
    :type ar: bool
    :param seed: Seed of the job.
    :type seed: str
    """
    path = results / job_id
    path.mkdir(parents=True, exist_ok=True)
    (path / "metrics.json").write_text(json.dumps({
        "job": {"job_id": job_id, "dataset": dataset, "ar": ar,
                "seed": seed, "rep": 1, "variant": variant},
        "primary": {"value": value, "name": "auroc_FUS"}}),
        encoding="utf-8")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                MAIN CLASS                                 #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class TestHodgesLehmann:
    """
    Tests for the Hodges-Lehmann estimate and its exact interval.
    """

    def test_analytical_walsh_median(self) -> None:
        """
        Hand-computed case for the differences [1, 2, 4].

        By hand: the Walsh averages are the n (n + 1) / 2 = 6 pairwise
        means (x_i + x_j) / 2 with i <= j, the self-pairs included:
        (1 + 1) / 2 = 1, (1 + 2) / 2 = 1.5, (1 + 4) / 2 = 2.5,
        (2 + 2) / 2 = 2, (2 + 4) / 2 = 3 and (4 + 4) / 2 = 4. Sorted,
        they are 1, 1.5, 2, 2.5, 3, 4. With six values the median is the
        mean of the third and the fourth, (2 + 2.5) / 2 = 2.25, which is
        the estimate. Note that it is not the plain median of the
        differences, which is 2.
        """
        stats = PairedStats()
        assert stats.walsh([1, 2, 4]) == [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
        assert stats.hodges_lehmann([1, 2, 4]) == pytest.approx(2.25)
        assert stats.median([1, 2, 4]) == 2.0

    def test_walsh_count(self) -> None:
        """
        There are n (n + 1) / 2 Walsh averages, the self-pairs included.
        """
        stats = PairedStats()
        for n in (1, 2, 4, 7):
            assert len(stats.walsh(list(range(n)))) == n * (n + 1) // 2

    def test_analytical_interval_of_eight_differences(self) -> None:
        """
        Hand-computed interval for the differences [1, ..., 8].

        By hand. The 36 Walsh averages of 1..8 are the halves of every
        i + j with i <= j, so sorted they start 1 (from 1+1), 1.5 (1+2),
        2, 2 (1+3 and 2+2), 2.5, 2.5 (1+4 and 2+3), ... and end
        symmetrically 6.5, 6.5 (5+8 and 6+7), 7, 7, 7.5, 8. The fourth
        smallest, at index 3, is therefore 2 and the fourth largest, at
        index 32, is 7.

        The null distribution of W+ for eight differences counts the
        subsets of the ranks 1..8 that add up to each value: 1 subset
        reaches 0, 1 reaches 1, 1 reaches 2, 2 reach 3 ({3} and {1, 2})
        and 2 reach 4 ({4} and {1, 3}), out of 2^8 = 256 subsets. The
        two-sided alpha of the plan is 1 - 0.9609375 = 0.0390625, so
        alpha / 2 = 5 / 256 exactly. Walking the support, the cumulative
        mass reaches 5 / 256 at W+ = 3 without exceeding alpha / 2, and
        the next value would take it to 7 / 256, so the critical value
        is c = 3 and the cumulative mass is 5 / 256. The interval is
        [W_(c+1), W_(M-c)] = [W_(4), W_(33)], the Walsh averages at
        index 3 and at index 32, that is 2 and 7, and the realised
        coverage is 1 - 2 * 5 / 256 = 0.9609375, the confidence level of
        the plan. Indexing with c + 1 instead would give [2.5, 6.5],
        whose true coverage is only 1 - 2 * 7 / 256 = 0.9453: that was
        the defect corrected on 7 October. The estimate itself is the
        median of the 36 Walsh averages, 4.5.
        """
        stats = PairedStats()
        values = list(range(1, 9))
        lower, upper, coverage = stats.hl_interval(values)
        assert (lower, upper) == (2.0, 7.0)
        assert coverage == pytest.approx(0.9609375)
        assert stats.hodges_lehmann(values) == pytest.approx(4.5)

    def test_analytical_interval_degenerates_with_three(self) -> None:
        """
        Hand-computed interval for the differences [1, 2, 3].

        By hand: with three differences there are only 2^3 = 8 sign
        patterns, so the smallest mass any tail can have is 1 / 8 =
        0.125, well above alpha / 2 = 0.0195. The walk over the support
        therefore stops at once with no critical value and no mass
        accumulated, and
        the interval is the whole range of the six Walsh averages of
        1, 2, 3, which run from (1 + 1) / 2 = 1 to (3 + 3) / 2 = 3. The
        realised coverage is 1 - 0 = 1: with three cells the interval
        cannot say anything, and it says so instead of pretending.
        """
        lower, upper, coverage = PairedStats().hl_interval([1, 2, 3])
        assert (lower, upper, coverage) == (1.0, 3.0, 1.0)

    def test_empty_samples(self) -> None:
        """
        An empty sample has no median, no estimate and no interval.
        """
        stats = PairedStats()
        assert stats.median(tuple()) is None
        assert stats.walsh(tuple()) == list()
        with pytest.raises(ValueError, match="at least one"):
            stats.hodges_lehmann(tuple())
        with pytest.raises(ValueError, match="at least one"):
            stats.hl_interval(tuple())

    def test_single_difference(self) -> None:
        """
        One difference is its own median, estimate and interval.
        """
        stats = PairedStats()
        assert stats.median([0.5]) == 0.5
        assert stats.hodges_lehmann([0.5]) == 0.5
        assert stats.hl_interval([0.5]) == (0.5, 0.5, 1.0)

    def test_wrong_types_raise_and_log(
            self,
            caplog: pytest.LogCaptureFixture) -> None:
        """
        Text, booleans and non-iterables are not samples.

        :param caplog: pytest fixture that captures log records.
        :type caplog: pytest.LogCaptureFixture
        """
        stats = PairedStats(logger=logging.getLogger("test"))
        with caplog.at_level(logging.ERROR, logger="test"):
            with pytest.raises(TypeError,
                               match="iterable of real numbers"):
                stats.median("abc")
        assert "[PairedStats.median] Input error" in \
            caplog.records[-1].message
        with pytest.raises(TypeError, match="must be a real number"):
            stats.median([1.0, True])
        with pytest.raises(TypeError, match="must be a real number"):
            stats.walsh([1.0, None])

    def test_bad_settings(self) -> None:
        """
        The alpha and the exact threshold are checked at construction.
        """
        with pytest.raises(ValueError, match=r"in \(0, 1\)"):
            PairedStats(alpha=0.0)
        with pytest.raises(ValueError, match=r"in \(0, 1\)"):
            PairedStats(alpha=1.0)
        with pytest.raises(TypeError, match="must be a real number"):
            PairedStats(alpha="small")
        with pytest.raises(ValueError, match=">= 0"):
            PairedStats(exact_max_n=-1)
        with pytest.raises(TypeError, match="must be an int"):
            PairedStats(exact_max_n=1.5)


class TestWilcoxon:
    """
    Tests for the two-sided signed rank test over the paired cells.
    """

    def test_analytical_three_positive_differences(self) -> None:
        """
        Hand-computed case for the differences [1, 2, 3].

        By hand: no difference is zero, so all three are kept. Ranked by
        absolute value they get the ranks 1, 2 and 3, and all three are
        positive, so W+ = 1 + 2 + 3 = 6 and the rank sum is also 6. The
        exact p value enumerates the 2^3 = 8 subsets of the ranks and
        counts those at least as extreme as the observation, that is
        those whose sum is >= max(6, 0) = 6 or <= min(6, 0) = 0: only
        the full subset {1, 2, 3}, which sums to 6, and the empty one,
        which sums to 0. So p = 2 / 8 = 0.25, the smallest p value three
        paired cells can give.
        """
        statistic, p_value, n_eff = PairedStats().wilcoxon_exact(
            [1, 2, 3])
        assert (statistic, p_value, n_eff) == (6.0, 0.25, 3)

    def test_analytical_ties_and_a_negative_difference(self) -> None:
        """
        Hand-computed case for the differences [1, -1, 2].

        By hand: the absolute values are 1, 1 and 2. The two 1s tie over
        the rank positions 1 and 2, so each takes (1 + 2) / 2 = 1.5, and
        the 2 takes rank 3; the rank sum is 1.5 + 1.5 + 3 = 6. Only the
        first difference and the third are positive, so
        W+ = 1.5 + 3 = 4.5.

        The enumeration of the 2^3 = 8 subsets of the ranks
        [1.5, 1.5, 3] gives the sums 0, 1.5, 1.5, 3, 3, 4.5, 4.5 and 6.
        Those at least as extreme as the observation are the ones
        >= max(4.5, 1.5) = 4.5, namely 4.5, 4.5 and 6, and the ones
        <= min(4.5, 1.5) = 1.5, namely 0, 1.5 and 1.5: six in all, so
        p = 6 / 8 = 0.75.
        """
        statistic, p_value, n_eff = PairedStats().wilcoxon_exact(
            [1, -1, 2])
        assert (statistic, p_value, n_eff) == (4.5, 0.75, 3)

    def test_zero_differences_are_dropped(self) -> None:
        """
        Exact zeros leave the sample, as Wilcoxon's convention asks.

        The sample [0, 1, 2, 3, 0] therefore gives exactly the result of
        [1, 2, 3], with three effective cells and not five.
        """
        assert PairedStats().wilcoxon_exact([0, 1, 2, 3, 0]) == \
            (6.0, 0.25, 3)

    def test_sample_without_a_single_non_zero(self) -> None:
        """
        An empty or all-zero sample has no statistic to report.
        """
        stats = PairedStats()
        assert stats.wilcoxon_exact(tuple()) == (None, None, 0)
        assert stats.wilcoxon_exact([0.0, 0.0]) == (None, None, 0)

    def test_normal_branch_above_the_threshold(self) -> None:
        """
        Above the threshold the p value comes from the normal
        approximation.

        With the threshold lowered to 2 the same three differences
        [1, 2, 3] take the approximate branch, whose p value is not the
        exact 0.25 any more but still a probability.
        """
        _, exact, _ = PairedStats().wilcoxon_exact([1, 2, 3])
        _, normal, n_eff = PairedStats(
            exact_max_n=2).wilcoxon_exact([1, 2, 3])
        assert exact == 0.25
        assert normal != exact
        assert 0.0 < normal <= 1.0
        assert n_eff == 3


class TestWinCounting:
    """
    Tests for the pairing of the cells and the counting of the wins.
    """

    def test_analytical_block(self, tmp_path: Path) -> None:
        """
        Hand-computed block for the four cells of CELLS.

        By hand, the difference of a cell is the windowed ROC-AUC minus
        the per-record one, and the cells are read in the sorted order
        of their keys:

        - CIC2017, TSR, seed A: 0.75 - 0.50 = +0.25, a win;
        - CIC2017, TSR, seed B: 0.25 - 0.50 = -0.25, a loss;
        - CIC2017, AR, seed A: 0.50 - 0.50 = 0.00, a tie;
        - OS scan, TSR, seed A: 0.875 - 0.375 = +0.50, a win.

        So four cells, two wins and one tie; the losses are the
        remainder and are not printed. Sorted, the differences are
        -0.25, 0.00, +0.25 and +0.50, so the median is the mean of the
        two middle ones, (0.00 + 0.25) / 2 = +0.125, the minimum is
        -0.25 and the maximum +0.50.

        The ten Walsh averages of the four differences are -0.25,
        -0.125, 0.00, 0.00, +0.125, +0.125, +0.25, +0.25, +0.375 and
        +0.50, so the Hodges-Lehmann estimate is the mean of the fifth
        and the sixth, (+0.125 + 0.125) / 2 = +0.125, which here
        coincides with the median.

        Dropping the tie leaves the three differences +0.25, -0.25 and
        +0.50. Their absolute values 0.25 and 0.25 tie over the rank
        positions 1 and 2 and take 1.5 each, and 0.50 takes rank 3, so
        W+ = 1.5 + 3 = 4.5 over three effective cells and the exact
        two-sided p value is 6 / 8 = 0.75, as in
        test_analytical_ties_and_a_negative_difference.

        Averaging the seeds inside each (stream, paradigm) group gives
        (+0.25 - 0.25) / 2 = 0.00 for CIC2017 TSR, 0.00 for CIC2017 AR
        and +0.50 for OS scan TSR, so the window wins 1 of the 3
        combinations. Only CIC2017 TSR has two seeds, and its two
        differences have opposite signs, so it is the one sign flip.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        pair = RetrainedPair(results_root=tmp_path, repo_root=tmp_path)
        block = pair.report(_cells())["pairs"]["mlp_ctx_fix"]
        assert (block["n_cells"], block["wins_ctx"], block["ties"]) == \
            (4, 2, 1)
        assert block["median"] == pytest.approx(0.125)
        assert (block["min"], block["max"]) == (-0.25, 0.5)
        assert block["hl"] == pytest.approx(0.125)
        assert block["wilcoxon"] == {"W": 4.5, "p": 0.75, "n_eff": 3}
        assert [x["mean_delta"] for x in block["per_dataset_ar_mean"]] \
            == pytest.approx([0.0, 0.0, 0.5])
        assert block["wins_ctx_dataset_ar_mean"] == 1
        assert block["n_groups_two_seeds"] == 1
        assert len(block["sign_flips"]) == 1
        assert block["sign_flips"][0]["dataset"] == "CIC-IDS2017"

    def test_analytical_paradigm_halves(self, tmp_path: Path) -> None:
        """
        Hand-computed halves of the same four cells.

        By hand: the three TSR cells have the differences +0.25, -0.25
        and +0.50, so two of three are wins and the sorted differences
        -0.25, +0.25, +0.50 give the median +0.25. Their six Walsh
        averages are -0.25, 0.00, +0.125, +0.25, +0.375 and +0.50, so
        the Hodges-Lehmann estimate is (+0.125 + 0.25) / 2 = +0.1875.
        The single AR cell is the tie, so its half has one cell, no win,
        median 0.00 and estimate 0.00.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        pair = RetrainedPair(results_root=tmp_path, repo_root=tmp_path)
        block = pair.report(_cells())["pairs"]["mlp_ctx_fix"]
        assert (block["TSR"]["n"], block["TSR"]["wins_ctx"]) == (3, 2)
        assert block["TSR"]["median"] == pytest.approx(0.25)
        assert block["TSR"]["hl"] == pytest.approx(0.1875)
        assert (block["AR"]["n"], block["AR"]["wins_ctx"]) == (1, 0)
        assert block["AR"]["median"] == 0.0
        assert block["AR"]["hl"] == 0.0

    def test_pair_without_cells(self, tmp_path: Path) -> None:
        """
        A pair without any paired cell reports only its count.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        pair = RetrainedPair(results_root=tmp_path, repo_root=tmp_path)
        report = pair.report(_cells())
        assert set(report["pairs"]) == {p[0] for p in PAIRS}
        assert report["pairs"]["mlp_ctx_fix_e5"] == {"n_cells": 0}

    def test_half_paired_cells_are_left_out(self,
                                            tmp_path: Path) -> None:
        """
        A cell that holds only one of the two variants is not evidence.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        pair = RetrainedPair(results_root=tmp_path, repo_root=tmp_path)
        cells = _cells()
        del cells[("cic2017", True, "A", 1)]["mlp_fix"]
        rows = pair.paired_rows(cells, "mlp_ctx_fix", "mlp_fix")
        assert len(rows) == 3
        assert all(row["delta"] != 0.0 for row in rows)

    def test_wrong_types_raise(self, tmp_path: Path) -> None:
        """
        The cell map and the variant names are checked.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        pair = RetrainedPair(results_root=tmp_path, repo_root=tmp_path)
        with pytest.raises(TypeError, match="must be a mapping"):
            pair.paired_rows(list(), "a", "b")
        with pytest.raises(TypeError, match="must be"):
            pair.paired_rows(dict(), 1, "b")
        with pytest.raises(ValueError, match="at least one pair"):
            RetrainedPair(tmp_path, tmp_path, pairs=list())
        with pytest.raises(ValueError, match="two non-empty variant"):
            RetrainedPair(tmp_path, tmp_path, pairs=[("a", "b", "c")])


class TestLoadCells:
    """
    Tests for the reading of the metrics files of the pair.
    """

    def test_cells_are_keyed_by_what_the_pair_shares(
            self,
            tmp_path: Path) -> None:
        """
        A cell is (dataset, paradigm, seed, repetition), nothing else.

        The two variants of the pair land in the same cell, so they can
        be subtracted; a job without a primary metric is skipped
        instead of being imputed.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        _write_metrics(tmp_path, "T8-ctx", "cic2017", "mlp_ctx_fix",
                       0.75)
        _write_metrics(tmp_path, "T8-base", "cic2017", "mlp_fix", 0.50)
        _write_metrics(tmp_path, "T8-running", "cic2017", "mlp_fix",
                       None, seed="B")
        pair = RetrainedPair(results_root=tmp_path, repo_root=tmp_path)
        cells = pair.load_cells()
        assert list(cells) == [("cic2017", False, "A", 1)]
        assert set(cells[("cic2017", False, "A", 1)]) == {"mlp_ctx_fix",
                                                          "mlp_fix"}
        assert cells[("cic2017", False, "A", 1)]["mlp_ctx_fix"][
            "path"] == "T8-ctx/metrics.json"
        block = pair.report(cells)["pairs"]["mlp_ctx_fix"]
        assert block["n_cells"] == 1
        assert block["median"] == pytest.approx(0.25)

    def test_unreadable_file_is_skipped_with_a_warning(
            self,
            tmp_path: Path,
            caplog: pytest.LogCaptureFixture) -> None:
        """
        A broken metrics file is reported and the rest is still read.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        :param caplog: pytest fixture that captures log records.
        :type caplog: pytest.LogCaptureFixture
        """
        _write_metrics(tmp_path, "T8-ctx", "cic2017", "mlp_ctx_fix",
                       0.75)
        (tmp_path / "T8-broken").mkdir()
        (tmp_path / "T8-broken" / "metrics.json").write_text(
            "{ not json", encoding="utf-8")
        pair = RetrainedPair(results_root=tmp_path, repo_root=tmp_path,
                             logger=logging.getLogger("test"))
        with caplog.at_level(logging.WARNING, logger="test"):
            cells = pair.load_cells()
        assert len(cells) == 1
        assert "must be readable JSON" in caplog.records[-1].message

    def test_missing_results_root_only_warns(
            self,
            tmp_path: Path,
            caplog: pytest.LogCaptureFixture) -> None:
        """
        A results root that does not exist means no cells, not a crash.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        :param caplog: pytest fixture that captures log records.
        :type caplog: pytest.LogCaptureFixture
        """
        pair = RetrainedPair(results_root=tmp_path / "nope",
                             repo_root=tmp_path,
                             logger=logging.getLogger("test"))
        with caplog.at_level(logging.WARNING, logger="test"):
            assert pair.load_cells() == dict()
        assert "No cell is loaded" in caplog.records[-1].message


class TestDatasetLabel:
    """
    Tests for the stream labels the paired report prints.
    """

    def test_analytical_labels(self, tmp_path: Path) -> None:
        """
        Hand-computed labels of the three kinds of dataset name.

        By hand: a name that starts with "knad-" is relabelled by
        capitalizing every underscore-separated word of the rest and
        prefixing "KNAD:", so "knad-active_wiretap" becomes
        "KNAD:Active_Wiretap". A name in the lookup table is replaced by
        its entry, so "cic2017" becomes "CIC-IDS2017". Any other name is
        returned unchanged, which is why a runner that already writes
        "CIC2017" keeps that spelling in the report.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        pair = RetrainedPair(results_root=tmp_path, repo_root=tmp_path)
        assert pair.dataset_label("knad-active_wiretap") == \
            "KNAD:Active_Wiretap"
        assert pair.dataset_label("cic2017") == "CIC-IDS2017"
        assert pair.dataset_label("CIC2017") == "CIC2017"

    def test_non_string_name(self, tmp_path: Path) -> None:
        """
        A dataset name that is not a string is refused by type.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        pair = RetrainedPair(results_root=tmp_path, repo_root=tmp_path)
        with pytest.raises(TypeError, match="must be a string"):
            pair.dataset_label(5)


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
