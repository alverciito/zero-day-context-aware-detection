# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         tests/stats/test_confirmatory.py                          #
#   Version:      1.0.0                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-07                                                #
#   Modified:     2026-10-07                                                #
#   Description:  Recomputes the five comparisons of H1 from cells.csv      #
#                 and checks them against the shipped stats.json.           #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import csv
import json
from pathlib import Path

import pytest

from kitnetzday.audit.paired_stats import PairedStats

# The shipped aggregate, relative to the root of the package.
DATA: Path = Path(__file__).resolve().parents[2] / "data"
# Configurations averaged into each family of H1, and the reference.
FAMILIES: dict[str, tuple[str, ...]] = {
    "Conv1D": ("C1", "C2"), "Conv2D": ("C3", "C4"),
    "LSTM": ("C5", "C6"), "MHA": ("C7", "C8"), "MLP": ("C9",)}
REFERENCE: str = "C10"
DATASETS: tuple[str, ...] = tuple(f"D{i}" for i in range(1, 10))
# Two-sided alpha of the exact interval of the plan, 1 - 0.9609375.
ALPHA: float = 0.0390625


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                AUXILIARY                                  #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def _cells() -> dict[tuple[str, str], float]:
    """
    Mean ROC-AUC of the fused detector of every cell of ``cells.csv``.

    :return: (dataset id, configuration id) -> mean over the seeds.
    :rtype: dict[tuple[str, str], float]
    """
    with open(DATA / "aggregate" / "cells.csv", newline="",
              encoding="utf-8") as handle:
        return {(row["dataset_id"], row["config_id"]):
                float(row["auroc_FUS_mean"])
                for row in csv.DictReader(handle)
                if row["auroc_FUS_mean"]}


def _differences(cells: dict[tuple[str, str], float],
                 family: str) -> list[float]:
    """
    Paired differences of one family against the reference, by stream.

    :param cells: Output of ``_cells``.
    :type cells: dict[tuple[str, str], float]
    :param family: Family of H1, a key of ``FAMILIES``.
    :type family: str
    :return: One difference per dataset, in the order D1..D9: the mean
        of the configurations of the family minus the reference.
    :rtype: list[float]
    """
    members = FAMILIES[family]
    return [sum(cells[(d, c)] for c in members) / len(members)
            - cells[(d, REFERENCE)] for d in DATASETS]


def _holm(p_values: dict[str, float]) -> dict[str, float]:
    """
    Holm step-down adjustment of a family of p values.

    The p values are sorted increasingly, the i-th one (from 1) is
    multiplied by m - i + 1, and the adjusted values are made
    non-decreasing and capped at 1.

    :param p_values: Name -> raw p value.
    :type p_values: dict[str, float]
    :return: Name -> adjusted p value.
    :rtype: dict[str, float]
    """
    adjusted: dict[str, float] = dict()
    running = 0.0
    m = len(p_values)
    for i, name in enumerate(sorted(p_values, key=p_values.get)):
        running = max(running, min(1.0, p_values[name] * (m - i)))
        adjusted[name] = running
    return adjusted


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                  TESTS                                    #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
@pytest.mark.skipif(not (DATA / "stats.json").exists(),
                    reason="the shipped aggregate is not in data/")
class TestH1AgainstTheShippedStatistics:
    """
    H1 recomputed here, independently of ``frozen/runner/stats.py``.

    ``stats.json`` was written by the frozen statistics code with
    scipy. These tests rebuild the same five comparisons from
    ``cells.csv`` with the exact routines of this package, which share
    no code with it, and require the two to agree.
    """

    def test_analytical_holm_adjustment(self) -> None:
        """
        Hand-computed Holm adjustment of five p values.

        By hand, for the raw values 0.25, 25/256, 47/64, 127/256 and
        13/64 (the five exact p values of H1): sorted they are 25/256,
        13/64, 1/4, 127/256 and 47/64, and the multipliers are 5, 4, 3,
        2 and 1, which gives 125/256 = 0.48828125, 13/16 = 0.8125,
        3/4 = 0.75, 127/128 = 0.9921875 and 47/64 = 0.734375. Making
        them non-decreasing turns the third into 0.8125 and the fifth
        into 0.9921875.
        """
        adjusted = _holm({"a": 0.25, "b": 25 / 256, "c": 47 / 64,
                          "d": 127 / 256, "e": 13 / 64})
        assert adjusted == {"b": 0.48828125, "e": 0.8125, "a": 0.8125,
                            "d": 0.9921875, "c": 0.9921875}

    def test_differences_estimates_and_intervals(self) -> None:
        """
        The differences, the estimate and its exact interval agree.

        With nine differences the 45 Walsh averages are bounded by the
        6th and the 40th, because P(W+ <= 5) = 10 / 512 is exactly
        alpha / 2 = 0.01953125, so the coverage is 0.9609375.
        """
        stats = PairedStats(alpha=ALPHA)
        cells = _cells()
        shipped = json.loads((DATA / "stats.json").read_text(
            encoding="utf-8"))["H1"]["comparisons"]
        for family in FAMILIES:
            block = shipped[f"{family}_vs_original"]
            diffs = _differences(cells, family)
            assert diffs == pytest.approx(
                [block["d"][d] for d in DATASETS], abs=1e-12)
            assert stats.hodges_lehmann(diffs) == pytest.approx(
                block["hl"], abs=1e-12)
            lower, upper, coverage = stats.hl_interval(diffs)
            assert lower == pytest.approx(block["hl_ci"]["lower"],
                                          abs=1e-12)
            assert upper == pytest.approx(block["hl_ci"]["upper"],
                                          abs=1e-12)
            assert coverage == block["hl_ci"]["coverage"] == 0.9609375

    def test_exact_p_values_and_holm(self) -> None:
        """
        The exact signed rank p values and their Holm adjustment agree.
        """
        stats = PairedStats(alpha=ALPHA)
        cells = _cells()
        shipped = json.loads((DATA / "stats.json").read_text(
            encoding="utf-8"))["H1"]["comparisons"]
        raw = dict()
        for family in FAMILIES:
            block = shipped[f"{family}_vs_original"]
            w_plus, p_value, n_eff = stats.wilcoxon_exact(
                _differences(cells, family))
            assert w_plus == block["wilcoxon"]["t_plus"]
            assert n_eff == block["wilcoxon"]["n_eff"]
            assert p_value == pytest.approx(block["wilcoxon"]["p"],
                                            abs=1e-12)
            raw[family] = p_value
        for family, value in _holm(raw).items():
            assert value == pytest.approx(
                shipped[f"{family}_vs_original"]["p_holm"], abs=1e-12)


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
