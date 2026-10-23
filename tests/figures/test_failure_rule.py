# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         tests/figures/test_failure_rule.py                        #
#   Version:      1.0.0                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-13                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Unit tests for the selection rule of the failure cases    #
#                 when it is applied to the aggregate.                      #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import json
from pathlib import Path
from typing import Any

import pytest

from kitnetzday.figures.failure_rule import (
    FALLBACK_NOTE,
    FALLBACK_SRC,
    RULE_FALLBACK,
    RULE_MARGIN,
    STATS_SRC,
    FailureRule,
)
from kitnetzday.io.inputs import Inputs
from kitnetzday.io.outputs import Out
from kitnetzday.naming import Naming

# Toy vocabulary: two streams, two windowed configurations and one
# context-free configuration, which is also the control.
NAMING: Naming = Naming(datasets=("D1", "D2"),
                        configurations=("C1", "C2", "C9"),
                        context_free=("C9",), control="C9")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                AUXILIARY                                  #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def _inputs(folder: Path,
            values: dict[str, dict[str, float | None]],
            stats: dict[str, Any] | None = None) -> Inputs:
    """
    Build a reader whose aggregate holds the given ROC-AUC values.

    :param folder: Temporary folder the input files are written to.
    :type folder: Path
    :param values: Dataset id -> configuration id -> ROC-AUC, or None
        for a planned cell without a value.
    :type values: dict[str, dict[str, float | None]]
    :param stats: Content of ``stats.json``; None leaves the file out.
    :type stats: dict[str, Any] | None
    :return: The reader, with the toy vocabulary.
    :rtype: Inputs
    """
    main = {x: {c: {"auroc_FUS": v} for c, v in row.items()}
            for x, row in values.items()}
    (folder / "aggregate.json").write_text(json.dumps(
        {"complete": True, "value_sets": {"main": main}}),
        encoding="utf-8")
    if stats is not None:
        (folder / "stats.json").write_text(json.dumps(stats),
                                           encoding="utf-8")
    return Inputs(folder / "aggregate.json", folder / "stats.json",
                  folder / "cells.csv", folder / "jobs",
                  folder / "fig7.csv", folder / "fig7_trivial.csv",
                  naming=NAMING)


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                  TESTS                                    #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class TestSelect:
    """
    Tests for the rule applied to the aggregate, without statistics.
    """

    def test_analytical_margin_rule(self, tmp_path: Path) -> None:
        """
        Hand-computed selection over two streams.

        By hand, D1: the context-free mean is 0.80 and the windowed
        mean (0.60 + 0.70) / 2 = 0.65, so the difference is 0.15, above
        the margin of 0.01, and every planned cell has a value: D1 is a
        failure case. D2: the context-free mean is 0.95 and the only
        windowed value is 0.90, a difference of 0.05, but the cell C2
        has no value, so D2 is not flagged and the note says why.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        inputs = _inputs(tmp_path, {
            "D1": {"C1": 0.60, "C2": 0.70, "C9": 0.80},
            "D2": {"C1": 0.90, "C2": None, "C9": 0.95}})
        per, selected, rule, src, missing, notes = FailureRule(
            NAMING).select(inputs)
        assert selected == ["D1"]
        assert (rule, src, notes) == (RULE_MARGIN, FALLBACK_SRC,
                                      [FALLBACK_NOTE])
        assert per["D1"]["simple_mean"] == pytest.approx(0.80)
        assert per["D1"]["context_mean"] == pytest.approx(0.65)
        assert per["D1"]["simple_minus_context"] == pytest.approx(0.15)
        assert (per["D1"]["n_simple"], per["D1"]["n_context"]) == (1, 2)
        assert per["D1"]["cells_without_value"] == list()
        assert per["D1"]["is_failure_case"] is True
        assert per["D2"]["cells_without_value"] == ["D2/C2"]
        assert per["D2"]["incomplete"] is True
        assert per["D2"]["is_failure_case"] is False
        assert missing == [
            "D2: 1 planned cells of its families have no value: the "
            "stream is not flagged as a failure case even when the "
            "difference exceeds the margin"]

    def test_analytical_fallback_rule(self, tmp_path: Path) -> None:
        """
        When no stream meets the margin the closest one is selected.

        By hand, D1: 0.70 against (0.70 + 0.72) / 2 = 0.71, a
        difference of -0.01. D2: 0.60 against 0.80, a difference of
        -0.20. Neither exceeds +0.01, so the stream with the smallest
        (windowed - context-free) is taken: D1 with 0.01 against D2
        with 0.20. The rule is the fallback one.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        inputs = _inputs(tmp_path, {
            "D1": {"C1": 0.70, "C2": 0.72, "C9": 0.70},
            "D2": {"C1": 0.80, "C2": 0.80, "C9": 0.60}})
        rule = FailureRule(NAMING)
        per, selected, text, _, missing, _ = rule.select(inputs)
        assert selected == ["D1"]
        assert text == RULE_FALLBACK
        assert rule.is_fallback(text) is True
        assert rule.is_fallback(RULE_MARGIN) is False
        assert rule.is_fallback(None) is False
        assert per["D2"]["simple_minus_context"] == pytest.approx(-0.2)
        assert missing == list()

    def test_stored_selection_is_read_as_it_is(self,
                                               tmp_path: Path) -> None:
        """
        With statistics the stored record is returned untouched.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        stored = {"per_dataset": {"D2": {"simple_mean": 0.9}},
                  "selected": ["D2"], "selection_rule": RULE_MARGIN}
        inputs = _inputs(tmp_path, {"D1": {"C1": 0.5}},
                         stats={"run_type": "FINAL",
                                "failure_cases": stored})
        per, selected, rule, src, missing, notes = FailureRule(
            NAMING).select(inputs)
        assert (per, selected, rule) == (stored["per_dataset"], ["D2"],
                                         RULE_MARGIN)
        assert (src, missing, notes) == (STATS_SRC, list(), list())


class TestAddNumbers:
    """
    Tests for the rows of values of the family means.
    """

    def test_analytical_rows(self, tmp_path: Path) -> None:
        """
        Three rows per stream with a record, one note per stream without.

        By hand: only D1 has a record, so three rows are registered
        (context-free mean, windowed mean and difference) and D2 leaves
        one note. The reader has no statistics, so every source starts
        with the partial-inputs label.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        inputs = _inputs(tmp_path, {"D1": {"C1": 0.6, "C9": 0.8}})
        out = Out(tmp_path / "out", 50, "cmd", "ref")
        rule = FailureRule(NAMING)
        per, _, _, src, _, _ = rule.select(inputs)
        ids, missing = list(), list()
        rule.add_numbers(inputs, out, per, src, ids, missing)
        assert ids == ["R.FAIL.D1.SIMPLE", "R.FAIL.D1.CTX",
                       "R.FAIL.D1.DIFF"]
        assert missing == ["D2: no family means (a family without "
                           "values)"]
        value, source = out.numbers["R.FAIL.D1.DIFF"]
        assert value == pytest.approx(0.2)
        assert source == (
            "[PARTIAL INPUTS] [EXPLORATORIO plan §10] "
            f"{FALLBACK_SRC}/per_dataset/D1/simple_minus_context")

    def test_wrong_arguments(self, tmp_path: Path) -> None:
        """
        Wrong classes are refused with a message that names them.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        rule = FailureRule()
        inputs = _inputs(tmp_path, {"D1": {"C1": 0.6}})
        out = Out(tmp_path / "out", 50, "cmd", "ref")
        assert "failure_margin=0.01" in repr(rule)
        with pytest.raises(TypeError, match="naming must be a Naming"):
            FailureRule(naming="D1")
        with pytest.raises(TypeError, match="inputs must be an Inputs"):
            rule.select("inputs")
        with pytest.raises(TypeError, match="an Inputs and an Out"):
            rule.add_numbers(inputs, "out", dict(), "", list(), list())
        with pytest.raises(TypeError, match="must be lists"):
            rule.add_numbers(inputs, out, dict(), "", tuple(), list())


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
