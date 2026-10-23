# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         tests/figures/test_oracle.py                              #
#   Version:      1.0.1                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-13                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Unit tests for the figure of the submitted values: its    #
#                 count of bars, its caption and its partial inputs.        #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
import pytest

# The non-interactive backend has to be chosen before pyplot is
# imported, which the modules of the package do.
matplotlib.use("Agg")

from kitnetzday.figures.oracle import OracleFigure  # noqa: E402
from kitnetzday.io.inputs import Inputs  # noqa: E402
from kitnetzday.io.outputs import Out  # noqa: E402
from kitnetzday.naming import DATASET_IDS, FIG7_ORDER  # noqa: E402

NAME: str = "fig_appendix_fig7_oracle"
NUMBER_ID: str = "R.FIG7.AGG.BARS_LE_TRIVIAL"
LABEL: str = "[EXPLORATORIO] (audit made before the analysis plan was fixed)"
# Opening of the caption with the 216 values of the nine datasets.
OPENING: str = (
    "Oracle results of the submitted regime, redrawn from the 216 bar "
    "values of its figure (nine datasets, twelve variants, two "
    "metrics); not comparable with the main results. ")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                AUXILIARY                                  #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def _setup(folder: Path) -> tuple[Inputs, Out]:
    """
    Build the reader and the writer the figure asks for.

    :param folder: Temporary folder of the inputs and of the outputs.
    :type folder: Path
    :return: The reader over a toy aggregate, and the writer.
    :rtype: tuple[Inputs, Out]
    """
    (folder / "aggregate.json").write_text(
        json.dumps({"complete": True}), encoding="utf-8")
    inputs = Inputs(folder / "aggregate.json", folder / "stats.json",
                    folder / "cells.csv", folder / "jobs",
                    folder / "fig7.csv", folder / "fig7_trivial.csv")
    return inputs, Out(folder / "out", 40, "cmd", "ref")


def _bars(f1: float) -> dict[tuple[str, str, str], dict[str, Any]]:
    """
    The 108 bar pairs of the figure, all with the same two values.

    :param f1: F1 of every bar, in [0, 1].
    :type f1: float
    :return: (dataset, variant, paradigm) -> its record.
    :rtype: dict[tuple[str, str, str], dict[str, Any]]
    """
    return {(x, v, p): {"F1": f1, "WindowDiff": 0.1, "detector": "MVD"}
            for x in DATASET_IDS for v, p in FIG7_ORDER}


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                  TESTS                                    #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class TestOracleFigure:
    """
    Tests for the count of bars and for the status of the output.
    """

    def test_analytical_complete_inputs(self, tmp_path: Path) -> None:
        """
        With the 216 values and the nine references nothing is flagged.

        By hand: nine datasets times twelve bars give 108 F1 bars, all
        at 0.9. The trivial F1 is 0.5 in the first eight datasets and
        0.95 in the last one, so the twelve bars of that one do not
        exceed it: 12 of 108. The inputs are complete, so the output
        and its number carry no partial-inputs label, and the caption
        opens with the 216 values of nine datasets, twelve variants
        and two metrics (9 x 12 x 2 = 216) and cites no label.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        inputs, out = _setup(tmp_path)
        trivial = {x: 0.5 for x in DATASET_IDS}
        trivial[DATASET_IDS[-1]] = 0.95
        assert OracleFigure().build(inputs, out, _bars(0.9), 216,
                                    trivial) is True
        record = out.records[0]
        assert record["partial_inputs"] is False
        assert record["missing"] == list()
        assert record["notes"] == [
            "216 values read from fig7_valores_publicados.csv "
            "(216 = 9 x 12 x 2)."]
        assert record["caption"].startswith(OPENING)
        assert record["caption"].endswith(
            "downward triangles mark the 12 of the 108 F1 bars that do "
            "not exceed it, an exploratory count.")
        assert "\\ref{" not in record["caption"]
        value, source = out.numbers[NUMBER_ID]
        assert value == 12
        assert source.startswith(LABEL + " count over data/")
        tex = (tmp_path / "out" / f"{NAME}.tex").read_text(
            encoding="utf-8")
        assert "[PARTIAL INPUTS]" not in tex
        assert (tmp_path / "out" / f"{NAME}.pdf").stat().st_size > 0

    def test_analytical_short_inputs(self, tmp_path: Path) -> None:
        """
        A short file of values is declared and labelled as partial.

        By hand: two bars of D1 are given, four values, with F1 0.5
        and 0.9, and the trivial F1 of D1 alone, 0.6. One of the two
        covered bars does not exceed it. Four values are not the 216
        expected and one reference is not nine, so the output carries
        the label with the reason "4/216 values, 1/9 trivial F1
        values", and the caption gives both counts of values.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        inputs, out = _setup(tmp_path)
        bars = {("D1", "MLP", "AR"): {"F1": 0.5, "WindowDiff": 0.4,
                                      "detector": "KMD"},
                ("D1", "MLP", "TSR"): {"F1": 0.9, "WindowDiff": 0.2,
                                       "detector": "MVD"}}
        OracleFigure().build(inputs, out, bars, 4, {"D1": 0.6})
        record = out.records[0]
        assert record["partial_inputs"] is True
        assert len(record["missing"]) == 2
        assert "mark the 1 of the 2 F1 bars" in record["caption"]
        assert record["caption"].startswith(
            "Oracle results of the submitted regime, redrawn from 4 of "
            "the 216 bar values of its figure (")
        assert out.numbers[NUMBER_ID] == (
            1, "[PARTIAL INPUTS] " + LABEL + " count over "
            "data/fig7_valores_publicados.csv against the smoothed "
            "trivial F1 of data/fig7_trivial_f1.csv")
        tex = (tmp_path / "out" / f"{NAME}.tex").read_text(
            encoding="utf-8")
        assert ("% [PARTIAL INPUTS]: incomplete input: 4/216 values, "
                "1/9 trivial F1 values") in tex

    def test_counts_spelled_in_the_caption(self) -> None:
        """
        Counts up to twelve are spelled out and the rest are digits.

        By hand: the caption names nine datasets, twelve variants and
        two metrics. Zero and twelve are the two ends of the list of
        words; thirteen is beyond it and a negative count has no word,
        so both are written in digits.
        """
        assert OracleFigure._count_word(9) == "nine"
        assert OracleFigure._count_word(12) == "twelve"
        assert OracleFigure._count_word(2) == "two"
        assert OracleFigure._count_word(0) == "zero"
        assert OracleFigure._count_word(13) == "13"
        assert OracleFigure._count_word(-1) == "-1"

    def test_no_values_no_figure(self,
                                 tmp_path: Path,
                                 caplog: pytest.LogCaptureFixture
                                 ) -> None:
        """
        Without published values the figure is not written.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        :param caplog: pytest fixture that captures log records.
        :type caplog: pytest.LogCaptureFixture
        """
        inputs, out = _setup(tmp_path)
        figure = OracleFigure(logger=logging.getLogger("test"))
        with caplog.at_level(logging.WARNING, logger="test"):
            assert figure.build(inputs, out, dict(), 0, dict()) is False
        assert "The oracle figure is not generated" in \
            caplog.records[-1].message
        assert out.records == list()

    def test_wrong_arguments(self, tmp_path: Path) -> None:
        """
        Wrong types and a negative count are refused.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        inputs, out = _setup(tmp_path)
        figure = OracleFigure()
        with pytest.raises(TypeError, match="inputs must be an Inputs"):
            figure.build("inputs", out, dict(), 0, dict())
        with pytest.raises(TypeError, match="out must be an Out"):
            figure.build(inputs, "out", dict(), 0, dict())
        with pytest.raises(TypeError, match="fig7 must be a mapping"):
            figure.build(inputs, out, list(), 0, dict())
        with pytest.raises(TypeError, match="n_values must be an int"):
            figure.build(inputs, out, dict(), 1.5, dict())
        with pytest.raises(ValueError, match="zero or more"):
            figure.build(inputs, out, dict(), -1, dict())
        with pytest.raises(TypeError, match="f1_trivial must be a map"):
            figure.build(inputs, out, dict(), 0, list())
        with pytest.raises(ValueError, match="strictly positive"):
            OracleFigure(expected_values=0)
        with pytest.raises(ValueError, match="at least one"):
            OracleFigure(fig7_order=tuple())


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
