# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         tests/figures/test_print_style.py                         #
#   Version:      1.0.0                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-07                                                #
#   Modified:     2026-10-07                                                #
#   Description:  Unit tests for the print size of the figures: the         #
#                 width, the smallest lettering and the font type.          #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import json
import logging
from pathlib import Path

import pytest

from kitnetzday.figures.print_style import (
    FONT_SIZE_MIN,
    FONT_SIZE_SCRIPT,
    PRINT_RC_PARAMS,
    PRINT_WIDTH_INCHES,
    print_rc_params,
)
from kitnetzday.naming import RC_PARAMS

# Limits the figures are held to: Elsevier asks for 7 pt lettering at
# the printed size and 6 pt for sub and superscripts; mathtext draws a
# subscript at 0.7 of the size of its text.
ELSEVIER_MIN_POINTS: float = 7.0
ELSEVIER_MIN_SCRIPT_POINTS: float = 6.0
MATHTEXT_SHRINK: float = 0.7
# Text width of the manuscript, \geometry{total={170mm,257mm}}.
MANUSCRIPT_TEXT_WIDTH_MM: float = 170.0
CONFIG_PATH: Path = Path(__file__).resolve().parents[2] / "src" / "config.json"


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                MAIN CLASS                                 #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class TestPrintRcParams:
    """
    Tests for the settings of a figure drawn at its printed width.
    """

    def test_analytical_lettering_floor(self) -> None:
        """
        Hand-computed case for the two smallest sizes.

        By hand: the smallest normal lettering is 7.5 pt, above the
        7 pt Elsevier asks for. Lettering with a subscript is set at
        9.0 pt and mathtext draws the subscript at 0.7 of that:
        0.7 x 9.0 = 6.3 pt, above the 6 pt asked of subscripts. At the
        7.5 pt of normal lettering the subscript would be
        0.7 x 7.5 = 5.25 pt, which is why there are two sizes.
        """
        assert FONT_SIZE_MIN == 7.5
        assert FONT_SIZE_MIN >= ELSEVIER_MIN_POINTS
        assert MATHTEXT_SHRINK * FONT_SIZE_SCRIPT == pytest.approx(6.3)
        assert (MATHTEXT_SHRINK * FONT_SIZE_SCRIPT
                >= ELSEVIER_MIN_SCRIPT_POINTS)
        assert MATHTEXT_SHRINK * FONT_SIZE_MIN == pytest.approx(5.25)

    def test_analytical_width_fits_the_text_block(self) -> None:
        """
        Hand-computed case for the printed width.

        By hand: the text block is 170 mm wide, and 170 / 25.4 =
        6.6929 inches. The figures are drawn 6.69 inches wide, which
        is 0.0029 inches (0.07 mm) narrower, so including them at
        \\textwidth never shrinks them.
        """
        assert PRINT_WIDTH_INCHES == 6.69
        assert MANUSCRIPT_TEXT_WIDTH_MM / 25.4 == pytest.approx(
            6.6929, abs=1e-4)
        assert PRINT_WIDTH_INCHES <= MANUSCRIPT_TEXT_WIDTH_MM / 25.4

    def test_overrides_are_laid_over_the_base(self) -> None:
        """
        The base settings are kept, the print ones win, and the base
        dictionary is not modified.
        """
        base = {"lines.linewidth": 3.0, "font.size": 20.0,
                "pdf.fonttype": 3}
        merged = print_rc_params(base)
        assert merged["lines.linewidth"] == 3.0
        assert merged["font.size"] == 8.0
        assert merged["pdf.fonttype"] == 42
        assert merged["ps.fonttype"] == 42
        assert merged["font.family"] == ["DejaVu Sans"]
        assert base == {"lines.linewidth": 3.0, "font.size": 20.0,
                        "pdf.fonttype": 3}

    def test_empty_base_gives_only_the_overrides(self) -> None:
        """
        An empty base is valid: the result is the overrides alone.
        """
        merged = print_rc_params(dict())
        assert merged["pdf.fonttype"] == 42
        assert merged["xtick.labelsize"] == FONT_SIZE_MIN
        assert "lines.linewidth" not in merged

    def test_default_keeps_the_house_style(self) -> None:
        """
        The default settings start from the house style of the paper
        and no size of theirs falls under the smallest lettering.
        """
        sizes = [value for key, value in PRINT_RC_PARAMS.items()
                 if key.endswith("size") and isinstance(value, float)]
        assert PRINT_RC_PARAMS["hatch.linewidth"] == RC_PARAMS[
            "hatch.linewidth"]
        assert PRINT_RC_PARAMS["pdf.fonttype"] == 42
        assert sizes and min(sizes) >= FONT_SIZE_MIN

    def test_wrong_type_is_logged_and_raised(
            self, caplog: pytest.LogCaptureFixture) -> None:
        """
        A base that is not a mapping raises a TypeError with the
        logged message.

        :param caplog: pytest fixture that captures the log records.
        :type caplog: pytest.LogCaptureFixture
        """
        with caplog.at_level(logging.ERROR):
            with pytest.raises(TypeError, match="must be a mapping"):
                print_rc_params(["font.size", 8.0],
                                logger=logging.getLogger("test"))
        assert "[print_rc_params] Input error" in caplog.text
        with pytest.raises(TypeError, match="not NoneType"):
            print_rc_params(None)


class TestConfiguredSizes:
    """
    Tests that ``src/config.json`` asks for print-size figures.

    The scripts in ``bin/`` hand these values to the figure classes, so
    a configuration left at the old 11-inch canvas would shrink the
    lettering to about 4 pt on the page again.
    """

    def test_config_respects_the_print_floor(self) -> None:
        """
        The heat map is as wide as the text block and both font sizes
        of the configuration respect the floor.

        By hand: 0.7 x font_size_footnote has to reach 6 pt, so
        font_size_footnote has to be at least 6 / 0.7 = 8.58 pt.
        """
        style = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))[
            "figures"]
        assert style["heatmap_size_inches"][0] == PRINT_WIDTH_INCHES
        assert style["font_size_small"] >= FONT_SIZE_MIN
        assert style["font_size_footnote"] >= 8.58
        assert (MATHTEXT_SHRINK * style["font_size_footnote"]
                >= ELSEVIER_MIN_SCRIPT_POINTS)


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
