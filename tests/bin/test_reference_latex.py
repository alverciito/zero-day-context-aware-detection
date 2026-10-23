# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         tests/bin/test_reference_latex.py                         #
#   Version:      1.0.0                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-13                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Checks that the generated LaTeX of the reference          #
#                 outputs cites no label that nothing defines.              #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import re
from pathlib import Path

import pytest

# Root of the package and its reference outputs.
ROOT: Path = Path(__file__).resolve().parents[2]
REFERENCE: Path = ROOT / "data" / "reference"
# File with the fifteen floats of the supplementary material.
SUPPLEMENTARY: str = "supplementary_tables.tex"
# Labels that those fifteen floats cite and that the supplement defines
# outside them: a table and its section.
DEFINED_IN_THE_SUPPLEMENT: frozenset[str] = frozenset((
    "tab:s_calib_attack", "sec:s_calib_attack"))
# Number of LaTeX files among the reference outputs.
N_TEX_FILES: int = 10

pytestmark = pytest.mark.skipif(
    not REFERENCE.is_dir(),
    reason="data/reference is needed for these tests")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                  TESTS                                    #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class TestReferenceLatex:
    """
    Tests for the cross-references of the generated LaTeX.
    """

    def test_every_cross_reference_resolves(self) -> None:
        """
        No generated float cites a label that nothing defines.

        By hand: the fifteen supplementary floats cite one another and
        two labels of the supplement outside them, the table of the
        attack types in the calibration window and its section. The
        floats of ``bin/make_figures.py`` stand alone, so they cite no
        label at all. Any other target would be a pointer to a float
        that does not exist.
        """
        labels = set()
        cited = dict()
        for path in sorted(REFERENCE.glob("*.tex")):
            text = path.read_text(encoding="utf-8")
            labels.update(re.findall(r"\\label\{([^}]*)\}", text))
            cited[path.name] = set(
                re.findall(r"\\(?:eq)?ref\{([^}]*)\}", text))
        assert len(cited) == N_TEX_FILES
        for name, targets in cited.items():
            assert targets <= labels | DEFINED_IN_THE_SUPPLEMENT, name
            if name != SUPPLEMENTARY:
                assert targets == set(), name
        assert DEFINED_IN_THE_SUPPLEMENT <= cited[SUPPLEMENTARY]

    def test_every_label_is_defined_once(self) -> None:
        """
        A label is defined by one float only.

        By hand: the ten files hold fifteen supplementary floats and
        nine floats of ``bin/make_figures.py``, twenty-four labels in
        all, each written once.
        """
        found = list()
        for path in sorted(REFERENCE.glob("*.tex")):
            found += re.findall(r"\\label\{([^}]*)\}",
                                path.read_text(encoding="utf-8"))
        assert len(found) == 24
        assert len(set(found)) == len(found)


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
