# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         tests/figures/test_labels.py                              #
#   Version:      1.0.1                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-07                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Unit tests for the labels of the print-size figures:      #
#                 row groups of the heat map and bars of Figure A.1.        #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
from kitnetzday.figures.heatmap import Heatmap
from kitnetzday.figures.oracle import OracleFigure
from kitnetzday.naming import DATASET_IDS, DATASET_LONG, FIG7_ORDER


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                MAIN CLASS                                 #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class TestHeatmapRowGroups:
    """
    Tests for the runs of rows that share the prefix "KNAD ".
    """

    def test_analytical_study_vocabulary(self) -> None:
        """
        Hand-computed case for the nine datasets of the study.

        By hand: rows 0 to 5 are the six KNAD captures and rows 6 to 8
        the three flow corpora, whose names do not start with "KNAD ".
        There is one run, from row 0 to row 5.
        """
        names = [DATASET_LONG[x] for x in DATASET_IDS]
        assert Heatmap._row_groups(names) == [(0, 5)]

    def test_a_single_row_is_not_a_group(self) -> None:
        """
        A lone row with the prefix keeps its full name: a bracket
        around one row would say nothing.

        By hand: rows 0 and 1 form a run, row 2 breaks it, and row 3 is
        alone, so only (0, 1) is returned.
        """
        names = ["KNAD A", "KNAD B", "CIC-IDS2017", "KNAD C"]
        assert Heatmap._row_groups(names) == [(0, 1)]

    def test_no_prefix_and_empty(self) -> None:
        """
        Without the prefix, or without rows, there is no group.
        """
        assert Heatmap._row_groups(
            ["ACI-IoT-2023", "CIC-IDS2017"]) == list()
        assert Heatmap._row_groups(list()) == list()

    def test_two_separate_runs(self) -> None:
        """
        Two runs separated by another dataset are both returned.
        """
        names = ["KNAD A", "KNAD B", "X", "KNAD C", "KNAD D", "KNAD E"]
        assert Heatmap._row_groups(names) == [(0, 1), (3, 5)]


class TestOracleBarLabel:
    """
    Tests for the label of a bar pair of the submitted Fig. 7.
    """

    def test_names_follow_the_revised_manuscript(self) -> None:
        """
        The CSV of the submission writes the variants in capitals and
        calls the control "KitNET"; the label uses today's names.
        """
        assert OracleFigure._bar_label("CONV1D", "TSR") == "Conv1D-TSR"
        assert OracleFigure._bar_label("CONV2D", "AR") == "Conv2D-AR"
        assert OracleFigure._bar_label("MLP", "AR") == "MLP-AR"
        assert OracleFigure._bar_label("KitNET", "PbP") == "KitNET-style"
        assert OracleFigure._bar_label("STAT", "PbP") == "STAT"

    def test_twelve_distinct_labels(self) -> None:
        """
        The twelve bars of a panel get twelve different labels.
        """
        labels = [OracleFigure._bar_label(v, p) for v, p in FIG7_ORDER]
        assert len(labels) == 12
        assert len(set(labels)) == 12

    def test_unknown_variant_is_kept_as_written(self) -> None:
        """
        A variant with no display name is written as the CSV has it.
        """
        assert OracleFigure._bar_label("GRU", "AR") == "GRU-AR"


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
