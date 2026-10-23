# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         tests/tables/test_ablations.py                            #
#   Version:      1.0.1                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-13                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Unit tests for the ablation table: the summary of a       #
#                 block and the count of planned cells without a value.     #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import json
from pathlib import Path
from typing import Any

import pytest

from kitnetzday.io.inputs import Inputs
from kitnetzday.io.outputs import Out
from kitnetzday.tables.ablations import (
    MISSING_NO_CELLS,
    NOTE_NO_STATS,
    AblationsTable,
)

# Raw cells of one toy block: three cells of D1 at level 200, two of
# them with a difference against their reference, and one cell of D8 at
# level 2000 without one.
CELLS: tuple[dict[str, Any], ...] = (
    {"block": "T2-B", "factor": "window", "dataset_id": "D1",
     "value": 200, "job_id": "job-a", "delta_auroc": 0.1,
     "within_noise": True},
    {"block": "T2-B", "factor": "window", "dataset_id": "D1",
     "value": 200, "job_id": "job-b", "delta_auroc": 0.3,
     "within_noise": False},
    {"block": "T2-B", "factor": "window", "dataset_id": "D1",
     "value": "200", "job_id": "job-c"},
    {"block": "T2-B", "factor": "window", "dataset_id": "D8",
     "value": 2000, "job_id": "job-d"})


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                AUXILIARY                                  #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def _table(folder: Path,
           aggregate: dict[str, Any],
           stats: dict[str, Any] | None = None) -> AblationsTable:
    """
    Build the table over files written into a temporary folder.

    :param folder: Temporary folder of the inputs and of the outputs.
    :type folder: Path
    :param aggregate: Content of ``aggregate.json``.
    :type aggregate: dict[str, Any]
    :param stats: Content of ``stats.json``; None leaves the file out.
    :type stats: dict[str, Any] | None
    :return: The table, with nothing written.
    :rtype: AblationsTable
    """
    (folder / "aggregate.json").write_text(json.dumps(aggregate),
                                           encoding="utf-8")
    if stats is not None:
        (folder / "stats.json").write_text(json.dumps(stats),
                                           encoding="utf-8")
    inputs = Inputs(folder / "aggregate.json", folder / "stats.json",
                    folder / "cells.csv", folder / "jobs",
                    folder / "fig7.csv", folder / "fig7_trivial.csv")
    return AblationsTable(inputs, Out(folder / "out", 50, "cmd", "ref"))


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                  TESTS                                    #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class TestSummary:
    """
    Tests for the summary of the raw cells of one block.
    """

    def test_analytical_summary(self, tmp_path: Path) -> None:
        """
        Hand-computed summary of the four toy cells.

        By hand: the levels 200 and "200" are the same group, so D1 at
        level 200 has three cells. Two of them have a difference, 0.1
        and 0.3: the median is 0.2, the minimum 0.1 and the maximum
        0.3, and one of them is within noise. The third has no value
        and is listed by its job id. D8 at level 2000 has one cell and
        no difference at all.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        table = _table(tmp_path, {"complete": True})
        summary = table._summary_from_cells(CELLS)
        assert summary[0] == {
            "dataset_id": "D1", "value": "200", "n_configs": 3,
            "n_with_delta": 2,
            "delta_auroc_median": pytest.approx(0.2),
            "delta_auroc_min": 0.1, "delta_auroc_max": 0.3,
            "n_within_noise": 1, "without_value": ["job-c"]}
        assert summary[1] == {
            "dataset_id": "D8", "value": "2000", "n_configs": 1,
            "n_with_delta": 0, "delta_auroc_median": None,
            "delta_auroc_min": None, "delta_auroc_max": None,
            "n_within_noise": 0, "without_value": ["job-d"]}

    def test_empty_and_wrong_cells(self, tmp_path: Path) -> None:
        """
        No cells give no summary; anything but mappings is refused.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        table = _table(tmp_path, {"complete": True})
        assert table._summary_from_cells(tuple()) == list()
        with pytest.raises(TypeError, match="sequence of mappings"):
            table._summary_from_cells("cells")
        with pytest.raises(TypeError, match="must be a mapping"):
            table._summary_from_cells([1])
        with pytest.raises(TypeError, match="entry must be a mapping"):
            table._emit("T2-B", "window", 1, "src", list(), list())


class TestBuild:
    """
    Tests for the written table.
    """

    def test_analytical_cells_without_a_value(self,
                                              tmp_path: Path) -> None:
        """
        The last column counts the planned cells without a value.

        By hand: the block is rebuilt from the aggregate, because there
        are no statistics. Its two rows end with 1 and 1, the cells
        "job-c" and "job-d", so the table counts two and says so in a
        note next to the note of the fallback. The header of the last
        column reads "No value".

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        table = _table(tmp_path, {
            "complete": True, "ablations": list(CELLS),
            "datasets": {"D1": {"sd_pool": {"threshold": 0.101}}}})
        assert table.build() is True
        assert table.n_without_value == 2
        assert table.notes == [
            NOTE_NO_STATS,
            "2 planned cells without a value (not run, not completed "
            "or not executable)"]
        assert table.missing == list()
        tex = (tmp_path / "out" / "tab_ablations.tex").read_text(
            encoding="utf-8")
        assert r"& Within noise & No value \\" in tex
        assert r"T2-B & window & AW & 200 & 3 & 2 & 0.200 & 0.100 & " \
            r"0.300 & 1 & 1 \\" in tex
        assert r"T2-B & window & CIC17 & 2000 & 1 & 0 & -- & -- & -- " \
            r"& 0 & 1 \\" in tex
        assert "Noise reference per dataset: AW 0.101." in tex
        preview = (tmp_path / "out" / "tab_ablations.md").read_text(
            encoding="utf-8").splitlines()
        assert preview[1].endswith("| within noise | no value |")
        assert preview[3] == ("| T2-B | window | AW | 200 | 3 | 2 | "
                              "0.200 | 0.100 | 0.300 | 1 | 1 |")
        assert table.out.records[0]["partial_inputs"] is True
        assert "without_value=2" in repr(table)

    def test_final_statistics_are_not_labelled(self,
                                               tmp_path: Path) -> None:
        """
        With FINAL statistics the table is a final output.

        By hand: the statistics hold one summarised row with no cell
        without a value, so the count is zero, there is no note and
        the record of the table is not marked as partial.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        table = _table(tmp_path, {"complete": True}, stats={
            "run_type": "FINAL", "ablations": {"blocks": {"T3": {
                "cells": [{"factor": "k"}],
                "summary_by_dataset_and_value": [{
                    "dataset_id": "D1", "value": "2", "n_configs": 2,
                    "n_with_delta": 2, "delta_auroc_median": 0.0125,
                    "delta_auroc_min": 0.01, "delta_auroc_max": 0.015,
                    "n_within_noise": 2, "without_value": list()}]}}}})
        table.build()
        assert table.n_without_value == 0
        assert table.notes == list()
        assert table.out.records[0]["partial_inputs"] is False
        assert table.out.numbers["R.ABL.T3.D1.2.MED"][0] == 0.0125
        assert "[PARTIAL INPUTS]" not in (
            tmp_path / "out" / "tab_ablations.tex").read_text(
                encoding="utf-8")

    def test_no_cells_at_all(self, tmp_path: Path) -> None:
        """
        Without any ablation cell the table says so in one row.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        table = _table(tmp_path, {"complete": True})
        table.build()
        assert table.missing == [MISSING_NO_CELLS]
        assert "no ablation cells in the inputs" in (
            tmp_path / "out" / "tab_ablations.tex").read_text(
                encoding="utf-8")

    def test_wrong_arguments(self, tmp_path: Path) -> None:
        """
        The table needs a reader and a writer of the right classes.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        table = _table(tmp_path, {"complete": True})
        with pytest.raises(TypeError, match="inputs must be an Inputs"):
            AblationsTable("inputs", table.out)
        with pytest.raises(TypeError, match="out must be an Out"):
            AblationsTable(table.inputs, "out")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
