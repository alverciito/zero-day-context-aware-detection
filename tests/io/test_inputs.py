# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         tests/io/test_inputs.py                                   #
#   Version:      1.0.0                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-13                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Unit tests for the reader of the aggregate: when its      #
#                 inputs are the final ones, and its cross-checks.          #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import json
import logging
from pathlib import Path
from typing import Any

import pytest

from kitnetzday.io.inputs import Inputs
from kitnetzday.utils.formatting import InputError

# Aggregate of the toy study: one dataset, one configuration.
AGGREGATE: dict[str, Any] = {
    "complete": True,
    "cells": {"D1": {"C1": {"name": "Conv1D-AR",
                            "mean": {"auroc_FUS": 0.75},
                            "divergent_seeds": ["B"]}}},
    "value_sets": {"main": {"D1": {"C1": {"auroc_FUS": 0.75}}}},
    "datasets": {"D1": {"dataset": "toy stream", "S_star": ["A", "B"],
                        "trivial": {"auc_t": 0.5},
                        "sd_pool": {"sqrt2_sd_pool": 0.02,
                                    "threshold": 0.02}}}}


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                AUXILIARY                                  #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def _inputs(folder: Path,
            aggregate: dict[str, Any] | None = None,
            stats: dict[str, Any] | None = None,
            cells: str | None = None,
            logger: logging.Logger | None = None) -> Inputs:
    """
    Build a reader over files written into a temporary folder.

    :param folder: Folder the input files are written to.
    :type folder: Path
    :param aggregate: Content of ``aggregate.json``; None for the toy
        aggregate.
    :type aggregate: dict[str, Any] | None
    :param stats: Content of ``stats.json``; None leaves the file out.
    :type stats: dict[str, Any] | None
    :param cells: Text of ``cells.csv``; None leaves the file out.
    :type cells: str | None
    :param logger: Logger handed to the reader, or None.
    :type logger: logging.Logger | None
    :return: The reader.
    :rtype: Inputs
    """
    (folder / "aggregate.json").write_text(
        json.dumps(AGGREGATE if aggregate is None else aggregate),
        encoding="utf-8")
    if stats is not None:
        (folder / "stats.json").write_text(json.dumps(stats),
                                           encoding="utf-8")
    if cells is not None:
        (folder / "cells.csv").write_text(cells, encoding="utf-8")
    return Inputs(agg_path=folder / "aggregate.json",
                  stats_path=folder / "stats.json",
                  cells_path=folder / "cells.csv",
                  results_root=folder / "jobs",
                  fig7_csv=folder / "fig7.csv",
                  fig7_md=folder / "fig7_trivial.csv", logger=logger)


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                  TESTS                                    #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class TestPartialInputs:
    """
    Tests for the rule that tells the final inputs from any other.
    """

    def test_analytical_final_inputs(self, tmp_path: Path) -> None:
        """
        A complete aggregate with FINAL statistics is final.

        By hand: the three conditions hold (the statistics file is
        there, its run type is FINAL, the aggregate is complete), so
        the flag is False and there is no reason to give.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        inputs = _inputs(tmp_path, stats={"run_type": "FINAL"})
        assert inputs.partial_inputs is False
        assert inputs.partial_inputs_reasons == list()
        assert "partial_inputs=False" in repr(inputs)

    def test_analytical_without_statistics(self, tmp_path: Path) -> None:
        """
        Without the statistics file the inputs are partial.

        By hand: one condition fails, so the flag is True with the one
        reason "no stats.json". The reader also records the two inputs
        it did not find, the statistics and the flat cells.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        inputs = _inputs(tmp_path)
        assert inputs.partial_inputs is True
        assert inputs.partial_inputs_reasons == ["no stats.json"]
        assert len(inputs.missing) == 2
        assert inputs.missing[0].endswith("carry [PARTIAL INPUTS]")
        assert inputs.sha256["stats.json"] is None

    def test_analytical_other_run_type(self, tmp_path: Path) -> None:
        """
        Statistics of another run type give their own reasons.

        By hand: the run type is "EARLY" with the reasons "a" and "b",
        so the single reason reads "stats.json run_type = EARLY: a; b".

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        inputs = _inputs(tmp_path, stats={
            "run_type": "EARLY", "run_type_reasons": ["a", "b"]})
        assert inputs.partial_inputs is True
        assert inputs.partial_inputs_reasons == [
            "stats.json run_type = EARLY: a; b"]

    def test_analytical_incomplete_aggregate(self,
                                             tmp_path: Path) -> None:
        """
        An aggregate that is not complete makes the inputs partial.

        By hand: the statistics are FINAL but the aggregate says
        complete = false, so the only reason is
        "aggregate.complete = false".

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        aggregate = dict(AGGREGATE, complete=False)
        inputs = _inputs(tmp_path, aggregate=aggregate,
                         stats={"run_type": "FINAL"})
        assert inputs.partial_inputs is True
        assert inputs.partial_inputs_reasons == [
            "aggregate.complete = false"]


class TestReading:
    """
    Tests for the values the reader hands out and for its checks.
    """

    def test_values_and_pointers(self, tmp_path: Path) -> None:
        """
        Every value comes back with the pointer it was read from.

        By hand: without statistics the ROC-AUC of D1/C1 is the 0.75 of
        the "main" value set of the aggregate, the cell carries one
        diverged seed, two seeds entered the analysis set and the noise
        threshold is 0.02. A cell that is not there gives None.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        inputs = _inputs(tmp_path)
        assert inputs.auroc_src("D1", "C1") == (
            0.75, "aggregate.json#/value_sets/main/D1/C1/auroc_FUS")
        assert inputs.auroc("D1", "C9") is None
        assert inputs.auroc_ptr("D1", "C1").startswith("aggregate.json")
        assert inputs.imputed("D1", "C1") is True
        assert inputs.n_seeds("D1") == 2
        assert inputs.noise_threshold("D1") == 0.02
        assert inputs.sqrt2_sd_pool("D1") == 0.02
        assert inputs.trivial("D1", "auc_t") == 0.5
        assert inputs.cell_mean("D1", "C1", "auroc_FUS") == 0.75
        assert inputs.cell("D1", "C9") == dict()
        assert inputs.config_name("C1") == "Conv1D-AR"
        assert inputs.config_short("C10") == "dA"
        assert inputs.dataset_label("D1") == "KNAD Active Wiretap"

    def test_statistics_take_precedence(self, tmp_path: Path) -> None:
        """
        With statistics the heat-map value is the one that is read.

        By hand: the statistics hold 0.75 for D1/C1, the same value as
        the aggregate, so the cross-check counts one cell and no
        mismatch, and the pointer names ``stats.json``.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        inputs = _inputs(tmp_path, stats={
            "run_type": "FINAL",
            "R110": {"heatmap_auroc_fus": {"D1": {"C1": 0.75}}}})
        assert inputs.auroc_src("D1", "C1") == (
            0.75, "stats.json#/R110/heatmap_auroc_fus/D1/C1")
        assert inputs.cross_stats == {"checked": 1, "mismatch": list()}
        assert inputs.consistency == list()

    def test_cross_check_reports_a_mismatch(self, tmp_path: Path) -> None:
        """
        A flat cell that disagrees with the aggregate is reported.

        By hand: ``cells.csv`` gives 0.70 for D1/C1 and the aggregate
        0.75; the difference of 0.05 is above the tolerance of 1e-9,
        so one inconsistency is recorded and one note points at it.
        The statistics disagree as well (0.80), which adds a second.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        inputs = _inputs(
            tmp_path,
            stats={"run_type": "FINAL",
                   "R110": {"heatmap_auroc_fus": {"D1": {"C1": 0.80}}}},
            cells="dataset_id,config_id,auroc_FUS_mean\nD1,C1,0.70\n"
                  "D1,C2,0.5\n,C3,0.1\nD1,C4,\n")
        assert len(inputs.consistency) == 3
        assert inputs.consistency[0] == \
            "D1/C1: cells.csv 0.7 != aggregate.json 0.75"
        assert "present in one source" in inputs.consistency[1]
        assert inputs.consistency[2].startswith("D1/C1: stats.json 0.8")
        assert inputs.notes == [
            "3 mismatches between the inputs (see figures.json)"]

    def test_wrong_arguments(self,
                             tmp_path: Path,
                             caplog: pytest.LogCaptureFixture) -> None:
        """
        A missing aggregate, a bad tolerance and bad ids are refused.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        :param caplog: pytest fixture that captures log records.
        :type caplog: pytest.LogCaptureFixture
        """
        inputs = _inputs(tmp_path, logger=logging.getLogger("test"))
        with caplog.at_level(logging.ERROR, logger="test"):
            with pytest.raises(ValueError, match="non-empty string"):
                inputs.auroc("", "C1")
        assert "must be a non-empty string" in caplog.records[-1].message
        with pytest.raises(InputError, match="aggregate must exist"):
            Inputs(tmp_path / "nope.json", tmp_path / "s.json",
                   tmp_path / "c.csv", tmp_path, tmp_path / "a",
                   tmp_path / "b")
        with pytest.raises(ValueError, match="strictly positive"):
            Inputs(tmp_path / "aggregate.json", tmp_path / "s.json",
                   tmp_path / "c.csv", tmp_path, tmp_path / "a",
                   tmp_path / "b", tolerance=0.0)
        with pytest.raises(TypeError, match="must be a Naming or None"):
            Inputs(tmp_path / "aggregate.json", tmp_path / "s.json",
                   tmp_path / "c.csv", tmp_path, tmp_path / "a",
                   tmp_path / "b", naming="D1")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
