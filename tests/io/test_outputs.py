# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         tests/io/test_outputs.py                                  #
#   Version:      1.0.0                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-13                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Unit tests for the output directory, the three reports    #
#                 of a run and the label of the partial inputs.             #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import argparse
import json
import logging
from pathlib import Path

import pytest

from kitnetzday.io.inputs import Inputs
from kitnetzday.io.outputs import Out
from kitnetzday.io.reports import (
    IMPLEMENTATION_NOTES,
    write_index,
    write_manifest,
    write_numbers,
)

# Label of the analysis plan used by the toy outputs.
TAG: str = "[CONFIRMATORIO plan §3.1]"
# Label of an output built from partial inputs.
PARTIAL: str = "[PARTIAL INPUTS]"
COMMAND: str = "uv run bin/make_figures.py"


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                AUXILIARY                                  #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def _out(folder: Path, logger: logging.Logger | None = None) -> Out:
    """
    Build an output directory inside a temporary folder.

    :param folder: Temporary folder; the outputs go to ``folder/out``.
    :type folder: Path
    :param logger: Logger handed to the writer, or None.
    :type logger: logging.Logger | None
    :return: The writer.
    :rtype: Out
    """
    return Out(folder / "out", 50, COMMAND, "script sha256:0123",
               root=folder, logger=logger)


def _inputs(folder: Path, final: bool = True) -> Inputs:
    """
    Build a reader over a toy aggregate, final or without statistics.

    :param folder: Temporary folder the input files are written to.
    :type folder: Path
    :param final: Whether to write FINAL statistics next to it.
    :type final: bool
    :return: The reader.
    :rtype: Inputs
    """
    (folder / "aggregate.json").write_text(
        json.dumps({"complete": True}), encoding="utf-8")
    if final:
        (folder / "stats.json").write_text(
            json.dumps({"run_type": "FINAL"}), encoding="utf-8")
    return Inputs(folder / "aggregate.json", folder / "stats.json",
                  folder / "cells.csv", folder / "jobs",
                  folder / "fig7.csv", folder / "fig7_trivial.csv")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                  TESTS                                    #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class TestOut:
    """
    Tests for the register of values and for the written tables.
    """

    def test_analytical_register_of_values(self, tmp_path: Path) -> None:
        """
        A value is stored with its label and its pointer.

        By hand: the first value is final, so its source is the label,
        a space and the pointer. The second one comes from partial
        inputs, so the partial label and a space go in front. The
        third is None and leaves no row: two rows in all.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        out = _out(tmp_path)
        assert out.add_number("R.A", 0.5, "a.json#/x", TAG, False) == \
            "R.A"
        out.add_number("R.B", 0.25, "a.json#/y", TAG, True)
        assert out.add_number("R.C", None, "a.json#/z", TAG, False) == \
            "R.C"
        assert out.numbers == {
            "R.A": (0.5, f"{TAG} a.json#/x"),
            "R.B": (0.25, f"{PARTIAL} {TAG} a.json#/y")}
        assert "numbers=2" in repr(out)

    def test_table_comments(self, tmp_path: Path) -> None:
        """
        The provenance comments carry the label only when asked to.

        By hand: the final table has four comment lines (generator,
        plan label, ids and its one note) and the partial one has five,
        the extra one being the partial label with its two reasons
        joined by "; ".

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        out = _out(tmp_path)
        out.save_table("tab_a", "BODY", "# preview", "Caption.", "tab:a",
                       TAG, ["R.A"], False, {"out_id": "T1",
                                             "notes": ["a note"]})
        out.save_table("tab_b", "BODY", None, "Caption.", "tab:b", TAG,
                       list(), True,
                       {"partial_inputs_reasons": ["r1", "r2"]})
        first = (tmp_path / "out" / "tab_a.tex").read_text(
            encoding="utf-8").splitlines()
        second = (tmp_path / "out" / "tab_b.tex").read_text(
            encoding="utf-8").splitlines()
        assert first == [
            "% tab_a - generated by bin/make_figures.py "
            "(script sha256:0123)", f"% Plan label: {TAG}",
            "% numbers_used.md ids (1): R.A", "% a note", "BODY"]
        assert second[2] == f"% {PARTIAL}: r1; r2"
        assert second[3:] == ["% numbers_used.md ids: (none)", "BODY"]
        assert [r["partial_inputs"] for r in out.records] == [False,
                                                               True]
        assert out.records[0]["files"] == ["tab_a.tex", "tab_a.md"]
        assert out.output_files(exclude=("tab_a.md",)) == [
            "tab_a.tex", "tab_b.tex"]

    def test_wrong_arguments(self,
                             tmp_path: Path,
                             caplog: pytest.LogCaptureFixture) -> None:
        """
        A bad resolution, an empty id and an empty name are refused.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        :param caplog: pytest fixture that captures log records.
        :type caplog: pytest.LogCaptureFixture
        """
        out = _out(tmp_path, logger=logging.getLogger("test"))
        with pytest.raises(ValueError, match="positive integer"):
            Out(tmp_path / "x", 0, COMMAND, "ref")
        with caplog.at_level(logging.ERROR, logger="test"):
            with pytest.raises(ValueError, match="non-empty string"):
                out.add_number("", 1.0, "p", TAG, False)
        assert "idd must be a non-empty string" in \
            caplog.records[-1].message
        with pytest.raises(ValueError, match="name must be a non-empty"):
            out.write_text("", "text")

    def test_paths_are_shown_from_the_root(self, tmp_path: Path) -> None:
        """
        A path inside the root is shown relative to it, others as given.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        out = _out(tmp_path)
        assert out.shown(tmp_path / "data" / "a.json") == "data/a.json"
        assert out.shown("/elsewhere/a.json") == "/elsewhere/a.json"
        assert Out(tmp_path / "y", 50, COMMAND, "ref").shown(
            "a/b.json") == "a/b.json"


class TestReports:
    """
    Tests for the table of values, the index and the manifest.
    """

    def test_analytical_table_of_values(self, tmp_path: Path) -> None:
        """
        The rows are sorted by id and their location is a dash.

        By hand: two values are registered, "R.B" first and "R.A"
        second, so the table lists "R.A" and then "R.B": two rows. The
        integer 3 is written as the float 3.0, and the third column of
        every row is "-".

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        out = _out(tmp_path)
        inputs = _inputs(tmp_path)
        out.add_number("R.B", 3, "a.json#/b", TAG, False)
        out.add_number("R.A", 0.5, "a.json#/a", TAG, False)
        assert write_numbers(inputs, out) == 2
        lines = (tmp_path / "out" / "numbers_used.md").read_text(
            encoding="utf-8").splitlines()
        assert lines[0] == ("# Values used by the figures and tables, "
                            "with their source")
        assert lines[-2] == (f"| R.A | 0.5 | - | {TAG} a.json#/a | "
                             f"script sha256:0123 | `{COMMAND}` |")
        assert lines[-1].startswith("| R.B | 3.0 | - | ")

    def test_partial_inputs_head_the_table(self, tmp_path: Path) -> None:
        """
        With partial inputs the table opens with the label and why.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        out = _out(tmp_path)
        inputs = _inputs(tmp_path, final=False)
        assert write_numbers(inputs, out) == 0
        head = (tmp_path / "out" / "numbers_used.md").read_text(
            encoding="utf-8").splitlines()[0]
        assert head == (f"# {PARTIAL} Values read from inputs other "
                        "than the final aggregate and its statistics: "
                        "no stats.json")

    def test_index_and_manifest(self, tmp_path: Path) -> None:
        """
        The index and the manifest state the status of the run.

        By hand: one final table is written from final inputs, so the
        index says FINAL and "final", the manifest holds
        ``partial_inputs`` false with no reason, one output, one listed
        value and the seven implementation notes, and the checksum
        file lists the manifest itself.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        out = _out(tmp_path)
        inputs = _inputs(tmp_path)
        args = argparse.Namespace(note=["checked"], only=None)
        out.add_number("R.A", 0.5, "a.json#/a", TAG, False)
        out.save_table("tab_a", "BODY", None, "Caption.", "tab:a", TAG,
                       ["R.A"], False, {"out_id": "T1"})
        write_index(inputs, out, list(), args, dict())
        manifest = write_manifest(inputs, out, {"k": "v"}, args,
                                  [{"job_id": "J1"}])
        index = (tmp_path / "out" / "README_figures.md").read_text(
            encoding="utf-8")
        assert "`stats.json run_type` = FINAL; **FINAL**." in index
        assert "| T1 | `tab_a` | table | " + TAG + " | final | " in index
        assert "- Run note: checked" in index
        assert PARTIAL + "**" not in index
        assert manifest["partial_inputs"] is False
        assert manifest["partial_inputs_reasons"] == list()
        assert manifest["numbers_listed"] == 1
        assert manifest["implementation_notes"] == IMPLEMENTATION_NOTES
        assert len(manifest["implementation_notes"]) == 7
        assert manifest["timing_reference_jobs"] == ["J1"]
        assert manifest["inputs"]["aggregate.json"]["path"] == \
            "aggregate.json"
        sums = (tmp_path / "out" / "SHA256SUMS").read_text(
            encoding="utf-8")
        assert "  figures.json\n" in sums

    def test_index_of_partial_inputs(self, tmp_path: Path) -> None:
        """
        With partial inputs the index says so, output by output.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        out = _out(tmp_path)
        inputs = _inputs(tmp_path, final=False)
        out.save_table("tab_a", "BODY", None, "Caption.", "tab:a", TAG,
                       list(), True, {"out_id": "T1",
                                      "missing": ["one input"]})
        write_index(inputs, out, list(), argparse.Namespace(), dict())
        index = (tmp_path / "out" / "README_figures.md").read_text(
            encoding="utf-8")
        assert "**PARTIAL INPUTS**." in index
        assert "  - Reasons: no stats.json" in index
        assert "| partial inputs | `tab_a.tex` |" in index
        assert f"- Plan label: {TAG}  **{PARTIAL}**" in index
        assert "- Missing data (1): one input" in index

    def test_wrong_arguments(self, tmp_path: Path) -> None:
        """
        The reports refuse anything that is not a reader and a writer.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        with pytest.raises(TypeError, match="must be a kitnetzday.io."):
            write_numbers("inputs", _out(tmp_path))
        with pytest.raises(TypeError, match="parameter out must be"):
            write_numbers(_inputs(tmp_path), "out")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
