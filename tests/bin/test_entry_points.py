# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         tests/bin/test_entry_points.py                            #
#   Version:      1.0.1                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-07                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Runs every script of bin/ against the shipped data and    #
#                 compares what it writes with the reference outputs.       #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import importlib.util
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

# Root of the package, the shipped inputs and the reference outputs.
ROOT: Path = Path(__file__).resolve().parents[2]
JOBS: Path = ROOT / "data" / "jobs"
REFERENCE: Path = ROOT / "data" / "reference"
# Tables written by bin/make_figures.py from modules that only read the
# aggregate and the job files.
TABLES_TEX: tuple[str, ...] = (
    "tab_main_results.tex", "tab_main_results_compact.tex",
    "tab_best_per_dataset.tex", "tab_ablations.tex", "tab_cost.tex")
TABLES_MD: tuple[str, ...] = (
    "tab_main_results.md", "tab_best_per_dataset.md",
    "tab_ablations.md", "tab_cost.md")
FIGURES: tuple[str, ...] = (
    "fig_main_auroc_heatmap", "fig_mean_ranks",
    "fig_appendix_fig7_oracle", "fig_failure_cases")
# Relative tolerance of the numeric comparisons: the reference was
# written with one numpy build and a reader may run another.
TOLERANCE: float = 1e-9

pytestmark = pytest.mark.skipif(
    not JOBS.is_dir() or not REFERENCE.is_dir(),
    reason="data/jobs and data/reference are needed for these tests")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                AUXILIARY                                  #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def _main(name: str) -> Callable[[Sequence[str]], int]:
    """
    Import one script of ``bin/`` and return its ``main``.

    :param name: File name of the script, without the extension.
    :type name: str
    :return: The ``main`` function of that script.
    :rtype: Callable[[Sequence[str]], int]
    """
    path = ROOT / "bin" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"bin_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def _body(path: Path) -> list[str]:
    """
    Lines of a generated ``.tex`` without its provenance comments.

    The comments carry the sha256 of the generating script, which
    changes with any edit of it; the float itself must not.

    :param path: Path of the ``.tex`` file.
    :type path: Path
    :return: The lines that do not start with "%".
    :rtype: list[str]
    """
    return [line for line
            in path.read_text(encoding="utf-8").splitlines()
            if not line.startswith("%")]


def _assert_close(got: Any, expected: Any, where: str = "") -> None:
    """
    Compare two JSON structures, numbers within a relative tolerance.

    :param got: Structure produced by the run under test.
    :type got: Any
    :param expected: Structure of the reference output.
    :type expected: Any
    :param where: JSON pointer of the structures, for the message.
    :type where: str
    :raises AssertionError: If the structures differ.
    """
    if isinstance(expected, dict):
        assert isinstance(got, dict), where
        assert sorted(got) == sorted(expected), where
        for key in expected:
            _assert_close(got[key], expected[key], f"{where}/{key}")
    elif isinstance(expected, list):
        assert isinstance(got, list) and len(got) == len(expected), where
        for index, (a, b) in enumerate(zip(got, expected)):
            _assert_close(a, b, f"{where}[{index}]")
    elif isinstance(expected, float):
        assert got == pytest.approx(expected, rel=TOLERANCE,
                                    abs=1e-12), where
    else:
        assert got == expected, where


def _load(path: Path) -> Any:
    """
    Read a JSON file.

    :param path: Path of the file.
    :type path: Path
    :return: The parsed document.
    :rtype: Any
    """
    return json.loads(path.read_text(encoding="utf-8"))


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                  TESTS                                    #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class TestMakeFigures:
    """
    ``bin/make_figures.py``: the tables and the four figures.
    """

    def test_tables_match_the_reference(self, tmp_path: Path) -> None:
        """
        The three table groups come out as the reference has them.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        code = _main("make_figures")(["--only", "T1,T2,T3",
                                      "--out-dir", str(tmp_path)])
        assert code == 0
        for name in TABLES_TEX:
            assert _body(tmp_path / name) == _body(REFERENCE / name), \
                name
        for name in TABLES_MD:
            assert (tmp_path / name).read_text(encoding="utf-8") == \
                (REFERENCE / name).read_text(encoding="utf-8"), name

    def test_cost_table_has_the_twelve_reference_runs(
            self, tmp_path: Path) -> None:
        """
        The cost table holds one row per reference run, none missing.

        The Markdown preview has a title, a header and a separator
        line, and then one row per run: twelve.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        code = _main("make_figures")(["--only", "T3",
                                      "--out-dir", str(tmp_path)])
        assert code == 0
        rows = [line for line in (tmp_path / "tab_cost.md").read_text(
            encoding="utf-8").splitlines() if line.startswith("|")]
        assert len(rows) - 2 == 12
        assert "no reference timing" not in (
            tmp_path / "tab_cost.tex").read_text(encoding="utf-8")

    def test_cost_table_without_job_files_fails(self,
                                                tmp_path: Path) -> None:
        """
        A cost table without rows is not written and the run fails.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        (tmp_path / "empty").mkdir()
        code = _main("make_figures")([
            "--only", "T3", "--out-dir", str(tmp_path / "out"),
            "--results-root", str(tmp_path / "empty")])
        assert code == 1
        assert not (tmp_path / "out" / "tab_cost.tex").exists()
        assert not (tmp_path / "out" / "tab_cost.md").exists()

    def test_unknown_output_id(self, tmp_path: Path) -> None:
        """
        An id that is not one of the seven outputs is refused.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        assert _main("make_figures")(["--only", "T9", "--out-dir",
                                      str(tmp_path)]) == 2

    def test_every_output_is_written_and_relocatable(
            self, tmp_path: Path) -> None:
        """
        The full run writes the nine outputs and no path of this machine.

        The figures are checked for existence only: their rendering
        depends on the matplotlib build. The manifest must not hold the
        absolute location of the package.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        assert _main("make_figures")(["--out-dir", str(tmp_path)]) == 0
        for name in FIGURES:
            for suffix in (".pdf", ".png", ".tex"):
                assert (tmp_path / f"{name}{suffix}").stat().st_size > 0
        manifest = _load(tmp_path / "figures.json")
        assert len(manifest["outputs"]) == 9
        assert manifest["stats_run_type"] == "FINAL"
        assert manifest["inputs"]["cells.csv"]["path"] == \
            "data/aggregate/cells.csv"
        assert str(ROOT) not in json.dumps(manifest["provenance"])
        assert str(ROOT) not in json.dumps(manifest["inputs"])


class TestSupplementaryTables:
    """
    ``bin/make_supplementary_tables.py``.
    """

    def test_matches_the_reference(self, tmp_path: Path) -> None:
        """
        The fifteen floats and the JSON document match the reference.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        tex = tmp_path / "supplementary_tables.tex"
        doc = tmp_path / "supplementary_tables.json"
        code = _main("make_supplementary_tables")([
            "--out-tex", str(tex), "--out-json", str(doc)])
        assert code == 0
        text = tex.read_text(encoding="utf-8")
        assert text == (REFERENCE / tex.name).read_text(encoding="utf-8")
        assert text.count(r"\label{tab:s_") == 15
        _assert_close(_load(doc), _load(REFERENCE / doc.name))

    def test_without_job_files_fails(self, tmp_path: Path) -> None:
        """
        Without the per-job records nothing is written and the run fails.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        (tmp_path / "empty").mkdir()
        tex = tmp_path / "supplementary_tables.tex"
        code = _main("make_supplementary_tables")([
            "--results-root", str(tmp_path / "empty"),
            "--out-tex", str(tex),
            "--out-json", str(tmp_path / "supplementary_tables.json")])
        assert code == 1
        assert not tex.exists()


class TestBaselineTables:
    """
    ``bin/make_baseline_tables.py``.
    """

    def test_matches_the_reference(self, tmp_path: Path) -> None:
        """
        The two comparator tables match the reference.

        The reference KitNET was run on the six KNAD captures and the
        Isolation Forest on the nine streams, once each.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        out = tmp_path / "baselines.json"
        code = _main("make_baseline_tables")([
            "--out-json", str(out),
            "--out-md", str(tmp_path / "baselines.md")])
        assert code == 0
        got = _load(out)
        _assert_close(got, _load(REFERENCE / out.name))
        assert len(got["isolation_forest"]["by_dataset"]) == 9
        assert sorted(got["knad_auc_eer"]) == [
            "D1", "D2", "D3", "D4", "D5", "D6", "mean"]

    def test_without_job_files_fails(self, tmp_path: Path) -> None:
        """
        Without the comparator jobs nothing is written and the run fails.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        (tmp_path / "empty").mkdir()
        out = tmp_path / "baselines.json"
        code = _main("make_baseline_tables")([
            "--results-root", str(tmp_path / "empty"),
            "--out-json", str(out),
            "--out-md", str(tmp_path / "baselines.md")])
        assert code == 1
        assert not out.exists()


class TestTrainingAudit:
    """
    ``bin/audit_training.py``.
    """

    def test_matches_the_reference(self, tmp_path: Path) -> None:
        """
        The audit of the shipped jobs matches the reference result.

        The interpreter version is the one field allowed to differ.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        out = tmp_path / "training_audit.json"
        assert _main("audit_training")(["--out-json", str(out)]) == 0
        got, expected = _load(out), _load(REFERENCE / out.name)
        got["method"].pop("interpreter")
        expected["method"].pop("interpreter")
        _assert_close(got, expected)
        assert got["method"]["floor_source"]["stream cache"] == 0
        assert got["method"]["floor_source"]["stored"] > 0

    def test_spanish_note_is_optional(self, tmp_path: Path) -> None:
        """
        The Spanish note of the audit is written only when asked for.

        Its first table has one row per variant with the same counts
        the result file holds; the row of the per-record MLP is checked
        against the result.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        out = tmp_path / "training_audit.json"
        note = tmp_path / "training_audit.md"
        assert _main("audit_training")(["--out-json", str(out)]) == 0
        assert not note.exists()
        assert _main("audit_training")(["--out-json", str(out),
                                        "--note"]) == 0
        row = _load(out)["aggregates"]["by_variant"]["mlp"]
        counts = row["counts"]
        assert (f"| `mlp` | {row['n_jobs']} | {row['n_autoencoders']} | "
                f"{counts['trained']} | {counts['at the floor']} | "
                f"{counts['worse than the floor']} | "
                f"{counts['not computable']} |") in note.read_text(
                    encoding="utf-8")

    def test_without_job_files_fails(self, tmp_path: Path) -> None:
        """
        A tree with nothing to audit writes nothing and fails.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        (tmp_path / "empty").mkdir()
        out = tmp_path / "training_audit.json"
        code = _main("audit_training")([
            "--results-root", str(tmp_path / "empty"),
            "--out-json", str(out)])
        assert code == 1
        assert not out.exists()
        assert _main("audit_training")([
            "--results-root", str(tmp_path / "absent"),
            "--out-json", str(out)]) == 1


class TestRetrainedPair:
    """
    ``bin/analyse_retrained_pair.py``.
    """

    def test_matches_the_reference(self, tmp_path: Path) -> None:
        """
        The paired report of the retraining matches the reference.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        out = tmp_path / "retrained_pair.json"
        assert _main("analyse_retrained_pair")(["--json", str(out)]) == 0
        got = _load(out)
        _assert_close(got, _load(REFERENCE / out.name))
        assert got["pairs"]["mlp_ctx_fix"]["n_cells"] == 36

    def test_without_job_files_fails(self, tmp_path: Path) -> None:
        """
        Zero paired cells is a failure, not an empty report.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        (tmp_path / "empty").mkdir()
        out = tmp_path / "retrained_pair.json"
        code = _main("analyse_retrained_pair")([
            "--results-root", str(tmp_path / "empty"),
            "--json", str(out)])
        assert code == 1
        assert not out.exists()


class TestReconstruction:
    """
    ``bin/analyse_reconstruction.py``.
    """

    def test_correlations_match_the_reference(self,
                                              tmp_path: Path) -> None:
        """
        The two correlations match the reference.

        The run uses 200 permutations to stay fast, so the permutation
        p values are not compared: the correlations, the sample size
        and the per-stream values do not depend on the draws.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        out = tmp_path / "reconstruction_vs_detection.json"
        code = _main("analyse_reconstruction")([
            "--audit", str(REFERENCE / "training_audit.json"),
            "--out", str(out), "--draws", "200"])
        assert code == 0
        got, expected = _load(out), _load(REFERENCE / out.name)
        assert got["n"] == expected["n"]
        for outcome in ("raw", "fus_primary"):
            assert got[outcome]["rho"] == pytest.approx(
                expected[outcome]["rho"], rel=TOLERANCE)
            _assert_close(got[outcome]["rho_by_dataset"],
                          expected[outcome]["rho_by_dataset"])

    def test_without_the_audit_fails(self, tmp_path: Path) -> None:
        """
        A missing audit result ends the run with exit code 1.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        out = tmp_path / "reconstruction_vs_detection.json"
        code = _main("analyse_reconstruction")([
            "--audit", str(tmp_path / "absent.json"),
            "--out", str(out), "--draws", "10"])
        assert code == 1
        assert not out.exists()


class TestRunAll:
    """
    ``bin/run_all.py``.
    """

    def test_every_step_succeeds(self, tmp_path: Path) -> None:
        """
        The six entry points run in order and all of them succeed.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        code = _main("run_all")(["--out-dir", str(tmp_path),
                                 "--draws", "200"])
        assert code == 0
        for name in ("training_audit.json", "retrained_pair.json",
                     "reconstruction_vs_detection.json", "baselines.json",
                     "supplementary_tables.tex", "figures.json"):
            assert (tmp_path / name).stat().st_size > 0


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
