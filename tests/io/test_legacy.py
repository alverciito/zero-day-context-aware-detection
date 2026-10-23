# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         tests/io/test_legacy.py                                   #
#   Version:      1.0.1                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-07                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Unit tests for the readers of the archived Fig. 7         #
#                 values, the trivial F1 table and the timing jobs.         #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import json
from pathlib import Path
from typing import Any

from kitnetzday.io.legacy import (
    parse_f1_trivial_smoothed,
    parse_fig7_csv,
    read_timing_reference,
)

# Job id of the toy reference run: block, dataset, variant, paradigm.
JOB_ID: str = "T1-A-KNAD__knad-active_wiretap__lstm__arT__v1"


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                AUXILIARY                                  #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def _write_job(root: Path,
               name: str = JOB_ID,
               timing: dict[str, Any] | None = None) -> Path:
    """
    Write one job directory with the three files the cost table reads.

    :param root: Results root the job directory is created in.
    :type root: Path
    :param name: Name of the job directory.
    :type name: str
    :param timing: Contents of ``timing.json``; None for an exclusive
        reference run of 7200 seconds.
    :type timing: dict[str, Any] | None
    :return: The job directory.
    :rtype: Path
    """
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "timing.json").write_text(json.dumps(
        timing if timing is not None else {
            "timing_ref": 1, "wall_seconds": 7200.0, "device": "/GPU:0",
            "phases": {"train_s": 5000.0}, "rows": 1000,
            "exec_rows_per_s": 250.0,
            "gpu_peak_bytes": {"fm": 10, "train": 2048, "ckpt": 7},
            "rss_peak_bytes": 4096,
            "concurrent_jobs_same_host": {"start": 0, "end": 0}}),
        encoding="utf-8")
    (path / "config.json").write_text(json.dumps({
        "hyperparameters_effective": {"autoencoders": [
            {"count_params": 100}, {"count_params": 23},
            {"name": "numpy dA"}]}}), encoding="utf-8")
    (path / "metrics.json").write_text(json.dumps(
        {"timing": {"total_s": 21.5}}), encoding="utf-8")
    return path


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                  TESTS                                    #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class TestFig7Readers:
    """
    The two archived inputs of Figure A.1 of the manuscript.
    """

    def test_published_values(self, tmp_path: Path) -> None:
        """
        One row per metric is folded into one record per bar.

        The file keeps the column names of the archived artefact, which
        are in Spanish. A row of an unknown dataset and a row whose
        value is not a number are skipped, so three of the five rows
        below are used.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        path = tmp_path / "fig7.csv"
        path.write_text(
            "dataset,variante,paradigma,metrica,valor,detector\n"
            "CIC2017,MLP,AR,F1,0.75,KMD\n"
            "CIC2017,MLP,AR,WindowDiff,0.25,KMD\n"
            "CIC2018,LSTM,TSR,F1,0.5,MVD\n"
            "UNKNOWN,MLP,AR,F1,0.9,KMD\n"
            "CIC2018,LSTM,TSR,WindowDiff,not-a-number,MVD\n",
            encoding="utf-8")
        values, used = parse_fig7_csv(path)
        assert used == 3
        assert values == {
            ("D8", "MLP", "AR"): {"F1": 0.75, "WindowDiff": 0.25,
                                  "detector": "KMD"},
            ("D9", "LSTM", "TSR"): {"F1": 0.5, "detector": "MVD"}}

    def test_trivial_f1_table(self, tmp_path: Path) -> None:
        """
        The trivial F1 is read per dataset, skipping what is not usable.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        path = tmp_path / "trivial.csv"
        path.write_text(
            "dataset,f1_trivial_smoothed\n"
            "CIC2017,0.737\n"
            "OS-SCAN,0.220\n"
            "UNKNOWN,0.5\n"
            "CIC2018,n/a\n", encoding="utf-8")
        assert parse_f1_trivial_smoothed(path) == {"D8": 0.737,
                                                   "D3": 0.22}

    def test_missing_files_give_empty_results(self,
                                              tmp_path: Path) -> None:
        """
        A missing archived file gives an empty result, not an error.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        assert parse_fig7_csv(tmp_path / "absent.csv") == (dict(), 0)
        assert parse_f1_trivial_smoothed(tmp_path / "absent.csv") \
            == dict()

    def test_shipped_trivial_f1_has_the_nine_streams(self) -> None:
        """
        The file shipped in ``data/`` holds one value per stream.
        """
        path = (Path(__file__).resolve().parents[2] / "data"
                / "fig7_trivial_f1.csv")
        if not path.exists():
            return
        values = parse_f1_trivial_smoothed(path)
        assert sorted(values) == [f"D{i}" for i in range(1, 10)]
        assert all(0.0 < v < 1.0 for v in values.values())


class TestTimingReference:
    """
    The jobs that may be quoted in the cost table.
    """

    def test_analytical_reference_job(self, tmp_path: Path) -> None:
        """
        Hand-checked record of one exclusive reference run.

        By hand: the job id splits into block, dataset, variant "lstm"
        and "arT", which is the autoregressive paradigm. The largest of
        the three GPU peaks (10, 2048, 7) is 2048. Two of the three
        autoencoders publish a parameter count, 100 + 23 = 123, and the
        third does not, so it adds nothing. No job ran next to it at
        the start or at the end, so the run is exclusive.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        _write_job(tmp_path)
        rows = read_timing_reference(tmp_path)
        assert len(rows) == 1
        row = rows[0]
        assert (row["variant"], row["paradigm"]) == ("lstm", "AR")
        assert row["wall_seconds"] == 7200.0
        assert row["gpu_peak_bytes"] == 2048
        assert (row["n_ae"], row["params_total"]) == (3, 123)
        assert row["postproc_total_s"] == 21.5
        assert row["exclusive"] is True

    def test_non_reference_and_non_exclusive_jobs(
            self, tmp_path: Path) -> None:
        """
        Only reference runs are kept, and exclusivity fails closed.

        A job without ``timing_ref = 1`` is left out. A reference run
        with another job at its start is flagged, and one whose
        ``timing.json`` does not say is reported as unknown (None),
        never as exclusive.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        _write_job(tmp_path, "a__d__mlp__arF", {"timing_ref": 0})
        _write_job(tmp_path, "b__d__mlp__arF", {
            "timing_ref": 1,
            "concurrent_jobs_same_host": {"start": 1, "end": 0}})
        _write_job(tmp_path, "c__d__mlp__arT", {"timing_ref": 1})
        (tmp_path / "d__no_timing").mkdir()
        rows = read_timing_reference(tmp_path)
        assert [r["job_id"] for r in rows] == ["b__d__mlp__arF",
                                               "c__d__mlp__arT"]
        assert rows[0]["exclusive"] is False
        assert rows[0]["paradigm"] == "TSR"
        assert "1/0" in rows[0]["exclusive_reason"]
        assert rows[1]["exclusive"] is None
        assert "without" in rows[1]["exclusive_reason"]

    def test_missing_root_gives_no_rows(self, tmp_path: Path) -> None:
        """
        A results root that is not a directory gives an empty list.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        assert read_timing_reference(tmp_path / "absent") == list()


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
