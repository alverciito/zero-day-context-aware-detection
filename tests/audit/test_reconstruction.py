# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         tests/audit/test_reconstruction.py                        #
#   Version:      1.0.2                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Unit tests for the exploratory correlations and their     #
#                 within-group permutation p-value.                         #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import json
import logging
from pathlib import Path
from typing import Any

import pytest

from kitnetzday.audit.reconstruction import ReconstructionCorrelation

# Trained fractions written into the fabricated audit. The cut is 0.5 and
# the candidate variants are lstm and mlp, so only lstm survives: mlp is
# below the cut and conv1d is not a candidate at all.
FRACTIONS: dict[str, float] = {"lstm": 0.7778, "mlp": 0.4,
                               "conv1d": 1.0}
# Permutations used by the tests. The real analyses use the 100000 of
# the plan; nothing here depends on the count except through the
# add-one estimator, which is checked explicitly.
DRAWS: int = 20


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                AUXILIARY                                  #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def _write_audit(path: Path, jobs: list[dict[str, Any]]) -> None:
    """
    Write a fabricated ``training_audit.json`` with the given jobs.

    :param path: File to write.
    :type path: Path
    :param jobs: One record per job, each with its "autoencoders".
    :type jobs: list[dict[str, Any]]
    """
    path.write_text(json.dumps({
        "jobs": jobs,
        "aggregates": {"by_variant": {
            name: {"fraction_of_9_datasets_with_a_trained_ae": value}
            for name, value in FRACTIONS.items()}}}), encoding="utf-8")


def _job(job_id: str,
         dataset: str,
         ratios: list[float | None],
         degenerate: list[bool] | None = None,
         variant: str = "lstm",
         paradigm: str = "AR") -> dict[str, Any]:
    """
    One job record of the fabricated audit.

    :param job_id: Job id, which must match the directory of its
        metrics file.
    :type job_id: str
    :param dataset: Dataset of the job.
    :type dataset: str
    :param ratios: Loss-to-floor ratio of every autoencoder, None where
        the audit could not compute one.
    :type ratios: list[float | None]
    :param degenerate: Whether the floor of every group is degenerate;
        None for no degenerate group at all.
    :type degenerate: list[bool] | None
    :param variant: Variant of the job.
    :type variant: str
    :param paradigm: Paradigm of the job.
    :type paradigm: str
    :return: The job record.
    :rtype: dict[str, Any]
    """
    flags = degenerate if degenerate is not None else [False] * len(
        ratios)
    return {"job_id": job_id, "dataset": dataset, "variant": variant,
            "paradigm": paradigm,
            "autoencoders": [{"ratio": r, "floor_degenerate": d}
                             for r, d in zip(ratios, flags)]}


def _write_metrics(results: Path,
                   job_id: str,
                   raw: float | None,
                   fused: float | None) -> None:
    """
    Write one ``T1-*/metrics.json`` with the two ROC-AUC values.

    :param results: Results root the job directory is created in.
    :type results: Path
    :param job_id: Name of the job directory; it must match the glob.
    :type job_id: str
    :param raw: ROC-AUC of the raw reconstruction error, or None to
        leave it out.
    :type raw: float | None
    :param fused: ROC-AUC of the fused detector, or None to leave it
        out.
    :type fused: float | None
    """
    path = results / job_id
    path.mkdir(parents=True, exist_ok=True)
    (path / "metrics.json").write_text(json.dumps({
        "job": {"job_id": job_id},
        "E": {"RAW": {"auroc": raw}, "FUS": {"auroc": fused}}}),
        encoding="utf-8")


def _study(tmp_path: Path, **kwargs: Any) -> ReconstructionCorrelation:
    """
    Build the analysis over the fabricated inputs of a test.

    :param tmp_path: Temporary folder of the test.
    :type tmp_path: Path
    :param kwargs: Extra arguments for the constructor.
    :type kwargs: Any
    :return: The analysis, before ``run`` is called.
    :rtype: ReconstructionCorrelation
    """
    kwargs.setdefault("draws", DRAWS)
    return ReconstructionCorrelation(
        audit_path=tmp_path / "training_audit.json",
        results_root=tmp_path, **kwargs)


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                MAIN CLASS                                 #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class TestQuality:
    """
    Tests for the reconstruction quality read off the training audit.
    """

    def test_analytical_median_of_the_sound_groups(
            self,
            tmp_path: Path) -> None:
        """
        Hand-computed quality of a job with five autoencoders.

        By hand: the quality of a job is the median of the
        loss-to-floor ratios of its autoencoders, and two kinds of
        autoencoder are dropped first. One whose feature group has a
        degenerate floor is dropped, because its ratio measures the
        floor and not the model, and one with no ratio at all is
        dropped too. Of the five autoencoders here, 0.2, 0.4 and 0.6
        survive, 9.0 is degenerate and the last has no ratio. With
        three values left the median is the middle one, 0.4.

        A job whose only autoencoder is degenerate has no quality at
        all and does not appear.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        _write_audit(tmp_path / "training_audit.json", [
            _job("T1-a", "D8", [0.2, 0.4, 0.6, 9.0, None],
                 degenerate=[False, False, False, True, False]),
            _job("T1-b", "D8", [1.0], degenerate=[True])])
        quality, meta = _study(tmp_path).quality_by_job()
        assert quality == {"T1-a": 0.4}
        assert meta == {"T1-a": ("D8", "lstm", "AR")}

    def test_analytical_trained_variants(self, tmp_path: Path) -> None:
        """
        Hand-computed filter of the variants that leave the floor.

        By hand: a variant is kept when it is one of the two
        candidates, lstm and mlp, and when the audit itself gives it a
        trained autoencoder in at least half of the nine datasets.
        Here lstm reaches 0.7778, above the 0.5 cut, and is kept; mlp
        reaches only 0.4 and is dropped; conv1d reaches 1.0 but is not
        a candidate, so it is dropped as well. One variant survives.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        _write_audit(tmp_path / "training_audit.json",
                     [_job("T1-a", "D8", [0.2])])
        assert _study(tmp_path).trained_variants() == ["lstm"]

    def test_cut_and_candidates_are_parameters(self,
                                               tmp_path: Path) -> None:
        """
        Lowering the cut or widening the candidates lets more in.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        _write_audit(tmp_path / "training_audit.json",
                     [_job("T1-a", "D8", [0.2])])
        assert _study(tmp_path,
                      min_trained_fraction=0.3).trained_variants() == \
            ["lstm", "mlp"]
        assert _study(tmp_path,
                      candidate_variants=("conv1d",)
                      ).trained_variants() == ["conv1d"]

    def test_jobs_without_both_roc_auc_are_dropped(
            self,
            tmp_path: Path) -> None:
        """
        A job that does not publish both ROC-AUC values is left out.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        _write_metrics(tmp_path, "T1-a", 0.7, 0.8)
        _write_metrics(tmp_path, "T1-b", 0.7, None)
        _write_metrics(tmp_path, "T1-c", None, 0.8)
        assert _study(tmp_path).auroc_by_job() == {"T1-a": (0.7, 0.8)}


class TestRows:
    """
    Tests for the analysis table and for its de-duplication.
    """

    def test_analytical_deduplication(self, tmp_path: Path) -> None:
        """
        Hand-computed de-duplication of the MLP pair.

        By hand: the autoregressive and the time-series jobs of a
        per-record model are the same configuration and give identical
        numbers, so counting both would pseudo-replicate. A row is
        therefore keyed by (dataset, variant, quality, raw ROC-AUC,
        fused ROC-AUC), rounded to twelve decimals, and a repeated key
        is dropped. Here T1-a and T1-b agree in all five fields, so of
        the three jobs that pass the filters only two rows survive, and
        the count before de-duplicating is still reported as three.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        _write_audit(tmp_path / "training_audit.json", [
            _job("T1-a", "D8", [0.4]),
            _job("T1-b", "D8", [0.4], paradigm="TSR"),
            _job("T1-c", "D9", [0.8]),
            _job("T1-d", "D8", [0.5], variant="mlp")])
        for job_id, raw, fused in (("T1-a", 0.7, 0.8),
                                   ("T1-b", 0.7, 0.8),
                                   ("T1-c", 0.6, 0.5),
                                   ("T1-d", 0.9, 0.9)):
            _write_metrics(tmp_path, job_id, raw, fused)
        rows, n_jobs, variants = _study(tmp_path).rows()
        assert n_jobs == 3
        assert variants == ["lstm"]
        assert rows == [(0.4, 0.7, 0.8, "D8", "lstm"),
                        (0.8, 0.6, 0.5, "D9", "lstm")]


class TestPermutationP:
    """
    Tests for the within-group permutation p-value.
    """

    def test_analytical_one_row_per_group(self, tmp_path: Path) -> None:
        """
        Hand-computed p-value when every group holds a single row.

        By hand: the p-value permutes the outcome inside each group and
        counts how often the absolute Spearman rho of the permuted data
        reaches the observed one. A group of one row has exactly one
        permutation, the identity, so every draw reproduces the data
        unchanged and ties the observed statistic. With D draws the
        count is therefore D, and the add-one estimator gives
        (D + 1) / (D + 1) = 1 exactly. This is not a defect: it says
        that with one row per stream and variant the permutation scheme
        has no freedom left and the test is uninformative.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        p_value, count = _study(tmp_path).permutation_p(
            [1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0],
            ["a", "b", "c", "d"])
        assert count == DRAWS
        assert p_value == 1.0

    def test_add_one_estimator_and_reproducibility(
            self,
            tmp_path: Path) -> None:
        """
        The p-value is (count + 1) / (draws + 1) and never zero.

        The generator is restarted from the seed of the plan at every
        call, so two calls with the same seed return the same count:
        that is what makes the published numbers reproducible.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        study = _study(tmp_path, draws=50)
        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        y = [1.0, 3.0, 2.0, 5.0, 4.0, 6.0]
        groups = ["g", "g", "g", "h", "h", "h"]
        first = study.permutation_p(x, y, groups)
        second = study.permutation_p(x, y, groups)
        assert first == second
        assert first[0] == pytest.approx((first[1] + 1) / 51)
        assert 0.0 < first[0] <= 1.0

    def test_mismatched_or_too_short_inputs(self,
                                            tmp_path: Path) -> None:
        """
        The three sequences must be as long as each other, and longer
        than one.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        study = _study(tmp_path)
        with pytest.raises(ValueError, match="same length"):
            study.permutation_p([1.0, 2.0], [1.0], ["a", "b"])
        with pytest.raises(ValueError, match="at least two rows"):
            study.permutation_p([1.0], [1.0], ["a"])


class TestRun:
    """
    Tests for the two correlations the analysis publishes.
    """

    def test_analytical_perfect_monotone_correlation(
            self,
            tmp_path: Path) -> None:
        """
        Hand-computed rho of -1 for a perfectly ordered table.

        By hand, the four rows that survive are, by job id,
        (x = 0.4, raw = 0.70, fused = 0.80), (0.8, 0.60, 0.50),
        (0.2, 0.90, 0.95) and (0.6, 0.65, 0.60). Sorted by x they read
        0.2, 0.4, 0.6, 0.8 with raw 0.90, 0.70, 0.65, 0.60 and fused
        0.95, 0.80, 0.60, 0.50: both outcomes fall strictly as x grows,
        so each is a perfectly decreasing function of x and the
        Spearman rho of both is exactly -1. Smaller x means a better
        reconstruction, so this is the table in which better
        reconstruction always goes with better detection.

        Every row sits in its own (dataset, variant) group, so the
        permutation has nothing to shuffle and both p-values are
        exactly 1, as in test_analytical_one_row_per_group. Each stream
        contributes a single row, which is below the three a
        within-stream rho needs, so no stream gets its own rho and the
        sign check counts zero streams.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        _write_audit(tmp_path / "training_audit.json", [
            _job("T1-a", "D8", [0.4]),
            _job("T1-b", "D8", [0.4], paradigm="TSR"),
            _job("T1-c", "D9", [0.8]),
            _job("T1-d", "D1", [0.2]),
            _job("T1-e", "D2", [0.6])])
        for job_id, raw, fused in (("T1-a", 0.70, 0.80),
                                   ("T1-b", 0.70, 0.80),
                                   ("T1-c", 0.60, 0.50),
                                   ("T1-d", 0.90, 0.95),
                                   ("T1-e", 0.65, 0.60)):
            _write_metrics(tmp_path, job_id, raw, fused)
        result = _study(tmp_path).run()
        assert (result["n"], result["n_jobs_before_deduplication"]) == \
            (4, 5)
        assert result["variants"] == ["lstm"]
        assert result["B"] == DRAWS
        for name in ("raw", "fus_primary"):
            assert result[name]["rho"] == pytest.approx(-1.0)
            assert result[name]["p_permutation"] == 1.0
            assert result[name]["count_ge"] == DRAWS
            assert result[name]["of_streams"] == 0
            assert result[name]["negative_in"] == 0

    def test_analytical_per_stream_rho(self, tmp_path: Path) -> None:
        """
        Hand-computed global and per-stream rho of a four-row table.

        By hand, the rows by job id are (x = 0.2, y = 0.9),
        (0.4, 0.8), (0.6, 0.7) and (0.5, 0.5), the last one on another
        stream. The ranks of x are 1, 2, 4, 3 and the ranks of y are 4,
        3, 2, 1, so the rank differences are -3, -1, 2, 2 and the sum
        of their squares is 9 + 1 + 4 + 4 = 18. With n = 4 the Spearman
        rho is 1 - 6 * 18 / (4 * (16 - 1)) = 1 - 108 / 60 = -0.8.

        A stream gets its own rho only with at least three rows. The
        first stream has three, and inside it y falls as x grows, so
        its rho is exactly -1; the second has one row and gets none. So
        one stream is measured and its rho is negative.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        _write_audit(tmp_path / "training_audit.json", [
            _job("T1-a", "D8", [0.2]), _job("T1-b", "D8", [0.4]),
            _job("T1-c", "D8", [0.6]), _job("T1-d", "D9", [0.5])])
        for job_id, value in (("T1-a", 0.9), ("T1-b", 0.8),
                              ("T1-c", 0.7), ("T1-d", 0.5)):
            _write_metrics(tmp_path, job_id, value, value)
        result = _study(tmp_path).run()
        assert result["n"] == 4
        block = result["fus_primary"]
        assert block["rho"] == pytest.approx(-0.8)
        assert block["rho_by_dataset"]["D8"] == pytest.approx(-1.0)
        assert block["rho_by_dataset"]["D9"] is None
        assert (block["of_streams"], block["negative_in"]) == (1, 1)
        assert block["p_permutation"] == pytest.approx(
            (block["count_ge"] + 1) / (DRAWS + 1))

    def test_labels_of_the_result(self, tmp_path: Path) -> None:
        """
        The result carries the exploratory tag and the seed it used.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        _write_audit(tmp_path / "training_audit.json", [
            _job("T1-a", "D8", [0.2]), _job("T1-b", "D9", [0.4])])
        _write_metrics(tmp_path, "T1-a", 0.9, 0.9)
        _write_metrics(tmp_path, "T1-b", 0.8, 0.8)
        result = _study(tmp_path, seed=7).run()
        assert result["tag"].startswith("[EXPLORATORY]")
        assert result["seed"] == 7
        assert result["permutation_groups"] == "dataset x variant"
        assert "lower = better" in result["x_definition"]

    def test_too_few_rows_to_correlate(self, tmp_path: Path) -> None:
        """
        One surviving row is not a correlation and says so.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        _write_audit(tmp_path / "training_audit.json",
                     [_job("T1-a", "D8", [0.2])])
        _write_metrics(tmp_path, "T1-a", 0.9, 0.9)
        with pytest.raises(ValueError, match="at least two rows"):
            _study(tmp_path).run()

    def test_missing_audit_file(self, tmp_path: Path) -> None:
        """
        The audit file is only opened when it is needed, and then it
        must be there.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        with pytest.raises(OSError):
            _study(tmp_path).run()


class TestSettings:
    """
    Tests for the settings the analysis refuses to start with.
    """

    def test_bad_settings_raise_and_log(
            self,
            tmp_path: Path,
            caplog: pytest.LogCaptureFixture) -> None:
        """
        The draws, the cut, the candidates and the glob are checked.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        :param caplog: pytest fixture that captures log records.
        :type caplog: pytest.LogCaptureFixture
        """
        with caplog.at_level(logging.ERROR, logger="test"):
            with pytest.raises(ValueError, match="strictly positive"):
                _study(tmp_path, draws=0,
                       logger=logging.getLogger("test"))
        assert "[ReconstructionCorrelation.__init__] Input error" in \
            caplog.records[-1].message
        with pytest.raises(ValueError, match=r"in \[0, 1\]"):
            _study(tmp_path, min_trained_fraction=1.5)
        with pytest.raises(TypeError, match="sequence of strings"):
            _study(tmp_path, candidate_variants="lstm")
        with pytest.raises(ValueError, match="non-empty"):
            _study(tmp_path, metrics_glob="")

    def test_summary_lines_need_both_outcomes(self,
                                              tmp_path: Path) -> None:
        """
        A result without both outcomes cannot be summarized.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        study = _study(tmp_path)
        with pytest.raises(TypeError, match="must be a mapping"):
            study.summary_lines(list())
        with pytest.raises(ValueError, match="must hold the outcome"):
            study.summary_lines({"n": 1})


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
