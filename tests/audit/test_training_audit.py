# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         tests/audit/test_training_audit.py                        #
#   Version:      1.0.1                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Unit tests for the verdicts of the training audit and     #
#                 for their two thresholds.                                 #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from kitnetzday.audit.training_audit import (
    SKIP_REASON_NO_LOSS,
    SKIP_REASON_NO_STREAM,
    VERDICT_AT_FLOOR,
    VERDICT_TRAINED,
    VERDICT_UNDEFINED,
    VERDICT_WORSE,
    TrainingAudit,
)

# The toy stream every test audits. Column 0 is 0, 2, 4, 6; column 1 is
# 1, 1, 3, 3; column 2 is constant. The floors they give are worked out
# in the docstring of test_analytical_four_verdicts.
STREAM: list[list[float]] = [[0.0, 1.0, 5.0], [2.0, 1.0, 5.0],
                             [4.0, 3.0, 5.0], [6.0, 3.0, 5.0]]
STREAM_KEY: str = "teststream"
# Column groups of the toy feature map, one per autoencoder.
FEATURE_MAP: list[list[int]] = [[0], [1], [2], [0, 1]]
# Losses chosen so that the four groups land on the four verdicts.
LOSSES: list[str] = ["0.05", "0.26", "0.10", "0.50"]


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                AUXILIARY                                  #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def _write_cache(root: Path) -> Path:
    """
    Write the toy stream cache the audit reads the floors from.

    :param root: Temporary folder of the test.
    :type root: Path
    :return: The cache root, with one sub-directory per stream key.
    :rtype: Path
    """
    cache = root / "cache"
    (cache / STREAM_KEY).mkdir(parents=True, exist_ok=True)
    np.save(cache / STREAM_KEY / "x.npy",
            np.array(STREAM, dtype=np.float64))
    return cache


def _write_job(results: Path,
               name: str = "T1-toy",
               variant: str = "lstm",
               losses: list[str] | None = None,
               n_visible: list[str] | None = None,
               ae_train: list[int] | None = None,
               done: bool = True) -> Path:
    """
    Write one finished job directory the audit can read.

    :param results: Results root the job directory is created in.
    :type results: Path
    :param name: Name of the job directory.
    :type name: str
    :param variant: Variant written into ``spec.variant``.
    :type variant: str
    :param losses: One recorded loss per autoencoder, as the text the
        runner writes; None for the four losses of the toy case.
    :type losses: list[str] | None
    :param n_visible: One ``n_visible`` per autoencoder, as text; None
        to write the true size of every group.
    :type n_visible: list[str] | None
    :param ae_train: Training stretch as [start, stop); None for the
        whole toy stream, and an empty list to leave it out.
    :type ae_train: list[int] | None
    :param done: Whether to write the ``DONE`` marker.
    :type done: bool
    :return: The job directory.
    :rtype: Path
    """
    rows = losses if losses is not None else LOSSES
    sizes = (n_visible if n_visible is not None
             else [str(len(g)) for g in FEATURE_MAP])
    stretch = ae_train if ae_train is not None else [0, len(STREAM)]
    path = results / name
    path.mkdir(parents=True, exist_ok=True)
    if done:
        (path / "DONE").write_text("", encoding="utf-8")
    (path / "train_log.csv").write_text(
        "ae_index,class,n_visible,n_hidden,train_steps,loss_epoch_1,"
        "loss_source\n"
        + "".join(f"{i},ae,{sizes[i]},2,4,{loss},model.history\n"
                  for i, loss in enumerate(rows)), encoding="utf-8")
    (path / "config.json").write_text(json.dumps({
        "spec": {"block": "T1", "dataset": "TEST", "pipeline": "p",
                 "variant": variant, "ar": True, "window": 800, "k": 4,
                 "grouping": "g", "partition_seed": 1,
                 "contamination_pct": None, "seed": "A", "rep": 1},
        "regions": {"ae_train": stretch},
        "cache": {"key": STREAM_KEY},
        "manifest_row": {"paradigm": "AR"}}), encoding="utf-8")
    (path / "fm_stats.json").write_text(
        json.dumps({"feature_map": FEATURE_MAP}), encoding="utf-8")
    return path


def _audit(tmp_path: Path, **kwargs: Any) -> TrainingAudit:
    """
    Build an audit over a fresh toy tree, with its cache in place.

    :param tmp_path: Temporary folder of the test.
    :type tmp_path: Path
    :param kwargs: Extra arguments for the constructor of the audit.
    :type kwargs: Any
    :return: The audit, before ``run`` is called.
    :rtype: TrainingAudit
    """
    results = tmp_path / "results"
    results.mkdir(parents=True, exist_ok=True)
    return TrainingAudit(results_root=results,
                         cache_root=_write_cache(tmp_path), **kwargs)


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                MAIN CLASS                                 #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class TestVerdict:
    """
    Tests for the four verdicts and for the cuts 0.90 and 1.15.
    """

    def test_analytical_thresholds(self, tmp_path: Path) -> None:
        """
        Hand-computed boundaries of the two cuts of the plan.

        By hand, the rule is: "trained" for ratio < 0.90, "at the
        floor" for 0.90 <= ratio < 1.15 and "worse than the floor" for
        ratio >= 1.15. Both cuts are therefore closed on the right-hand
        band: exactly 0.90 is already at the floor, not trained, and
        exactly 1.15 is already worse than the floor. A value just
        below each cut stays in the band below it.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        audit = _audit(tmp_path)
        assert audit.verdict(0.0) == VERDICT_TRAINED
        assert audit.verdict(0.8999999999) == VERDICT_TRAINED
        assert audit.verdict(0.90) == VERDICT_AT_FLOOR
        assert audit.verdict(1.1499999999) == VERDICT_AT_FLOOR
        assert audit.verdict(1.15) == VERDICT_WORSE
        assert audit.verdict(1000.0) == VERDICT_WORSE

    def test_undefined_ratios(self, tmp_path: Path) -> None:
        """
        No ratio at all, or a non-finite one, is "not computable".

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        audit = _audit(tmp_path)
        assert audit.verdict(None) == VERDICT_UNDEFINED
        assert audit.verdict(float("nan")) == VERDICT_UNDEFINED
        assert audit.verdict(float("inf")) == VERDICT_UNDEFINED

    def test_cuts_come_from_the_configuration(self,
                                              tmp_path: Path) -> None:
        """
        The two cuts are parameters, read from src/config.json.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        audit = _audit(tmp_path, trained_below=0.5, at_floor_below=2.0)
        assert audit.verdict(0.4) == VERDICT_TRAINED
        assert audit.verdict(0.6) == VERDICT_AT_FLOOR
        assert audit.verdict(2.0) == VERDICT_WORSE

    def test_wrong_ratio_type_raises_and_logs(
            self,
            tmp_path: Path,
            caplog: pytest.LogCaptureFixture) -> None:
        """
        Text and booleans are not ratios and are refused by type.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        :param caplog: pytest fixture that captures log records.
        :type caplog: pytest.LogCaptureFixture
        """
        audit = _audit(tmp_path, logger=logging.getLogger("test"))
        with caplog.at_level(logging.ERROR, logger="test"):
            with pytest.raises(TypeError,
                               match="must be a real number or None"):
                audit.verdict("0.5")
        assert "[TrainingAudit.verdict] Input error" in \
            caplog.records[-1].message
        with pytest.raises(TypeError, match="not bool"):
            audit.verdict(True)


class TestConstructor:
    """
    Tests for the settings the audit refuses to start with.
    """

    def test_missing_results_root(self, tmp_path: Path) -> None:
        """
        A results root that is not a directory stops the audit.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        with pytest.raises(FileNotFoundError,
                           match="existing directory"):
            TrainingAudit(results_root=tmp_path / "nope",
                          cache_root=tmp_path)

    def test_unordered_or_negative_cuts(self, tmp_path: Path) -> None:
        """
        The cuts must be positive and ordered, the degenerate one >= 0.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        with pytest.raises(ValueError, match="strictly positive"):
            TrainingAudit(tmp_path, tmp_path, trained_below=0.0)
        with pytest.raises(ValueError, match="at least trained_below"):
            TrainingAudit(tmp_path, tmp_path, trained_below=1.0,
                          at_floor_below=0.5)
        with pytest.raises(ValueError, match="must not be negative"):
            TrainingAudit(tmp_path, tmp_path, degenerate_cut=-1.0)

    def test_missing_cache_root_only_warns(
            self,
            tmp_path: Path,
            caplog: pytest.LogCaptureFixture) -> None:
        """
        A cache root that does not exist warns and does not raise.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        :param caplog: pytest fixture that captures log records.
        :type caplog: pytest.LogCaptureFixture
        """
        with caplog.at_level(logging.WARNING, logger="test"):
            TrainingAudit(results_root=tmp_path,
                          cache_root=tmp_path / "nope",
                          logger=logging.getLogger("test"))
        assert "[ConstantFloor.__init__] Input warning" in \
            caplog.records[-1].message


class TestAuditedRatios:
    """
    Tests for the ratios the audit computes from the recorded loss.
    """

    def test_analytical_four_verdicts(self, tmp_path: Path) -> None:
        """
        Hand-computed floors and ratios of the toy stream.

        The floor of a feature group is the mean, over its columns, of
        the variance of the column after the min-max scaling of its own
        training stretch. The toy stream has four rows and three
        columns, and the whole stream is the training stretch.

        Column 0 is 0, 2, 4, 6: its minimum is 0 and its maximum 6, so
        the spread is 6; its mean is 3 and the squared distances to it
        are 9, 1, 1 and 9, which add up to 20, so the raw variance is
        20 / 4 = 5. Scaling divides by the spread, so the scaled
        variance is 5 / 6^2 = 5 / 36 = 0.138889. The 1e-16 the code adds
        to the spread is far below the resolution of 36 in float64 and
        disappears.

        Column 1 is 1, 1, 3, 3: spread 3 - 1 = 2, mean 2, squared
        distances 1, 1, 1, 1, raw variance 4 / 4 = 1, scaled variance
        1 / 2^2 = 0.25.

        Column 2 is constant: spread 0 and variance 0, so its scaled
        variance is 0 / (1e-16)^2 = 0 and the group has no floor at all.

        The four groups are therefore [0] -> 5 / 36, [1] -> 1 / 4,
        [2] -> 0 and [0, 1] -> (5 / 36 + 9 / 36) / 2 = 7 / 36 =
        0.194444. With the recorded losses 0.05, 0.26, 0.10 and 0.50 the
        ratios are

        - 0.05 / (5 / 36) = 0.05 * 36 / 5 = 0.36, below 0.90:
          "trained";
        - 0.26 / 0.25 = 1.04, in [0.90, 1.15): "at the floor";
        - group [2] has a zero floor, so there is no ratio:
          "not computable", and the group is flagged as degenerate;
        - 0.50 / (7 / 36) = 0.50 * 36 / 7 = 18 / 7 = 2.571429, at or
          above 1.15: "worse than the floor".

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        audit = _audit(tmp_path)
        _write_job(audit.results_root)
        audit.run()
        aes = audit.jobs[0]["autoencoders"]
        assert [ae["floor_mse"] for ae in aes] == pytest.approx(
            [5.0 / 36.0, 0.25, 0.0, 7.0 / 36.0])
        assert [ae["ratio"] for ae in aes[:2]] == pytest.approx(
            [0.36, 1.04])
        assert aes[2]["ratio"] is None
        assert aes[3]["ratio"] == pytest.approx(18.0 / 7.0)
        assert [ae["verdict"] for ae in aes] == [
            VERDICT_TRAINED, VERDICT_AT_FLOOR, VERDICT_UNDEFINED,
            VERDICT_WORSE]
        assert [ae["floor_degenerate"] for ae in aes] == [
            False, False, True, False]

    def test_flat_records_and_group_sizes(self, tmp_path: Path) -> None:
        """
        One flat record per autoencoder, with the size of its group.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        audit = _audit(tmp_path)
        _write_job(audit.results_root)
        audit.run()
        assert len(audit.flat) == len(FEATURE_MAP)
        assert [r["n_features_group"] for r in audit.flat] == [1, 1, 1, 2]
        assert {r["variant"] for r in audit.flat} == {"lstm"}
        assert audit.warnings == list()

    def test_missing_loss_is_not_a_trained_model(self,
                                                 tmp_path: Path) -> None:
        """
        An "NA" or empty loss gives no ratio, never a zero one.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        audit = _audit(tmp_path)
        _write_job(audit.results_root,
                   losses=["NA", "", "nan", "0.05"])
        audit.run()
        aes = audit.jobs[0]["autoencoders"]
        assert [ae["loss_mse"] for ae in aes[:3]] == [None, None, None]
        assert [ae["verdict"] for ae in aes[:3]] == [
            VERDICT_UNDEFINED] * 3
        assert aes[3]["verdict"] == VERDICT_TRAINED

    def test_group_size_mismatch_is_a_warning(self,
                                              tmp_path: Path) -> None:
        """
        A recorded n_visible that disagrees with the map is reported.

        The group of the feature map prevails, so the ratio is still
        computed; the disagreement is only recorded as a warning.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        audit = _audit(tmp_path)
        _write_job(audit.results_root,
                   n_visible=["99", "1", "1", "2"])
        audit.run()
        assert len(audit.warnings) == 1
        assert "n_visible=99 != |group|=1" in audit.warnings[0]
        assert audit.jobs[0]["autoencoders"][0]["verdict"] == \
            VERDICT_TRAINED


class TestSkipped:
    """
    Tests for the jobs that are counted but cannot be audited.
    """

    def test_variants_without_a_comparable_loss(self,
                                                tmp_path: Path) -> None:
        """
        The numpy and statistical variants are skipped, not failed.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        audit = _audit(tmp_path)
        _write_job(audit.results_root, name="T1-dA", variant="original")
        _write_job(audit.results_root, name="T1-stat", variant="stat")
        audit.run()
        assert audit.jobs == list()
        assert [s["reason"] for s in audit.skipped] == \
            [SKIP_REASON_NO_LOSS] * 2
        assert [s["job_id"] for s in audit.skipped] == \
            ["T1-dA", "T1-stat"]

    def test_job_without_its_training_region(self,
                                             tmp_path: Path) -> None:
        """
        No ``regions.ae_train`` means no floor and so no audit.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        audit = _audit(tmp_path)
        _write_job(audit.results_root, ae_train=list())
        audit.run()
        assert audit.jobs == list()
        assert audit.skipped[0]["reason"] == SKIP_REASON_NO_STREAM

    def test_job_without_done_file_is_ignored(self,
                                              tmp_path: Path) -> None:
        """
        A directory without its DONE file is not a completed run and is
        passed over.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        audit = _audit(tmp_path)
        _write_job(audit.results_root, name="T1-no-done", done=False)
        audit.run()
        assert audit.jobs == list()
        assert audit.skipped == list()

    def test_unreadable_configuration(self, tmp_path: Path) -> None:
        """
        A job whose config.json is broken is skipped with its reason.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        audit = _audit(tmp_path)
        path = _write_job(audit.results_root)
        (path / "config.json").write_text("{ not json",
                                          encoding="utf-8")
        audit.run()
        assert audit.jobs == list()
        assert audit.skipped[0]["reason"].startswith(
            "unreadable config.json:")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
