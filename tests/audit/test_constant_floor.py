# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         tests/audit/test_constant_floor.py                        #
#   Version:      1.0.0                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-07                                                #
#   Modified:     2026-10-07                                                #
#   Description:  Unit tests for the constant-predictor floor, from the     #
#                 stream cache and from the stored vectors.                 #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import json
from pathlib import Path

import numpy as np
import pytest

from kitnetzday.audit.constant_floor import SCALING_EPSILON, ConstantFloor
from kitnetzday.audit.training_audit import (
    VERDICT_AT_FLOOR,
    VERDICT_TRAINED,
    VERDICT_UNDEFINED,
    VERDICT_WORSE,
    TrainingAudit,
)

# The toy stream. Column 0 is 0, 2, 4, 6; column 1 is 1, 1, 3, 3; column
# 2 is constant. Its vector is worked out in
# test_analytical_vector_from_the_cache.
STREAM: list[list[float]] = [[0.0, 1.0, 5.0], [2.0, 1.0, 5.0],
                             [4.0, 3.0, 5.0], [6.0, 3.0, 5.0]]
STREAM_KEY: str = "teststream"
# The vector of that stream over its four rows, computed by hand.
VECTOR: list[float] = [5.0 / 36.0, 0.25, 0.0]
# A vector that is deliberately wrong, to tell which source was used.
WRONG: list[float] = [9.0, 9.0, 9.0]


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                AUXILIARY                                  #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def _write_cache(root: Path) -> Path:
    """
    Write the toy stream cache.

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


def _write_vectors(root: Path,
                   vector: list[float] | None = None,
                   epsilon: float = SCALING_EPSILON) -> Path:
    """
    Write a stored-vectors file for the toy stream.

    :param root: Temporary folder of the test.
    :type root: Path
    :param vector: Vector stored for rows [0, 4); None for the true one.
    :type vector: list[float] | None
    :param epsilon: Epsilon the file declares.
    :type epsilon: float
    :return: Path of the JSON file.
    :rtype: Path
    """
    path = root / "floor_vectors.json"
    path.write_text(json.dumps({
        "epsilon": epsilon,
        "stretches": {STREAM_KEY: {"0:4": {
            "vector": VECTOR if vector is None else vector}}},
        "contaminated": {"T4-E-toy": {
            "vector": [0.5, 0.5, 0.5],
            "block_sha256_match": True}}}), encoding="utf-8")
    return path


def _write_job(results: Path, name: str = "T1-toy") -> Path:
    """
    Write one finished job whose four groups land on the four verdicts.

    :param results: Results root the job directory is created in.
    :type results: Path
    :param name: Name of the job directory.
    :type name: str
    :return: The job directory.
    :rtype: Path
    """
    path = results / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "DONE").write_text("", encoding="utf-8")
    (path / "train_log.csv").write_text(
        "ae_index,class,n_visible,n_hidden,train_steps,loss_epoch_1,"
        "loss_source\n"
        "0,ae,1,2,4,0.05,model.history\n"
        "1,ae,1,2,4,0.26,model.history\n"
        "2,ae,1,2,4,0.10,model.history\n"
        "3,ae,2,2,4,0.50,model.history\n", encoding="utf-8")
    (path / "config.json").write_text(json.dumps({
        "spec": {"block": "T1", "dataset": "TEST", "variant": "lstm"},
        "regions": {"ae_train": [0, 4]},
        "cache": {"key": STREAM_KEY}}), encoding="utf-8")
    (path / "fm_stats.json").write_text(
        json.dumps({"feature_map": [[0], [1], [2], [0, 1]]}),
        encoding="utf-8")
    return path


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                  TESTS                                    #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class TestFromTheCache:
    """
    The floor computed from the preprocessed stream itself.
    """

    def test_analytical_vector_from_the_cache(self,
                                              tmp_path: Path) -> None:
        """
        Hand-computed vector of the toy stream.

        By hand. Column 0 holds 0, 2, 4, 6: its mean is 3, its variance
        (9 + 1 + 1 + 9) / 4 = 5 and its spread 6, so the variance of
        the min-max scaled column is 5 / 6^2 = 5 / 36. Column 1 holds
        1, 1, 3, 3: mean 2, variance 1, spread 2, so 1 / 2^2 = 0.25.
        Column 2 is constant: variance 0, and 0 divided by the squared
        epsilon is 0. The epsilon of 1e-16 does not move the first two
        values within double precision.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        floor = ConstantFloor(_write_cache(tmp_path))
        vector = floor.vector(STREAM_KEY, 0, 4)
        assert vector == pytest.approx(VECTOR)
        assert floor.source == {f"{STREAM_KEY}|0:4": "stream cache"}

    def test_analytical_group_floor(self, tmp_path: Path) -> None:
        """
        Hand-computed floor of a group of two columns.

        By hand: the floor of the group [0, 1] is the mean of its two
        columns, (5 / 36 + 1 / 4) / 2 = (5 / 36 + 9 / 36) / 2 = 7 / 36.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        floor = ConstantFloor(_write_cache(tmp_path))
        vector = floor.vector(STREAM_KEY, 0, 4)
        assert floor.group_floor(vector, [0, 1]) == pytest.approx(7 / 36)
        with pytest.raises(ValueError, match="at least one"):
            floor.group_floor(vector, list())

    def test_cache_wins_over_the_stored_vector(self,
                                               tmp_path: Path) -> None:
        """
        A stream that is on disk is never served from the stored file.

        The stored vector is wrong on purpose, so the true values can
        only come from the cache.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        floor = ConstantFloor(_write_cache(tmp_path),
                              vectors_path=_write_vectors(tmp_path,
                                                          WRONG))
        assert floor.vector(STREAM_KEY, 0, 4) == pytest.approx(VECTOR)
        assert floor.source[f"{STREAM_KEY}|0:4"] == "stream cache"


class TestFromTheStoredVectors:
    """
    The floor served from the file shipped with the package.
    """

    def test_stored_vector_when_the_cache_is_absent(
            self, tmp_path: Path) -> None:
        """
        Without the stream the stored vector is returned, and said so.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        floor = ConstantFloor(tmp_path / "no-cache",
                              vectors_path=_write_vectors(tmp_path))
        assert not floor.has_cache(STREAM_KEY)
        assert floor.has_stream(STREAM_KEY)
        assert not floor.has_stream("another-stream")
        assert list(floor.vector(STREAM_KEY, 0, 4)) == VECTOR
        assert floor.source == {f"{STREAM_KEY}|0:4": "stored"}
        assert "stored=1" in repr(floor)

    def test_unknown_stretch_without_the_cache(self,
                                               tmp_path: Path) -> None:
        """
        A stretch that is neither on disk nor stored cannot be served.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        floor = ConstantFloor(tmp_path / "no-cache",
                              vectors_path=_write_vectors(tmp_path))
        with pytest.raises(FileNotFoundError, match="x.npy"):
            floor.vector(STREAM_KEY, 0, 3)
        with pytest.raises(ValueError, match="greater than start"):
            floor.vector(STREAM_KEY, 4, 4)

    def test_stored_contaminated_vector(self, tmp_path: Path) -> None:
        """
        The vector of a training-contamination job is looked up by id.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        floor = ConstantFloor(tmp_path / "no-cache",
                              vectors_path=_write_vectors(tmp_path))
        vector, matches = floor.stored_contaminated("T4-E-toy")
        assert list(vector) == [0.5, 0.5, 0.5]
        assert matches is True
        assert floor.stored_contaminated("T4-E-other") is None
        with pytest.raises(TypeError, match="must be a string"):
            floor.stored_contaminated(7)

    def test_wrong_settings(self, tmp_path: Path) -> None:
        """
        A negative epsilon, or vectors of another epsilon, are refused.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        with pytest.raises(ValueError, match="must not be negative"):
            ConstantFloor(tmp_path, epsilon=-1.0)
        with pytest.raises(ValueError, match="computed with epsilon"):
            ConstantFloor(tmp_path, vectors_path=_write_vectors(
                tmp_path, epsilon=1e-8))
        with pytest.raises(TypeError, match="must be a string"):
            ConstantFloor(tmp_path).has_cache(3)

    def test_analytical_audit_from_the_stored_vectors(
            self, tmp_path: Path) -> None:
        """
        The four verdicts come out the same without the stream.

        By hand, with the floors 5 / 36, 1 / 4, 0 and 7 / 36 of the
        four groups and the recorded losses 0.05, 0.26, 0.10 and 0.50:
        0.05 / (5 / 36) = 0.36, below 0.90, "trained"; 0.26 / 0.25 =
        1.04, in [0.90, 1.15), "at the floor"; a zero floor gives no
        ratio, "not computable"; and 0.50 / (7 / 36) = 18 / 7 = 2.571,
        at or above 1.15, "worse than the floor".

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        results = tmp_path / "results"
        _write_job(results)
        audit = TrainingAudit(results_root=results,
                              cache_root=tmp_path / "no-cache",
                              vectors_path=_write_vectors(tmp_path))
        audit.run()
        verdicts = [a["verdict"] for a
                    in audit.jobs[0]["autoencoders"]]
        ratios = [a["ratio"] for a in audit.jobs[0]["autoencoders"]]
        assert verdicts == [VERDICT_TRAINED, VERDICT_AT_FLOOR,
                            VERDICT_UNDEFINED, VERDICT_WORSE]
        assert ratios[0] == pytest.approx(0.36)
        assert ratios[1] == pytest.approx(1.04)
        assert ratios[2] is None
        assert ratios[3] == pytest.approx(18 / 7)
        assert set(audit.floor.source.values()) == {"stored"}

    def test_job_without_any_floor_is_skipped(self,
                                              tmp_path: Path) -> None:
        """
        With neither the stream nor a stored vector the job is skipped.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        results = tmp_path / "results"
        _write_job(results)
        audit = TrainingAudit(results_root=results,
                              cache_root=tmp_path / "no-cache")
        audit.run()
        assert audit.jobs == list()
        assert len(audit.skipped) == 1


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
