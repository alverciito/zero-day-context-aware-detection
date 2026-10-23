# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         src/kitnetzday/audit/constant_floor.py                    #
#   Version:      1.0.2                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Error floor of the constant predictor over the scaled     #
#                 training stretch of an autoencoder.                       #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import hashlib
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from kitnetzday.utils.loggable import Loggable

# The 1e-16 of train() in frozen/models/networks/*.py, with the same
# value here: the floor has to be the floor of the very scaling the
# networks used.
SCALING_EPSILON: float = 1e-16


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                MAIN CLASS                                 #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class ConstantFloor(Loggable):
    """
    How small the loss would be if the model only learned the mean.

    An autoencoder that learned nothing at all still reaches a definite
    reconstruction error: the one of answering, for every record, the
    mean of each feature over its training data. That value is the floor
    against which a recorded loss has to be read, and this class computes
    it from the read-only stream caches.

    The networks of the study scale their input min-max with the minimum
    and the maximum of their own training stretch, ``x_scaled = (x - min)
    / (max - min + 1e-16)``, and the optimal constant per feature is its
    mean, so the floor of a feature group g is

    ``floor(g) = mean_{f in g} Var(x_scaled[:, f])``.

    It is evaluated as ``Var(x_raw[:, f]) / (max_f - min_f + 1e-16)^2``,
    which is the same number without ever materialising the scaled
    matrix. The class returns the per-column vector and leaves the mean
    over each group to the caller, because the vector does not depend on
    the architecture and is shared by every job of a stream: one read per
    (stream, stretch) serves them all.

    The targets of the training are the rows of the stretch in the three
    windowing schemes of the study, so this floor is the same for the
    autoregressive and the time-series-reconstruction paradigms.

    The preprocessed streams are 26 GB and do not travel with the code,
    so the vectors computed from them are shipped in a small JSON file
    (``data/floor_vectors.json``). A stream whose cache is on disk is
    always read from the cache; the stored vector is used only when the
    cache is absent, and ``source`` says which of the two served every
    request, so that a result never hides where its floor came from.

    :param cache_root: Directory with one sub-directory per stream key,
        each holding the preprocessed ``x.npy``.
    :type cache_root: str | Path
    :param epsilon: Term added to the min-max spread before dividing; it
        must be the one the networks used, 1e-16.
    :type epsilon: float
    :param vectors_path: JSON file with the vectors computed in advance
        from the stream caches, or None to work from the caches alone.
    :type vectors_path: str | Path | None
    :param logger: Logger created by the calling script, or None to stay
        silent.
    :type logger: logging.Logger | None
    :raises TypeError: If logger is not a logging.Logger or None.
    :raises ValueError: If epsilon is negative, or if the stored vectors
        were computed with another epsilon.
    """

    def __init__(self,
                 cache_root: str | Path,
                 epsilon: float = SCALING_EPSILON,
                 vectors_path: str | Path | None = None,
                 logger: logging.Logger | None = None) -> None:
        """
        Store the cache root and read the stored vectors, if any.

        :param cache_root: Directory with one sub-directory per stream.
        :type cache_root: str | Path
        :param epsilon: Term added to the min-max spread, 1e-16.
        :type epsilon: float
        :param vectors_path: JSON file with stored vectors, or None.
        :type vectors_path: str | Path | None
        :param logger: Logger created by the calling script, or None.
        :type logger: logging.Logger | None
        :raises TypeError: If logger is not a logging.Logger or None.
        :raises ValueError: If epsilon is negative, or if the stored
            vectors were computed with another epsilon.
        """
        super().__init__(logger=logger)
        if float(epsilon) < 0.0:
            self._error("[ConstantFloor.__init__] Input error: the "
                        "parameter epsilon must not be negative (>=0), "
                        f"not {epsilon}.")
        self.cache_root = Path(cache_root)
        self.epsilon = float(epsilon)
        self.cache: dict[tuple[str, int, int], np.ndarray] = dict()
        self.stored: dict[str, Any] = dict()
        self.source: dict[str, str] = dict()
        self.vectors_path = (None if vectors_path is None
                             else Path(vectors_path))
        if self.vectors_path is not None and self.vectors_path.exists():
            self.stored = json.loads(self.vectors_path.read_text(
                encoding="utf-8"))
            if float(self.stored.get("epsilon", epsilon)) != self.epsilon:
                self._error("[ConstantFloor.__init__] Input error: the "
                            "stored vectors must have been computed "
                            f"with epsilon {self.epsilon}, not "
                            f"{self.stored.get('epsilon')}.")
        if not self.cache_root.is_dir() and not self.stored:
            self._warning("[ConstantFloor.__init__] Input warning: the "
                          "parameter cache_root must be an existing "
                          f"directory, and it is not: {cache_root}; "
                          "there are no stored vectors either. No floor "
                          "can be computed without one of them.")

    def has_cache(self, stream_key: str) -> bool:
        """
        Whether the preprocessed stream itself is on disk.

        :param stream_key: Sub-directory of cache_root, for example
            "cic2017__corrected__v1".
        :type stream_key: str
        :return: True when that sub-directory exists.
        :rtype: bool
        :raises TypeError: If stream_key is not a string.
        """
        if not isinstance(stream_key, str):
            self._error("[ConstantFloor.has_cache] Input error: the "
                        "parameter stream_key must be a string, not "
                        f"{type(stream_key).__name__}.", TypeError)
        return (self.cache_root / stream_key).is_dir()

    def has_stream(self, stream_key: str) -> bool:
        """
        Whether a floor can be given for one stream.

        :param stream_key: Sub-directory of cache_root, for example
            "cic2017__corrected__v1".
        :type stream_key: str
        :return: True when the cache of the stream is on disk or a
            vector of that stream is stored.
        :rtype: bool
        :raises TypeError: If stream_key is not a string.
        """
        return (self.has_cache(stream_key)
                or stream_key in self.stored.get("stretches", dict()))

    def stored_contaminated(self, job_id: str) -> tuple[np.ndarray,
                                                        bool] | None:
        """
        Stored vector of one job of the training-contamination block.

        :param job_id: Job id of the T4-E job.
        :type job_id: str
        :return: The per-column variance of its contaminated stretch
            and whether the rebuilt block reproduced the recorded hash
            when the vector was computed, or None when nothing is
            stored for that job.
        :rtype: tuple[np.ndarray, bool] | None
        :raises TypeError: If job_id is not a string.
        """
        if not isinstance(job_id, str):
            self._error("[ConstantFloor.stored_contaminated] Input "
                        "error: the parameter job_id must be a string, "
                        f"not {type(job_id).__name__}.", TypeError)
        block = self.stored.get("contaminated", dict()).get(job_id)
        if block is None:
            return None
        self.source[job_id] = "stored"
        return (np.asarray(block["vector"], dtype=np.float64),
                bool(block["block_sha256_match"]))

    def vector(self,
               stream_key: str,
               start: int,
               stop: int) -> np.ndarray:
        """
        Per-column variance of the min-max scaled training stretch.

        The result is cached per (stream_key, start, stop): it does not
        depend on the variant, so a stretch shared by many jobs is read
        from disk only once. The returned array is the cached one and
        must not be modified in place.

        :param stream_key: Sub-directory of cache_root that holds
            ``x.npy``.
        :type stream_key: str
        :param start: First row of the training stretch of the
            autoencoders, as ``regions.ae_train[0]`` of the job.
        :type start: int
        :param stop: One past the last row of that stretch; it must be
            greater than start.
        :type stop: int
        :return: Variance of every column of the scaled stretch, shape
            (n_features,), in float64.
        :rtype: np.ndarray
        :raises TypeError: If stream_key is not a string.
        :raises ValueError: If stop is not greater than start.
        :raises FileNotFoundError: If the stream cache has no ``x.npy``
            and no vector of that stretch is stored.
        """
        key = (str(stream_key), int(start), int(stop))
        if not isinstance(stream_key, str):
            self._error("[ConstantFloor.vector] Input error: the "
                        "parameter stream_key must be a string, not "
                        f"{type(stream_key).__name__}.", TypeError)
        if int(stop) <= int(start):
            self._error("[ConstantFloor.vector] Input error: the "
                        f"parameter stop must be greater than start "
                        f"({start}), not {stop}.")
        if key in self.cache:
            return self.cache[key]
        label = f"{key[0]}|{key[1]}:{key[2]}"
        block = self.stored.get("stretches", dict()).get(
            key[0], dict()).get(f"{key[1]}:{key[2]}")
        if not self.has_cache(stream_key) and block is not None:
            self.cache[key] = np.asarray(block["vector"],
                                         dtype=np.float64)
            self.source[label] = "stored"
            return self.cache[key]
        seg = np.asarray(self._rows(stream_key)[start:stop],
                         dtype=np.float64)
        self.cache[key] = self._scaled_variance(seg)
        self.source[label] = "stream cache"
        return self.cache[key]

    def vector_contaminated(self,
                            stream_key: str,
                            start: int,
                            stop: int,
                            injected: Sequence[int],
                            donor_rows: Any,
                            digest: str | None = None) -> tuple[np.ndarray,
                                                                bool]:
        """
        The same vector over a stretch with an injected block of rows.

        The training-contamination block (T4-E) trains on a stretch into
        which a block of attack rows was copied, so its floor has to be
        the floor of the contaminated data. The block is rebuilt in
        memory exactly as the runner did it, ``x[start:start + B] =
        x[donor]``, and the rebuilt block is hashed so that a floor is
        never computed over a stretch that does not reproduce the run.

        This vector is not cached: it belongs to one job only.

        :param stream_key: Sub-directory of cache_root with ``x.npy``.
        :type stream_key: str
        :param start: First row of the training stretch.
        :type start: int
        :param stop: One past the last row of that stretch.
        :type stop: int
        :param injected: Pair (first, last + 1) of absolute row indices
            of the injected block, as ``regions
            .t4e_training_contamination.injected_rows``; it must lie
            inside [start, stop).
        :type injected: Sequence[int]
        :param donor_rows: Absolute indices of the donor rows, the
            contents of ``t4e_donor_rows.npy``; as many as the block.
        :type donor_rows: Any
        :param digest: Expected sha256 of the injected block, as
            ``x_block_sha256_after``, or None to skip the check.
        :type digest: str | None
        :return: The per-column variance of the scaled contaminated
            stretch, and whether the rebuilt block reproduces digest
            (False when digest is None).
        :rtype: tuple[np.ndarray, bool]
        :raises TypeError: If stream_key is not a string.
        :raises ValueError: If stop is not greater than start, or if the
            injected block is not a pair inside [start, stop).
        :raises FileNotFoundError: If the stream cache has no ``x.npy``.
        """
        block = np.empty(0, dtype=np.float64)
        if not isinstance(stream_key, str):
            self._error("[ConstantFloor.vector_contaminated] Input "
                        "error: the parameter stream_key must be a "
                        f"string, not {type(stream_key).__name__}.",
                        TypeError)
        if int(stop) <= int(start):
            self._error("[ConstantFloor.vector_contaminated] Input "
                        "error: the parameter stop must be greater than "
                        f"start ({start}), not {stop}.")
        if len(injected) != 2 or not (int(start) <= int(injected[0])
                                      <= int(injected[1]) <= int(stop)):
            self._error("[ConstantFloor.vector_contaminated] Input "
                        "error: the parameter injected must be a pair "
                        f"inside [{start}, {stop}], not "
                        f"{list(injected)}.")
        rows = self._rows(stream_key)
        seg = np.array(rows[start:stop], dtype=np.float64, copy=True)
        first = int(injected[0]) - int(start)
        last = int(injected[1]) - int(start)
        seg[first:last] = np.asarray(rows[donor_rows], dtype=np.float64)
        block = np.ascontiguousarray(seg[first:last])
        return (self._scaled_variance(seg),
                hashlib.sha256(block.tobytes()).hexdigest() == digest)

    def group_floor(self, vector: np.ndarray, columns: Any) -> float:
        """
        Floor of one feature group: the mean over its columns.

        :param vector: Per-column variance of the scaled stretch, as
            ``vector`` returns it.
        :type vector: np.ndarray
        :param columns: Column indices of the group, as one entry of
            ``fm_stats.json['feature_map']``; non-empty.
        :type columns: Any
        :return: The floor of the group, in MSE units of the scaled
            data; 0.0 when every column of the group is constant over
            the stretch.
        :rtype: float
        :raises ValueError: If the group is empty or holds an index
            outside the vector.
        """
        cols = list(columns)
        if not cols:
            self._error("[ConstantFloor.group_floor] Input error: the "
                        "parameter columns must hold at least one "
                        "column index, not an empty group.")
        if max(cols) >= len(vector) or min(cols) < -len(vector):
            self._error("[ConstantFloor.group_floor] Input error: every "
                        "index in the parameter columns must address one "
                        f"of the {len(vector)} features, and "
                        f"{[c for c in cols if c >= len(vector)]} does "
                        "not.")
        return float(np.mean(vector[cols]))

    def _rows(self, stream_key: str) -> np.ndarray:
        """
        Memory-mapped view of the preprocessed stream of one dataset.

        :param stream_key: Sub-directory of cache_root with ``x.npy``.
        :type stream_key: str
        :return: Read-only memory map of shape (n_rows, n_features).
        :rtype: np.ndarray
        :raises FileNotFoundError: If the stream cache has no ``x.npy``.
        """
        path = self.cache_root / stream_key / "x.npy"
        if not path.exists():
            self._error("[ConstantFloor._rows] Input error: the stream "
                        f"cache must hold x.npy, and it does not: "
                        f"{path}.", FileNotFoundError)
        return np.load(path, mmap_mode="r")

    def _scaled_variance(self, segment: np.ndarray) -> np.ndarray:
        """
        Variance of every column after the min-max scaling of train().

        :param segment: Raw rows of the training stretch, shape
            (n_rows, n_features), in float64.
        :type segment: np.ndarray
        :return: Variance of every column of the scaled stretch, shape
            (n_features,).
        :rtype: np.ndarray
        """
        spread = segment.max(axis=0) - segment.min(axis=0) + self.epsilon
        return segment.var(axis=0) / spread ** 2

    def __repr__(self) -> str:
        """
        Show the cache root and how many stretches are held in memory.

        :return: Text like "ConstantFloor(cache_root='data/cache/
            streams', stored=11, cached=9)".
        :rtype: str
        """
        return (f"ConstantFloor(cache_root={str(self.cache_root)!r}, "
                f"stored={len(self.stored.get('stretches', dict()))}, "
                f"cached={len(self.cache)})")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
