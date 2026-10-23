# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         src/kitnetzday/audit/training_audit.py                    #
#   Version:      1.0.0                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-07                                                #
#   Description:  Label-free training audit: the recorded loss of every     #
#                 autoencoder against its constant-predictor floor.         #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import csv
import json
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from kitnetzday.audit.constant_floor import ConstantFloor
from kitnetzday.utils.loggable import Loggable

# Variants that write "loss_epoch_1 = NA": the numpy denoising
# autoencoder, the statistical layer and the isolation forest. They are
# reported as skipped, never as "not trained".
SKIP_VARIANTS: frozenset[str] = frozenset(("original", "stat", "iforest"))
# Verdict cuts on ratio = loss / floor (src/config.json, analysis_plan:
# floor_trained_below and floor_at_floor_above).
TRAINED_BELOW: float = 0.90
AT_FLOOR_BELOW: float = 1.15
# A feature group whose floor falls below this is reported as degenerate:
# the ratio stops being informative there (secondary table of the report).
DEGENERATE_CUT: float = 1e-3
# The four verdicts, as the result file and the console summary write
# them. The archived audit of the revision used the Spanish labels
# "entrenado", "en el suelo", "peor que el suelo" and "no calculable";
# the cuts and the counts are the same.
VERDICT_TRAINED: str = "trained"
VERDICT_AT_FLOOR: str = "at the floor"
VERDICT_WORSE: str = "worse than the floor"
VERDICT_UNDEFINED: str = "not computable"
VERDICTS: list[str] = [VERDICT_TRAINED, VERDICT_AT_FLOOR, VERDICT_WORSE,
                       VERDICT_UNDEFINED]
# Windowed variants expected to stay at the floor: the ones whose input is
# a window of L records. Used to list the exceptions and to flag the cases
# that contradict the expected pattern.
WINDOWED_VARIANTS: tuple[str, ...] = ("conv1d", "conv2d", "mha", "mlp_ctx")
# Reasons written verbatim into the "skipped" list of the report.
SKIP_REASON_NO_LOSS: str = ("variant without a recorded gradient loss "
                            "(loss_epoch_1 = NA)")
SKIP_REASON_NO_STREAM: str = ("no regions.ae_train, or neither the stream "
                              "cache nor a stored floor vector")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                AUXILIARY                                  #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def _opt_int(row: Mapping[str, Any], key: str) -> int | None:
    """
    Read an integer column of ``train_log.csv`` that may be absent.

    The runner writes the literal "NA" for a field that does not apply to
    a variant, and an older runner leaves the column out altogether; both
    have to read as "not available" and not as zero.

    Note: a value that is present but does not parse as an integer (for
    example "7.0") raises, exactly as in the audited script, so that a
    malformed log is noticed instead of silently becoming None.

    :param row: One row of ``train_log.csv`` as ``csv.DictReader`` gives
        it, mapping column name -> text.
    :type row: Mapping[str, Any]
    :param key: Column to read, for example "n_visible".
    :type key: str
    :return: The integer, or None when the column is missing, empty or
        the literal "NA".
    :rtype: int | None
    :raises ValueError: If the text is present and is not an integer.
    """
    if row.get(key) in (None, "", "NA"):
        return None
    return int(row[key])


def _opt_float(raw: Any) -> float | None:
    """
    Read the recorded loss, which may be absent or unparsable.

    ``loss_epoch_1`` is the mean loss of the single training epoch. It is
    "NA" for the variants without gradient descent, and a run that never
    reached the training phase leaves it empty.

    Note: only the lower-case "nan" is treated as missing, as in the
    audited script. A "NaN" written by another tool parses to a float
    NaN, which gives a NaN ratio and the verdict "not computable", so
    nothing is ever reported as trained because of it.

    :param raw: Text of the column, or None when it is missing.
    :type raw: Any
    :return: The loss as a float, or None when it is missing, empty,
        "NA", "nan" or does not parse as a number.
    :rtype: float | None
    """
    if raw in (None, "", "NA", "nan"):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                MAIN CLASS                                 #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class TrainingAudit(Loggable):
    """
    Reconstruction error of each autoencoder relative to a constant
    predictor, computed without using any label.

    This is one of the three checks the article applies. A
    reconstruction error that looks small proves nothing on its own: an
    autoencoder that only learned the mean of every feature already
    reaches a definite error, which is the reference the ratio below is
    read against. For every autoencoder of every executed cell the audit
    therefore compares the loss the runner recorded in
    ``train_log.csv`` with the loss of the best constant predictor of
    that autoencoder's feature group (see
    :class:`kitnetzday.audit.constant_floor.ConstantFloor`) and reports

    ``ratio = loss_epoch_1 / floor(feature group)``,

    which gives four verdicts: "trained" below 0.90, "at the floor" in
    [0.90, 1.15), "worse than the floor" at or above 1.15, and "not
    computable" when there is no recorded loss or the floor is zero,
    which happens when every column of the group is constant over the
    training stretch.

    The comparison is MSE against MSE: the compiled Keras loss is the
    mean squared error in the six variants that do gradient descent, so
    the recorded loss is an MSE even where the per-record error of the
    pipeline is a RMSE. The training stretch is read from
    ``regions.ae_train`` of each job and never assumed, because it is
    shorter than the usual one in ACI-IoT-2023.

    No label is used anywhere, nothing is re-executed and no GPU is
    needed: the audit reads ``train_log.csv``, ``fm_stats.json`` and
    ``config.json`` of each job, plus the floor of its stream, which
    comes from the read-only stream caches when they are on disk and
    from the vectors stored in ``data/floor_vectors.json`` otherwise.

    Note: ``loss_epoch_1`` is the mean over the single epoch and not the
    final loss, so a model that converged late drags the untrained start
    of the epoch into the average and the ratios are biased upwards. A
    verdict of "trained" is therefore conservative, while "at the
    floor" does not tell a model that never moved from one that learned
    at the very end of the epoch.

    :param results_root: Directory with one sub-directory per job; a job
        is audited only when it holds both ``DONE`` and
        ``train_log.csv``.
    :type results_root: str | Path
    :param cache_root: Directory with one sub-directory per stream key,
        each holding the preprocessed ``x.npy``.
    :type cache_root: str | Path
    :param vectors_path: JSON file with the floor vectors computed in
        advance, used for the streams whose cache is not on disk, or
        None to work from the caches alone.
    :type vectors_path: str | Path | None
    :param skip_variants: Variants without a comparable recorded loss.
    :type skip_variants: Iterable[str]
    :param trained_below: Ratio below which an autoencoder counts as
        trained; strictly positive.
    :type trained_below: float
    :param at_floor_below: Ratio below which it counts as sitting at the
        floor; at least trained_below.
    :type at_floor_below: float
    :param degenerate_cut: Floor below which the feature group is
        flagged as degenerate, in MSE units of the scaled data; not
        negative.
    :type degenerate_cut: float
    :param logger: Logger created by the calling script, or None to stay
        silent.
    :type logger: logging.Logger | None
    :raises TypeError: If logger is not a logging.Logger or None.
    :raises FileNotFoundError: If results_root is not a directory.
    :raises ValueError: If trained_below is not strictly positive, if
        at_floor_below is below trained_below, or if degenerate_cut is
        negative.
    """

    def __init__(self,
                 results_root: str | Path,
                 cache_root: str | Path,
                 vectors_path: str | Path | None = None,
                 skip_variants: Iterable[str] = SKIP_VARIANTS,
                 trained_below: float = TRAINED_BELOW,
                 at_floor_below: float = AT_FLOOR_BELOW,
                 degenerate_cut: float = DEGENERATE_CUT,
                 logger: logging.Logger | None = None) -> None:
        """
        Check the roots and the cuts, and start with nothing audited.

        :param results_root: Directory with one sub-directory per job.
        :type results_root: str | Path
        :param cache_root: Directory with one sub-directory per stream.
        :type cache_root: str | Path
        :param vectors_path: JSON file with stored floor vectors, or
            None.
        :type vectors_path: str | Path | None
        :param skip_variants: Variants without a comparable loss.
        :type skip_variants: Iterable[str]
        :param trained_below: Ratio below which an AE counts as trained.
        :type trained_below: float
        :param at_floor_below: Ratio below which it sits at the floor.
        :type at_floor_below: float
        :param degenerate_cut: Floor below which a group is degenerate.
        :type degenerate_cut: float
        :param logger: Logger created by the calling script, or None.
        :type logger: logging.Logger | None
        :raises TypeError: If logger is not a logging.Logger or None.
        :raises FileNotFoundError: If results_root is not a directory.
        :raises ValueError: If the cuts are not positive and ordered.
        """
        super().__init__(logger=logger)
        if not Path(results_root).is_dir():
            self._error("[TrainingAudit.__init__] Input error: the "
                        "parameter results_root must be an existing "
                        f"directory, and it is not: {results_root}.",
                        FileNotFoundError)
        if not float(trained_below) > 0.0:
            self._error("[TrainingAudit.__init__] Input error: the "
                        "parameter trained_below must be strictly "
                        f"positive (>0), not {trained_below}.")
        if float(at_floor_below) < float(trained_below):
            self._error("[TrainingAudit.__init__] Input error: the "
                        "parameter at_floor_below must be at least "
                        f"trained_below ({trained_below}), not "
                        f"{at_floor_below}.")
        if float(degenerate_cut) < 0.0:
            self._error("[TrainingAudit.__init__] Input error: the "
                        "parameter degenerate_cut must not be negative "
                        f"(>=0), not {degenerate_cut}.")
        self.results_root = Path(results_root)
        self.floor = ConstantFloor(cache_root, vectors_path=vectors_path,
                                   logger=logger)
        self.skip_variants = frozenset(str(v) for v in skip_variants)
        self.trained_below = float(trained_below)
        self.at_floor_below = float(at_floor_below)
        self.degenerate_cut = float(degenerate_cut)
        self.jobs: list[dict[str, Any]] = list()
        self.skipped: list[dict[str, Any]] = list()
        self.warnings: list[str] = list()
        self.flat: list[dict[str, Any]] = list()

    def verdict(self, ratio: float | None) -> str:
        """
        Turn a loss-to-floor ratio into one of the four verdicts.

        :param ratio: Recorded loss divided by the floor of the feature
            group, or None when either of the two is unavailable.
        :type ratio: float | None
        :return: "trained" for ratio < trained_below, "at the floor"
            for ratio in [trained_below, at_floor_below), "worse than
            the floor" at or above at_floor_below, and "not computable"
            for None and for a non-finite ratio.
        :rtype: str
        :raises TypeError: If ratio is neither a real number nor None.
        """
        if ratio is None:
            return VERDICT_UNDEFINED
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
            self._error("[TrainingAudit.verdict] Input error: the "
                        "parameter ratio must be a real number or None, "
                        f"not {type(ratio).__name__}.", TypeError)
        if not np.isfinite(ratio):
            return VERDICT_UNDEFINED
        if ratio < self.trained_below:
            return VERDICT_TRAINED
        if ratio < self.at_floor_below:
            return VERDICT_AT_FLOOR
        return VERDICT_WORSE

    def run(self) -> list[dict[str, Any]]:
        """
        Audit every job of the results tree, in alphabetical order.

        The job directories are visited in the alphabetical order of
        their names, which is the order their records keep in the
        report. Calling it again starts from scratch, so a tree that
        grew since the last call is audited whole.

        :return: One record per audited job, with its autoencoders under
            the key "autoencoders"; the same list that stays in the
            attribute jobs.
        :rtype: list[dict[str, Any]]
        """
        self.jobs = list()
        self.skipped = list()
        self.warnings = list()
        self.flat = list()
        for name in sorted(p.name for p in self.results_root.iterdir()):
            self._audit_job(name)
        self.flat = self._flatten()
        if self.logger is not None:
            self.logger.info("[TrainingAudit.run] Audited %d job(s), "
                             "skipped %d, %d autoencoder(s), %d "
                             "warning(s).", len(self.jobs),
                             len(self.skipped), len(self.flat),
                             len(self.warnings))
        return self.jobs

    def _audit_job(self, name: str) -> None:
        """
        Audit one job directory and file it under jobs or under skipped.

        A directory without ``DONE`` and ``train_log.csv`` is not a
        finished training run and is passed over in silence. A finished
        run that cannot be audited is appended to skipped together with
        its reason, so that the totals of the report always add up: a
        variant without a comparable loss, a job without its training
        region or its stream cache, and a job whose ``config.json`` or
        ``fm_stats.json`` cannot be read.

        Note: a ``train_log.csv`` that exists but cannot be read raises,
        as in the audited script; only the two JSON files fail softly.

        :param name: Name of the job directory inside results_root.
        :type name: str
        :raises OSError: If ``train_log.csv`` exists but cannot be read.
        """
        path = self.results_root / name
        base: dict[str, Any] = dict()
        rows: list[dict[str, Any]] = list()
        var: np.ndarray = np.empty(0, dtype=float)
        stored: tuple[np.ndarray, bool] | None = None
        matches = True
        if not path.is_dir():
            return
        if not ((path / "DONE").exists()
                and (path / "train_log.csv").exists()):
            return
        try:
            with open(path / "config.json", encoding="utf-8") as fh:
                cfg = json.load(fh)
        except Exception as ex:
            self.skipped.append({"job_id": name,
                                 "reason": f"unreadable config.json: {ex}"})
            return
        spec = cfg.get("spec", dict())
        variant = spec.get("variant")
        regions = cfg.get("regions", dict())
        ae_train = regions.get("ae_train")
        stream_key = (cfg.get("cache") or dict()).get("key")
        with open(path / "train_log.csv", newline="",
                  encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        base = {
            "job_id": name, "block": spec.get("block"),
            "dataset": spec.get("dataset"), "stream_key": stream_key,
            "pipeline": spec.get("pipeline"), "variant": variant,
            "ar": spec.get("ar"),
            "paradigm": (cfg.get("manifest_row")
                         or dict()).get("paradigm"),
            "window": spec.get("window"), "k": spec.get("k"),
            "grouping": spec.get("grouping"),
            "partition_seed": spec.get("partition_seed"),
            "contamination_pct": spec.get("contamination_pct"),
            "seed": spec.get("seed"), "rep": spec.get("rep"),
            "ae_train": ae_train, "n_train_log_rows": len(rows)}
        if variant in self.skip_variants:
            self.skipped.append(dict(base, reason=SKIP_REASON_NO_LOSS))
            return
        if not ae_train or not stream_key \
                or not self.floor.has_stream(str(stream_key)):
            self.skipped.append(dict(base, reason=SKIP_REASON_NO_STREAM))
            return
        try:
            with open(path / "fm_stats.json", encoding="utf-8") as fh:
                feature_map = json.load(fh)["feature_map"]
        except Exception as ex:
            self.skipped.append(dict(
                base, reason=f"fm_stats.json without feature_map: {ex}"))
            return
        t4e = regions.get("t4e_training_contamination")
        if t4e:
            if self.floor.has_cache(str(stream_key)):
                var, matches = self.floor.vector_contaminated(
                    str(stream_key), int(ae_train[0]), int(ae_train[1]),
                    t4e["injected_rows"],
                    np.load(path / "t4e_donor_rows.npy"),
                    t4e.get("x_block_sha256_after"))
            else:
                stored = self.floor.stored_contaminated(name)
                if stored is None:
                    self.skipped.append(dict(
                        base, reason=SKIP_REASON_NO_STREAM))
                    return
                var, matches = stored
            base["t4e_block_sha256_match"] = bool(matches)
            if not matches:
                self.warnings.append(
                    f"{name}: the rebuilt T4-E block does not reproduce "
                    "x_block_sha256_after")
        else:
            var = self.floor.vector(str(stream_key), int(ae_train[0]),
                                    int(ae_train[1]))
        base["autoencoders"] = self._autoencoders(name, rows,
                                                  feature_map, var)
        base["loss_source"] = (rows[0].get("loss_source") if rows
                               else None)
        self.jobs.append(base)

    def _autoencoders(self,
                      name: str,
                      rows: list[dict[str, Any]],
                      feature_map: list[Any],
                      var: np.ndarray) -> list[dict[str, Any]]:
        """
        Compare the loss of every autoencoder of one job with its floor.

        The rows are taken in the order ``train_log.csv`` wrote them, one
        per autoencoder of the feature map. A row whose ``ae_index`` is
        outside the feature map is reported as a warning and left out,
        because its floor cannot be built; a mismatch between the
        recorded ``n_visible`` and the size of the group is reported as a
        warning too, and the group of the feature map prevails.

        Note: the output autoencoder of the ensemble is not in
        ``train_log.csv`` and is therefore not audited.

        :param name: Job id, used in the warning messages.
        :type name: str
        :param rows: Rows of ``train_log.csv`` of this job.
        :type rows: list[dict[str, Any]]
        :param feature_map: Feature groups of ``fm_stats.json``, each one
            the list of column indices of one autoencoder.
        :type feature_map: list[Any]
        :param var: Per-column variance of the scaled training stretch,
            shape (n_features,).
        :type var: np.ndarray
        :return: One record per audited autoencoder, with the keys
            "ae_index", "class", "n_visible", "n_hidden", "train_steps",
            "loss_mse", "floor_mse", "ratio", "verdict",
            "n_features_group" and "floor_degenerate".
        :rtype: list[dict[str, Any]]
        """
        out: list[dict[str, Any]] = list()
        index = 0
        floor = 0.0
        ratio: float | None = None
        for row in rows:
            index = int(row["ae_index"])
            if index >= len(feature_map):
                self.warnings.append(
                    f"{name}: ae_index {index} outside the feature_map "
                    f"({len(feature_map)} groups)")
                continue
            cols = list(feature_map[index])
            n_visible = _opt_int(row, "n_visible")
            if n_visible is not None and n_visible != len(cols):
                self.warnings.append(
                    f"{name} ae{index}: n_visible={n_visible} != "
                    f"|group|={len(cols)}")
            floor = self.floor.group_floor(var, cols)
            loss = _opt_float(row.get("loss_epoch_1"))
            ratio = ((loss / floor)
                     if (loss is not None and floor > 0) else None)
            out.append({
                "ae_index": index, "class": row.get("class"),
                "n_visible": n_visible,
                "n_hidden": _opt_int(row, "n_hidden"),
                "train_steps": _opt_int(row, "train_steps"),
                "loss_mse": loss, "floor_mse": floor, "ratio": ratio,
                "verdict": self.verdict(ratio),
                "n_features_group": len(cols),
                "floor_degenerate": bool(floor < self.degenerate_cut)})
        return out

    def _flatten(self) -> list[dict[str, Any]]:
        """
        One flat record per audited autoencoder, for the aggregates.

        :return: One record per autoencoder, in the order the jobs were
            audited, with the keys "variant", "dataset", "verdict",
            "ratio", "job_id", "ae_index", "floor_degenerate",
            "floor_mse", "loss_mse", "n_features_group", "ar" and
            "block".
        :rtype: list[dict[str, Any]]
        """
        return [{"variant": job["variant"], "dataset": job["dataset"],
                 "verdict": ae["verdict"], "ratio": ae["ratio"],
                 "job_id": job["job_id"], "ae_index": ae["ae_index"],
                 "floor_degenerate": ae["floor_degenerate"],
                 "floor_mse": ae["floor_mse"],
                 "loss_mse": ae["loss_mse"],
                 "n_features_group": ae["n_features_group"],
                 "ar": job["ar"], "block": job["block"]}
                for job in self.jobs for ae in job["autoencoders"]]

    def __repr__(self) -> str:
        """
        Show the tree audited and how much of it came out.

        :return: Text like "TrainingAudit(results_root='results/
            revision', jobs=391, skipped=89, autoencoders=1564)".
        :rtype: str
        """
        return (f"TrainingAudit(results_root={str(self.results_root)!r}, "
                f"jobs={len(self.jobs)}, skipped={len(self.skipped)}, "
                f"autoencoders={len(self.flat)})")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
