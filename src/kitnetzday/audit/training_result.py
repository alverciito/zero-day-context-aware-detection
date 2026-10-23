# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         src/kitnetzday/audit/training_result.py                   #
#   Version:      1.0.1                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Result file of the training audit: the method block and   #
#                 the structure the manuscript points at.                   #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
from collections.abc import Mapping
from typing import Any, NoReturn

from kitnetzday.audit.training_audit import TrainingAudit
from kitnetzday.audit.training_summary import TrainingSummary

# Method block written verbatim into the result file: it states what was
# asked, where the loss comes from, how the floor is defined and which
# rows it is computed over, so that the file can be read on its own. The
# calling script fills in "script", "interpreter" and "floor_source".
METHOD: dict[str, Any] = {
    "question": ("which of the executed cells produced a trained "
                 "model"),
    "loss_source": (
        "<results_root>/<job>/train_log.csv, column loss_epoch_1 (mean "
        "over epoch 1, not the final loss). loss_source is "
        "'model.history' (mlp) or \"model.metrics['loss']\" (windowed "
        "variants)."),
    "error_convention": (
        "MSE is compared with MSE. The compiled loss is "
        "tf.keras.losses.MeanSquaredError in the six variants trained "
        "by gradient descent (frozen/models/networks/multilayer.py, "
        "conv1d.py, conv2d.py, lstm.py, mha.py and "
        "frozen/runner/networks_ctx.py), so loss_epoch_1 is an MSE even "
        "where the variant reports the per-record error as RMSE "
        "(conv1d, conv2d, lstm, mha and mlp_deep_ctx store RMSE in "
        "ae_errors.npy; mlp and mlp_ctx store MSE). The RMSE/MSE "
        "convention affects ae_errors.npy and the fused score, NOT the "
        "loss audited here."),
    "floor": (
        "MSE of the constant predictor = per-feature mean of the "
        "min-max scaled training stretch. floor(g) = mean_{f in g} "
        "Var(x_scaled[:, f]), with x_scaled = (x - min)/(max - min + "
        "1e-16) and min/max taken over the SAME stretch, as train() of "
        "frozen/models/networks/*.py does. Evaluated as Var(x_raw[:, "
        "f]) / (max_f - min_f + 1e-16)^2 so that the scaled matrix is "
        "never materialised."),
    "row_range": (
        "regions.ae_train of each config.json (never assumed to be "
        "[50000, 200000)): it is [50000, 200000) in the 8 KNAD/CIC "
        "streams and [50000, 72264) in ACI-IoT-2023, where the "
        "corrected training cut is the whole day of 31 October (feature "
        "map 50,000 + autoencoders 22,264). Using [50000, 200000) on "
        "ACI would have put evaluation rows into the floor."),
    "targets": (
        "The training targets are exactly the rows of the stretch in "
        "the three windowing schemes: create_windowed_data (TSR) maps y "
        "= window[-1] = x[j]; create_windowed_data_ar (AR) maps y = "
        "full_x[j+L] = x[j] (only the INPUT changes, not the target); "
        "and multilayer.py trains fit(x, x). The floor is therefore "
        "identical for AR and TSR."),
    "feature_groups": (
        "fm_stats.json['feature_map'] of each job (it depends on the "
        "dataset, k, the grouping and the partition seed)."),
    "floor_cache": (
        "one read per (stream_key, ae_train): the per-column vector of "
        "scaled variances is cached and the floor of each group is the "
        "mean of its columns. It does not depend on the variant."),
    "t4e": (
        "The 4 gradient-trained jobs of T4-E are audited with the "
        "CONTAMINATED stretch rebuilt in memory (x[start:start+B] = "
        "x[donor], frozen/runner/run_job.py and instrument.py), "
        "verified against "
        "regions.t4e_training_contamination.x_block_sha256_after."),
    "cuts": {"trained": "ratio < 0.90",
             "at the floor": "0.90 <= ratio < 1.15",
             "worse than the floor": "ratio >= 1.15"},
    "excluded_variants": ["iforest", "original", "stat"],
    "script": "bin/audit_training.py",
    "floor_code": (
        "x = np.load(CACHE/<stream_key>/x.npy, mmap_mode='r'); seg = "
        "np.asarray(x[a:b], np.float64); mn, mx = seg.min(0), "
        "seg.max(0); v = seg.var(0) / (mx - mn + 1e-16)**2; floor[g] = "
        "float(np.mean(v[feature_map[g]]))  # checked against the "
        "brute-force mean((xs - xs.mean(0))**2) over the scaled "
        "matrix, |diff| < 1.4e-17"),
    "floor_source": dict(),
    "interpreter": "python",
    "no_gpu": True,
    "no_retraining": True}
# Key of the aggregate that does not fit in one line below.
_NEVER_NONDEGENERATE: str = ("variants_never_trained_on_any_dataset_"
                             "nondegenerate_groups")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                AUXILIARY                                  #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def _fail(message: str,
          logger: logging.Logger | None = None,
          exception: type[Exception] = ValueError) -> NoReturn:
    """
    Log an error, if there is a logger, and raise it with the same text.

    This is the plain-function counterpart of ``Loggable._error``: this
    module is built around a function, so it cannot inherit from
    ``Loggable``.

    :param message: Text in the house format, used for both the log
        record and the exception.
    :type message: str
    :param logger: Logger created by the calling script, or None to stay
        silent.
    :type logger: logging.Logger | None
    :param exception: Exception class to raise.
    :type exception: type[Exception]
    :raises Exception: Always, of the class given in exception.
    """
    if logger is not None:
        logger.error(message)
    raise exception(message)


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                              MAIN FUNCTION                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def build_result(audit: TrainingAudit,
                 summary: TrainingSummary,
                 method: Mapping[str, Any] = METHOD,
                 logger: logging.Logger | None = None) -> dict[str, Any]:
    """
    Assemble the result file of the training audit.

    The audit leaves one JSON file that the manuscript and its
    supplementary material point at, and it has to be
    readable on its own: it opens with the method block, which states
    the question, the source of the loss, the definition of the floor
    and the rows it is computed over; then the totals; then the
    aggregates; and it closes with the warnings, the jobs that could not
    be audited and the full per-job, per-autoencoder table, so that
    every number of the report can be traced to one autoencoder of one
    job.

    The list of excluded variants is taken from the audit instead of
    from the method block, so that the file cannot claim to have skipped
    something else than it did.

    Note: the keys ``script``, ``interpreter`` and ``floor_source`` of
    the method block hold defaults. The calling script replaces them
    with the script that ran, the version of its interpreter and, for
    every stretch, whether its floor came from the stream cache or from
    the stored vectors.

    :param audit: Training audit whose ``run`` has been called; its
        jobs, warnings and skipped list go into the result as they are.
    :type audit: TrainingAudit
    :param summary: Aggregates of that audit, with ``run`` already
        called on them.
    :type summary: TrainingSummary
    :param method: Method block to write, verbatim.
    :type method: Mapping[str, Any]
    :param logger: Logger created by the calling script, or None to stay
        silent.
    :type logger: logging.Logger | None
    :return: The structure to serialize, with the keys "method",
        "totals", "aggregates", "warnings", "skipped" and "jobs".
    :rtype: dict[str, Any]
    :raises TypeError: If audit is not a TrainingAudit, if summary is
        not a TrainingSummary, or if method is not a mapping.
    :raises ValueError: If ``run`` was not called on the summary, or if
        it does not belong to this audit.
    """
    notes: dict[str, Any] = dict()
    if not isinstance(audit, TrainingAudit):
        _fail("[build_result] Input error: the parameter audit must be "
              f"a TrainingAudit, not {type(audit).__name__}.", logger,
              TypeError)
    if not isinstance(summary, TrainingSummary):
        _fail("[build_result] Input error: the parameter summary must "
              f"be a TrainingSummary, not {type(summary).__name__}.",
              logger, TypeError)
    if not isinstance(method, Mapping):
        _fail("[build_result] Input error: the parameter method must be "
              f"a mapping, not {type(method).__name__}.", logger,
              TypeError)
    if summary.audit is not audit:
        _fail("[build_result] Input error: the parameter summary must "
              "aggregate the parameter audit, and it aggregates another "
              "one.", logger)
    if not summary.totals:
        _fail("[build_result] Input error: the parameter summary must "
              "have been built with its run method before the result "
              "can be assembled, and its totals are empty.", logger)
    notes = dict(method)
    notes["excluded_variants"] = sorted(audit.skip_variants)
    return {
        "method": notes,
        "totals": summary.totals,
        "aggregates": {
            "by_variant": summary.by_variant,
            "by_dataset": summary.by_dataset,
            "median_ratio_variant_x_dataset": summary.matrix,
            "variants_never_trained_on_any_dataset": summary.never,
            _NEVER_NONDEGENERATE: summary.never_nondegenerate,
            "degeneracy": summary.degeneracy,
            "windowed_variants_below_the_floor": summary.trained,
            "contradictions": summary.contradictions},
        "warnings": audit.warnings,
        "skipped": audit.skipped,
        "jobs": audit.jobs}


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
