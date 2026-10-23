# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         src/kitnetzday/audit/training_summary.py                  #
#   Version:      1.0.1                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Aggregates of the training audit by variant, by dataset   #
#                 and by degeneracy of the feature group.                   #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from kitnetzday.audit.training_audit import (
                                          VERDICT_TRAINED,
                                          VERDICTS,
                                          WINDOWED_VARIANTS,
                                          TrainingAudit,
)
from kitnetzday.utils.loggable import Loggable

# Nine datasets of the study: the denominator of the "how many datasets
# did this variant ever train on" fraction. The key of that fraction
# names the nine, so it is reported as n/9 whatever is passed.
N_DATASETS: int = 9
# Variants that were expected to train, and the label each one gets when
# it does not. Output strings of the result file.
CONTRADICTION_LABELS: dict[str, str] = {
    "lstm": "lstm not trained", "mlp": "per-record mlp not trained"}
# Label of the opposite surprise: a windowed variant below the floor.
CONTRADICTION_WINDOWED: str = "windowed variant below the floor"


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                AUXILIARY                                  #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def median_finite(values: Iterable[Any] = tuple()) -> float | None:
    """
    Median of the values that are there, ignoring None and non-finite.

    The ratios of the audit are missing whenever the loss or the floor
    is missing, and the median of a group has to describe the ratios it
    does have instead of becoming NaN because of one hole.

    :param values: Ratios of a group of autoencoders, which may hold
        None and non-finite entries.
    :type values: Iterable[Any]
    :return: The median of the finite values, or None when there is
        none.
    :rtype: float | None
    """
    kept = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.median(kept)) if kept else None


def _counts(records: Sequence[dict[str, Any]]) -> dict[str, int]:
    """
    How many autoencoders of a group got each of the four verdicts.

    :param records: Flat autoencoder records, as ``TrainingAudit.flat``
        holds them.
    :type records: Sequence[dict[str, Any]]
    :return: Verdict -> count, with the four verdicts always present and
        in their canonical order.
    :rtype: dict[str, int]
    """
    return {v: sum(1 for r in records if r["verdict"] == v)
            for v in VERDICTS}


def _summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """
    The three figures every group of the report opens with.

    :param records: Flat autoencoder records of one group.
    :type records: Sequence[dict[str, Any]]
    :return: The keys "n_autoencoders", "counts" and "median_ratio", in
        that order.
    :rtype: dict[str, Any]
    """
    return {"n_autoencoders": len(records), "counts": _counts(records),
            "median_ratio": median_finite(r["ratio"] for r in records)}


def _trained_datasets(records: Sequence[dict[str, Any]]) -> list[str]:
    """
    Datasets on which a group has at least one trained autoencoder.

    :param records: Flat autoencoder records of one group.
    :type records: Sequence[dict[str, Any]]
    :return: Dataset names in alphabetical order; empty when the group
        never got below its floor.
    :rtype: list[str]
    """
    return sorted({r["dataset"] for r in records
                   if r["verdict"] == VERDICT_TRAINED})


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                MAIN CLASS                                 #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class TrainingSummary(Loggable):
    """
    Reads the per-autoencoder verdicts as a statement about the study.

    One autoencoder below or above its floor says little; what the paper
    reports is how the verdicts fall across the twelve variants and the
    nine datasets, which is what this class tabulates from a
    :class:`kitnetzday.audit.training_audit.TrainingAudit` that has
    already run.

    It produces six views:

    - ``by_variant``: counts, median, extremes and the datasets on which
      each variant ever got below its floor;
    - ``by_dataset``: the same seen from the dataset side;
    - ``matrix``: median ratio per (variant, dataset) cell;
    - ``by_variant_nondegenerate``: the variant view restricted to the
      feature groups whose floor is not degenerate;
    - ``degeneracy``: how many groups are degenerate and where the
      "worse than the floor" verdicts concentrate;
    - ``contradictions``: the cases that go against the expected
      pattern, and ``trained``, the windowed autoencoders that did get
      below their floor.

    A degenerate group is one whose floor falls below the cut: after the
    min-max scaling of the training stretch its columns are nearly
    constant, because outliers stretch the range, so the constant
    predictor already makes almost no error and the ratio explodes
    without saying anything about learning. The main tables keep those
    groups, the secondary view drops them, and the paper reports both.

    Note: the fraction of datasets with a trained autoencoder is keyed
    as ``fraction_of_9_datasets_with_a_trained_ae``, with the nine in
    the name; passing another n_datasets changes the value and not the
    key, as in the audited script.

    :param audit: Training audit whose ``run`` has already been called;
        an audit with no autoencoder gives empty views.
    :type audit: TrainingAudit
    :param n_datasets: Number of datasets of the study, the denominator
        of the coverage fraction; strictly positive.
    :type n_datasets: int
    :param windowed_variants: Variants whose input is a window, the ones
        expected to stay at the floor.
    :type windowed_variants: Sequence[str]
    :param logger: Logger created by the calling script, or None to stay
        silent.
    :type logger: logging.Logger | None
    :raises TypeError: If audit is not a TrainingAudit, or if logger is
        not a logging.Logger or None.
    :raises ValueError: If n_datasets is not strictly positive.
    """

    def __init__(self,
                 audit: TrainingAudit,
                 n_datasets: int = N_DATASETS,
                 windowed_variants: Sequence[str] = WINDOWED_VARIANTS,
                 logger: logging.Logger | None = None) -> None:
        """
        Keep the audit and the study size; ``run`` tabulates.

        :param audit: Training audit whose ``run`` has been called.
        :type audit: TrainingAudit
        :param n_datasets: Number of datasets of the study; positive.
        :type n_datasets: int
        :param windowed_variants: Variants whose input is a window.
        :type windowed_variants: Sequence[str]
        :param logger: Logger created by the calling script, or None.
        :type logger: logging.Logger | None
        :raises TypeError: If audit is not a TrainingAudit, or if logger
            is not a logging.Logger or None.
        :raises ValueError: If n_datasets is not strictly positive.
        """
        super().__init__(logger=logger)
        if not isinstance(audit, TrainingAudit):
            self._error("[TrainingSummary.__init__] Input error: the "
                        "parameter audit must be a TrainingAudit, not "
                        f"{type(audit).__name__}.", TypeError)
        if isinstance(n_datasets, bool) or not isinstance(n_datasets,
                                                          int):
            self._error("[TrainingSummary.__init__] Input error: the "
                        "parameter n_datasets must be an integer, not "
                        f"{type(n_datasets).__name__}.", TypeError)
        if n_datasets <= 0:
            self._error("[TrainingSummary.__init__] Input error: the "
                        "parameter n_datasets must be strictly positive "
                        f"(>0), not {n_datasets}.")
        self.audit = audit
        self.n_datasets = int(n_datasets)
        self.windowed_variants = tuple(str(v) for v in windowed_variants)
        self.variants: list[str] = list()
        self.datasets: list[str] = list()
        self.overall: dict[str, int] = dict()
        self.by_variant: dict[str, dict[str, Any]] = dict()
        self.by_dataset: dict[str, dict[str, Any]] = dict()
        self.matrix: dict[str, dict[str, Any]] = dict()
        self.by_variant_nondegenerate: dict[str, dict[str, Any]] = dict()
        self.degeneracy: dict[str, Any] = dict()
        self.never: list[str] = list()
        self.never_nondegenerate: list[str] = list()
        self.trained: list[dict[str, Any]] = list()
        self.contradictions: list[dict[str, Any]] = list()
        self.totals: dict[str, Any] = dict()

    def run(self) -> None:
        """
        Build every view from the flat records of the audit.

        The variants and the datasets are taken from the records that
        are there, in alphabetical order, so a partially executed grid
        tabulates without holes. Calling it again rebuilds every view.
        """
        flat = self.audit.flat
        self.variants = sorted({r["variant"] for r in flat})
        self.datasets = sorted({r["dataset"] for r in flat})
        self.overall = _counts(flat)
        self.by_variant = self._by_variant()
        self.by_dataset = self._by_dataset()
        self.matrix = self._matrix()
        self.by_variant_nondegenerate = self._nondegenerate()
        self.degeneracy = self._degeneracy()
        self.never = [v for v in self.variants if not self.by_variant[v][
            "datasets_with_at_least_one_trained_ae"]]
        self.never_nondegenerate = [
            v for v in self.variants
            if not self.by_variant_nondegenerate[v][
                "datasets_with_at_least_one_trained_ae"]]
        self.trained = self._trained()
        self.contradictions = self._contradictions()
        self.totals = {
            "job_dirs_with_DONE_and_train_log": (len(self.audit.jobs)
                                                 + len(
                                                     self.audit.skipped)),
            "jobs_audited": len(self.audit.jobs),
            "jobs_skipped": len(self.audit.skipped),
            "autoencoders_audited": len(flat),
            "verdict_counts": self.overall}

    def records(self,
                variant: str | None = None,
                dataset: str | None = None,
                nondegenerate: bool = False) -> list[dict[str, Any]]:
        """
        Flat autoencoder records of one cell, one row or one column.

        :param variant: Variant to keep, or None for every variant.
        :type variant: str | None
        :param dataset: Dataset to keep, or None for every dataset.
        :type dataset: str | None
        :param nondegenerate: True to drop the autoencoders whose
            feature group has a degenerate floor.
        :type nondegenerate: bool
        :return: The matching records, in the order of the audit.
        :rtype: list[dict[str, Any]]
        """
        return [r for r in self.audit.flat
                if (variant is None or r["variant"] == variant)
                and (dataset is None or r["dataset"] == dataset)
                and not (nondegenerate and r["floor_degenerate"])]

    def _by_variant(self) -> dict[str, dict[str, Any]]:
        """
        One row per variant: counts, median, extremes and coverage.

        The extremes keep every ratio that is not None, non-finite ones
        included, as in the audited script; the median drops those.

        :return: Variant -> the keys "n_jobs", "n_autoencoders",
            "counts", "median_ratio", "min_ratio", "max_ratio",
            "datasets_run", "datasets_with_at_least_one_trained_ae",
            "fraction_of_9_datasets_with_a_trained_ae" and
            "fraction_of_datasets_run_with_a_trained_ae".
        :rtype: dict[str, dict[str, Any]]
        """
        out: dict[str, dict[str, Any]] = dict()
        for name in self.variants:
            sub = self.records(variant=name)
            known = [r["ratio"] for r in sub if r["ratio"] is not None]
            runs = sorted({r["dataset"] for r in sub})
            trained = _trained_datasets(sub)
            out[name] = {
                "n_jobs": len({j["job_id"] for j in self.audit.jobs
                               if j["variant"] == name}),
                **_summary(sub),
                "min_ratio": (min(known) if known else None),
                "max_ratio": (max(known) if known else None),
                "datasets_run": runs,
                "datasets_with_at_least_one_trained_ae": trained,
                "fraction_of_9_datasets_with_a_trained_ae": round(
                    len(trained) / float(self.n_datasets), 4),
                "fraction_of_datasets_run_with_a_trained_ae": (
                    round(len(trained) / len(runs), 4) if runs
                    else None)}
        return out

    def _by_dataset(self) -> dict[str, dict[str, Any]]:
        """
        One row per dataset: counts, median and which variants trained.

        :return: Dataset -> the keys "n_jobs", "n_autoencoders",
            "counts", "median_ratio", "variants_run" and
            "variants_with_a_trained_ae".
        :rtype: dict[str, dict[str, Any]]
        """
        out: dict[str, dict[str, Any]] = dict()
        for name in self.datasets:
            sub = self.records(dataset=name)
            out[name] = {
                "n_jobs": len({j["job_id"] for j in self.audit.jobs
                               if j["dataset"] == name}),
                **_summary(sub),
                "variants_run": sorted({r["variant"] for r in sub}),
                "variants_with_a_trained_ae": sorted(
                    {r["variant"] for r in sub
                     if r["verdict"] == VERDICT_TRAINED})}
        return out

    def _matrix(self) -> dict[str, dict[str, Any]]:
        """
        Median ratio of every (variant, dataset) cell that was run.

        A cell with no autoencoder is left out instead of being written
        as null, so that the table shows what was executed.

        :return: Variant -> dataset -> the keys "median_ratio", "n" and
            "n_trained".
        :rtype: dict[str, dict[str, Any]]
        """
        out: dict[str, dict[str, Any]] = dict()
        for name in self.variants:
            out[name] = dict()
            for dataset in self.datasets:
                sub = self.records(variant=name, dataset=dataset)
                if sub:
                    out[name][dataset] = {
                        "median_ratio": median_finite(r["ratio"]
                                                      for r in sub),
                        "n": len(sub),
                        "n_trained": sum(
                            1 for r in sub
                            if r["verdict"] == VERDICT_TRAINED)}
        return out

    def _nondegenerate(self) -> dict[str, dict[str, Any]]:
        """
        The variant view again, over the groups with a usable floor.

        :return: Variant -> the keys "n_autoencoders", "counts",
            "median_ratio" and
            "datasets_with_at_least_one_trained_ae".
        :rtype: dict[str, dict[str, Any]]
        """
        out: dict[str, dict[str, Any]] = dict()
        for name in self.variants:
            sub = self.records(variant=name, nondegenerate=True)
            out[name] = {
                **_summary(sub),
                "datasets_with_at_least_one_trained_ae":
                    _trained_datasets(sub)}
        return out

    def _degeneracy(self) -> dict[str, Any]:
        """
        How many feature groups have a floor too small to divide by.

        :return: The keys "cut", "definition", "n_degenerate",
            "n_total", "verdicts_in_degenerate_groups",
            "verdicts_in_nondegenerate_groups" and
            "by_variant_nondegenerate".
        :rtype: dict[str, Any]
        """
        cut = self.audit.degenerate_cut
        degenerate = [r for r in self.audit.flat
                      if r["floor_degenerate"]]
        return {
            "cut": cut,
            "definition": (
                f"group with floor < {cut:g}: after the min-max scaling "
                "of the training stretch its columns are almost "
                "constant (outliers stretch the range), so the constant "
                "predictor makes almost no error and the ratio blows up "
                "without saying anything about learning."),
            "n_degenerate": len(degenerate),
            "n_total": len(self.audit.flat),
            "verdicts_in_degenerate_groups": _counts(degenerate),
            "verdicts_in_nondegenerate_groups": _counts(
                self.records(nondegenerate=True)),
            "by_variant_nondegenerate": self.by_variant_nondegenerate}

    def _trained(self) -> list[dict[str, Any]]:
        """
        The windowed autoencoders that did get below their floor.

        These are the exceptions to the pattern of the paper, listed
        from the smallest ratio upwards so that the marginal ones, just
        under the cut, are told apart from a real one.

        :return: One record per exception, with the keys "variant",
            "dataset", "job_id", "ae_index", "ratio", "floor_mse",
            "loss_mse", "n_features_group" and "ar".
        :rtype: list[dict[str, Any]]
        """
        rows = [{"variant": r["variant"], "dataset": r["dataset"],
                 "job_id": r["job_id"], "ae_index": r["ae_index"],
                 "ratio": r["ratio"], "floor_mse": r["floor_mse"],
                 "loss_mse": r["loss_mse"],
                 "n_features_group": r["n_features_group"],
                 "ar": r["ar"]}
                for r in self.audit.flat
                if r["verdict"] == VERDICT_TRAINED
                and r["variant"] in self.windowed_variants]
        return sorted(rows, key=lambda r: r["ratio"])

    def _contradictions(self) -> list[dict[str, Any]]:
        """
        Autoencoders that fall on the wrong side of the expectation.

        Two surprises are reported: a variant that was expected to
        train and did not, and a windowed variant that did train. An
        autoencoder can only be reported once, as in the audited
        script, because the two conditions are checked in order.

        :return: One record per surprise, with the keys "kind",
            "job_id", "ae_index", "ratio" and "verdict".
        :rtype: list[dict[str, Any]]
        """
        out: list[dict[str, Any]] = list()
        label = ""
        for r in self.audit.flat:
            label = ""
            if r["variant"] in CONTRADICTION_LABELS \
                    and r["verdict"] != VERDICT_TRAINED:
                label = CONTRADICTION_LABELS[r["variant"]]
            elif r["variant"] in self.windowed_variants \
                    and r["verdict"] == VERDICT_TRAINED:
                label = CONTRADICTION_WINDOWED
            if label:
                out.append({"kind": label, "job_id": r["job_id"],
                            "ae_index": r["ae_index"],
                            "ratio": r["ratio"],
                            "verdict": r["verdict"]})
        return out

    def __repr__(self) -> str:
        """
        Show the size of the grid and the global verdict counts.

        :return: Text like "TrainingSummary(variants=9, datasets=9,
            autoencoders=1564, trained=18)".
        :rtype: str
        """
        return (f"TrainingSummary(variants={len(self.variants)}, "
                f"datasets={len(self.datasets)}, "
                f"autoencoders={len(self.audit.flat)}, "
                f"trained={self.overall.get(VERDICT_TRAINED, 0)})")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
