# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         src/kitnetzday/audit/reconstruction.py                    #
#   Version:      1.0.2                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Exploratory correlations between reconstruction quality   #
#                 and detection, with within-group permutation p-values.    #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import json
import logging
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from kitnetzday.utils.loggable import Loggable

# Defaults of the analysis plan (section 5.2, permutation block). They
# also live in src/config.json under "analysis_plan"; the scripts in
# bin/ read that file and hand them to the constructor.
PERMUTATION_DRAWS: int = 100000
PERMUTATION_SEED: int = 20260917
# Only the variants that do leave the constant-predictor floor say
# anything about reconstruction quality.
CANDIDATE_VARIANTS: tuple[str, ...] = ("lstm", "mlp")
MIN_TRAINED_FRACTION: float = 0.5
# Where the two inputs are read from, relative to their roots.
METRICS_GLOB: str = "T1-*/metrics.json"
AUDIT_VARIANT_KEY: str = "fraction_of_9_datasets_with_a_trained_ae"
# Outcome name as written in the result -> position of the ROC-AUC in
# the row tuples, in the order the original script reported them.
OUTCOME_COLUMNS: tuple[tuple[str, int], ...] = (("raw", 1),
                                                ("fus_primary", 2))
# Fewest points for a within-stream rho; with fewer the sign is noise.
MIN_DATASET_POINTS: int = 3
# Slack of the permutation counter, so that a tie still counts.
TIE_TOLERANCE: float = 1e-15
# Decimals of the de-duplication key.
DEDUP_DECIMALS: int = 12
# Labels written verbatim into the result file. Both analyses are
# exploratory: they are outside the analysis plan and their p values
# are not adjusted.
RESULT_TAG: str = ("[EXPLORATORY] two correlations outside the plan, "
                   "unadjusted")
X_DEFINITION: str = ("median loss/floor ratio of the non-degenerate "
                     "autoencoders (lower = better)")
PERMUTATION_GROUPS: str = "dataset x variant"


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                MAIN CLASS                                 #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class ReconstructionCorrelation(Loggable):
    """
    Does reconstructing too well leave less error to detect?

    This is the exploratory pair of correlations on the cases with
    limited detection (supplementary material, Section 10).
    Per job, x is reconstruction quality: the median loss-over-floor
    ratio of the non-degenerate autoencoders, as the training audit
    wrote it, so a *smaller* x is a *better* reconstruction. The two
    outcomes are the ROC-AUC of the raw error (sensitivity) and the
    ROC-AUC of the fused detector, the primary metric of the plan.

    Only the variants that really leave the floor take part: for one
    that never trains the ratio says nothing about reconstruction.
    The p-value permutes y *within* each (dataset, variant) group,
    the unit where the comparison means anything, since permuting
    across datasets would mix streams of different prevalence and
    difficulty.

    Note: the keys of the result and the summary lines are the ones
    the supplementary material quotes.

    :param audit_path: Path of ``training_audit.json``, the audit that
        holds the per-autoencoder ratios.
    :type audit_path: str | Path
    :param results_root: Directory with one sub-directory per job.
    :type results_root: str | Path
    :param draws: Permutations per correlation; 100000 in the paper,
        strictly positive.
    :type draws: int
    :param seed: Seed of the permutation generator; 20260917 in the
        paper, the one the plan uses everywhere.
    :type seed: int
    :param candidate_variants: Variants allowed before the
        trained-fraction filter.
    :type candidate_variants: Sequence[str]
    :param min_trained_fraction: Smallest fraction of the nine
        datasets with a trained autoencoder; 0.5, in [0, 1].
    :type min_trained_fraction: float
    :param metrics_glob: Pattern of the ``metrics.json`` files.
    :type metrics_glob: str
    :param logger: Logger of the calling script, or None.
    :type logger: logging.Logger | None
    :raises TypeError: If logger or candidate_variants are of the
        wrong type.
    :raises ValueError: If draws, seed, min_trained_fraction or
        metrics_glob are out of range.
    """

    def __init__(self,
                 audit_path: str | Path,
                 results_root: str | Path,
                 draws: int = PERMUTATION_DRAWS,
                 seed: int = PERMUTATION_SEED,
                 candidate_variants: Sequence[str] = CANDIDATE_VARIANTS,
                 min_trained_fraction: float = MIN_TRAINED_FRACTION,
                 metrics_glob: str = METRICS_GLOB,
                 logger: logging.Logger | None = None) -> None:
        """
        Check the settings and remember where the inputs are.

        Nothing is read here: the files are opened by ``run``, so a
        missing input fails where it is used.

        :param audit_path: Path of ``training_audit.json``.
        :type audit_path: str | Path
        :param results_root: Directory with one sub-directory per job.
        :type results_root: str | Path
        :param draws: Number of permutations per correlation.
        :type draws: int
        :param seed: Seed of the permutation generator.
        :type seed: int
        :param candidate_variants: Variants allowed in the analysis.
        :type candidate_variants: Sequence[str]
        :param min_trained_fraction: Trained-fraction cut in [0, 1].
        :type min_trained_fraction: float
        :param metrics_glob: Pattern of the ``metrics.json`` files.
        :type metrics_glob: str
        :param logger: Logger created by the calling script, or None.
        :type logger: logging.Logger | None
        :raises TypeError: If logger or candidate_variants are of the
            wrong type.
        :raises ValueError: If draws, seed, min_trained_fraction or
            metrics_glob are out of range or empty.
        """
        super().__init__(logger=logger)
        if isinstance(candidate_variants, str) \
                or not isinstance(candidate_variants, Sequence) \
                or not all(isinstance(v, str)
                           for v in candidate_variants):
            self._error("[ReconstructionCorrelation.__init__] Input "
                        "error: candidate_variants must be a sequence "
                        f"of strings, not {candidate_variants!r}.",
                        TypeError)
        if isinstance(draws, bool) or not isinstance(draws, int) \
                or draws <= 0:
            self._error("[ReconstructionCorrelation.__init__] Input "
                        "error: draws must be a strictly positive "
                        f"integer (>0), not {draws!r}.")
        if isinstance(seed, bool) or not isinstance(seed, int):
            self._error("[ReconstructionCorrelation.__init__] Input "
                        "error: seed must be an integer, not "
                        f"{seed!r}.")
        if not 0.0 <= float(min_trained_fraction) <= 1.0:
            self._error("[ReconstructionCorrelation.__init__] Input "
                        "error: min_trained_fraction must be in "
                        f"[0, 1], not {min_trained_fraction}.")
        if not isinstance(metrics_glob, str) or not metrics_glob:
            self._error("[ReconstructionCorrelation.__init__] Input "
                        "error: metrics_glob must be a non-empty "
                        f"string, not {metrics_glob!r}.")
        self.audit_path = Path(audit_path)
        self.results_root = Path(results_root)
        self.draws = int(draws)
        self.seed = int(seed)
        self.candidate_variants = tuple(candidate_variants)
        self.min_trained_fraction = float(min_trained_fraction)
        self.metrics_glob = metrics_glob
        self._audit_cache: dict[str, Any] | None = None

    def quality_by_job(
            self) -> tuple[dict[str, float],
                           dict[str, tuple[str, str, str]]]:
        """
        Reconstruction quality of every job, from the training audit.

        It is the median of the loss-over-floor ratios of the
        autoencoders of the job, keeping only those whose floor is not
        degenerate: a group whose features barely move has a floor
        near zero and its ratio measures the floor, not the model.

        :return: Two mappings with the same keys: job id -> median
            ratio, and job id -> (dataset, variant, paradigm).
        :rtype: tuple[dict[str, float], dict[str, tuple[str, str,
            str]]]
        :raises OSError: If the audit file cannot be read.
        :raises ValueError: If the audit file is not valid JSON.
        """
        quality: dict[str, float] = dict()
        meta: dict[str, tuple[str, str, str]] = dict()
        ratios: list[float] = list()
        for job in self._audit()["jobs"]:
            ratios = [a["ratio"] for a in job.get("autoencoders", list())
                      if a.get("ratio") is not None
                      and not a.get("floor_degenerate")]
            if not ratios:
                continue
            quality[job["job_id"]] = statistics.median(ratios)
            meta[job["job_id"]] = (job["dataset"], job["variant"],
                                   job["paradigm"])
        return quality, meta

    def trained_variants(self) -> list[str]:
        """
        Variants that do leave the floor, as the audit itself reports.

        A variant is kept when it is a candidate and the audit gives
        it a trained autoencoder in at least ``min_trained_fraction``
        of the nine datasets.

        :return: The kept variant names, sorted alphabetically.
        :rtype: list[str]
        :raises OSError: If the audit file cannot be read.
        :raises ValueError: If the audit file is not valid JSON.
        """
        by_variant: Mapping[str, Any] = \
            self._audit()["aggregates"]["by_variant"]
        return sorted(
            v for v, d in by_variant.items()
            if d.get(AUDIT_VARIANT_KEY, 0) >= self.min_trained_fraction
            and v in self.candidate_variants)

    def auroc_by_job(self) -> dict[str, tuple[float, float]]:
        """
        ROC-AUC of the raw error and of the fused detector per job.

        Both come from the evaluation block ``E`` of the job's
        ``metrics.json``: ``RAW`` is the plain reconstruction error and
        ``FUS`` is the primary metric of the plan. A job that does not
        publish both is skipped, and so is an unreadable file.

        Note: the directories are visited in directory order, as the
        original ``glob.glob`` did, and the job id comes from the file
        itself, or from the directory name.

        :return: Job id -> (ROC-AUC of the raw error, ROC-AUC of the
            fused detector), both as floats.
        :rtype: dict[str, tuple[float, float]]
        """
        out: dict[str, tuple[float, float]] = dict()
        metrics: Any = None
        for path in self.results_root.glob(self.metrics_glob):
            try:
                metrics = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                # Fidelity: the original swallowed every read or parse
                # error of a metrics file and went on with the rest.
                continue
            block = metrics.get("E") or dict()
            raw = (block.get("RAW") or dict()).get("auroc")
            fused = (block.get("FUS") or dict()).get("auroc")
            if raw is None or fused is None:
                continue
            jid = ((metrics.get("job") or dict()).get("job_id")
                   or path.parent.name)
            out[jid] = (float(raw), float(fused))
        return out

    def rows(self) -> tuple[list[tuple[float, float, float, str, str]],
                            int, list[str]]:
        """
        Build the analysis table, without duplicated MLP rows.

        A row is kept when the job has a quality value and the two
        ROC-AUCs and its variant leaves the floor. The AR and the TSR
        jobs of the MLP are the same configuration and give identical
        results, so the rows are de-duplicated by (dataset,
        variant, quality, raw ROC-AUC, fused ROC-AUC) rounded to
        twelve decimals, not to pseudo-replicate.

        Note: the key holds neither the seed, although the original
        comment mentions it, nor the paradigm, which collapses the MLP
        pair.

        :return: The rows as (quality, raw ROC-AUC, fused ROC-AUC,
            dataset, variant), ordered by job id; how many jobs there
            were before de-duplicating; and the kept variants.
        :rtype: tuple[list[tuple[float, float, float, str, str]], int,
            list[str]]
        :raises OSError: If the audit file cannot be read.
        :raises ValueError: If the audit file is not valid JSON.
        """
        kept: dict[str, tuple[Any, ...]] = dict()
        rows: list[tuple[float, float, float, str, str]] = list()
        seen: set[tuple[Any, ...]] = set()
        quality, meta = self.quality_by_job()
        aurocs, variants = self.auroc_by_job(), self.trained_variants()
        for jid, value in quality.items():
            if jid not in aurocs:
                continue
            dataset, variant, paradigm = meta[jid]
            if variant not in variants:
                continue
            kept[jid] = (value, aurocs[jid][0], aurocs[jid][1],
                         dataset, variant, paradigm)
        for _, row in sorted(kept.items()):
            value, raw, fused, dataset, variant, _ = row
            key = (dataset, variant, round(value, DEDUP_DECIMALS),
                   round(raw, DEDUP_DECIMALS),
                   round(fused, DEDUP_DECIMALS))
            if key in seen:
                continue
            seen.add(key)
            rows.append((value, raw, fused, dataset, variant))
        return rows, len(kept), variants

    def permutation_p(self,
                      x: Sequence[float],
                      y: Sequence[float],
                      groups: Sequence[Any]) -> tuple[float, int]:
        """
        Permutation p-value of |rho|, permuting y inside each group.

        The observed statistic is the absolute Spearman rho of x
        against y. Then y is shuffled ``draws`` times, independently
        inside every group, counting how often the absolute rho of the
        shuffle reaches the observed one. The p-value is the add-one
        estimator (count + 1) / (draws + 1), never zero.

        :param x: Reconstruction quality, one value per row.
        :type x: Sequence[float]
        :param y: Detection metric per row, as long as x, two rows up.
        :type y: Sequence[float]
        :param groups: Hashable group label per row, normally the
            (dataset, variant) pair; as long as x.
        :type groups: Sequence[Any]
        :return: The p-value in (0, 1] and the count of permutations
            that reached the observed |rho|.
        :rtype: tuple[float, int]
        :raises ValueError: If x, y and groups do not have the same
            length, or if there are fewer than two rows.
        """
        count = 0
        index: dict[Any, list[int]] = defaultdict(list)
        values = np.empty(0)
        trial = np.empty(0)
        if len(x) != len(y) or len(x) != len(groups):
            self._error("[ReconstructionCorrelation.permutation_p] "
                        "Input error: x, y and groups must have the "
                        f"same length, not {len(x)}, {len(y)} and "
                        f"{len(groups)}.")
        if len(x) < 2:
            self._error("[ReconstructionCorrelation.permutation_p] "
                        "Input error: at least two rows are needed to "
                        f"correlate, not {len(x)}.")
        observed = abs(spearmanr(x, y).statistic)
        for i, group in enumerate(groups):
            index[group].append(i)
        values = np.asarray(y, float)
        rng = np.random.default_rng(self.seed)
        for _ in range(self.draws):
            trial = values.copy()
            for _, positions in index.items():
                trial[positions] = rng.permutation(values[positions])
            if abs(spearmanr(x, trial).statistic) \
                    >= observed - TIE_TOLERANCE:
                count += 1
        return (count + 1) / (self.draws + 1), count

    def run(self) -> dict[str, Any]:
        """
        Compute both correlations and their permutation p-values.

        Besides the global rho it reports the rho of every stream on
        its own and in how many it is negative, the sign check that
        Section 10 of the supplement quotes. Note: every correlation
        restarts the generator from the same seed, so both p-values use
        the very same permutations.
        :return: The result, with the keys of the original file in the
            original order: ``tag``, ``n_jobs_before_deduplication``,
            ``n``, ``variants``, ``B``, ``seed``, ``x_definition``,
            ``permutation_groups``, ``raw`` and ``fus_primary``.
        :rtype: dict[str, Any]
        :raises OSError: If the audit file cannot be read.
        :raises ValueError: If the audit is not valid JSON, or if
            fewer than two rows survive the filters.
        """
        result: dict[str, Any] = dict()
        per_dataset: dict[str, float | None] = dict()
        rows, n_jobs, variants = self.rows()
        x = [t[0] for t in rows]
        groups = [(t[3], t[4]) for t in rows]
        result = {"tag": RESULT_TAG,
                  "n_jobs_before_deduplication": n_jobs,
                  "n": len(rows),
                  "variants": variants,
                  "B": self.draws,
                  "seed": self.seed,
                  "x_definition": X_DEFINITION,
                  "permutation_groups": PERMUTATION_GROUPS}
        for name, column in OUTCOME_COLUMNS:
            y = [t[column] for t in rows]
            rho = spearmanr(x, y)
            p_value, count = self.permutation_p(x, y, groups)
            per_dataset = dict()
            for dataset in sorted({t[3] for t in rows}):
                xs = [t[0] for t in rows if t[3] == dataset]
                ys = [y[i] for i, t in enumerate(rows) if t[3] == dataset]
                per_dataset[dataset] = (
                    float(spearmanr(xs, ys).statistic)
                    if len(xs) >= MIN_DATASET_POINTS else None)
            result[name] = {
                "rho": float(rho.statistic),
                "p_asymptotic": float(rho.pvalue),
                "p_permutation": p_value,
                "count_ge": count,
                "negative_in": sum(1 for v in per_dataset.values()
                                   if v is not None and v < 0),
                "of_streams": sum(v is not None
                                  for v in per_dataset.values()),
                "rho_by_dataset": per_dataset}
        return result

    def summary_lines(self, result: Mapping[str, Any]) -> list[str]:
        """
        The console summary of the original script, line by line.

        Library code does not print, so the lines are returned and
        the script in ``bin/`` logs them, with the wording and the
        number formats the original used.

        :param result: A result as ``run`` returns it; it must carry
            the two outcome blocks.
        :type result: Mapping[str, Any]
        :return: One string per printed line, without line breaks.
        :rtype: list[str]
        :raises TypeError: If result is not a mapping.
        :raises ValueError: If result does not hold both outcomes.
        """
        lines: list[str] = list()
        block: Mapping[str, Any] = dict()
        if not isinstance(result, Mapping):
            self._error("[ReconstructionCorrelation.summary_lines] "
                        "Input error: result must be a mapping, not "
                        f"{type(result).__name__}.", TypeError)
        for name, _ in OUTCOME_COLUMNS:
            if name not in result:
                self._error("[ReconstructionCorrelation."
                            "summary_lines] Input error: result must "
                            f"hold the outcome {name!r}, and it holds "
                            f"{sorted(result)}.")
        lines.append(f"n = {result['n']} (of "
                     f"{result['n_jobs_before_deduplication']} jobs "
                     "before de-duplication), variants "
                     f"{result['variants']}")
        for name, _ in OUTCOME_COLUMNS:
            block = result[name]
            lines.append(f"  {name:13s} rho = {block['rho']:+.4f}  "
                         f"p_perm = {block['p_permutation']:.4f}  "
                         f"negative in {block['negative_in']} of "
                         f"{block['of_streams']} streams")
        return lines

    def _audit(self) -> dict[str, Any]:
        """
        Read the retrospective training audit once and keep it.

        :return: The parsed ``training_audit.json``.
        :rtype: dict[str, Any]
        :raises OSError: If the file cannot be read.
        :raises ValueError: If the file is not valid JSON.
        """
        if self._audit_cache is None:
            self._audit_cache = json.loads(
                self.audit_path.read_text(encoding="utf-8"))
        return self._audit_cache

    def __repr__(self) -> str:
        """
        Show the inputs and the permutation settings.

        :return: Text like "ReconstructionCorrelation(audit_path=...,
            draws=100000, ...)".
        :rtype: str
        """
        return (f"ReconstructionCorrelation(audit_path="
                f"{str(self.audit_path)!r}, results_root="
                f"{str(self.results_root)!r}, draws={self.draws}, "
                f"seed={self.seed}, candidate_variants="
                f"{self.candidate_variants!r})")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
