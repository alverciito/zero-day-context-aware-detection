# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         src/kitnetzday/figures/ranks.py                           #
#   Version:      1.1.2                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Figure F2: mean rank of every configuration with its      #
#                 descriptive interval, above the H1 contrasts.             #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from kitnetzday.figures.contrasts import ContrastsPanel
from kitnetzday.figures.print_style import (
    FONT_SIZE_MIN,
    PRINT_RC_PARAMS,
    PRINT_WIDTH_INCHES,
)
from kitnetzday.figures.ranks_caption import mean_ranks_caption
from kitnetzday.io.inputs import Inputs
from kitnetzday.io.outputs import Out
from kitnetzday.naming import ROPE, Naming
from kitnetzday.utils.formatting import get, ranks_avg
from kitnetzday.utils.loggable import Loggable

# Label of the analysis plan for the between-seed noise reference: not
# one of naming.PLAN_TAGS, written inline for this single value.
_NOISE_TAG = ("[SENSIBILIDAD plan §7.1 — referencia de ruido entre "
              "semillas]")
_INTERVAL_NOTE = ("The interval (+-1.96 standard errors of the mean "
                  "rank) is not fixed by the plan: it is descriptive "
                  "(note FIG-1).")
# Canvas at the text width of the manuscript (170 mm), which does not
# include the figure, and the two
# panels as [left, bottom, width, height] in figure fractions. Side by
# side they do not fit with 7.5 pt lettering (the robustness line of H1
# alone is 241 pt wide), so the ranks go on top and the H1 contrasts
# below, with their labels in a text column to the right of the panel.
_FIG_SIZE: tuple[float, float] = (PRINT_WIDTH_INCHES, 6.1)
_RANKS_AXES: tuple[float, float, float, float] = (
    0.335, 0.560, 0.650, 0.395)
_H1_AXES: tuple[float, float, float, float] = (
    0.105, 0.075, 0.500, 0.370)
# Base name, LaTeX label and plan id, as in the submission.
_NAME = "fig_mean_ranks"
_LABEL = "fig:mean-ranks"
_OUT_ID = "F2"


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                MAIN CLASS                                 #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class RanksFigure(Loggable):
    """
    Output F2: mean ranks and the five H1 contrasts.

    The figure is not a float of the article: its bottom panel draws
    the H1 comparisons that Table 6 of the manuscript prints.
    The top panel is the descriptive omnibus of section 5.3 of the
    analysis plan: each configuration is ranked within every dataset
    (rank 1 = lowest ROC-AUC, ties share the average rank) and the mean
    of those ranks across datasets is drawn with the per-dataset ranks
    behind it and an interval of +-1.96 standard errors of that mean.
    Only datasets with all eleven configurations are ranked, and the
    ones left out are reported as missing inputs. The bottom panel,
    delegated to :class:`ContrastsPanel`, shows the five pre-specified
    H1 comparisons against the control, so that no rank is read without
    the effect size, its interval, the ROPE and the noise reference.

    Nothing is recomputed that ``stats.py`` already computed: the mean
    ranks of ``stats.json``, when present, are the ones drawn, and any
    difference becomes a note of the figure. The figure is drawn at
    the text width of the manuscript, 170 mm, with no lettering under
    7.5 pt.

    :param inputs: Reader of the aggregate and of the statistics.
    :type inputs: Inputs
    :param out: Output directory of the run.
    :type out: Out
    :param naming: Vocabulary of the plan, or None for that of inputs.
    :type naming: Naming | None
    :param rope: Half width of the ROPE in ROC-AUC points, 0.01 in
        the plan (section 5.2); strictly positive.
    :type rope: float
    :param interval_z: Normal quantile of the interval of the mean
        rank, 1.96 in the submission; strictly positive.
    :type interval_z: float
    :param alpha: Level at which a Holm-adjusted p value counts as a
        rejection in the bottom panel, 0.05; in (0, 1].
    :type alpha: float
    :param tolerance: Largest difference in rank units still read as
        identical to ``stats.json``; 1e-9, positive.
    :type tolerance: float
    :param rc_params: Matplotlib settings of the paper style.
    :type rc_params: Mapping[str, object]
    :param figsize: Size in inches, (width, height); (6.69, 6.1) in
        the paper, both strictly positive.
    :type figsize: tuple[float, float]
    :param logger: Logger of the calling script, or None.
    :type logger: logging.Logger | None
    :raises TypeError: If a collaborator or rc_params are of the wrong
        class.
    :raises ValueError: If rope, interval_z, tolerance or figsize are
        not strictly positive, or alpha is outside (0, 1].
    """

    def __init__(self,
                 inputs: Inputs,
                 out: Out,
                 naming: Naming | None = None,
                 rope: float = ROPE,
                 interval_z: float = 1.96,
                 alpha: float = 0.05,
                 tolerance: float = 1e-9,
                 rc_params: Mapping[str, object] = PRINT_RC_PARAMS,
                 figsize: tuple[float, float] = _FIG_SIZE,
                 logger: logging.Logger | None = None) -> None:
        """
        Check the collaborators and the settings, and store them.

        :param inputs: Reader of the aggregate and of the statistics.
        :type inputs: Inputs
        :param out: Output directory of the run.
        :type out: Out
        :param naming: Vocabulary of the plan, or None.
        :type naming: Naming | None
        :param rope: Half width of the ROPE, strictly positive.
        :type rope: float
        :param interval_z: Normal quantile, strictly positive.
        :type interval_z: float
        :param alpha: Rejection level, in (0, 1].
        :type alpha: float
        :param tolerance: Reconciliation tolerance, positive.
        :type tolerance: float
        :param rc_params: Matplotlib settings of the paper style.
        :type rc_params: Mapping[str, object]
        :param figsize: Size in inches, both values positive.
        :type figsize: tuple[float, float]
        :param logger: Logger of the calling script, or None.
        :type logger: logging.Logger | None
        :raises TypeError: If a parameter is of the wrong class.
        :raises ValueError: If a setting is out of range, figsize
            included.
        """
        super().__init__(logger=logger)
        # inputs, out, naming, rope and alpha are checked by the panel
        # of the bottom half, which is given the very same ones.
        self.contrasts = ContrastsPanel(inputs=inputs, out=out,
                                        naming=naming, rope=rope,
                                        alpha=alpha, logger=logger)
        for name, value in (("interval_z", interval_z),
                            ("tolerance", tolerance)):
            if isinstance(value, bool) or not isinstance(value,
                                                         (int, float)):
                self._error("[RanksFigure.__init__] Input error: the "
                            f"parameter {name} must be a real number, "
                            f"not {type(value).__name__}.", TypeError)
            if not float(value) > 0.0:
                self._error("[RanksFigure.__init__] Input error: the "
                            f"parameter {name} must be strictly "
                            f"positive (>0), not {value}.")
        if not isinstance(rc_params, Mapping):
            self._error("[RanksFigure.__init__] Input error: the "
                        "parameter rc_params must be a mapping, not "
                        f"{type(rc_params).__name__}.", TypeError)
        if not isinstance(figsize, (tuple, list)) or len(figsize) != 2 \
                or not all(float(v) > 0.0 for v in figsize):
            self._error("[RanksFigure.__init__] Input error: the "
                        "parameter figsize must be a pair (width, "
                        "height) of strictly positive numbers, not "
                        f"{figsize!r}.")
        self.inputs = inputs
        self.out = out
        self.naming = self.contrasts.naming
        self.rope = self.contrasts.rope
        self.alpha = self.contrasts.alpha
        self.interval_z = float(interval_z)
        self.tolerance = float(tolerance)
        self.rc_params = dict(rc_params)
        self.figsize = (float(figsize[0]), float(figsize[1]))

    def build(self) -> bool:
        """
        Build figure F2 and write it, with its caption and its numbers.

        The figure is produced whatever the inputs hold: a dataset
        without the eleven configurations is dropped from the ranking
        and listed as a missing input, and a run without ``stats.json``
        gets the top panel plus a notice in place of the bottom one.

        :return: True when the figure was written, False when no
            dataset is complete.
        :rtype: bool
        :raises OSError: If a file cannot be written.
        """
        ids = list()
        notes = list()
        missing = list()
        rows = list()
        ranks = np.empty((0, 0))
        mean_rank = np.empty(0)
        se = np.empty(0)
        half = np.empty(0)
        configs = self.naming.configurations
        stats = self.inputs.stats or dict()
        rows = self._rank_rows(missing)
        if not rows:
            missing.append("no dataset with the 11 configurations: the "
                           "mean-ranks figure is not generated")
            self._warning("[RanksFigure.build] Input warning: at least "
                          "one dataset with all "
                          f"{len(configs)} configurations was "
                          "expected, and none has them. The mean-ranks "
                          "figure is not generated.")
            return False
        ranks = np.vstack(rows)
        mean_rank = ranks.mean(axis=0)
        n = ranks.shape[0]
        se = (ranks.std(axis=0, ddof=1) / np.sqrt(n) if n > 1
              else np.zeros(len(configs)))
        half = self.interval_z * se
        mean_rank, has_reference = self._reconcile(mean_rank, notes)
        ids += self._add_rank_numbers(mean_rank, has_reference)
        comparisons = get(stats, "H1", "comparisons",
                          default=dict()) or dict()
        informative = get(stats, "H1", "informative", default=None)
        status = get(stats, "H1", "status", default=None)
        if not comparisons:
            missing.append("stats.json without H1: the bottom panel "
                           "is empty")
        noise_median, n_no_noise, noise_ids = self._noise(notes)
        ids += noise_ids
        order = np.argsort(mean_rank)
        significant, not_robust = self.contrasts.families(comparisons)
        with plt.rc_context(self.rc_params):
            fig = plt.figure(figsize=self.figsize)
            self._draw_ranks(fig.add_axes(list(_RANKS_AXES)), ranks,
                             mean_rank, half, order, significant,
                             not_robust)
            ids += self.contrasts.draw(fig.add_axes(list(_H1_AXES)),
                                       comparisons, noise_median,
                                       informative, status)
            # Saved inside the context: pdf.fonttype is read when the
            # file is written, and outside it the fonts are Type 3.
            self.out.save_figure(
                fig, _NAME,
                mean_ranks_caption(
                    n_ranked=n, noise_median=noise_median,
                    n_no_noise=n_no_noise,
                    n_datasets=len(self.naming.datasets),
                    robustness_note=self.contrasts.robustness_note(
                        informative, status),
                    rope=self.rope, logger=self.logger),
                _LABEL,
                self.naming.plan_tag("omni") + " / "
                + self.naming.plan_tag("h1"),
                sorted(set(ids)), self.inputs.partial_inputs,
                {"out_id": _OUT_ID,
                 "partial_inputs_reasons":
                     self.inputs.partial_inputs_reasons,
                 "missing": missing,
                 "notes": notes + [_INTERVAL_NOTE]})
        return True

    def __repr__(self) -> str:
        """
        Show the vocabulary and the settings of the figure.

        :return: Text like "RanksFigure(datasets=9, configurations=11,
            interval_z=1.96, rope=0.01)".
        :rtype: str
        """
        return (f"RanksFigure(datasets={len(self.naming.datasets)}, "
                f"configurations={len(self.naming.configurations)}, "
                f"interval_z={self.interval_z}, rope={self.rope})")

    # ----- private helpers ------------------------------------------- #
    def _rank_rows(self, missing: list[str]) -> list[np.ndarray]:
        """
        Rank the configurations within every complete dataset.

        A dataset enters the ranking only when all the configurations
        have a primary metric: a partial row would rank a hole against
        a number. The dropped ones go to ``missing``.

        Note: the message says "las 11 configuraciones" literally, as
        in the submission, whatever the size of the vocabulary.

        :param missing: Missing inputs, extended in place with one line
            per dropped dataset.
        :type missing: list[str]
        :return: One array of average ranks per complete dataset, in
            vocabulary order; empty when none is complete.
        :rtype: list[np.ndarray]
        """
        rows = list()
        used = list()
        values = list()
        for dataset in self.naming.datasets:
            values = [self.inputs.auroc(dataset, c)
                      for c in self.naming.configurations]
            if any(v is None for v in values):
                missing.append(f"{dataset}: without the 11 "
                               "configurations (left out of the ranks, "
                               "plan §5.3)")
                continue
            rows.append(ranks_avg(values, logger=self.logger))
            used.append(dataset)
        return rows

    def _reconcile(self,
                   mean_rank: np.ndarray,
                   notes: list[str]) -> tuple[np.ndarray, bool]:
        """
        Compare the recomputed mean ranks with those of stats.json.

        The statistics of the paper are produced by ``stats.py``, so a
        figure that recomputed them silently could disagree with the
        text. The stored values win whenever they differ by more than
        the tolerance, and the outcome becomes a note either way.

        Note: the "identical" note says "(<= 1e-9)" literally, so it
        does not follow another tolerance. Note: when ``stats.json``
        carries the branch but no configuration of it, the maximum runs
        over an empty sequence and raises.

        :param mean_rank: Recomputed mean ranks, one per
            configuration; modified in place when the stored ones win.
        :type mean_rank: np.ndarray
        :param notes: Notes of the figure, extended in place.
        :type notes: list[str]
        :return: The mean ranks to draw, and whether ``stats.json``
            provided them, which decides the pointer of their rows.
        :rtype: tuple[np.ndarray, bool]
        :raises ValueError: If the stored branch holds no configuration
            of the vocabulary (see the note above).
        """
        configs = self.naming.configurations
        dmax = 0.0
        reference = get(self.inputs.stats or dict(), "H1_descriptive",
                        "omnibus", "mean_ranks_by_config", default=None)
        if not reference:
            return mean_rank, False
        dmax = max(abs(mean_rank[j] - float(reference.get(c, np.nan)))
                   for j, c in enumerate(configs)
                   if reference.get(c) is not None)
        if not np.isfinite(dmax) or dmax > self.tolerance:
            notes.append("WARNING: the recomputed mean ranks differ "
                         "from stats.json by "
                         f"{dmax:.2e} (those of stats.py are drawn)")
            for j, c in enumerate(configs):
                if reference.get(c) is not None:
                    mean_rank[j] = float(reference[c])
        else:
            notes.append("Mean ranks identical to those of stats.json "
                         "(<= 1e-9).")
        return mean_rank, True

    def _add_rank_numbers(self,
                          mean_rank: np.ndarray,
                          has_reference: bool) -> list[str]:
        """
        Register one row of numbers_used.md per mean rank.

        :param mean_rank: Mean ranks as drawn, one per configuration.
        :type mean_rank: np.ndarray
        :param has_reference: Whether the values come from
            ``stats.json``, which selects the pointer of the row.
        :type has_reference: bool
        :return: Ids of the rows, in vocabulary order.
        :rtype: list[str]
        """
        ids = list()
        pointer = ""
        tag = self.naming.plan_tag("omni")
        for j, c in enumerate(self.naming.configurations):
            if has_reference:
                pointer = ("stats.json#/H1_descriptive/omnibus/"
                           "mean_ranks_by_config/" + c)
            else:
                pointer = ("make_figures.py (ranks over "
                           "aggregate.json#/value_sets/main)")
            ids.append(self.out.add_number(
                f"R.RANK.AGG.{c}.MEANRANK", float(mean_rank[j]),
                pointer, tag, self.inputs.partial_inputs))
        return ids

    def _noise(self,
               notes: list[str]) -> tuple[float | None, int, list[str]]:
        """
        Between-seed noise reference drawn next to the ROPE.

        Section 5.2 of the analysis plan asks for the ROPE to be
        reported next to the median across datasets of sqrt(2) SD_pool
        (section 7.1): without it, +-0.01 reads as the resolution of
        the experiment while the run-to-run noise reaches +-0.10.

        :param notes: Notes of the figure, extended in place when the
            recomputed median disagrees with ``stats.json``.
        :type notes: list[str]
        :return: The median in ROC-AUC points (None when no dataset
            has an estimate), how many have none, and the ids of the
            rows of ``numbers_used.md``.
        :rtype: tuple[float | None, int, list[str]]
        """
        ids = list()
        pointer = ""
        datasets = self.naming.datasets
        values = [self.inputs.sqrt2_sd_pool(x) for x in datasets]
        present = sorted(v for v in values if v is not None)
        median = float(np.median(present)) if present else None
        n_no_noise = len(datasets) - len(present)
        if median is None:
            return None, n_no_noise, ids
        reference = get(self.inputs.stats or dict(), "rope_vs_noise",
                        "median_sqrt2_sd_pool", default=None)
        if reference is not None:
            pointer = "stats.json#/rope_vs_noise/median_sqrt2_sd_pool"
        else:
            pointer = ("make_figures.py: median of aggregate.json#/"
                       "datasets/<D>/sd_pool/sqrt2_sd_pool")
        ids.append(self.out.add_number(
            "R.ROPE.AGG.NOISE.MEDIAN_SQRT2_SDPOOL", median, pointer,
            _NOISE_TAG, self.inputs.partial_inputs))
        if reference is not None and \
                abs(float(reference) - median) > self.tolerance:
            notes.append("WARNING: recomputed median of sqrt(2) "
                         f"SD_pool {median:.6f} != stats.json "
                         f"{reference}")
        return median, n_no_noise, ids

    def _draw_ranks(self,
                    ax: Axes,
                    ranks: np.ndarray,
                    mean_rank: np.ndarray,
                    half: np.ndarray,
                    order: np.ndarray,
                    significant: set[str],
                    not_robust: set[str]) -> None:
        """
        Draw the top panel: per-dataset ranks, means and intervals.

        The configurations go from the lowest mean rank at the top to
        the highest at the bottom. The control is a filled square and
        the rest open diamonds, so the baseline is visible without a
        legend; an asterisk marks a family whose H1 comparison is a
        robust rejection and a parenthesised one a rejection that is
        not robust (section 5.8).

        Note: the x label reads "11 = highest" literally, as in the
        submission, whatever the size of the vocabulary.

        :param ax: Axes to draw on, the top panel of F2.
        :type ax: Axes
        :param ranks: Per-dataset average ranks, shape (n datasets,
            n configurations).
        :type ranks: np.ndarray
        :param mean_rank: Mean rank per configuration, one value per
            configuration.
        :type mean_rank: np.ndarray
        :param half: Half width of each interval in rank units.
        :type half: np.ndarray
        :param order: Configuration indices, lowest mean rank first.
        :type order: np.ndarray
        :param significant: Families with a robust H1 rejection.
        :type significant: set[str]
        :param not_robust: Families that reject without robustness.
        :type not_robust: set[str]
        """
        configs = self.naming.configurations
        control = self.naming.control
        labels = list()
        mark = ""
        n = ranks.shape[0]
        for k, j in enumerate(order):
            c = configs[j]
            for i in range(n):
                ax.plot(ranks[i, j], k, marker="o", markersize=2.6,
                        markerfacecolor="none", markeredgecolor="0.6",
                        linestyle="none", zorder=1)
            ax.errorbar(mean_rank[j], k, xerr=half[j],
                        fmt="s" if c == control else "D",
                        color="black",
                        markersize=5.0 if c == control else 4.0,
                        capsize=2.5, elinewidth=1.0, zorder=3,
                        markerfacecolor=("black" if c == control
                                         else "white"))
            architecture = self.naming.config_architecture(c)
            mark = ""
            if architecture in significant:
                mark = "  *"
            elif architecture in not_robust:
                mark = "  (*)"
            labels.append(f"{c} {self.inputs.config_name(c)}"
                          + ("  (control)" if c == control else "")
                          + mark)
        ax.set_yticks(np.arange(len(configs)))
        ax.set_yticklabels(labels, fontsize=FONT_SIZE_MIN)
        ax.invert_yaxis()
        ax.set_xlabel(f"Mean rank over the {n} datasets (1 = lowest "
                      "ROC-AUC, 11 = highest)")
        ax.set_title("Mean ranks of the eleven configurations "
                     "(descriptive)")
        ax.set_xlim(0.5, len(configs) + 0.5)
        ax.grid(axis="x", color="0.9", linewidth=0.6)
        ax.set_axisbelow(True)


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
