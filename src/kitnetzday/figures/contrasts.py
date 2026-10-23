# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         src/kitnetzday/figures/contrasts.py                       #
#   Version:      1.1.1                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Bottom panel of figure F2: the five pre-specified H1      #
#                 paired contrasts with their intervals and labels.         #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
from matplotlib.axes import Axes

from kitnetzday.figures.h1_numbers import add_h1_numbers
from kitnetzday.figures.print_style import FONT_SIZE_MIN
from kitnetzday.io.inputs import Inputs
from kitnetzday.io.outputs import Out
from kitnetzday.naming import GREY_FILL, GREY_LIGHT, ROPE, Naming
from kitnetzday.utils.formatting import fnum, get
from kitnetzday.utils.loggable import Loggable

# Text printed instead of a Holm p value when the family cannot reject
# by construction, and fallback of the "status" field of stats.json.
_UNINFORMATIVE = "uninformative by construction"
# The labels of every interval go in a text column to the right of the
# panel, at this fraction of the axes width: the Holm p value and the
# count on top, the two robustness readings under it, one per line.
_TEXT_X: float = 1.04
_TEXT_RISE: float = 0.27
_TEXT_DROP: float = 0.15
# Separator of the two robustness readings inside the label; it becomes
# a line break when the label is drawn.
_READING_SEPARATOR: str = " | "


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                MAIN CLASS                                 #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class ContrastsPanel(Loggable):
    """
    Panel of the five pre-specified H1 contrasts against the control.

    Section 5.3 of the analysis plan fixes one comparison per
    autoencoder family (Conv1D, Conv2D, LSTM, MHA, MLP) against the
    KitNET-style control C10, with an exact Wilcoxon signed-rank test
    Holm-adjusted over the five of them. The panel draws, per family,
    the Hodges-Lehmann difference in ROC-AUC with its exact confidence
    interval, the ROPE band, the between-seed noise band of section
    7.1 and the two robustness readings required whenever a difference
    between configurations is stated (the six KNAD captures as one
    unit, and the KNAD testbed removed), and registers one row of
    ``numbers_used.md`` per value it prints.

    Nothing is computed here: every number comes from ``stats.json``,
    written by ``stats.py``. The labels are printed at 7.5 pt in a
    text column to the right of the panel instead of over the
    intervals.

    :param inputs: Inputs of the run; only ``partial_inputs`` is read.
    :type inputs: Inputs
    :param out: Output directory, which collects the rows of values.
    :type out: Out
    :param naming: Vocabulary of the plan, or None for that of inputs.
    :type naming: Naming | None
    :param rope: Half width of the ROPE in ROC-AUC points, 0.01 in
        the plan (section 5.2); strictly positive.
    :type rope: float
    :param alpha: Level at which a Holm-adjusted p value counts as a
        rejection, 0.05; in (0, 1].
    :type alpha: float
    :param grey_rope: Matplotlib grey of the ROPE band, "0.85" here.
    :type grey_rope: str
    :param grey_noise: Grey of the edge of the noise band, "0.55".
    :type grey_noise: str
    :param logger: Logger of the calling script, or None.
    :type logger: logging.Logger | None
    :raises TypeError: If a collaborator or grey has a wrong class.
    :raises ValueError: If rope is not strictly positive or alpha is
        outside (0, 1].
    """

    def __init__(self,
                 inputs: Inputs,
                 out: Out,
                 naming: Naming | None = None,
                 rope: float = ROPE,
                 alpha: float = 0.05,
                 grey_rope: str = GREY_LIGHT,
                 grey_noise: str = GREY_FILL,
                 logger: logging.Logger | None = None) -> None:
        """
        Check the collaborators and the thresholds, and store them.

        :param inputs: Inputs of the run.
        :type inputs: Inputs
        :param out: Output directory of the run.
        :type out: Out
        :param naming: Vocabulary of the plan, or None.
        :type naming: Naming | None
        :param rope: Half width of the ROPE, strictly positive.
        :type rope: float
        :param alpha: Rejection level, in (0, 1].
        :type alpha: float
        :param grey_rope: Grey of the ROPE band.
        :type grey_rope: str
        :param grey_noise: Grey of the edge of the noise band.
        :type grey_noise: str
        :param logger: Logger created by the calling script, or None.
        :type logger: logging.Logger | None
        :raises TypeError: If a collaborator or a grey is of the wrong
            class.
        :raises ValueError: If rope or alpha are out of range.
        """
        super().__init__(logger=logger)
        if not isinstance(inputs, Inputs):
            self._error("[ContrastsPanel.__init__] Input error: the "
                        "parameter inputs must be an Inputs, not "
                        f"{type(inputs).__name__}.", TypeError)
        if not isinstance(out, Out):
            self._error("[ContrastsPanel.__init__] Input error: the "
                        "parameter out must be an Out, not "
                        f"{type(out).__name__}.", TypeError)
        if naming is not None and not isinstance(naming, Naming):
            self._error("[ContrastsPanel.__init__] Input error: the "
                        "parameter naming must be a Naming or None, not "
                        f"{type(naming).__name__}.", TypeError)
        for name, value in (("rope", rope), ("alpha", alpha)):
            if isinstance(value, bool) or not isinstance(value,
                                                         (int, float)):
                self._error("[ContrastsPanel.__init__] Input error: "
                            f"the parameter {name} must be a real "
                            f"number, not {type(value).__name__}.",
                            TypeError)
        if not float(rope) > 0.0:
            self._error("[ContrastsPanel.__init__] Input error: the "
                        "parameter rope must be strictly positive "
                        f"(>0), not {rope}.")
        if not 0.0 < float(alpha) <= 1.0:
            self._error("[ContrastsPanel.__init__] Input error: the "
                        "parameter alpha must be in (0, 1], not "
                        f"{alpha}.")
        if not isinstance(grey_rope, str) \
                or not isinstance(grey_noise, str):
            self._error("[ContrastsPanel.__init__] Input error: the "
                        "parameters grey_rope and grey_noise must be "
                        f"strings, not {type(grey_rope).__name__} and "
                        f"{type(grey_noise).__name__}.", TypeError)
        self.inputs = inputs
        self.out = out
        self.naming = inputs.naming if naming is None else naming
        self.rope = float(rope)
        self.alpha = float(alpha)
        self.grey_rope = grey_rope
        self.grey_noise = grey_noise

    def families(self,
                 comparisons: Mapping[str, Any]
                 ) -> tuple[set[str], set[str]]:
        """
        Split the families that reject into robust and not robust.

        Section 5.8 of the plan only lets a rejection reach the
        abstract, the highlights and the contributions when it also
        passes both pre-specified robustness rules (sign of the
        Hodges-Lehmann difference preserved within blocks, and
        direction preserved with the six KNAD captures as one unit and
        with KNAD removed), so the panel marks the two differently.

        :param comparisons: Content of
            ``stats.json#/H1/comparisons``, keyed
            "<architecture>_vs_original"; a missing or null record
            counts as "no rejection".
        :type comparisons: Mapping[str, Any]
        :return: The families whose Holm-adjusted p value is at most
            alpha and whose "robust" flag is true, and those that
            reject without it.
        :rtype: tuple[set[str], set[str]]
        :raises TypeError: If comparisons is not a mapping.
        """
        significant = set()
        not_robust = set()
        record = dict()
        p_holm = None
        if not isinstance(comparisons, Mapping):
            self._error("[ContrastsPanel.families] Input error: the "
                        "parameter comparisons must be a mapping, not "
                        f"{type(comparisons).__name__}.", TypeError)
        for architecture in self.naming.architectures:
            record = comparisons.get(f"{architecture}_vs_original") \
                or dict()
            p_holm = record.get("p_holm")
            if p_holm is not None and float(p_holm) <= self.alpha:
                if record.get("robust"):
                    significant.add(architecture)
                else:
                    not_robust.add(architecture)
        return significant, not_robust

    def draw(self,
             ax: Axes,
             comparisons: Mapping[str, Any],
             noise_median: float | None = None,
             informative: Any = None,
             status: Any = None) -> list[str]:
        """
        Draw the five contrasts and register their rows of values.

        One horizontal interval per family, in the order the plan
        lists the architectures: the Hodges-Lehmann difference as a
        diamond, its exact confidence interval as a line with end caps,
        the ROPE as a shaded band and the between-seed noise reference
        as a hatched band with dashed edges. Each interval carries the
        two labels of :meth:`_labels`, printed to the right of the
        panel. Without comparisons the axes are switched off and a
        notice is printed instead.

        Note: the x label names the control "C10" literally, whatever
        the control is. Note: a comparison storing an explicit null
        under "hl" raises a TypeError here, because the value is
        converted with ``float`` and only a *missing* key is treated as
        not-a-number.

        :param ax: Axes to draw on, the bottom panel of F2; the
            labels are written to its right, outside the axes.
        :type ax: Axes
        :param comparisons: Content of
            ``stats.json#/H1/comparisons``; empty gives the notice.
        :type comparisons: Mapping[str, Any]
        :param noise_median: Median across datasets of sqrt(2) SD_pool
            in ROC-AUC points, or None when no band can be drawn.
        :type noise_median: float | None
        :param informative: Value of ``stats.json#/H1/informative``;
            only the literal False switches the labels.
        :type informative: Any
        :param status: Value of ``stats.json#/H1/status``, printed when
            the family is uninformative; a falsy value falls back to
            "uninformative by construction".
        :type status: Any
        :return: Ids of the rows of ``numbers_used.md``, in the
            order they were produced.
        :rtype: list[str]
        :raises TypeError: If ax is not an Axes, comparisons is not a
            mapping, or noise_median is neither real nor None.
        """
        ids = list()
        names = list()
        hl = list()
        lo = list()
        hi = list()
        txt = list()
        txt2 = list()
        record = dict()
        tag = self.naming.plan_tag("h1")
        if not isinstance(ax, Axes):
            self._error("[ContrastsPanel.draw] Input error: the "
                        "parameter ax must be a matplotlib Axes, not "
                        f"{type(ax).__name__}.", TypeError)
        if not isinstance(comparisons, Mapping):
            self._error("[ContrastsPanel.draw] Input error: the "
                        "parameter comparisons must be a mapping, not "
                        f"{type(comparisons).__name__}.", TypeError)
        if noise_median is not None and (isinstance(noise_median, bool)
                                         or not isinstance(noise_median,
                                                           (int, float))):
            self._error("[ContrastsPanel.draw] Input error: the "
                        "parameter noise_median must be a real number "
                        f"or None, not {type(noise_median).__name__}.",
                        TypeError)
        for architecture in self.naming.architectures:
            record = comparisons.get(f"{architecture}_vs_original") \
                or dict()
            if not record:
                continue
            names.append(architecture)
            hl.append(float(record.get("hl", np.nan)))
            lo.append(float(get(record, "hl_ci", "lower",
                                default=np.nan)))
            hi.append(float(get(record, "hl_ci", "upper",
                                default=np.nan)))
            above, below = self._labels(record, informative, status)
            txt.append(above)
            txt2.append(below)
            ids += add_h1_numbers(self.out, architecture, record,
                                  tag, self.inputs.partial_inputs,
                                  self.logger)
        if names:
            self._draw_intervals(ax, names, hl, lo, hi, txt, txt2,
                                 noise_median)
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, "stats.json not available:\nH1 panel not "
                              "generated", ha="center", va="center")
        return ids

    def robustness_note(self,
                        informative: Any = None,
                        status: Any = None) -> str:
        """
        Sentence of the caption that explains the asterisks.

        The caption has to say what an asterisk means before a reader
        uses one, and differently when the family cannot reject at all:
        section 5.8 bars a non-robust rejection from the abstract, the
        highlights and the contributions, and an uninformative family
        has nothing to mark.

        :param informative: Value of ``stats.json#/H1/informative``;
            only the literal False selects the other sentence.
        :type informative: Any
        :param status: Value of ``stats.json#/H1/status``, inserted in
            that sentence; a falsy value falls back to "uninformative
            by construction".
        :type status: Any
        :return: The sentence, ready to be concatenated into the
            caption of F2.
        :rtype: str
        """
        if informative is False:
            return ("The family is " + str(status or _UNINFORMATIVE)
                    + ", so no comparison can be rejected and no "
                      "asterisk is drawn.")
        return ("An asterisk in the top panel marks a family whose H1 "
                "comparison is rejected at the "
                "Holm-adjusted 5\\% level \\emph{and} passes both "
                "pre-specified robustness rules (sign of the "
                "Hodges--Lehmann difference preserved on the "
                "within-block evaluation, and direction preserved "
                "when the KNAD captures are treated as one unit and "
                "when they are removed); a parenthesised "
                "asterisk marks a rejection that does not pass them, "
                "which the analysis plan bars from the "
                "abstract, the highlights and the contributions.")

    def __repr__(self) -> str:
        """
        Show the thresholds and how many families are drawn.

        :return: Text like "ContrastsPanel(rope=0.01, alpha=0.05,
            architectures=5)".
        :rtype: str
        """
        return (f"ContrastsPanel(rope={self.rope}, alpha={self.alpha}, "
                f"architectures={len(self.naming.architectures)})")

    # ----- private helpers ------------------------------------------- #
    def _labels(self,
                record: Mapping[str, Any],
                informative: Any,
                status: Any) -> tuple[str, str]:
        """
        Build the two labels printed around one interval.

        The first one carries the Holm p value, or the pre-specified
        string of a family that cannot reject by construction.
        The second one carries the two dependence readings that
        sections 1.2 and 5.8.2 ask for in the very figure that states
        the claim: the difference with the six KNAD captures collapsed
        into one unit (n = 4) and without the KNAD testbed (n = 3).

        :param record: One comparison record of stats.json.
        :type record: Mapping[str, Any]
        :param informative: Value of ``/H1/informative``.
        :type informative: Any
        :param status: Value of ``/H1/status``.
        :type status: Any
        :return: The label above the interval, e.g. "Holm $p$ = 0.031;
            higher in 8/9", and the one below it, empty when neither
            dependence reading is stored.
        :rtype: tuple[str, str]
        """
        above = ""
        below = ""
        p_holm = record.get("p_holm")
        k_pos = record.get("k_pos")
        n_pairs = record.get("n")
        collapsed = get(record, "robustness", "dependence",
                        "KNAD_collapsed", default=dict()) or dict()
        without = get(record, "robustness", "dependence",
                      "without_KNAD", default=dict()) or dict()
        if informative is False:
            above = str(status or _UNINFORMATIVE)
        else:
            above = (("Holm $p$ = " + (f"{float(p_holm):.3f}"
                                       if p_holm is not None
                                       else "n/a"))
                     + (f"; higher in {k_pos}/{n_pairs}"
                        if k_pos is not None and n_pairs else "")
                     # The robustness caveat only says something about
                     # a REJECTION: in a comparison that is not
                     # significant there is nothing to be robust or
                     # not (section 5.8).
                     + ("; rejection not robust (§5.8)"
                        if (record.get("reject")
                            and record.get("robust") is False)
                        else ""))
        if collapsed or without:
            below = ("KNAD as one unit: HL %s (%s/%s) | without KNAD: "
                     "HL %s (%s/%s)"
                     % (fnum(collapsed.get("hl")),
                        collapsed.get("k_pos"), collapsed.get("n"),
                        fnum(without.get("hl")),
                        without.get("k_pos"), without.get("n")))
        return above, below

    def _draw_intervals(self,
                        ax: Axes,
                        names: list[str],
                        hl: list[float],
                        lo: list[float],
                        hi: list[float],
                        txt: list[str],
                        txt2: list[str],
                        noise_median: float | None) -> None:
        """
        Draw the bands, the intervals and the annotations.

        :param ax: Axes to draw on.
        :type ax: Axes
        :param names: Families, top to bottom, at least one.
        :type names: list[str]
        :param hl: Hodges-Lehmann differences, as long as names.
        :type hl: list[float]
        :param lo: Lower ends of the confidence intervals.
        :type lo: list[float]
        :param hi: Upper ends of the confidence intervals.
        :type hi: list[float]
        :param txt: First line of the label of each interval.
        :type txt: list[str]
        :param txt2: The two robustness readings of each interval,
            printed under the first line, one per line; empty without
            the dependence readings.
        :type txt2: list[str]
        :param noise_median: Median of sqrt(2) SD_pool, or None.
        :type noise_median: float | None
        """
        y = np.arange(len(names))
        ax.axvspan(-self.rope, self.rope, color=self.grey_rope,
                   zorder=0)
        if noise_median is not None:
            for sign in (-1, 1):
                ax.axvline(sign * noise_median, color="0.35",
                           linestyle=(0, (4, 2)), linewidth=1.0,
                           zorder=1)
            ax.axvspan(-noise_median, noise_median, facecolor="none",
                       edgecolor=self.grey_noise, hatch="\\\\\\",
                       linewidth=0.0, zorder=0)
        ax.axvline(0.0, color="black", linewidth=0.9, zorder=1)
        for i in range(len(names)):
            ax.plot([lo[i], hi[i]], [i, i], color="black",
                    linewidth=1.1, zorder=2)
            ax.plot([lo[i], lo[i]], [i - 0.12, i + 0.12], color="black",
                    linewidth=1.1)
            ax.plot([hi[i], hi[i]], [i - 0.12, i + 0.12], color="black",
                    linewidth=1.1)
            ax.plot(hl[i], i, marker="D", color="black", markersize=4.2,
                    zorder=3)
            ax.annotate(txt[i],
                        xy=(_TEXT_X,
                            i - (_TEXT_RISE if txt2[i] else 0.0)),
                        xycoords=("axes fraction", "data"), ha="left",
                        va="center", fontsize=FONT_SIZE_MIN,
                        annotation_clip=False)
            if txt2[i]:
                ax.annotate(txt2[i].replace(_READING_SEPARATOR, "\n"),
                            xy=(_TEXT_X, i + _TEXT_DROP),
                            xycoords=("axes fraction", "data"),
                            ha="left", va="center",
                            fontsize=FONT_SIZE_MIN, color="0.25",
                            linespacing=1.2, annotation_clip=False)
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.set_ylim(-0.6, len(names) - 0.4)
        ax.invert_yaxis()
        ax.set_xlabel("HL difference in ROC-AUC vs. control C10")
        ax.set_title("H1: five pre-specified comparisons (Holm, "
                     "$m$ = 5)", loc="left")
        ax.grid(axis="x", color="0.9", linewidth=0.6)
        ax.set_axisbelow(True)


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
