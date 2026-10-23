# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         src/kitnetzday/figures/failures.py                        #
#   Version:      1.1.3                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Figure F4: failure cases per stream, with the             #
#                 pre-specified criterion of plan section 10.               #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from kitnetzday.figures.failure_rule import FailureRule
from kitnetzday.figures.print_style import (
    FONT_SIZE_MIN,
    FONT_SIZE_SCRIPT,
    PRINT_RC_PARAMS,
    PRINT_WIDTH_INCHES,
)
from kitnetzday.io.inputs import Inputs
from kitnetzday.io.outputs import Out
from kitnetzday.naming import GREY_LIGHT, Naming
from kitnetzday.utils.loggable import Loggable

# Identity of the output (plan section 10).
FIGURE_NAME: str = "fig_failure_cases"
FIGURE_LABEL: str = "fig:failure-cases"
OUT_ID: str = "F4"
# Label of the per-cell ROC-AUC values of the bottom panel. It is not one
# of Naming.plan_tags: it is spelled out here.
PER_CELL_TAG: str = ("[CONFIRMATORIO plan §3.1 — valor por celda, "
                     "descriptivo, sin test]")
# The three configurations examined in every failure case, in the order
# they are drawn: the control, the context-free network and one windowed
# network. PROBE_GREYS and PROBE_HATCHES are indexed by that position.
PROBE_CONFIGS: tuple[tuple[str, str], ...] = (
    ("C10", "original (KitNET-style)"), ("C9", "MLP"), ("C5", "LSTM-AR"))
PROBE_GREYS: tuple[str, ...] = ("0.25", "0.55", "0.85")
PROBE_HATCHES: tuple[str, ...] = ("", "///", "...")
# Geometry and greys of the two panels. The figure is drawn at the text
# width of the manuscript (170 mm), although the manuscript does not
# include it, so the point sizes are those of a full-width print.
FIGURE_SIZE_INCHES: tuple[float, float] = (PRINT_WIDTH_INCHES, 7.0)
HEIGHT_RATIOS: tuple[float, float] = (1.0, 0.95)
HSPACE: float = 0.62
MARGINS: dict[str, float] = {
    "left": 0.125, "right": 0.995, "top": 0.955, "bottom": 0.185}
BAR_WIDTH: float = 0.6
PROBE_WIDTH: float = 0.26
NOISE_HALF_WIDTH: float = 0.42
# Room under the bars, in ROC-AUC units, for the four lines of counts.
BOTTOM_ROOM: float = 0.285
TOP_ROOM: float = 0.035
TICK_ROTATION: float = 28.0
LEGEND_COLUMNS: int = 2
LEGEND_ANCHOR: tuple[float, float] = (0.45, -0.50)
CHANCE_LEVEL: float = 0.5
SELECTED_GREY: str = "0.25"
NOISE_EDGE_GREY: str = "0.45"
COUNT_GREY: str = "0.25"
GRID_GREY: str = "0.92"
# The printed differences carry no subscript; the legend names AUC_t, so
# it is set at the size that keeps the subscript above 6 pt.
FONT_SIZE_SMALL: float = FONT_SIZE_MIN
FONT_SIZE_FOOTNOTE: float = FONT_SIZE_SCRIPT
FONT_SIZE_TICK: float = FONT_SIZE_MIN
FONT_SIZE_COUNT: float = FONT_SIZE_MIN
FONT_SIZE_CHANCE: float = FONT_SIZE_MIN
# Titles and axis labels, in academic English.
AXIS_TITLE: str = ("Failure-case criterion (pre-specified): "
                   "context-free configurations minus windowed ones")
AXIS_YLABEL: str = "Mean ROC-AUC:\ncontext-free $-$ context"
PROBE_TITLE: str = ("The three configurations examined in each failure "
                    "case")
PROBE_YLABEL: str = "ROC-AUC (FUS)"
TRIVIAL_LABEL: str = "AUC$_t$ (row index)"


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                MAIN CLASS                                 #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class FailureCases(Loggable):
    """
    Output F4: the failure cases of plan §10 (manuscript, Section 6.7).

    The plan fixes, before any result is seen, how a stream becomes a
    failure case: the mean ROC-AUC of the context-free configurations
    minus the mean of the windowed ones has to exceed a margin of 0.01
    points. The top panel draws that difference for every stream,
    together with the margin band and the run-to-run noise reference of
    the stream itself, so that a flag falling inside its own noise
    cannot be read as an effect. The bottom panel draws the three
    configurations that are then examined in each flagged stream, with
    the row-index baseline AUC_t and the chance level, so that a bar at
    or below AUC_t is not read as detection either.

    The selection itself is not made here: it is read, or applied, by
    ``kitnetzday.figures.failure_rule.FailureRule``, which takes it from
    ``stats.json#/failure_cases`` when that document is among the
    inputs and otherwise applies the same rule to the ROC-AUC values of
    the aggregate. No metric is recomputed in either case; a figure
    drawn without ``stats.json`` carries the partial-inputs label.

    Note: the titles, the axis labels and the caption are output
    strings of the pipeline that produced the submitted artefacts and
    are kept verbatim.

    :param naming: Vocabulary of the study, which also carries the
        failure margin of plan §10. None builds the default vocabulary.
        The same vocabulary is handed to the selection rule.
    :type naming: Naming | None
    :param probe_configs: The configurations examined in each failure
        case, as (configuration id, label) pairs in drawing order.
        Exactly three pairs, because the offsets of the grouped bars
        and the grey and hatch tables are fixed to three.
    :type probe_configs: Sequence[tuple[str, str]]
    :param chance_level: ROC-AUC of a random scorer, drawn as a dotted
        line in the bottom panel; in [0, 1].
    :type chance_level: float
    :param font_size_small: Font size of the printed differences, in
        points; strictly positive.
    :type font_size_small: float
    :param font_size_footnote: Font size of the legend of the bottom
        panel, in points; strictly positive.
    :type font_size_footnote: float
    :param rc_params: Matplotlib settings applied while drawing, so
        that the global state of the caller is left untouched.
    :type rc_params: Mapping[str, object]
    :param logger: Logger created by the calling script, or None to
        stay silent.
    :type logger: logging.Logger | None
    :raises TypeError: If naming is not a Naming or None, or logger is
        not a logging.Logger or None.
    :raises ValueError: If probe_configs does not hold exactly three
        pairs, if chance_level is outside [0, 1], or if a font size is
        not strictly positive.
    """

    def __init__(self,
                 naming: Naming | None = None,
                 probe_configs: Sequence[tuple[str, str]] = (
                     PROBE_CONFIGS),
                 chance_level: float = CHANCE_LEVEL,
                 font_size_small: float = FONT_SIZE_SMALL,
                 font_size_footnote: float = FONT_SIZE_FOOTNOTE,
                 rc_params: Mapping[str, object] = PRINT_RC_PARAMS,
                 logger: logging.Logger | None = None) -> None:
        """
        Store the vocabulary and the presentation settings.

        :param naming: Vocabulary of the study, or None for the default.
        :type naming: Naming | None
        :param probe_configs: Three (configuration id, label) pairs.
        :type probe_configs: Sequence[tuple[str, str]]
        :param chance_level: ROC-AUC of a random scorer, in [0, 1].
        :type chance_level: float
        :param font_size_small: Font size of the printed differences.
        :type font_size_small: float
        :param font_size_footnote: Font size of the legend.
        :type font_size_footnote: float
        :param rc_params: Matplotlib settings applied while drawing.
        :type rc_params: Mapping[str, object]
        :param logger: Logger created by the calling script, or None.
        :type logger: logging.Logger | None
        :raises TypeError: If naming or logger has the wrong type.
        :raises ValueError: If a presentation setting is out of range.
        """
        probes = list()
        super().__init__(logger=logger)
        if naming is not None and not isinstance(naming, Naming):
            self._error("[FailureCases.__init__] Input error: the "
                        "parameter naming must be a Naming or None, not "
                        f"{type(naming).__name__}.", TypeError)
        probes = [(str(c), str(name)) for c, name in probe_configs]
        if len(probes) != len(PROBE_GREYS):
            self._error("[FailureCases.__init__] Input error: the "
                        "parameter probe_configs must hold exactly "
                        f"{len(PROBE_GREYS)} pairs, not {len(probes)}.")
        if not 0.0 <= float(chance_level) <= 1.0:
            self._error("[FailureCases.__init__] Input error: the "
                        "parameter chance_level must lie in [0, 1], not "
                        f"{chance_level}.")
        if min(float(font_size_small), float(font_size_footnote)) <= 0.0:
            self._error("[FailureCases.__init__] Input error: the font "
                        "sizes must be strictly positive (>0), not "
                        f"{font_size_small} and {font_size_footnote}.")
        self.naming = naming if naming is not None else Naming(
            logger=logger)
        self.rule = FailureRule(naming=self.naming, logger=logger)
        self.probe_configs = probes
        self.chance_level = float(chance_level)
        self.font_size_small = float(font_size_small)
        self.font_size_footnote = float(font_size_footnote)
        self.rc_params = dict(rc_params)

    def build(self, inputs: Inputs, out: Out) -> bool:
        """
        Draw figure F4 and hand it to the writer, with its caption.

        The method asks FailureRule for the failure cases (from
        ``stats.json`` when it is there, with the rule of plan §10
        applied to the aggregate otherwise), registers the family
        means and the per-cell ROC-AUC values it draws as rows of
        ``numbers_used.md``, and writes the figure as
        ``fig_failure_cases``.

        Note: the caption depends on which branch was really applied.
        With the fallback branch no stream meets the +0.01 criterion, so
        the caption says that the marked stream is only the closest one
        and not a failure case.

        :param inputs: Reader of the aggregate and of ``stats.json``.
        :type inputs: Inputs
        :param out: Writer of the figures and of the rows of
            ``numbers_used.md``.
        :type out: Out
        :return: True once the figure has been written.
        :rtype: bool
        :raises TypeError: If inputs is not an Inputs or out is not an
            Out.
        :raises OSError: If a file of the figure cannot be written.
        """
        ids = list()
        if not isinstance(inputs, Inputs):
            self._error("[FailureCases.build] Input error: the "
                        "parameter inputs must be an Inputs, not "
                        f"{type(inputs).__name__}.", TypeError)
        if not isinstance(out, Out):
            self._error("[FailureCases.build] Input error: the "
                        "parameter out must be an Out, not "
                        f"{type(out).__name__}.", TypeError)
        per, selected, rule, src, missing, notes = self.rule.select(
            inputs)
        # The branch that was really applied decides the text of the
        # caption. With the fallback branch the marked
        # streams do NOT meet the +0.01 criterion and saying otherwise
        # would be false.
        fallback = self.rule.is_fallback(rule)
        self.rule.add_numbers(inputs, out, per, src, ids, missing)
        xs = [x for x in self.naming.datasets if x in per]
        with plt.rc_context(self.rc_params):
            fig, (ax, ax2) = plt.subplots(
                2, 1, figsize=FIGURE_SIZE_INCHES,
                gridspec_kw={"height_ratios": list(HEIGHT_RATIOS),
                             "hspace": HSPACE, **MARGINS})
            self._top_panel(ax, inputs, per, selected, xs)
            self._bottom_panel(ax2, inputs, out, selected, xs, ids)
            # Saved inside the context: pdf.fonttype is read when the
            # file is written, and outside it the fonts are Type 3.
            out.save_figure(
                fig, FIGURE_NAME, self._caption(selected, fallback),
                FIGURE_LABEL, self.naming.plan_tag("fail"),
                sorted(set(ids)), inputs.partial_inputs,
                {"out_id": OUT_ID,
                 "partial_inputs_reasons": inputs.partial_inputs_reasons,
                 "missing": missing,
                 "notes": notes
                 + [f"Selection rule applied: {rule}"]})
        return True

    def __repr__(self) -> str:
        """
        Show the vocabulary size, the margin and the probes.

        :return: Text like "FailureCases(datasets=9,
            failure_margin=0.01, probes=['C10', 'C9', 'C5'])".
        :rtype: str
        """
        return (f"FailureCases(datasets={len(self.naming.datasets)}, "
                f"failure_margin={self.naming.failure_margin}, "
                f"probes={[c for c, _ in self.probe_configs]})")

    # ----- panels ----------------------------------------------------- #
    def _top_panel(self,
                   ax: Axes,
                   inputs: Inputs,
                   per: Mapping[str, Any],
                   selected: Sequence[str],
                   xs: Sequence[str]) -> None:
        """
        Draw the difference bars, the margin band and the noise bands.

        Flagged streams are drawn dark and hatched. Behind every bar
        sits that stream's own noise reference, max(sqrt(2) SD_pool,
        0.01), as a hatched rectangle, and under every bar the number
        of seeds, the size of each family and that reference.

        Note: the vertical limits leave a fixed hole of 0.285 under the
        lowest drawn element for those counts, which take four short
        lines ("noise 0.101" on one line is as wide as the gap between
        two bars at the printed width), the axis has no tick in that
        hole, and the printed value is pushed outside the margin band
        even when the bar is shorter than the margin.

        :param ax: Axes of the top panel.
        :type ax: Axes
        :param inputs: Reader of the noise references and seed counts.
        :type inputs: Inputs
        :param per: Per-stream record of the selection.
        :type per: Mapping[str, Any]
        :param selected: Ids of the flagged streams.
        :type selected: Sequence[str]
        :param xs: Dataset ids with a record, in presentation order.
        :type xs: Sequence[str]
        """
        margin = self.naming.failure_margin
        diff = [per[x]["simple_minus_context"] for x in xs]
        bars = ax.bar(range(len(xs)), diff, width=BAR_WIDTH,
                      color=[(SELECTED_GREY if x in selected
                              else GREY_LIGHT) for x in xs],
                      edgecolor="black")
        for i, x in enumerate(xs):
            if x in selected:
                bars[i].set_hatch("///")
        ax.axhline(0.0, color="black", linewidth=0.9)
        ax.axhspan(-margin, margin, color=GREY_LIGHT, alpha=0.7,
                   zorder=0)
        ax.axhline(margin, color="black", linestyle="--", linewidth=0.8)
        # Per-stream noise band max(sqrt(2) SD_pool, 0.01) behind the
        # bars. Without it a difference of +0.055 in D1 reads as an
        # effect when its own noise reference is 0.101.
        for i, x in enumerate(xs):
            thr = inputs.noise_threshold(x)
            if thr is None:
                continue
            ax.add_patch(Rectangle((i - NOISE_HALF_WIDTH, -thr),
                                   2 * NOISE_HALF_WIDTH, 2 * thr,
                                   facecolor="none",
                                   edgecolor=NOISE_EDGE_GREY,
                                   hatch="\\\\\\", linewidth=0.0,
                                   zorder=2))
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([self.naming.dataset_label(x) for x in xs],
                           rotation=TICK_ROTATION, ha="right",
                           rotation_mode="anchor",
                           fontsize=FONT_SIZE_TICK)
        ax.set_ylabel(AXIS_YLABEL)
        ax.set_title(AXIS_TITLE)
        # Room under the bars for the per-stream counts (seeds and the
        # size of each family).
        lo_v = min([0.0] + [v for v in diff]
                   + [-(inputs.noise_threshold(x) or 0.0) for x in xs])
        hi_v = max([0.0] + [v for v in diff]
                   + [(inputs.noise_threshold(x) or 0.0) for x in xs])
        ax.set_ylim(lo_v - BOTTOM_ROOM, hi_v + TOP_ROOM)
        # No tick in the hole of the counts: there is no bar to read.
        ax.set_yticks([t for t in np.round(np.arange(-1.0, 1.01, 0.1), 1)
                       if lo_v - 0.02 <= t <= hi_v + 0.02])
        for i, v in enumerate(diff):
            xd = xs[i]
            y = ((max(v, margin) + 0.006) if v >= 0
                 else (min(v, -margin) - 0.006))
            ax.text(i, y, f"{v:+.3f}", ha="center",
                    va=("bottom" if v >= 0 else "top"),
                    fontsize=self.font_size_small, clip_on=False,
                    bbox=dict(facecolor="white", edgecolor="none",
                              pad=0.5, alpha=0.85))
            thr = inputs.noise_threshold(xd)
            seeds = inputs.n_seeds(xd)
            ax.text(i, 0.015,
                    f"n={seeds} seed{'' if seeds == 1 else 's'}\n"
                    f"{per[xd].get('n_simple', 0)} vs "
                    f"{per[xd].get('n_context', 0)} cfg\nnoise\n"
                    f"{('%.3f' % thr) if thr is not None else 'n/a'}",
                    transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=FONT_SIZE_COUNT,
                    color=COUNT_GREY, linespacing=1.1,
                    bbox=dict(facecolor="white", edgecolor="none",
                              pad=0.6, alpha=0.85))

    def _bottom_panel(self,
                      ax2: Axes,
                      inputs: Inputs,
                      out: Out,
                      selected: Sequence[str],
                      xs: Sequence[str],
                      ids: list[str]) -> None:
        """
        Draw the examined configurations, AUC_t and the chance line.

        Every finite bar also becomes a row of ``numbers_used.md``,
        so that each per-cell value printed in the figure is
        traceable. A missing cell leaves a gap, not a zero.

        :param ax2: Axes of the bottom panel.
        :type ax2: Axes
        :param inputs: Reader of the per-cell ROC-AUC and of AUC_t.
        :type inputs: Inputs
        :param out: Writer that collects the rows.
        :type out: Out
        :param selected: Ids of the flagged streams, marked with an
            asterisk on the axis.
        :type selected: Sequence[str]
        :param xs: Dataset ids with a record, in presentation order.
        :type xs: Sequence[str]
        :param ids: List of row ids, extended in place.
        :type ids: list[str]
        """
        w = PROBE_WIDTH
        for j, (c, name) in enumerate(self.probe_configs):
            vals = [inputs.auroc(x, c)
                    if inputs.auroc(x, c) is not None else np.nan
                    for x in xs]
            ax2.bar(np.arange(len(xs)) + (j - 1) * w, vals, width=w,
                    color=PROBE_GREYS[j], edgecolor="black",
                    hatch=PROBE_HATCHES[j], label=f"{c} {name}")
            for i, v in enumerate(vals):
                if np.isfinite(v):
                    ids.append(out.add_number(
                        f"R.T1.{xs[i]}.{c}.AUROC_FUS", float(v),
                        inputs.auroc_ptr(xs[i], c), PER_CELL_TAG,
                        inputs.partial_inputs))
        auct = [inputs.trivial(x, "auc_t") for x in xs]
        ax2.plot(np.arange(len(xs)), auct, linestyle="none", marker="_",
                 markersize=18, color="black", label=TRIVIAL_LABEL)
        ax2.axhline(self.chance_level, color="black", linestyle=":",
                    linewidth=0.9)
        ax2.text(-0.45, self.chance_level + 0.015, "chance",
                 fontsize=FONT_SIZE_CHANCE, ha="left",
                 bbox=dict(facecolor="white", edgecolor="none", pad=0.8,
                           alpha=0.85))
        ax2.set_xticks(range(len(xs)))
        ax2.set_xticklabels(
            [(self.naming.dataset_label(x)
              + (" *" if x in selected else "")) for x in xs],
            rotation=TICK_ROTATION, ha="right", rotation_mode="anchor",
            fontsize=FONT_SIZE_TICK)
        ax2.set_ylim(0, 1.02)
        ax2.set_ylabel(PROBE_YLABEL)
        ax2.set_title(PROBE_TITLE)
        ax2.legend(frameon=False, ncol=LEGEND_COLUMNS,
                   fontsize=self.font_size_footnote, loc="upper center",
                   bbox_to_anchor=LEGEND_ANCHOR, columnspacing=2.0,
                   handlelength=2.2)
        ax2.grid(axis="y", color=GRID_GREY, linewidth=0.5)
        ax2.set_axisbelow(True)

    # ----- caption ---------------------------------------------------- #
    def _caption(self,
                 selected: Sequence[str],
                 fallback: bool) -> str:
        """
        Build the caption in academic English, branch included.

        :param selected: Ids of the selected streams, in the order they
            are listed in the caption.
        :type selected: Sequence[str]
        :param fallback: True when no stream met the criterion and the
            closest one is analysed instead.
        :type fallback: bool
        :return: The caption, with its LaTeX markup already written.
        :rtype: str
        """
        sel_txt = (", ".join(self.naming.dataset_label(x)
                             for x in selected)
                   if selected else "none")
        if fallback:
            rule_txt = (
                "No stream met the pre-specified $+0.01$ criterion "
                "(dashed line; the shaded band is the $\\pm 0.01$ "
                "region), so the stream with the smallest difference "
                "is analysed instead, as pre-specified: " + sel_txt +
                ". It is therefore not a failure case, only the "
                "closest one.")
        else:
            rule_txt = (
                "The pre-specified criterion, fixed before looking at "
                "any result, flags a stream when this difference "
                "exceeds $+0.01$ (dashed line; the shaded band is the "
                "$\\pm 0.01$ region); flagged streams are hatched and "
                "marked with an asterisk below: " + sel_txt + ".")
        return ("Failure cases. Top: for each stream, the mean ROC-AUC "
                "of the three context-free configurations (MLP, the "
                "KitNET-style control and the statistical layer) minus "
                "the mean of the eight windowed ones. " + rule_txt +
                " The hatched band behind each bar is that stream's own "
                "run-to-run noise reference, "
                "$\\max(\\sqrt{2}\\,\\mathrm{SD}_{\\mathrm{pool}}, "
                "0.01)$; the number of seeds and the number of "
                "configurations averaged in each family are printed "
                "under each bar. A flag whose difference falls inside "
                "its own noise band selects the stream for inspection "
                "and is not evidence of an effect: with two "
                "seeds per cell these family means carry no usable "
                "interval. Bottom: ROC-AUC of the three configurations "
                "examined in every failure case, together with the "
                "row-index baseline AUC$_t$ of each stream and the "
                "chance level; a value at or below AUC$_t$ is not "
                "evidence of detection. The subsequent per-attack-type "
                "and false-positive analyses are exploratory and "
                "counted as such.")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
