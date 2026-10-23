# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         src/kitnetzday/figures/oracle.py                          #
#   Version:      1.2.1                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Figure A.1 of the manuscript: the oracle results of the   #
#                 submitted regime, with the always-alarm F1 line.          #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from kitnetzday.figures.print_style import (
    FONT_SIZE_LEGEND,
    FONT_SIZE_MIN,
    PRINT_RC_PARAMS,
    PRINT_WIDTH_INCHES,
)
from kitnetzday.io.inputs import Inputs
from kitnetzday.io.outputs import Out
from kitnetzday.naming import FIG7_ORDER, GREY_FILL, Naming
from kitnetzday.utils.loggable import Loggable

# Bar values published for the submitted Fig. 7: nine datasets times
# twelve variants times two metrics (F1 and WindowDiff).
FIG7_EXPECTED_VALUES: int = 216
_METRICS_PER_BAR: int = 2
# Counts that the caption spells out, as the manuscript does; a count
# beyond the list is written in digits.
_COUNT_WORDS: tuple[str, ...] = (
    "zero", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "eleven", "twelve")
# Paradigm of the two packet-by-packet variants, which take no suffix.
_PBP: str = "PbP"
# Names of the variants as the revised manuscript writes them. The CSV of
# the submission has them in capitals and calls the control "KitNET",
# which is "KitNET-style" today. Only the label changes: the key that
# reads the CSV is the same.
_DISPLAY_NAMES: dict[str, str] = {
    "CONV1D": "Conv1D", "CONV2D": "Conv2D", "KitNET": "KitNET-style"}
# Presentation: panel grid, canvas in inches (drawn at the width the
# manuscript includes it at, so the point sizes are the printed ones),
# half the distance between the two bars of a group, and the width of
# one bar.
_GRID_SHAPE: tuple[int, int] = (3, 3)
_FIG_SIZE: tuple[float, float] = (PRINT_WIDTH_INCHES, 7.3)
_BAR_OFFSET: float = 0.2
_BAR_WIDTH: float = 0.38
_HATCH: str = "////"
# Dashed trivial-F1 reference line. Its value goes on the second line of
# the panel title: printed inside the panel it covered the bars.
_TRIVIAL_LINEWIDTH: float = 0.9
# Downward triangle drawn over every F1 bar that does not beat the line.
_MARKER_RISE: float = 0.06
_MARKER_CEIL: float = 1.02
_MARKER_SIZE: float = 3.4
# Axes, grid, legend and super title.
_TICK_LABEL_SIZE: float = FONT_SIZE_MIN
_Y_MAX: float = 1.05
_Y_TICKS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
_TITLE_SIZE: float = 8.0
_GRID_GREY: str = "0.9"
_GRID_LINEWIDTH: float = 0.5
_LEGEND_COLUMNS: int = 2
_LEGEND_ANCHOR: tuple[float, float] = (0.5, 0.972)
_SUPTITLE_Y: float = 0.995
_SUPTITLE_SIZE: float = 8.5
# Margins of the panel grid, in figure fractions.
_MARGINS: dict[str, float] = {
    "left": 0.075, "right": 0.995, "bottom": 0.085, "top": 0.868,
    "wspace": 0.07, "hspace": 1.02}
# Legend entries without a bar of their own.
_LEGEND_TRIVIAL: str = "trivial F1 (always-alarm classifier)"
_LEGEND_BELOW: str = "F1 bar not above the trivial F1"
_LEGEND_KMD: str = "* reported detector is the centroid detector (KMD)"
_SUPTITLE: str = ("Figure 7 of the submitted version, reproduced: "
                  "threshold and detector selected with evaluation "
                  "labels (oracle)")
# Identity of the generated output.
_FIG_NAME: str = "fig_appendix_fig7_oracle"
_FIG_LABEL: str = "fig:appendix-fig7"
_OUT_ID: str = "F3"
_NUMBER_ID: str = "R.FIG7.AGG.BARS_LE_TRIVIAL"


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                MAIN CLASS                                 #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class OracleFigure(Loggable):
    """
    Figure A.1 of the manuscript: oracle results of the submitted regime.

    Figure 7 of the submission reported, per dataset and variant, the
    smoothed F1 and WindowDiff of whichever of the two detectors scored
    better, with a decision threshold picked on the whole stream using
    the evaluation labels. Both choices used labels of the evaluation
    period, so the values are a record of that regime: they are neither
    bounds on the revised regime nor comparable with the main results.
    The manuscript keeps them in its Appendix A (Figure A.1), says so in
    the caption, and adds the F1 of the trivial always-alarm classifier
    as a dashed reference, with a downward triangle over every bar that
    does not beat it.

    The bars are not recomputed: they are the values published back then
    (``fig7_valores_publicados.csv``, read by
    ``kitnetzday.io.legacy.parse_fig7_csv``), and the trivial reference
    comes from ``fig7_trivial_f1.csv``, read by
    ``kitnetzday.io.legacy.parse_f1_trivial_smoothed``.

    :param naming: Vocabulary of the labels, or None for the default
        nine datasets of the study.
    :type naming: Naming | None
    :param fig7_order: The twelve (variant, paradigm) pairs in the
        order the submitted figure draws them; non-empty.
    :type fig7_order: Sequence[tuple[str, str]]
    :param rc_params: Matplotlib settings of the house style.
    :type rc_params: Mapping[str, object]
    :param grey_fill: Grey level of the F1 bars, a matplotlib grey
        string between "0" (black) and "1" (white).
    :type grey_fill: str
    :param expected_values: Bar values the published CSV should hold,
        216 in the submission; strictly positive. Any other count is
        declared in the caption.
    :type expected_values: int
    :param logger: Logger created by the calling script, or None to stay
        silent.
    :type logger: logging.Logger | None
    :raises TypeError: If naming is not a Naming or None, or logger is
        not a logging.Logger or None.
    :raises ValueError: If fig7_order is empty, if expected_values is
        not strictly positive, or if the 3 x 3 panel grid of the
        submitted figure does not cover every dataset.
    """

    def __init__(self,
                 naming: Naming | None = None,
                 fig7_order: Sequence[tuple[str, str]] = tuple(
                     FIG7_ORDER),
                 rc_params: Mapping[str, object] = PRINT_RC_PARAMS,
                 grey_fill: str = GREY_FILL,
                 expected_values: int = FIG7_EXPECTED_VALUES,
                 logger: logging.Logger | None = None) -> None:
        """
        Store the vocabulary and check that the panel grid fits.

        :param naming: Vocabulary of the labels, or None.
        :type naming: Naming | None
        :param fig7_order: The twelve (variant, paradigm) pairs.
        :type fig7_order: Sequence[tuple[str, str]]
        :param rc_params: Matplotlib settings of the house style.
        :type rc_params: Mapping[str, object]
        :param grey_fill: Grey level of the F1 bars.
        :type grey_fill: str
        :param expected_values: Bar values the CSV should hold.
        :type expected_values: int
        :param logger: Logger created by the calling script, or None.
        :type logger: logging.Logger | None
        :raises TypeError: If naming or logger are of the wrong type.
        :raises ValueError: If the vocabulary is empty or too large.
        """
        super().__init__(logger=logger)
        panels = _GRID_SHAPE[0] * _GRID_SHAPE[1]
        if naming is not None and not isinstance(naming, Naming):
            self._error("[OracleFigure.__init__] Input error: the "
                        "parameter naming must be a Naming or None, not "
                        f"{type(naming).__name__}.", TypeError)
        if not fig7_order:
            self._error("[OracleFigure.__init__] Input error: the "
                        "parameter fig7_order must hold at least one "
                        "(variant, paradigm) pair, not an empty "
                        "sequence.")
        if int(expected_values) <= 0:
            self._error("[OracleFigure.__init__] Input error: the "
                        "parameter expected_values must be strictly "
                        f"positive (>0), not {expected_values}.")
        self.naming = naming if naming is not None else Naming(
            logger=logger)
        if len(self.naming.datasets) > panels:
            self._error("[OracleFigure.__init__] Input error: the "
                        "parameter naming must hold at most "
                        f"{panels} datasets, the panels of the "
                        f"submitted figure, not "
                        f"{len(self.naming.datasets)}.")
        self.fig7_order = [(str(v), str(p)) for v, p in fig7_order]
        self.rc_params = dict(rc_params)
        self.grey_fill = str(grey_fill)
        self.grid_shape = _GRID_SHAPE
        self.fig_size = _FIG_SIZE
        self.expected_values = int(expected_values)

    def build(self,
              inputs: Inputs,
              out: Out,
              fig7: Mapping[tuple[str, str, str], Mapping[str, Any]],
              n_values: int,
              f1_trivial: Mapping[str, float]) -> bool:
        """
        Draw the oracle figure and register its one derived number.

        The figure, its caption and its row of ``numbers_used.md``
        always carry the oracle label of the plan, because the bars
        were picked with evaluation labels. When the published CSV is
        short, or the trivial reference misses a dataset, the caption
        declares the numerator and the denominator actually read
        instead of the 216 bar values, and the output carries the
        partial-inputs label with that reason. With the shipped files
        the 216 values and the nine references are read, so the label
        is not written.

        Note: the inputs object is not read. It stays in the signature
        so that every figure is built the same way and so that the
        caller cannot draw one without having loaded and hashed the
        inputs first.

        Note: the bars below the trivial reference are counted only for
        the datasets whose reference could be read, and only over finite
        F1 values, so the counters of the caption run over the bars
        actually covered, not over the twelve bars of every panel.

        Note: when the published values are not available the figure
        is not generated; the reason is logged and does not reach the
        manifest.

        :param inputs: Loaded and hashed inputs; not read, see above.
        :type inputs: Inputs
        :param out: Output directory that writes the figure and
            registers its number.
        :type out: Out
        :param fig7: Published values keyed by (dataset id, variant,
            paradigm), each with the keys "F1", "WindowDiff" and
            "detector", as ``parse_fig7_csv`` returns them. An empty
            mapping skips the figure.
        :type fig7: Mapping[tuple[str, str, str], Mapping[str, Any]]
        :param n_values: Rows read from the published CSV, 216 in the
            submission; zero or more.
        :type n_values: int
        :param f1_trivial: Dataset id -> F1 of the trivial always-alarm
            classifier against the smoothed labels, in [0, 1], as
            ``parse_f1_trivial_smoothed`` returns it; it may cover only
            some datasets.
        :type f1_trivial: Mapping[str, float]
        :return: True when the figure was written, False when the
            published values were not available.
        :rtype: bool
        :raises TypeError: If any argument is of the wrong type.
        :raises ValueError: If n_values is negative.
        :raises OSError: If the figure cannot be written.
        """
        ids = list()
        missing = list()
        notes = list()
        reasons = list()
        partial_f3 = False
        pointer = ""
        below = 0
        covered = 0
        if not isinstance(inputs, Inputs):
            self._error("[OracleFigure.build] Input error: the parameter"
                        " inputs must be an Inputs, not "
                        f"{type(inputs).__name__}.", TypeError)
        if not isinstance(out, Out):
            self._error("[OracleFigure.build] Input error: the parameter"
                        f" out must be an Out, not "
                        f"{type(out).__name__}.", TypeError)
        if not isinstance(fig7, Mapping):
            self._error("[OracleFigure.build] Input error: the parameter"
                        " fig7 must be a mapping of published values, "
                        f"not {type(fig7).__name__}.", TypeError)
        if isinstance(n_values, bool) or not isinstance(n_values, int):
            self._error("[OracleFigure.build] Input error: the parameter"
                        " n_values must be an integer, not "
                        f"{type(n_values).__name__}.", TypeError)
        if n_values < 0:
            self._error("[OracleFigure.build] Input error: the parameter"
                        f" n_values must be zero or more (>=0), not "
                        f"{n_values}.")
        if not isinstance(f1_trivial, Mapping):
            self._error("[OracleFigure.build] Input error: the parameter"
                        " f1_trivial must be a mapping of dataset id to "
                        f"trivial F1, not {type(f1_trivial).__name__}.",
                        TypeError)
        if not fig7:
            missing.append("fig7_valores_publicados.csv not available: "
                           "the oracle figure is not generated")
            self._warning("[OracleFigure.build] Input warning: the "
                          "parameter fig7 must hold the published values"
                          " of the submitted figure, and it is empty. "
                          "The oracle figure is not generated.")
            return False
        notes.append(f"{n_values} values read from "
                     "fig7_valores_publicados.csv "
                     f"({self.expected_values} = "
                     f"{len(self.naming.datasets)} x "
                     f"{len(self.fig7_order)} x {_METRICS_PER_BAR}).")
        if not f1_trivial:
            missing.append("the smoothed trivial F1 could not be read "
                           "from fig7_trivial_f1.csv")
        if n_values != self.expected_values:
            missing.append("fig7_valores_publicados.csv holds "
                           f"{n_values} values ({self.expected_values} "
                           "expected): the caption says so")
            partial_f3 = True
        if len(f1_trivial) != len(self.naming.datasets):
            missing.append("smoothed trivial F1 for "
                           f"{len(f1_trivial)} of "
                           f"{len(self.naming.datasets)} datasets: the "
                           "count of bars at or below it covers only "
                           "those datasets")
            partial_f3 = True
        figure, below, covered = self._draw(fig7, f1_trivial)
        pointer = ("count over data/fig7_valores_publicados.csv "
                   "against the smoothed trivial F1 of "
                   "data/fig7_trivial_f1.csv")
        ids.append(out.add_number(
            _NUMBER_ID, below, pointer,
            self.naming.plan_tag("expl_preplan"), partial_f3))
        if partial_f3:
            reasons.append(f"incomplete input: {n_values}/"
                           f"{self.expected_values} values, "
                           f"{len(f1_trivial)}/"
                           f"{len(self.naming.datasets)} trivial F1 "
                           "values")
        # pdf.fonttype is read when the file is written, not when the
        # figure is drawn: saving outside this context embeds Type 3
        # fonts.
        with plt.rc_context(self.rc_params):
            out.save_figure(
                figure, _FIG_NAME,
                self._caption(n_values, below, covered), _FIG_LABEL,
                self.naming.plan_tag("o1") + " "
                + self.naming.plan_tag("expl_preplan"),
                sorted(set(ids)), partial_f3,
                {"out_id": _OUT_ID, "partial_inputs_reasons": reasons,
                 "missing": missing, "notes": notes})
        return True

    # ----- drawing --------------------------------------------------- #
    def _draw(self,
              fig7: Mapping[tuple[str, str, str], Mapping[str, Any]],
              f1_trivial: Mapping[str, float],
              ) -> tuple[Figure, int, int]:
        """
        Draw one panel per dataset and count the bars below trivial.

        :param fig7: Published values keyed by (dataset, variant,
            paradigm).
        :type fig7: Mapping[tuple[str, str, str], Mapping[str, Any]]
        :param f1_trivial: Dataset id -> trivial F1, in [0, 1].
        :type f1_trivial: Mapping[str, float]
        :return: The finished figure, how many covered F1 bars do not
            exceed their trivial reference, and how many it covers.
        :rtype: tuple[Figure, int, int]
        """
        below = 0
        covered = 0
        rows, columns = self.grid_shape
        with plt.rc_context(self.rc_params):
            figure, axes = plt.subplots(rows, columns,
                                        figsize=self.fig_size)
            for k, dataset in enumerate(self.naming.datasets):
                ax = axes[k // columns][k % columns]
                panel = self._draw_panel(ax, dataset, fig7,
                                         f1_trivial.get(dataset))
                below += panel[0]
                covered += panel[1]
                # The nine panels share the scale: only the left column
                # prints it.
                ax.tick_params(axis="y", labelleft=(k % columns == 0))
                if k % columns == 0:
                    ax.set_ylabel("Score")
            handles, labels = axes[0][0].get_legend_handles_labels()
            handles += [
                Line2D(list(), list(), color="black", linestyle="--",
                       linewidth=_TRIVIAL_LINEWIDTH),
                Line2D(list(), list(), color="black", linestyle="none",
                       marker="v", markersize=_MARKER_SIZE),
                Line2D(list(), list(), linestyle="none")]
            labels += [_LEGEND_TRIVIAL, _LEGEND_BELOW, _LEGEND_KMD]
            figure.suptitle(_SUPTITLE, y=_SUPTITLE_Y,
                            fontsize=_SUPTITLE_SIZE)
            figure.legend(handles, labels, loc="upper center",
                          ncol=_LEGEND_COLUMNS, frameon=False,
                          fontsize=FONT_SIZE_LEGEND,
                          bbox_to_anchor=_LEGEND_ANCHOR,
                          columnspacing=1.6, handlelength=2.2,
                          borderaxespad=0.0, labelspacing=0.35)
            figure.subplots_adjust(**_MARGINS)
        return figure, below, covered

    def _draw_panel(self,
                    ax: Axes,
                    dataset: str,
                    fig7: Mapping[tuple[str, str, str],
                                  Mapping[str, Any]],
                    trivial: float | None) -> tuple[int, int]:
        """
        Draw the twelve bar pairs of one dataset and its trivial line.

        A variant with no published value is drawn as a NaN, which
        matplotlib leaves as a gap, and is not counted anywhere. The
        value of the trivial F1 is written on the second line of the
        panel title, without a subscript.

        :param ax: Axes of this dataset panel.
        :type ax: Axes
        :param dataset: Dataset id such as "D8".
        :type dataset: str
        :param fig7: Published values keyed by (dataset, variant,
            paradigm).
        :type fig7: Mapping[tuple[str, str, str], Mapping[str, Any]]
        :param trivial: Trivial F1 of this dataset, in [0, 1], or None
            when the archived table does not give it; without it there is
            no line, no triangle and no count for this panel.
        :type trivial: float | None
        :return: How many finite F1 bars of this panel do not exceed the
            trivial reference, and how many bars it covers.
        :rtype: tuple[int, int]
        """
        below = 0
        covered = 0
        f1s = list()
        wds = list()
        kmd = list()
        for variant, paradigm in self.fig7_order:
            rec = fig7.get((dataset, variant, paradigm), dict())
            f1s.append(rec.get("F1", np.nan))
            wds.append(rec.get("WindowDiff", np.nan))
            kmd.append(rec.get("detector") == "KMD")
        pos = np.arange(len(self.fig7_order))
        ax.bar(pos - _BAR_OFFSET, f1s, width=_BAR_WIDTH,
               color=self.grey_fill, edgecolor="black",
               label="F1 (Smooth)")
        ax.bar(pos + _BAR_OFFSET, wds, width=_BAR_WIDTH, color="white",
               edgecolor="black", hatch=_HATCH,
               label="WindowDiff (Smooth)")
        if trivial is not None:
            ax.axhline(trivial, color="black", linestyle="--",
                       linewidth=_TRIVIAL_LINEWIDTH)
            for i, value in enumerate(f1s):
                if not np.isfinite(value):
                    continue
                covered += 1
                if value <= trivial:
                    below += 1
                    ax.plot(pos[i] - _BAR_OFFSET,
                            min(value + _MARKER_RISE, _MARKER_CEIL),
                            marker="v", color="black",
                            markersize=_MARKER_SIZE)
        ax.set_xticks(pos)
        # The reported detector is an oracle choice and is marked with
        # an asterisk on the label ("LSTM-TSR*"): the hatching of the
        # submission is withdrawn (R1.5).
        ax.set_xticklabels([self._bar_label(v, p)
                            + ("*" if kmd[i] else "")
                            for i, (v, p) in enumerate(self.fig7_order)],
                           rotation=90, fontsize=_TICK_LABEL_SIZE)
        ax.set_ylim(0, _Y_MAX)
        ax.set_yticks(list(_Y_TICKS))
        ax.set_title(self.naming.dataset_label(dataset)
                     + (f"\ntrivial F1 = {trivial:.2f}"
                        if trivial is not None else ""),
                     fontsize=_TITLE_SIZE, linespacing=1.15, pad=3.0)
        ax.grid(axis="y", color=_GRID_GREY, linewidth=_GRID_LINEWIDTH)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", length=2.5, pad=2.0)
        return below, covered

    @staticmethod
    def _bar_label(variant: str, paradigm: str) -> str:
        """
        Label of one bar pair of the submitted Fig. 7.

        :param variant: Variant as the published CSV names it, for
            example "CONV1D" or "KitNET".
        :type variant: str
        :param paradigm: "AR", "TSR", or "PbP" for the two
            packet-by-packet variants, which take no suffix.
        :type paradigm: str
        :return: Text like "Conv1D-TSR", "MLP-AR", "KitNET-style" or
            "STAT".
        :rtype: str
        """
        name = _DISPLAY_NAMES.get(variant, variant)
        return name if paradigm == _PBP else f"{name}-{paradigm}"

    # ----- caption --------------------------------------------------- #
    def _caption(self, n_values: int, below: int, covered: int) -> str:
        """
        Caption of the generated fragment: that of Figure A.1.

        The text states what the caption of Figure A.1 of the
        manuscript states, in its own wording, with its counts taken
        from what was read. It states what the figure
        is, that both the threshold and the detector were chosen with
        evaluation labels, what the two kinds of bars are, how many
        values were read and how many bars do not beat the trivial
        classifier. The two sections it names are those of the
        manuscript. When fewer values than expected were read, the
        caption gives both counts ("4 of the 216 bar values").

        :param n_values: Rows actually read from the published CSV.
        :type n_values: int
        :param below: Covered F1 bars that do not exceed the trivial
            reference.
        :type below: int
        :param covered: F1 bars the trivial reference covers.
        :type covered: int
        :return: The caption, with its LaTeX commands written out.
        :rtype: str
        """
        values = ""
        values = (f"the {n_values}" if n_values == self.expected_values
                  else f"{n_values} of the {self.expected_values}")
        return ("Oracle results of the submitted regime, redrawn from "
                + values + " bar values of its figure ("
                + self._count_word(len(self.naming.datasets))
                + " datasets, "
                + self._count_word(len(self.fig7_order)) + " variants, "
                + self._count_word(_METRICS_PER_BAR) + " metrics); not "
                "comparable with the main results. "
                "For every dataset and variant the threshold was chosen "
                "on the whole stream with the evaluation labels, and "
                "the detector shown is the one with the lower "
                "WindowDiff, again chosen with those labels: an "
                "asterisk on the axis label marks the Centroid Detector "
                "(KMD), and the other bars show the Distribution "
                "Detector (MVD). "
                "Grey bars are the F1 and hatched bars the WindowDiff, "
                "both against the extended labels of that regime "
                "(`Smooth' in the legend), the latter from the "
                "submitted, non-standard implementation "
                "($k=\\lfloor 2\\cdot 10^{10}/N \\rfloor$ records, "
                "window sums instead of boundary counts, 8-bit "
                "overflow; Section~5.6). "
                "MLP-AR and MLP-TSR are replicates of one per-record "
                "model (Section~4.3.2). "
                "The dashed line is the F1 of the always-alarm "
                "classifier against the same extended labels, with its "
                "value under each panel title, and downward triangles "
                "mark the " + str(below) + " of the " + str(covered)
                + " F1 bars that do not exceed it, an exploratory "
                "count.")

    @staticmethod
    def _count_word(count: int) -> str:
        """
        Write a small count as the caption spells it.

        :param count: Number of datasets, variants or metrics; zero or
            more.
        :type count: int
        :return: Text like "nine" or "twelve"; the digits ("13") for a
            count beyond twelve or below zero.
        :rtype: str
        """
        if 0 <= count < len(_COUNT_WORDS):
            return _COUNT_WORDS[count]
        return str(count)

    def __repr__(self) -> str:
        """
        Show the geometry of the figure and the expected bar count.

        :return: Text like "OracleFigure(datasets=9, bars=12,
            grid=(3, 3), expected_values=216)".
        :rtype: str
        """
        return (f"OracleFigure(datasets={len(self.naming.datasets)}, "
                f"bars={len(self.fig7_order)}, grid={self.grid_shape}, "
                f"expected_values={self.expected_values})")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
