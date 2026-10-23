# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         src/kitnetzday/figures/heatmap.py                         #
#   Version:      1.1.2                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Main figure F1: ROC-AUC heat map of dataset by            #
#                 configuration with the trivial AUC_t column.              #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from kitnetzday.figures.print_style import (
    FONT_SIZE_MIN,
    FONT_SIZE_SCRIPT,
    PRINT_RC_PARAMS,
    PRINT_WIDTH_INCHES,
)
from kitnetzday.io.inputs import Inputs
from kitnetzday.io.outputs import Out
from kitnetzday.utils.loggable import Loggable

# Presentation constants. The figure is drawn at the width the manuscript
# includes it at, so every point size below is the printed one (R1.10).
# The size, the AUC_t ratio, bottom, top and the two font sizes also live
# in src/config.json (section "figures"); the scripts in bin/ read that
# file and hand the values to the constructor, library code never reads
# it.
FIGURE_SIZE_INCHES: tuple[float, float] = (PRINT_WIDTH_INCHES, 4.0)
# Width of the AUC_t column in units of one configuration column; the
# left panel always gets one unit per configuration, as the original did
# (config.json keeps the pair [11, 1.4], of which only 1.4 is free here).
TRIVIAL_WIDTH_RATIO: float = 1.4
# Width of the colour bar, in the same units.
COLORBAR_WIDTH_RATIO: float = 0.36
PANEL_SPACING: float = 0.045
AXES_LEFT: float = 0.202
AXES_RIGHT: float = 0.925
AXES_BOTTOM: float = 0.243
AXES_TOP: float = 0.935
CHANCE_LEVEL: float = 0.5
# Above this ROC-AUC the "Greys" background is dark enough for white text.
WHITE_TEXT_ABOVE: float = 0.62
# Tick labels, family names and row-group name: lettering without
# subscripts. The footnote and the AUC_t tick label carry a subscript.
FONT_SIZE_SMALL: float = FONT_SIZE_MIN
FONT_SIZE_FOOTNOTE: float = FONT_SIZE_SCRIPT
FONT_SIZE_CELL: float = FONT_SIZE_MIN
FONT_SIZE_NA: float = FONT_SIZE_MIN
FONT_SIZE_COLORBAR_LABEL: float = 8.0
# Family line under the x tick labels, as a fraction of the axes height,
# how far each line stops short of the edge of its columns (in columns),
# the gap down to the family name, and the room the x label leaves for
# them, in points.
GROUP_LINE_Y: float = -0.127
GROUP_LINE_INSET: float = 0.38
GROUP_NAME_DROP: float = 0.012
XLABEL_PAD: float = 17.0
# Name of the block of configurations without temporal context.
CONTEXT_FREE_GROUP: str = "no temporal context"
# Rows whose name starts with this prefix share it: it is written once,
# rotated, next to a bracket placed at this fraction of the axes width.
ROW_GROUP_PREFIX: str = "KNAD "
ROW_GROUP_X: float = -0.257
FOOTNOTE_XY: tuple[float, float] = (0.004, 0.012)
FIGURE_NAME: str = "fig_main_auroc_heatmap"
FIGURE_LABEL: str = "fig:auroc-heatmap"
OUT_ID: str = "F1"
# Label of the analysis plan attached to every per-cell value. It is an
# output string of the pipeline and stays in Spanish, as the plan has it.
CELL_TAG: str = ("[CONFIRMATORIO plan §3.1 — valor por celda, "
                 "descriptivo, sin test]")
NOTES: tuple[str, ...] = (
    "Takes the place of Fig. 7 of the submitted version as the main "
    "figure (R1.10).",
    "Values = the 'main' set of aggregate.py (plan §4.5 and §12.2, with "
    "the intention-to-treat imputation of diverged seeds, if any).")
AXIS_TITLE: str = ("ROC-AUC of the fused detector on the evaluation"
                   " segment (mean over {seeds})")
AXIS_XLABEL: str = "Configuration (ensemble variant)"
TRIVIAL_TICK: str = "AUC$_t$\n(row\nindex)"
COLORBAR_LABEL: str = "ROC-AUC (0.5 = chance, marked on the bar)"
# Symbol footnote; the six-space gap separates the two legends.
SYMBOL_FOOTNOTE: str = (
    "* ROC-AUC $\\leq$ 0.5 (chance, hatched)      "
    "† ROC-AUC $\\leq$ max(AUC$_t$, 1 $-$ AUC$_t$) (row-index baseline)")
#: Second line of the footnote, only when at least one cell is imputed.
SYMBOL_FOOTNOTE_IMPUTED: str = (
    "\n‡ cell with an intention-to-treat imputed seed")

CAPTION_HEAD: str = (
    "ROC-AUC of the fused detector (FUS) on the evaluation segment of "
    "each stream, for the eleven analysis configurations (mean over the "
    "two seeds of the nine stochastic configurations C1--C9; C10 and "
    "C11 have no random initialisation and are run once per stream).")
CAPTION_TAIL: str = (
    " Darker is higher. Cells with ROC-AUC $\\leq 0.5$ (chance) are "
    "hatched and marked with $*$; cells that do not exceed the "
    "row-index baseline of their stream, read as $\\max($AUC$_t$, 1 $-$ "
    "AUC$_t)$ because a detector returning the decreasing index is "
    "equally free, are marked with $\\dagger$; the rightmost column "
    "shows AUC$_t$ itself, as in the main results table. "
    "The threshold-free primary metric and the "
    "two trivial references are shown together so that no value is read "
    "without its baseline. Configurations: C1--C8 use a temporal "
    "context window, C9 (MLP) ignores it, C10 is the KitNET-style "
    "control and C11 the statistical layer. No detector or threshold is "
    "selected with evaluation labels in this figure.")
CAPTION_NO_IMPUTED: str = (
    " No cell required the intention-to-treat imputation of a diverged "
    "seed.")
CAPTION_IMPUTED: str = (
    " Cells marked with $\\ddagger$ ({n} of {total}) contain a seed "
    "that diverged and is imputed intention-to-treat as specified in "
    "the analysis plan, so their mean is not a plain average of the "
    "observed runs.")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                AUXILIARY                                  #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def _is_positive(value: Any = None) -> bool:
    """
    Tell whether a value is a strictly positive real number.

    Booleans are rejected: ``True`` would otherwise pass as the number
    1 and silently become a font size or a figure width.

    :param value: Value to check, of any type.
    :type value: Any
    :return: True for a non-boolean int or float greater than zero.
    :rtype: bool
    """
    return (isinstance(value, (int, float))
            and not isinstance(value, bool) and value > 0.0)


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                MAIN CLASS                                 #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class Heatmap(Loggable):
    """
    Main figure of the revised manuscript: the ROC-AUC heat map.

    One row per dataset, one column per analysis configuration, and the
    mean ROC-AUC of the fused detector printed inside every cell. The
    trivial row-index baseline AUC$_t$ of each stream is a separate
    panel glued to the right of the map, so that no value is read
    without its baseline, and the colour bar carries a line at the
    chance level. Cells at or below chance are hatched and marked with
    an asterisk, cells that do not beat the row-index reference
    max(AUC$_t$, 1 - AUC$_t$) with a dagger, cells
    whose mean includes an intention-to-treat imputed seed with a double
    dagger, and missing cells are white with a cross hatch and "n/a".
    This figure replaces the submitted Fig. 7 as the main result
    (comment R1.10): it uses no threshold and selects no detector with
    evaluation labels.

    The figure is drawn at the width the manuscript prints it, 170 mm,
    with no lettering under 7.5 pt (R1.10 asked for larger fonts). At
    that width a column is 28 pt wide, so the x axis has two levels:
    the id and the paradigm under each column ("C2" over "TSR") and the
    family under each pair ("Conv1D"), and the rows that share the
    prefix "KNAD " show it once, rotated, next to a bracket. The number
    of seeds goes in the title when it is the same for every dataset.

    The class only reads what the aggregation already computed, through
    ``Inputs``, and writes through ``Out``; it computes no metric. The
    caller must have selected a non-interactive matplotlib backend
    ("Agg"), as the original did. The scripts in ``bin/`` read
    ``src/config.json``.

    :param size_inches: Figure size (width, height) in inches, both
        strictly positive; (6.69, 4.0) in the paper.
    :type size_inches: tuple[float, float]
    :param trivial_width_ratio: Width of the AUC$_t$ panel in units of
        one configuration column; strictly positive, 1.4 in the paper.
    :type trivial_width_ratio: float
    :param bottom: Bottom of the axes area in figure fractions, in
        [0, 1); 0.243 in the paper, to leave room for the two-level
        tick labels and the symbol footnote.
    :type bottom: float
    :param top: Top of the axes area in figure fractions, in
        (bottom, 1]; 0.935 in the paper.
    :type top: float
    :param chance_level: ROC-AUC of a coin flip, 0.5, in [0, 1]. It is
        the cut of the asterisk mark and of the hatch, and the line
        drawn across the colour bar.
    :type chance_level: float
    :param font_size_small: Point size of the tick labels, the family
        names and the row-group name; strictly positive, 7.5 in the
        paper.
    :type font_size_small: float
    :param font_size_footnote: Point size of the lettering that carries
        a subscript, the symbol footnote and the AUC$_t$ tick label;
        strictly positive, 9.0 in the paper, so that the subscript
        (0.7 of it) stays above 6 pt.
    :type font_size_footnote: float
    :param rc_params: matplotlib settings for ``plt.rc_context``, used
        while drawing and while saving.
    :type rc_params: Mapping[str, Any]
    :param logger: Logger of the calling script, or None to stay silent.
    :type logger: logging.Logger | None
    :raises TypeError: If logger is not a logging.Logger or None, or if
        size_inches is not a pair of numbers.
    :raises ValueError: If any size is not strictly positive, if bottom
        and top are not an increasing pair inside [0, 1], or if
        chance_level falls outside [0, 1].
    """

    def __init__(self,
                 size_inches: tuple[float, float] = FIGURE_SIZE_INCHES,
                 trivial_width_ratio: float = TRIVIAL_WIDTH_RATIO,
                 bottom: float = AXES_BOTTOM,
                 top: float = AXES_TOP,
                 chance_level: float = CHANCE_LEVEL,
                 font_size_small: float = FONT_SIZE_SMALL,
                 font_size_footnote: float = FONT_SIZE_FOOTNOTE,
                 rc_params: Mapping[str, Any] = PRINT_RC_PARAMS,
                 logger: logging.Logger | None = None) -> None:
        """
        Store the presentation settings after checking them.

        :param size_inches: Figure size (width, height) in inches.
        :type size_inches: tuple[float, float]
        :param trivial_width_ratio: Width of the AUC$_t$ panel.
        :type trivial_width_ratio: float
        :param bottom: Bottom of the axes area, in [0, 1).
        :type bottom: float
        :param top: Top of the axes area, in (bottom, 1].
        :type top: float
        :param chance_level: ROC-AUC of a coin flip, in [0, 1].
        :type chance_level: float
        :param font_size_small: Point size of the tick labels.
        :type font_size_small: float
        :param font_size_footnote: Point size of the footnote.
        :type font_size_footnote: float
        :param rc_params: matplotlib settings used while drawing.
        :type rc_params: Mapping[str, Any]
        :param logger: Logger created by the calling script, or None.
        :type logger: logging.Logger | None
        :raises TypeError: If logger or size_inches has the wrong type.
        :raises ValueError: If a size, a fraction or a level is out of
            range.
        """
        super().__init__(logger=logger)
        if not isinstance(size_inches, (tuple, list)) or \
                len(size_inches) != 2:
            self._error("[Heatmap.__init__] Input error: the parameter "
                        "size_inches must be a pair (width, height) in "
                        f"inches, not {size_inches!r}.", TypeError)
        if not all(_is_positive(v) for v in size_inches):
            self._error("[Heatmap.__init__] Input error: both values of "
                        "the parameter size_inches must be strictly "
                        f"positive numbers, not {size_inches!r}.")
        if not _is_positive(trivial_width_ratio):
            self._error("[Heatmap.__init__] Input error: the parameter "
                        "trivial_width_ratio must be a strictly "
                        f"positive number, not {trivial_width_ratio!r}.")
        if not 0.0 <= bottom < top <= 1.0:
            self._error("[Heatmap.__init__] Input error: the parameters "
                        "bottom and top must satisfy 0 <= bottom < top "
                        f"<= 1, not bottom={bottom!r} and top={top!r}.")
        if not 0.0 <= chance_level <= 1.0:
            self._error("[Heatmap.__init__] Input error: the parameter "
                        "chance_level must be a ROC-AUC in [0, 1], not "
                        f"{chance_level!r}.")
        if not _is_positive(font_size_small) or \
                not _is_positive(font_size_footnote):
            self._error("[Heatmap.__init__] Input error: the font sizes "
                        "must be strictly positive point sizes, not "
                        f"font_size_small={font_size_small!r} and "
                        f"font_size_footnote={font_size_footnote!r}.")
        self.size_inches = (float(size_inches[0]), float(size_inches[1]))
        self.trivial_width_ratio = float(trivial_width_ratio)
        self.bottom = float(bottom)
        self.top = float(top)
        self.chance_level = float(chance_level)
        self.font_size_small = float(font_size_small)
        self.font_size_footnote = float(font_size_footnote)
        self.rc_params = dict(rc_params)

    def build(self, inputs: Inputs, out: Out) -> bool:
        """
        Collect the values, draw the figure and write it out.

        The cell values and the AUC$_t$ column are also registered as
        rows of ``numbers_used.md``, so every printed number is traceable,
        and the caption states how many cells carry an imputed seed.
        Cells missing from the aggregate are listed in the "missing"
        field of the output record instead of stopping the figure.

        :param inputs: Reader of the aggregate and of the statistics.
        :type inputs: Inputs
        :param out: Output directory and register of what it holds.
        :type out: Out
        :return: True, always, once the figure has been written; the
            original signalled "figure generated" this way.
        :rtype: bool
        :raises TypeError: If inputs is not an Inputs, or out an Out.
        :raises ValueError: If the vocabulary has no dataset or none.
        :raises OSError: If a file cannot be written.
        """
        values = np.empty(0)
        imputed = np.empty(0, dtype=bool)
        auc_t = np.empty(0)
        ids: list[str] = list()
        missing: list[str] = list()
        imputation = ""
        n_imputed = 0
        n_cells = 0
        if not isinstance(inputs, Inputs) or not isinstance(out, Out):
            self._error("[Heatmap.build] Input error: the parameters "
                        "inputs and out must be an Inputs and an Out "
                        f"instance, not {type(inputs).__name__} and "
                        f"{type(out).__name__}.", TypeError)
        if not inputs.naming.datasets or not inputs.naming.configurations:
            self._error("[Heatmap.build] Input error: the vocabulary of "
                        "inputs must have at least one dataset and one "
                        f"configuration, not {len(inputs.naming.datasets)}"
                        f" and {len(inputs.naming.configurations)}.")
        values, imputed, auc_t, ids, missing = self._collect(inputs, out)
        n_imputed = int(imputed.sum())
        n_cells = values.size
        if missing:
            self._warning("[Heatmap.build] Input warning: the aggregate "
                          f"must hold all {n_cells} cells, and "
                          f"{len(missing)} are absent. They are drawn as "
                          "n/a and listed in the output record.")
        imputation = (CAPTION_IMPUTED.format(n=n_imputed, total=n_cells)
                      if n_imputed else CAPTION_NO_IMPUTED)
        fig = self._draw(inputs, values, imputed, auc_t)
        # pdf.fonttype is read when the file is written, not when the
        # figure is drawn: saving outside this context embeds Type 3
        # fonts.
        with plt.rc_context(self.rc_params):
            out.save_figure(
                fig, FIGURE_NAME,
                CAPTION_HEAD + imputation + CAPTION_TAIL, FIGURE_LABEL,
                inputs.naming.plan_tag("primary") + " / "
                + inputs.naming.plan_tag("r110"),
                sorted(set(ids)), inputs.partial_inputs,
                {"out_id": OUT_ID,
                 "partial_inputs_reasons": inputs.partial_inputs_reasons,
                 "missing": missing, "notes": list(NOTES)})
        return True

    def __repr__(self) -> str:
        """
        Show the presentation settings of this heat map.

        :return: Text like "Heatmap(size_inches=(6.69, 4.0),
            trivial_width_ratio=1.4, chance_level=0.5)".
        :rtype: str
        """
        return (f"Heatmap(size_inches={self.size_inches}, "
                f"trivial_width_ratio={self.trivial_width_ratio}, "
                f"chance_level={self.chance_level})")

    def _collect(self, inputs: Inputs, out: Out) -> tuple[
            np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
        """
        Read the ROC-AUC matrix, the imputation mask and AUC$_t$.

        The matrix is filled dataset by dataset and, inside a dataset,
        configuration by configuration: that is the order in which the
        rows of ``numbers_used.md`` are created.

        :param inputs: Reader of the aggregate.
        :type inputs: Inputs
        :param out: Register the rows of values are added to.
        :type out: Out
        :return: The ROC-AUC matrix, shape (datasets, configurations),
            NaN where the aggregate has no value; the boolean mask of
            cells with an imputed seed; the AUC$_t$ column, shape
            (datasets,), NaN where missing; the ids of the rows; and
            the "dataset/config" names of the missing cells.
        :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, list[str],
            list[str]]
        """
        datasets = list(inputs.naming.datasets)
        configurations = list(inputs.naming.configurations)
        shape = (len(datasets), len(configurations))
        values = np.full(shape, np.nan)
        imputed = np.zeros(shape, dtype=bool)
        auc_t = np.full(len(datasets), np.nan)
        ids: list[str] = list()
        missing: list[str] = list()
        trivial_tag = inputs.naming.plan_tag("trivial")
        for i, x in enumerate(datasets):
            for j, c in enumerate(configurations):
                value, pointer = inputs.auroc_src(x, c)
                if value is None:
                    missing.append(f"{x}/{c}")
                    continue
                values[i, j] = value
                imputed[i, j] = inputs.imputed(x, c)
                ids.append(out.add_number(
                    f"R.T1.{x}.{c}.AUROC_FUS", value, pointer, CELL_TAG,
                    inputs.partial_inputs))
        for i, x in enumerate(datasets):
            reference = inputs.trivial(x, "auc_t")
            if reference is not None:
                auc_t[i] = reference
            ids.append(out.add_number(
                f"R.TRIV.{x}.AUC_t", reference,
                f"aggregate.json#/datasets/{x}/trivial/auc_t",
                trivial_tag, inputs.partial_inputs))
        return values, imputed, auc_t, ids, missing

    def _draw(self,
              inputs: Inputs,
              values: np.ndarray,
              imputed: np.ndarray,
              auc_t: np.ndarray) -> Figure:
        """
        Draw the two panels, the grid, the colour bar and the footnote.

        :param inputs: Reader used for the axis labels and seed counts.
        :type inputs: Inputs
        :param values: ROC-AUC matrix, NaN where the value is missing.
        :type values: np.ndarray
        :param imputed: Boolean mask of the same shape.
        :type imputed: np.ndarray
        :param auc_t: Trivial baseline per dataset, shape (datasets,).
        :type auc_t: np.ndarray
        :return: The finished figure; the caller closes it.
        :rtype: Figure
        """
        datasets = list(inputs.naming.datasets)
        configurations = list(inputs.naming.configurations)
        seeds = [inputs.n_seeds(x) for x in datasets]
        same_seeds = len(set(seeds)) == 1
        footnote = SYMBOL_FOOTNOTE
        with plt.rc_context(self.rc_params):
            fig = plt.figure(figsize=self.size_inches)
            grid = fig.add_gridspec(
                1, 3,
                width_ratios=[len(configurations),
                              self.trivial_width_ratio,
                              COLORBAR_WIDTH_RATIO],
                wspace=PANEL_SPACING, left=AXES_LEFT, right=AXES_RIGHT,
                bottom=self.bottom, top=self.top)
            ax = fig.add_subplot(grid[0, 0])
            axt = fig.add_subplot(grid[0, 1])
            cax = fig.add_subplot(grid[0, 2])
            image = ax.imshow(values, cmap="Greys", vmin=0.0, vmax=1.0,
                              aspect="auto")
            self._draw_cells(ax, image, values, imputed, auc_t)
            self._label_axes(ax, inputs, seeds, same_seeds)
            ax.set_title(AXIS_TITLE.format(
                seeds=(f"$n$ = {seeds[0]} seeds" if same_seeds
                       else "seeds")))
            for k in range(len(datasets) + 1):
                ax.axhline(k - 0.5, color="white", linewidth=0.8)
            for k in range(len(configurations) + 1):
                ax.axvline(k - 0.5, color="white", linewidth=0.8)
            axt.imshow(auc_t.reshape(-1, 1), cmap="Greys", vmin=0.0,
                       vmax=1.0, aspect="auto")
            for i, value in enumerate(auc_t):
                axt.text(0, i, "n/a" if not np.isfinite(value)
                         else f"{value:.2f}", ha="center", va="center",
                         fontsize=FONT_SIZE_CELL,
                         color=("white" if np.isfinite(value)
                                and value > WHITE_TEXT_ABOVE
                                else "black"))
            axt.set_xticks([0])
            axt.set_xticklabels([TRIVIAL_TICK],
                                fontsize=self.font_size_footnote,
                                linespacing=0.95)
            axt.set_yticks(list())
            axt.tick_params(axis="x", length=2.5, pad=2.5)
            for k in range(len(datasets) + 1):
                axt.axhline(k - 0.5, color="white", linewidth=0.8)
            bar = fig.colorbar(image, cax=cax,
                               ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
            bar.ax.axhline(self.chance_level, color="black",
                           linewidth=1.2)
            bar.ax.tick_params(labelsize=FONT_SIZE_CELL, length=2.5,
                               pad=2.0)
            bar.set_label(COLORBAR_LABEL, labelpad=3,
                          fontsize=FONT_SIZE_COLORBAR_LABEL)
            if bool(imputed.any()):
                footnote += SYMBOL_FOOTNOTE_IMPUTED
            fig.text(FOOTNOTE_XY[0], FOOTNOTE_XY[1], footnote,
                     fontsize=self.font_size_footnote, ha="left",
                     va="bottom")
        return fig

    def _label_axes(self,
                    ax: Any,
                    inputs: Inputs,
                    seeds: list[int],
                    same_seeds: bool) -> None:
        """
        Write the two-level labels of the columns and of the rows.

        Columns: the id over the paradigm under each one ("C2" over
        "TSR"), then a line and the family name under each pair, and
        "no temporal context" under the contiguous block of
        configurations without context. "C2" over "Conv1D-TSR" is 45 pt
        wide at 7.5 pt and a column is 28 pt, so the one-level label of
        the 11-inch figure does not fit at the printed width.

        Rows: the rows that share the prefix "KNAD " show it once,
        rotated, next to a bracket; the number of seeds goes under the
        name only when it differs between datasets.

        :param ax: Axes of the heat map panel.
        :type ax: Any
        :param inputs: Reader used for the names and the vocabulary.
        :type inputs: Inputs
        :param seeds: Number of seeds per dataset, in dataset order.
        :type seeds: list[int]
        :param same_seeds: Whether every dataset has the same number.
        :type same_seeds: bool
        """
        datasets = list(inputs.naming.datasets)
        configurations = list(inputs.naming.configurations)
        names = [inputs.dataset_label(x) for x in datasets]
        row_groups = self._row_groups(names)
        grouped = {i for first, last in row_groups
                   for i in range(first, last + 1)}
        along_x = ax.get_xaxis_transform()
        along_y = ax.get_yaxis_transform()
        ax.set_xticks(range(len(configurations)))
        ax.set_xticklabels(
            [f"{c}\n{self._tick(inputs, c)}" for c in configurations],
            fontsize=self.font_size_small)
        ax.set_yticks(range(len(datasets)))
        ax.set_yticklabels(
            [(name[len(ROW_GROUP_PREFIX):] if i in grouped else name)
             + ("" if same_seeds else f"\n(n = {seeds[i]})")
             for i, name in enumerate(names)])
        ax.tick_params(axis="both", length=2.5, pad=2.5)
        for first, last in row_groups:
            ax.plot([ROW_GROUP_X, ROW_GROUP_X],
                    [first - GROUP_LINE_INSET, last + GROUP_LINE_INSET],
                    transform=along_y, clip_on=False, color="0.25",
                    linewidth=0.7, solid_capstyle="butt")
            ax.text(ROW_GROUP_X - GROUP_NAME_DROP, 0.5 * (first + last),
                    ROW_GROUP_PREFIX.strip(), transform=along_y,
                    ha="right", va="center", rotation=90,
                    fontsize=self.font_size_small)
        for name, first, last in self._column_groups(inputs):
            ax.plot([first - GROUP_LINE_INSET, last + GROUP_LINE_INSET],
                    [GROUP_LINE_Y, GROUP_LINE_Y], transform=along_x,
                    clip_on=False, color="0.25", linewidth=0.7,
                    solid_capstyle="butt")
            ax.text(0.5 * (first + last), GROUP_LINE_Y - GROUP_NAME_DROP,
                    name, transform=along_x, ha="center", va="top",
                    fontsize=self.font_size_small)
        ax.set_xlabel(AXIS_XLABEL, labelpad=XLABEL_PAD)

    @staticmethod
    def _tick(inputs: Inputs, config: str) -> str:
        """
        Second line of a column label.

        :param inputs: Reader used for the configuration names.
        :type inputs: Inputs
        :param config: Configuration id such as "C2".
        :type config: str
        :return: The paradigm ("AR" or "TSR") when the family of the
            configuration has two of them, and the short name ("MLP",
            "dA", "STAT") when it has one.
        :rtype: str
        """
        name = inputs.config_name(config)
        for family, members in inputs.naming.arch_configs.items():
            if config in members and len(members) > 1 \
                    and name.startswith(family + "-"):
                return name[len(family) + 1:]
        return inputs.config_short(config)

    @staticmethod
    def _column_groups(inputs: Inputs) -> list[tuple[str, int, int]]:
        """
        Groups of columns named under the x axis.

        :param inputs: Reader used for the vocabulary.
        :type inputs: Inputs
        :return: (name, first column, last column) per group: one per
            family with an AR and a TSR configuration, and a single
            "no temporal context" group when the configurations
            without context are contiguous (plan section 10).
        :rtype: list[tuple[str, int, int]]
        """
        groups = list()
        columns = list()
        configurations = list(inputs.naming.configurations)
        for family, members in inputs.naming.arch_configs.items():
            columns = [configurations.index(c) for c in members
                       if c in configurations]
            if len(columns) > 1 and all(
                    inputs.config_name(configurations[k]).startswith(
                        family + "-") for k in columns):
                groups.append((family, min(columns), max(columns)))
        columns = sorted(configurations.index(c)
                         for c in inputs.naming.context_free
                         if c in configurations)
        if columns and columns == list(range(columns[0],
                                             columns[-1] + 1)):
            groups.append((CONTEXT_FREE_GROUP, columns[0], columns[-1]))
        return groups

    @staticmethod
    def _row_groups(names: list[str]) -> list[tuple[int, int]]:
        """
        Runs of consecutive rows whose name starts with "KNAD ".

        The prefix is written once for the run, which leaves 67 pt of
        row labels instead of 91 pt, and that width goes to the cells.

        :param names: Long names of the datasets, in row order.
        :type names: list[str]
        :return: (first row, last row) of every run of two rows or
            more; empty when no two consecutive rows share the prefix.
        :rtype: list[tuple[int, int]]
        """
        groups = list()
        first = 0
        last = 0
        while first < len(names):
            if not names[first].startswith(ROW_GROUP_PREFIX):
                first += 1
                continue
            last = first
            while last + 1 < len(names) and \
                    names[last + 1].startswith(ROW_GROUP_PREFIX):
                last += 1
            if last > first:
                groups.append((first, last))
            first = last + 1
        return groups

    def _draw_cells(self,
                    ax: Any,
                    image: Any,
                    values: np.ndarray,
                    imputed: np.ndarray,
                    auc_t: np.ndarray) -> None:
        """
        Write the number and the marks of every cell of the map.

        A missing cell becomes a white square with a cross hatch and
        the text "n/a". A cell at or below chance gets a diagonal hatch,
        visible in black and white too, and an asterisk; one that does
        not exceed the row-index reference a dagger; one with an
        imputed seed a double dagger. Over the hatch the number sits in
        a box of the colour of its cell, so that no line crosses it.

        The reference is max(AUC_t, 1 - AUC_t) and not AUC_t: a
        detector that returns the DECREASING record index is as free as
        one that returns the increasing index, so a model has to beat
        the better of the two. It differs only where AUC_t < 0.5, which
        on this grid is ACI-IoT-2023 (0.047, reference 0.953) and
        CSE-CIC-IDS2018 (0.391, reference 0.609), and it moves the
        count of marked cells from 56 to 66 of 99.

        Note: the marks use "value <= chance_level" and "value <=
        reference", so a cell exactly at chance, or exactly at the
        reference, counts as not beating it.

        :param ax: Axes of the heat map panel.
        :type ax: Any
        :param image: Image drawn by ``imshow`` on those axes; its
            colour map gives the colour of the box behind the numbers
            of the hatched cells.
        :type image: Any
        :param values: ROC-AUC matrix, NaN where a value is missing.
        :type values: np.ndarray
        :param imputed: Boolean mask of the same shape.
        :type imputed: np.ndarray
        :param auc_t: Trivial baseline per dataset, shape (datasets,).
        :type auc_t: np.ndarray
        """
        value = 0.0
        mark = ""
        box = None
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value = values[i, j]
                if not np.isfinite(value):
                    ax.add_patch(Rectangle(
                        (j - 0.5, i - 0.5), 1, 1, facecolor="white",
                        edgecolor="0.3", hatch="xx", linewidth=0.5))
                    ax.text(j, i, "n/a", ha="center", va="center",
                            fontsize=FONT_SIZE_NA, color="0.2")
                    continue
                reference = max(auc_t[i], 1.0 - auc_t[i])
                mark = (("*" if value <= self.chance_level else "")
                        + ("†" if np.isfinite(reference)
                           and value <= reference else "")
                        + ("‡" if imputed[i, j] else ""))
                box = None
                if value <= self.chance_level:
                    ax.add_patch(Rectangle(
                        (j - 0.5, i - 0.5), 1, 1, fill=False,
                        edgecolor="0.15", hatch="///", linewidth=0.0))
                    box = dict(facecolor=image.cmap(image.norm(value)),
                               edgecolor="none", pad=0.5)
                ax.text(j, i, f"{value:.2f}{mark}", ha="center",
                        va="center", fontsize=FONT_SIZE_CELL,
                        color=("white" if value > WHITE_TEXT_ABOVE
                               else "black"), bbox=box)


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
