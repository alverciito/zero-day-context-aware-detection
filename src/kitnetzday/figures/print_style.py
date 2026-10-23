# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         src/kitnetzday/figures/print_style.py                     #
#   Version:      1.0.1                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-07                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Print size of the figures: the width the manuscript       #
#                 includes them at and the smallest lettering allowed.      #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
from collections.abc import Mapping
from typing import Any

from kitnetzday.naming import RC_PARAMS

# Comment R1.10 asked for larger lettering inside the figures. A figure
# drawn wider than the page is shrunk when it is included at \textwidth
# (170 mm), and its lettering with it. Every figure is therefore drawn
# at the width it is included at: the point sizes below are the printed
# ones.
# \textwidth of the manuscript, \geometry{total={170mm,257mm}}, in inches.
PRINT_WIDTH_INCHES: float = 6.69
# Smallest normal lettering of any figure, in points.
FONT_SIZE_MIN: float = 7.5
# Lettering that carries a mathtext subscript such as AUC_t. Mathtext
# draws subscripts at 0.7 of the size: 9.0 pt gives 6.3 pt, above the
# 6 pt Elsevier asks of sub and superscripts.
FONT_SIZE_SCRIPT: float = 9.0
FONT_SIZE_LEGEND: float = 8.0
FONT_SIZE_TITLE: float = 9.0
# Settings laid over the house style of ``naming.RC_PARAMS``. The font
# family is pinned so that no system font is picked as a fallback, and
# the padding of the tight bounding box is kept small so that the PDF is
# as wide as the drawing, not 0.2 inches wider.
_PRINT_OVERRIDES: dict[str, Any] = {
    "savefig.pad_inches": 0.02, "font.family": ["DejaVu Sans"],
    "mathtext.fontset": "dejavusans", "font.size": 8.0,
    "axes.titlesize": FONT_SIZE_TITLE, "axes.labelsize": 8.5,
    "xtick.labelsize": FONT_SIZE_MIN, "ytick.labelsize": FONT_SIZE_MIN,
    "legend.fontsize": FONT_SIZE_LEGEND, "pdf.fonttype": 42,
    "ps.fonttype": 42}


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                              MAIN FUNCTION                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def print_rc_params(base: Mapping[str, Any] = RC_PARAMS,
                    logger: logging.Logger | None = None,
                    ) -> dict[str, Any]:
    """
    Matplotlib settings of a figure drawn at its printed width.

    The house style of ``naming.RC_PARAMS`` is kept and the print
    overrides are laid on top: lettering of 7.5 pt or more, the DejaVu
    Sans family pinned, and TrueType (Type 42) fonts in the PDF. The
    result must be active while the figure is drawn AND while it is
    saved: matplotlib reads ``pdf.fonttype`` when it writes the file,
    so a figure drawn inside ``plt.rc_context`` and saved outside it
    embeds Type 3 fonts, which is what happened until version 1.1.0.

    :param base: Settings to start from, for example
        ``naming.RC_PARAMS``; a mapping of rcParams names to values.
    :type base: Mapping[str, Any]
    :param logger: Logger of the calling script, or None to stay silent.
    :type logger: logging.Logger | None
    :return: A new dictionary with the base settings and the print
        overrides; the base is not modified.
    :rtype: dict[str, Any]
    :raises TypeError: If base is not a mapping.
    """
    merged = dict()
    message = ""
    if not isinstance(base, Mapping):
        message = ("[print_rc_params] Input error: the parameter base "
                   "must be a mapping of rcParams, not "
                   f"{type(base).__name__}.")
        if logger is not None:
            logger.error(message)
        raise TypeError(message)
    merged = dict(base)
    merged.update(_PRINT_OVERRIDES)
    return merged


# Settings every figure of the package uses by default.
PRINT_RC_PARAMS: dict[str, Any] = print_rc_params()


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
