# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         src/kitnetzday/figures/ranks_caption.py                   #
#   Version:      1.1.1                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Caption of figure F2, assembled verbatim from the         #
#                 submitted script.                                         #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
from typing import NoReturn

from kitnetzday.naming import ROPE


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                AUXILIARY                                  #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def _fail(message: str,
          logger: logging.Logger | None = None,
          exception: type[Exception] = ValueError) -> NoReturn:
    """
    Log an error, if there is a logger, and raise it with the same text.

    Plain-function counterpart of ``Loggable._error``: this module is a
    single stateless function, so it cannot inherit from ``Loggable``.

    :param message: Text in the house format, used for both the log
        record and the exception.
    :type message: str
    :param logger: Logger created by the calling script, or None to
        stay silent.
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
def mean_ranks_caption(n_ranked: int = 0,
                       noise_median: float | None = None,
                       n_no_noise: int = 0,
                       n_datasets: int = 0,
                       robustness_note: str = "",
                       rope: float = ROPE,
                       logger: logging.Logger | None = None) -> str:
    """
    Assemble the caption of figure F2, in academic English.

    The caption states what is drawn in both panels and, above all, what
    may not be read from them: the interval of the mean rank is
    descriptive and unadjusted, no critical-difference diagram is drawn
    because with 55 pairs the Bonferroni level falls below the smallest
    attainable exact p value, and the between-seed noise band is given
    next to the ROPE so that +-0.01 is not mistaken for the resolution
    of the experiment. The sentence on the datasets with fewer than two
    seeds is written only when there is at least one.

    :param n_ranked: Datasets that entered the ranking, that is, those
        with all the configurations; zero or more.
    :type n_ranked: int
    :param noise_median: Median across datasets of sqrt(2) SD_pool, in
        ROC-AUC points, or None when no dataset has a between-seed
        estimate and no band is drawn.
    :type noise_median: float | None
    :param n_no_noise: Datasets with fewer than two seeds and therefore
        without a between-seed estimate; zero or more.
    :type n_no_noise: int
    :param n_datasets: Datasets of the vocabulary, the denominator of
        the previous count; zero or more.
    :type n_datasets: int
    :param robustness_note: Sentence that explains the asterisks of the
        top panel, as produced by
        ``ContrastsPanel.robustness_note``.
    :type robustness_note: str
    :param rope: Half width of the ROPE in ROC-AUC points, 0.01 in the
        plan; the noise band is reported as a multiple of it, so it has
        to be strictly positive.
    :type rope: float
    :param logger: Logger created by the calling script, or None to
        stay silent.
    :type logger: logging.Logger | None
    :return: The caption, ready to be written into the ``.tex`` block
        of the figure.
    :rtype: str
    :raises TypeError: If a count is not an integer, noise_median is
        neither a real number nor None, or robustness_note is not a
        string.
    :raises ValueError: If a count is negative or rope is not strictly
        positive.
    """
    noise_txt = ""
    for name, value in (("n_ranked", n_ranked),
                        ("n_no_noise", n_no_noise),
                        ("n_datasets", n_datasets)):
        if isinstance(value, bool) or not isinstance(value, int):
            _fail("[mean_ranks_caption] Input error: the parameter "
                  f"{name} must be an integer, not "
                  f"{type(value).__name__}.", logger, TypeError)
        if value < 0:
            _fail("[mean_ranks_caption] Input error: the parameter "
                  f"{name} must be zero or more, not {value}.", logger)
    if noise_median is not None and (isinstance(noise_median, bool)
                                     or not isinstance(noise_median,
                                                       (int, float))):
        _fail("[mean_ranks_caption] Input error: the parameter "
              "noise_median must be a real number or None, not "
              f"{type(noise_median).__name__}.", logger, TypeError)
    if not isinstance(robustness_note, str):
        _fail("[mean_ranks_caption] Input error: the parameter "
              "robustness_note must be a string, not "
              f"{type(robustness_note).__name__}.", logger, TypeError)
    if isinstance(rope, bool) or not isinstance(rope, (int, float)):
        _fail("[mean_ranks_caption] Input error: the parameter rope "
              f"must be a real number, not {type(rope).__name__}.",
              logger, TypeError)
    if not float(rope) > 0.0:
        _fail("[mean_ranks_caption] Input error: the parameter rope "
              f"must be strictly positive (>0), not {rope}.", logger)
    if noise_median is not None:
        noise_txt = ("The dashed vertical lines and the hatched band "
                     "are the median across datasets of "
                     "$\\sqrt{2}\\,\\mathrm{SD}_{\\mathrm{pool}}$, the "
                     "run-to-run noise reference of the analysis "
                     "plan, at $\\pm " + f"{noise_median:.3f}"
                     + "$; it is " + f"{noise_median / rope:.0f}"
                     + " times wider than the ROPE"
                     + (", and " + str(n_no_noise) + " of "
                        + str(n_datasets) + " datasets have fewer than "
                        "two seeds and therefore no between-seed "
                        "estimate at all" if n_no_noise else "")
                     + ". Differences smaller than this band are not "
                       "resolved with two seeds.")
    else:
        noise_txt = ("No dataset has two or more seeds, so no "
                     "between-seed noise reference can be drawn next "
                     "to the ROPE.")
    return ("Top: mean rank of each configuration across the "
            + str(n_ranked) + " datasets with all eleven "
            "configurations (1 = lowest ROC-AUC). Open circles are the "
            "per-dataset ranks; the marker is the mean rank and the "
            "bar spans $\\pm 1.96$ standard errors of that mean across "
            "datasets. The interval is descriptive and is not adjusted "
            "for multiplicity; no critical-difference diagram is "
            "drawn, because with 55 pairs the Bonferroni level falls "
            "below the smallest attainable exact $p$. Ranks among the "
            "Conv1D and Conv2D variants are within run-to-run noise. "
            "Bottom: the five pre-specified H1 comparisons of each "
            "autoencoder family against the KitNET-style control "
            "(C10), as Hodges--Lehmann differences in ROC-AUC with "
            "their exact unadjusted 96.1\\% confidence intervals and "
            "Holm-adjusted exact Wilcoxon $p$ values; the shaded band "
            "is the pre-specified $\\pm 0.01$ ROPE. " + noise_txt + " "
            + robustness_note
            + " To the right of each interval, the same difference is "
              "given with "
              "the six KNAD captures collapsed into a single unit "
              "($n=4$) and with the KNAD testbed removed ($n=3$), as "
              "the analysis plan requires whenever a difference "
              "between configurations is stated.")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
