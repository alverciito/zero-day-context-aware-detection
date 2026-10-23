# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         src/kitnetzday/utils/formatting.py                        #
#   Version:      1.0.1                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Number and LaTeX formatting, tolerant dictionary          #
#                 lookup and average ranks with ties.                       #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
from collections.abc import Sequence
from typing import Any, NoReturn

import numpy as np

# Replacements applied by tex_escape, in this exact order: the backslash
# has to go first, otherwise the backslashes introduced by the other
# replacements would be escaped again.
_TEX_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("\\", r"\textbackslash{}"), ("_", r"\_"), ("&", r"\&"),
    ("%", r"\%"), ("#", r"\#"), ("$", r"\$"), ("{", r"\{"),
    ("}", r"\}"), ("~", r"\textasciitilde{}"),
    ("^", r"\textasciicircum{}"))


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                MAIN CLASS                                 #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class InputError(RuntimeError):
    """
    An input of the figure pipeline is missing or inconsistent.

    The scripts of the paper map it to exit code 1 ("invalid or
    inconsistent input"), which is separate from the guard failures
    (code 3) and from a wrong command line (code 2). It is raised, for
    example, when ``aggregate.json`` is not at the path given.
    """


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                AUXILIARY                                  #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def _fail(message: str,
          logger: logging.Logger | None = None,
          exception: type[Exception] = ValueError) -> NoReturn:
    """
    Log an error, if there is a logger, and raise it with the same text.

    This is the plain-function counterpart of ``Loggable._error``: the
    helpers of this module are stateless functions, so they cannot
    inherit from ``Loggable``.

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
def fnum(v: Any = None, nd: int = 3, na: str = "--") -> str:
    """
    Write one number for a LaTeX table or a text report.

    A cell of the paper is either a number with a fixed number of
    decimals or an explicit "missing" marker. Anything that is not a
    finite number becomes that marker, so a hole in the aggregate can
    never print as "0.000" or as "nan" in the manuscript. The function
    is deliberately tolerant and never raises: it is called once per
    table cell, and a single missing seed must not stop a table.

    :param v: Value to write. None, a string that does not parse, an
        array with more than one element, NaN and the infinities all
        give the missing marker.
    :type v: Any
    :param nd: Number of decimals, as in the ``:.{nd}f`` format; the
        tables of the paper use 3.
    :type nd: int
    :param na: Marker printed instead of a number when the value is
        missing; "--" in the LaTeX tables.
    :type na: str
    :return: The formatted number, for example "0.873", or na.
    :rtype: str
    """
    f = 0.0
    if v is None:
        return na
    try:
        f = float(v)
    except (TypeError, ValueError):
        return na
    if not np.isfinite(f):
        return na
    return f"{f:.{nd}f}"


def tex_escape(s: Any = "") -> str:
    """
    Escape a text so that LaTeX typesets it literally.

    Dataset and configuration labels travel into captions and table
    cells, and some of them carry underscores or percent signs that
    would otherwise break the build or start a comment.

    :param s: Value to escape; anything not a string is passed through
        ``str`` first.
    :type s: Any
    :return: The escaped text, safe inside a LaTeX paragraph.
    :rtype: str
    """
    out = str(s)
    for a, b in _TEX_REPLACEMENTS:
        out = out.replace(a, b)
    return out


def get(o: Any, *path: str, default: Any = None) -> Any:
    """
    Walk a chain of dictionary keys without ever raising.

    The aggregate and the statistics are nested JSON documents whose
    optional branches depend on the run: a figure must not break
    because one of them is absent. A missing key, a branch that is not
    a dictionary and a stored ``null`` all give the default.

    :param o: Object to walk, normally the parsed ``aggregate.json`` or
        ``stats.json``.
    :type o: Any
    :param path: Keys to follow in order, for example "cells", "D1",
        "C7", "name".
    :type path: str
    :param default: Value returned when the path is not there or holds
        a JSON ``null``.
    :type default: Any
    :return: The value at the end of the path, or default.
    :rtype: Any
    """
    cur = o
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur if cur is not None else default


def ids_comment(ids: Sequence[str] = tuple(),
                name: str = "",
                k: int = 12) -> str:
    """
    Build the LaTeX comment that lists the value ids of an output.

    Every figure and table carries, as a comment inside its ``.tex``,
    the ids of the rows of ``numbers_used.md`` it uses, so that a
    reader of the sources can trace each printed value. Long lists are
    truncated and the comment points at the full one.

    :param ids: Ids of ``numbers_used.md`` used by the output, in the
        order the output produced them; an empty sequence is allowed.
    :type ids: Sequence[str]
    :param name: Base name of the output, used to point into
        ``figures.json#/outputs/<name>/numbers_ids``.
    :type name: str
    :param k: How many ids are listed before truncating; 12 in the
        paper pipeline.
    :type k: int
    :return: A one-line LaTeX comment starting with "%".
    :rtype: str
    """
    head = ""
    tail = ""
    if not ids:
        return "% numbers_used.md ids: (none)"
    head = ", ".join(ids[:k])
    tail = (f" ... (+{len(ids) - k}; full list in "
            f"README_figures.md and figures.json#/outputs/{name}"
            f"/numbers_ids)" if len(ids) > k else "")
    return f"% numbers_used.md ids ({len(ids)}): {head}{tail}"


def ranks_avg(values: Any = tuple(),
              logger: logging.Logger | None = None) -> np.ndarray:
    """
    Rank a row of values, giving tied values their average rank.

    This is the ranking that feeds the Friedman omnibus of plan section
    5.3 and the mean-ranks figure: rank 1 is the smallest value, and a
    group of ties shares the mean of the ranks it occupies, so the ranks
    of a row always add up to n (n + 1) / 2. The sort is stable
    (mergesort), which keeps the order of equal values reproducible.

    Ties are detected by comparing each sorted value with the first
    value of the group it is joining, so a group only grows while the
    values are exactly equal. Note: NaN sorts to the end and never
    compares equal, so every NaN becomes its own group and receives a
    distinct, largest rank instead of being flagged; the caller is the
    one that must keep NaN out, as the statistics of the paper do.

    :param values: One-dimensional array-like of real numbers, for
        example the eleven configuration values of one dataset. An
        empty input gives an empty result.
    :type values: Any
    :param logger: Logger created by the calling script, or None to stay
        silent.
    :type logger: logging.Logger | None
    :return: Array of shape (n,) of float ranks in 1..n, aligned with
        the input order.
    :rtype: np.ndarray
    :raises ValueError: If values cannot be read as an array of floats,
        or is not one-dimensional.
    """
    a = np.empty(0, dtype=float)
    order = np.empty(0, dtype=int)
    r = np.empty(0, dtype=float)
    i = 0
    j = 0
    try:
        a = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        _fail("[ranks_avg] Input error: the parameter values must be "
              "an array-like of real numbers, and it could not be "
              f"converted: {exc}.", logger)
    if a.ndim != 1:
        _fail("[ranks_avg] Input error: the parameter values must be "
              f"one-dimensional, not of shape {a.shape}.", logger)
    if a.size == 0:
        if logger is not None:
            logger.warning("[ranks_avg] Input warning: the parameter "
                           "values is empty. An empty array of ranks "
                           "is returned.")
        return np.empty(0, dtype=float)
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), dtype=float)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        r[order[i:j + 1]] = 0.5 * ((i + 1) + (j + 1))
        i = j + 1
    return r


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
