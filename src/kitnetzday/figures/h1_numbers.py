# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         src/kitnetzday/figures/h1_numbers.py                      #
#   Version:      2.0.0                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Rows of numbers_used.md for one H1 comparison of          #
#                 the analysis plan.                                        #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
from collections.abc import Mapping
from typing import Any, NoReturn

from kitnetzday.io.outputs import Out
from kitnetzday.utils.formatting import get

# Values of each H1 comparison copied into the rows of numbers_used.md,
# as (suffix of the id, path inside the comparison record, JSON pointer
# of the "Source" column), in the order they are written.
_H1_FIELDS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("HL", ("hl",), "hl"),
    ("CIlo", ("hl_ci", "lower"), "hl_ci/lower"),
    ("CIhi", ("hl_ci", "upper"), "hl_ci/upper"),
    ("pHolm", ("p_holm",), "p_holm"),
    ("kpos", ("k_pos",), "k_pos"),
    ("HL_KNADcollapsed",
     ("robustness", "dependence", "KNAD_collapsed", "hl"),
     "robustness/dependence/KNAD_collapsed/hl"),
    ("kpos_KNADcollapsed",
     ("robustness", "dependence", "KNAD_collapsed", "k_pos"),
     "robustness/dependence/KNAD_collapsed/k_pos"),
    ("HL_noKNAD",
     ("robustness", "dependence", "without_KNAD", "hl"),
     "robustness/dependence/without_KNAD/hl"),
    ("kpos_noKNAD",
     ("robustness", "dependence", "without_KNAD", "k_pos"),
     "robustness/dependence/without_KNAD/k_pos"),
    ("HL_AUClocRAW",
     ("robustness", "drift", "hl_auc_loc_raw"),
     "robustness/drift/hl_auc_loc_raw"))


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
def add_h1_numbers(out: Out,
                   architecture: str = "",
                   record: Mapping[str, Any] | None = None,
                   tag: str = "",
                   partial: bool = False,
                   logger: logging.Logger | None = None) -> list[str]:
    """
    Register one row of numbers_used.md per value of one H1 comparison.

    Every number printed by an output is traceable to a row of
    ``numbers_used.md`` with its source. A comparison of plan section
    5.3 prints ten of them: the Hodges-Lehmann
    difference and its two interval ends, the Holm-adjusted p value,
    the count of datasets in which the family is higher, the same
    difference and count under the two dependence robustness readings,
    and the difference on the within-block evaluation. A value that is
    not stored leaves no row, which ``Out.add_number`` takes care of.

    :param out: Output directory of the run, which keeps the rows and
        writes them to ``numbers_used.md``.
    :type out: Out
    :param architecture: Family of the comparison, such as "LSTM"; the
        id uses it upper-cased.
    :type architecture: str
    :param record: One comparison record of
        ``stats.json#/H1/comparisons``, or None when the family has
        none, which leaves no row.
    :type record: Mapping[str, Any] | None
    :param tag: Label of the analysis plan for these values, normally
        ``naming.PLAN_TAGS["h1"]``.
    :type tag: str
    :param partial: Whether the run reads partial inputs, which puts
        the partial-inputs label in front of the source of every row.
    :type partial: bool
    :param logger: Logger created by the calling script, or None to
        stay silent.
    :type logger: logging.Logger | None
    :return: Ids of the rows, in the order of ``_H1_FIELDS``; one per
        field, whether or not the value was stored.
    :rtype: list[str]
    :raises TypeError: If out is not an Out, architecture or tag are
        not strings, record is neither a mapping nor None, or partial
        is not a boolean.
    :raises ValueError: If architecture is empty.
    """
    ids = list()
    pointer = ""
    fields = dict()
    if not isinstance(out, Out):
        _fail("[add_h1_numbers] Input error: the parameter out must be "
              f"an Out, not {type(out).__name__}.", logger, TypeError)
    if not isinstance(architecture, str) or not isinstance(tag, str):
        _fail("[add_h1_numbers] Input error: the parameters "
              "architecture and tag must be strings, not "
              f"{type(architecture).__name__} and "
              f"{type(tag).__name__}.", logger, TypeError)
    if not architecture:
        _fail("[add_h1_numbers] Input error: the parameter "
              "architecture must be a non-empty string, not "
              f"{architecture!r}.", logger)
    if record is not None and not isinstance(record, Mapping):
        _fail("[add_h1_numbers] Input error: the parameter record must "
              f"be a mapping or None, not {type(record).__name__}.",
              logger, TypeError)
    if not isinstance(partial, bool):
        _fail("[add_h1_numbers] Input error: the parameter partial "
              f"must be a boolean, not {type(partial).__name__}.",
              logger, TypeError)
    fields = dict(record) if record is not None else dict()
    for key, path, ptr in _H1_FIELDS:
        pointer = (f"stats.json#/H1/comparisons/"
                   f"{architecture}_vs_original/{ptr}")
        ids.append(out.add_number(
            f"R.H1.AGG.{architecture.upper()}vsORIG.{key}",
            get(fields, *path), pointer, tag, partial))
    return ids


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
