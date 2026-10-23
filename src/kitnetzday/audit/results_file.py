# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         src/kitnetzday/audit/results_file.py                      #
#   Version:      1.0.1                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Writer of the JSON result files of the audits, with the   #
#                 indentation of the archived originals.                    #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NoReturn

# Indentation of the archived audit files, so that a re-run shows an
# empty diff against them.
RESULT_INDENT: int = 1


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                AUXILIARY                                  #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def _fail(message: str,
          logger: logging.Logger | None = None,
          exception: type[Exception] = ValueError) -> NoReturn:
    """
    Log an error, if there is a logger, and raise it with the same text.

    This is the plain-function counterpart of ``Loggable._error``: this
    module is a stateless function, so it cannot inherit from
    ``Loggable``.

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
def write_result(result: Mapping[str, Any],
                 path: str | Path,
                 indent: int = RESULT_INDENT,
                 logger: logging.Logger | None = None) -> Path:
    """
    Write the result of an audit as JSON, as the original scripts did.

    The exploratory audits of the revision each leave one JSON file
    that the manuscript and its supplementary material point at. The
    indentation of one space is the one of the archived files, so that
    re-running an audit shows an empty diff when nothing changed.

    Note: non-finite floats are written as ``NaN`` and ``Infinity``,
    which is what ``json.dumps`` does by default and what the archived
    files already contain; they are not replaced by ``null``.

    :param result: Structure to serialize, as the ``run`` method of an
        audit returns it.
    :type result: Mapping[str, Any]
    :param path: Destination file; its directory must exist.
    :type path: str | Path
    :param indent: Spaces per nesting level; 1 in the archived files,
        not negative.
    :type indent: int
    :param logger: Logger created by the calling script, or None to
        stay silent.
    :type logger: logging.Logger | None
    :return: The path written.
    :rtype: Path
    :raises TypeError: If result is not a mapping or indent is not an
        integer.
    :raises ValueError: If indent is negative.
    :raises OSError: If the file cannot be written.
    """
    target = Path(path)
    if not isinstance(result, Mapping):
        _fail("[write_result] Input error: result must be a mapping, "
              f"not {type(result).__name__}.", logger, TypeError)
    if isinstance(indent, bool) or not isinstance(indent, int):
        _fail("[write_result] Input error: indent must be an integer, "
              f"not {type(indent).__name__}.", logger, TypeError)
    if indent < 0:
        _fail("[write_result] Input error: indent must not be "
              f"negative (>=0), not {indent}.", logger)
    target.write_text(json.dumps(result, indent=indent),
                      encoding="utf-8")
    return target


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
