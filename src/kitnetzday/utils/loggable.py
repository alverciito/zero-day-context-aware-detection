# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         src/kitnetzday/utils/loggable.py                          #
#   Version:      1.0.0                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-09-22                                                #
#   Modified:     2026-09-22                                                #
#   Description:  Base class that lets any class warn and fail through      #
#                 the logger handed down by the bin script.                 #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
from typing import NoReturn


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                MAIN CLASS                                 #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class Loggable:
    """
    Base class that gives any class a safe way to warn and to fail.

    Inherit from it and call ``super().__init__(logger=logger)`` at the
    start of your own ``__init__``. You then get two helpers:

    - ``_warning(message)`` writes a warning, if there is a logger, and
      the code goes on with the fallback the message describes.
    - ``_error(message, exception)`` writes an error, if there is a
      logger, and then always raises ``exception`` with the very same
      message, so the log and the traceback never disagree.

    The logger is created by the script in ``bin/`` and handed down. With
    ``logger=None`` (the default) nothing is written anywhere, which is
    handy in tests and notebooks.

    :param logger: Logger created by the calling script, or None to stay
        silent.
    :type logger: logging.Logger | None
    :raises TypeError: If logger is neither a logging.Logger nor None.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """
        Store the logger after checking that it really is one.

        :param logger: Logger created by the calling script, or None.
        :type logger: logging.Logger | None
        :raises TypeError: If logger is neither a logging.Logger nor None.
        """
        if logger is not None and not isinstance(logger, logging.Logger):
            raise TypeError("[Loggable.__init__] Input error: the parameter "
                            "logger must be a logging.Logger or None, not "
                            f"{type(logger).__name__}.")
        self.logger = logger

    def _warning(self, message: str) -> None:
        """
        Write a warning if there is a logger; do nothing otherwise.

        :param message: Text in the house format, for example "[Cls.meth]
            Input error: x must be > 0, not -1. x is truncated to 0.".
        :type message: str
        """
        if self.logger is not None:
            self.logger.warning(message)

    def _error(self,
               message: str,
               exception: type[Exception] = ValueError) -> NoReturn:
        """
        Write an error if there is a logger, then raise it.

        :param message: Text in the house format, used for both the log
            and the exception.
        :type message: str
        :param exception: Exception class to raise. ValueError for wrong
            values, TypeError for wrong types, FileNotFoundError for
            missing files, and so on.
        :type exception: type[Exception]
        :raises Exception: Always, of the class given in exception.
        """
        if self.logger is not None:
            self.logger.error(message)
        raise exception(message)

    def __repr__(self) -> str:
        """
        Show the class name and the name of its logger.

        :return: Text like "Loggable(logger='kitnetzday')".
        :rtype: str
        """
        name = None if self.logger is None else self.logger.name
        return f"{type(self).__name__}(logger={name!r})"


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
