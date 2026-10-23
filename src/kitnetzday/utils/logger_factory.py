# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         src/kitnetzday/utils/logger_factory.py                    #
#   Version:      1.0.0                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-09-22                                                #
#   Modified:     2026-09-22                                                #
#   Description:  Builds the logger of one run: a timestamped file in       #
#                 logs/ plus, optionally, the console.                      #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
from datetime import datetime
from pathlib import Path


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                MAIN CLASS                                 #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class LoggerFactory:
    """
    Builds the logger of one run. Only the scripts in bin/ use it.

    The logger writes to a new file inside ``log_dir`` named after the
    moment the run started, for example ``logs/2026-09-22_17-05-43.log``,
    and can also print every message on the console. Library classes
    never call this factory: they receive the finished logger as a
    parameter.

    :param log_dir: Folder for the log file. It is created if missing.
    :type log_dir: str | Path
    :param name: Name of the logger, usually the project name.
    :type name: str
    :param level: Minimum level to write, as a name ("INFO") or a number
        (logging.INFO).
    :type level: str | int
    :param console: Also print the messages on the terminal when True.
    :type console: bool
    :param timestamp_format: strftime pattern for the file name. Avoid
        colons: Windows does not allow them in file names.
    :type timestamp_format: str
    :param message_format: logging pattern for each line.
    :type message_format: str
    :raises ValueError: If name is empty or level is unknown.
    """

    def __init__(self,
                 log_dir: str | Path = "logs",
                 name: str = "app",
                 level: str | int = "INFO",
                 console: bool = True,
                 timestamp_format: str = "%Y-%m-%d_%H-%M-%S",
                 message_format: str = "%(asctime)s | %(levelname)-8s | "
                                       "%(message)s") -> None:
        """
        Validate the settings. Nothing is created until build() is called.

        :param log_dir: Folder for the log file. It is created if missing.
        :type log_dir: str | Path
        :param name: Name of the logger, usually the project name.
        :type name: str
        :param level: Minimum level to write, as a name or a number.
        :type level: str | int
        :param console: Also print the messages on the terminal when True.
        :type console: bool
        :param timestamp_format: strftime pattern for the file name.
        :type timestamp_format: str
        :param message_format: logging pattern for each line.
        :type message_format: str
        :raises ValueError: If name is empty or level is unknown.
        """
        numeric_level = level
        if not isinstance(name, str) or not name.strip():
            raise ValueError("[LoggerFactory.__init__] Input error: the "
                             "parameter name must be a non-empty string, "
                             f"not {name!r}.")
        if isinstance(level, str):
            numeric_level = logging.getLevelName(level.upper())
        if not isinstance(numeric_level, int) or \
                isinstance(numeric_level, bool):
            raise ValueError("[LoggerFactory.__init__] Input error: the "
                             "parameter level must be a logging level such "
                             f"as 'INFO' or 20, not {level!r}.")
        self.log_dir = Path(log_dir)
        self.name = name
        self.level = numeric_level
        self.console = console
        self.timestamp_format = timestamp_format
        self.message_format = message_format
        self.log_path = None

    def build(self) -> logging.Logger:
        """
        Create the log folder and file and return the configured logger.

        Calling build() twice replaces the handlers instead of adding
        more, so messages are never written twice.

        :return: Logger ready to be passed to the library classes.
        :rtype: logging.Logger
        """
        timestamp = datetime.now().strftime(self.timestamp_format)
        formatter = logging.Formatter(self.message_format)
        logger = logging.getLogger(self.name)
        handlers = list()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / f"{timestamp}.log"
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
        handlers.append(logging.FileHandler(self.log_path, encoding="utf-8"))
        if self.console:
            handlers.append(logging.StreamHandler())
        for handler in handlers:
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(self.level)
        logger.propagate = False
        return logger

    def __repr__(self) -> str:
        """
        Show where and at which level the logger writes.

        :return: Text like "LoggerFactory(name='app', level='INFO', ...)".
        :rtype: str
        """
        shown_path = None if self.log_path is None else str(self.log_path)
        return (f"LoggerFactory(name={self.name!r}, "
                f"level={logging.getLevelName(self.level)!r}, "
                f"log_dir={str(self.log_dir)!r}, console={self.console}, "
                f"log_path={shown_path!r})")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
