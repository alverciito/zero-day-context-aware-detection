# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         tests/utils/test_logging_tools.py                         #
#   Version:      1.0.0                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-09-22                                                #
#   Modified:     2026-09-22                                                #
#   Description:  Unit tests for Loggable and LoggerFactory.                #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
import re
from pathlib import Path

import pytest

from kitnetzday.utils.loggable import Loggable
from kitnetzday.utils.logger_factory import LoggerFactory


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                MAIN CLASS                                 #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class TestLoggingTools:
    """
    Tests for the logging helpers shared by every class.
    """

    def test_loggable_rejects_non_logger(self) -> None:
        """
        A logger that is not a logging.Logger raises TypeError.
        """
        with pytest.raises(TypeError, match="logging.Logger or None"):
            Loggable(logger="not a logger")

    def test_loggable_error_logs_and_raises_same_text(
            self,
            caplog: pytest.LogCaptureFixture) -> None:
        """
        _error writes the message and raises the chosen exception with it.

        :param caplog: pytest fixture that captures log records.
        :type caplog: pytest.LogCaptureFixture
        """
        helper = Loggable(logger=logging.getLogger("test"))
        with caplog.at_level(logging.ERROR, logger="test"):
            with pytest.raises(KeyError):
                helper._error("[Test] Input error: boom.", KeyError)
        assert caplog.records[-1].message == "[Test] Input error: boom."

    def test_loggable_repr(self) -> None:
        """
        __repr__ shows the logger name, or None without logger.
        """
        assert repr(Loggable()) == "Loggable(logger=None)"
        assert repr(Loggable(logging.getLogger("x"))) == \
            "Loggable(logger='x')"

    def test_factory_writes_timestamped_file(self, tmp_path: Path) -> None:
        """
        build() creates logs/<timestamp>.log and writes messages to it.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        factory = LoggerFactory(log_dir=tmp_path / "logs", name="t1",
                                console=False)
        logger = factory.build()
        logger.info("hello")
        for handler in logger.handlers:
            handler.flush()
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.log",
                            factory.log_path.name)
        assert "hello" in factory.log_path.read_text(encoding="utf-8")
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)

    def test_factory_build_twice_does_not_duplicate(
            self,
            tmp_path: Path) -> None:
        """
        Building twice leaves one file handler and one console handler.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        factory = LoggerFactory(log_dir=tmp_path, name="t2", console=True)
        factory.build()
        logger = factory.build()
        assert len(logger.handlers) == 2
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)

    def test_factory_rejects_bad_settings(self) -> None:
        """
        Empty names and unknown levels raise ValueError.
        """
        with pytest.raises(ValueError, match="non-empty string"):
            LoggerFactory(name="  ")
        with pytest.raises(ValueError, match="logging level"):
            LoggerFactory(level="LOUD")
        with pytest.raises(ValueError, match="logging level"):
            LoggerFactory(level=True)

    def test_factory_repr(self, tmp_path: Path) -> None:
        """
        __repr__ shows name, level, folder, console and file.

        :param tmp_path: pytest fixture with a temporary folder.
        :type tmp_path: Path
        """
        factory = LoggerFactory(log_dir=tmp_path, name="t3", level=10,
                                console=False)
        assert "level='DEBUG'" in repr(factory)
        assert "log_path=None" in repr(factory)


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
