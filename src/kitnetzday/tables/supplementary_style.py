# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         src/kitnetzday/tables/supplementary_style.py              #
#   Version:      1.0.0                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-06                                                #
#   Description:  House style of the supplementary tables: labels,          #
#                 number format and the preamble of every float.            #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
import statistics as st
from collections.abc import Mapping, Sequence
from typing import Any

from kitnetzday.tables.supplementary_inputs import (
                                                 SUPPL_CONFIG_RENAME,
                                                 SUPPL_DATASET_SHORT,
)
from kitnetzday.utils.loggable import Loggable


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                MAIN CLASS                                 #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class SupplementaryStyle(Loggable):
    """
    How the supplementary tables are written, as opposed to what is in
    them.

    One place for the four decisions that are repeated in every float of
    the supplement: which short label a stream gets, which label a
    configuration gets, how many decimals a number has and what is
    printed when there is no number, plus the preamble and the closing
    line of a float. The tables themselves only decide which value goes
    in which cell.

    The defaults reproduce the submitted supplement. The script in
    ``bin/`` reads ``src/config.json`` (section ``tables``) and passes
    the values here; library code never reads the configuration file.

    :param dataset_short: Dataset id -> short label of the rows and the
        column headers; the supplement labels by default.
    :type dataset_short: Mapping[str, str]
    :param config_rename: Long configuration label of the aggregate ->
        the label the supplement prints instead.
    :type config_rename: Mapping[str, str]
    :param decimals: Decimals of every printed number; 3 in the paper,
        0 or greater.
    :type decimals: int
    :param missing: Marker printed instead of a number that is not
        there; "--" in the paper.
    :type missing: str
    :param column_sep: Column separation in points of the full-width
        floats; 3.5 in the paper, strictly positive.
    :type column_sep: float
    :param seeds_column_sep: Column separation in points of the two
        per-seed floats; 3.0 in the paper, strictly positive.
    :type seeds_column_sep: float
    :param day_order: Days of the per-day table in print order; the
        days a corpus has that are not listed here are appended sorted.
    :type day_order: Sequence[str]
    :param logger: Logger created by the calling script, or None to stay
        silent.
    :type logger: logging.Logger | None
    :raises TypeError: If logger is not a logging.Logger or None.
    :raises ValueError: If decimals is not a non-negative integer or a
        column separation is not strictly positive.
    """

    def __init__(self,
                 dataset_short: Mapping[str, str] = SUPPL_DATASET_SHORT,
                 config_rename: Mapping[str, str] = SUPPL_CONFIG_RENAME,
                 decimals: int = 3,
                 missing: str = "--",
                 column_sep: float = 3.5,
                 seeds_column_sep: float = 3.0,
                 day_order: Sequence[str] = ("Mon", "Tue", "Wed", "Thu",
                                             "Fri"),
                 logger: logging.Logger | None = None) -> None:
        """
        Check the settings and keep them.

        :param dataset_short: Dataset id -> short label.
        :type dataset_short: Mapping[str, str]
        :param config_rename: Long label -> label of the supplement.
        :type config_rename: Mapping[str, str]
        :param decimals: Decimals of every number; 0 or greater.
        :type decimals: int
        :param missing: Marker of a value that is not there.
        :type missing: str
        :param column_sep: Column separation in points; positive.
        :type column_sep: float
        :param seeds_column_sep: Column separation of the per-seed
            floats in points; positive.
        :type seeds_column_sep: float
        :param day_order: Days of the per-day table in print order.
        :type day_order: Sequence[str]
        :param logger: Logger created by the calling script, or None.
        :type logger: logging.Logger | None
        :raises TypeError: If logger is not a logging.Logger or None.
        :raises ValueError: If decimals is negative or a column
            separation is not strictly positive.
        """
        super().__init__(logger=logger)
        if isinstance(decimals, bool) or not isinstance(decimals, int) \
                or decimals < 0:
            self._error("[SupplementaryStyle.__init__] Input error: the "
                        "parameter decimals must be a non-negative "
                        f"integer, not {decimals!r}.")
        for name, value in (("column_sep", column_sep),
                            ("seeds_column_sep", seeds_column_sep)):
            if not float(value) > 0.0:
                self._error("[SupplementaryStyle.__init__] Input error: "
                            f"the parameter {name} must be strictly "
                            f"positive (>0), not {value}.")
        self.dataset_shorts = dict(dataset_short)
        self.config_renames = dict(config_rename)
        self.decimals = int(decimals)
        self.missing = str(missing)
        self.column_sep = f"{float(column_sep):g}pt"
        self.seeds_column_sep = f"{float(seeds_column_sep):g}pt"
        self.day_order = [str(x) for x in day_order]

    # ----- numbers --------------------------------------------------- #
    def number(self,
               value: Any = None,
               decimals: int | None = None) -> str:
        """
        Write one number for a cell of the supplement.

        Note: this is not ``kitnetzday.utils.fnum`` and must not be
        replaced by it. ``fnum`` prints the missing marker for NaN and
        for the infinities and never raises; the original printed only
        None as the marker, so a NaN that reached a cell was printed as
        "nan" and a value that is not a number raised. The behaviour is
        kept, so that a broken cell stays visible instead of silent.

        :param value: Value to write, or None for the missing marker.
        :type value: Any
        :param decimals: Decimals to use, or None for the decimals of
            the instance (3 in the paper).
        :type decimals: int | None
        :return: The formatted number, for example "0.873", or the
            missing marker.
        :rtype: str
        :raises TypeError: If value is neither None nor a number, as in
            the original.
        """
        nd = self.decimals if decimals is None else int(decimals)
        if value is None:
            return self.missing
        return f"{value:.{nd}f}"

    def signed(self,
               value: Any = None,
               decimals: int | None = None) -> str:
        """
        Write one change, always with its sign, as the deltas need.

        :param value: Change to write, or None for the missing marker.
        :type value: Any
        :param decimals: Decimals to use, or None for the decimals of
            the instance.
        :type decimals: int | None
        :return: The formatted change, for example "-0.004" or
            "+0.000", or the missing marker.
        :rtype: str
        :raises TypeError: If value is neither None nor a number.
        """
        nd = self.decimals if decimals is None else int(decimals)
        if value is None:
            return self.missing
        return f"{value:+.{nd}f}"

    def median(self, values: Sequence[Any] | None = None) -> Any:
        """
        Median of a list of changes, or None when there is none.

        :param values: Changes collected for one cell, or None when
            nothing was collected for it.
        :type values: Sequence[Any] | None
        :return: The median as ``statistics.median`` computes it, or
            None for an empty or missing list.
        :rtype: Any
        """
        kept = values or list()
        return st.median(kept) if kept else None

    # ----- labels ---------------------------------------------------- #
    def stream(self, dataset: str) -> str:
        """
        Short label of a stream, as the supplement prints it.

        :param dataset: Dataset id such as "D2".
        :type dataset: str
        :return: The short label ("Fuzz"), or the id itself.
        :rtype: str
        :raises ValueError: If dataset is not a non-empty string.
        """
        if not isinstance(dataset, str) or not dataset:
            self._error("[SupplementaryStyle.stream] Input error: the "
                        "parameter dataset must be a non-empty string, "
                        f"not {dataset!r}.")
        return self.dataset_shorts.get(dataset, dataset)

    def row(self, name: str) -> str:
        """
        Label of a configuration row, as the supplement prints it.

        :param name: Long label of the configuration, as ``cells.csv``
            writes it, for example "original (KitNET-style)".
        :type name: str
        :return: The renamed label ("KitNET-style dA"), or the long
            label when there is nothing to rename.
        :rtype: str
        :raises ValueError: If name is not a non-empty string.
        """
        if not isinstance(name, str) or not name:
            self._error("[SupplementaryStyle.row] Input error: the "
                        "parameter name must be a non-empty string, "
                        f"not {name!r}.")
        return self.config_renames.get(name, name)

    def header(self, first: str, datasets: Sequence[str]) -> str:
        """
        Header row of a table whose columns are the streams.

        :param first: Title of the first column, "Configuration" in
            every table of the supplement that uses this header.
        :type first: str
        :param datasets: Dataset ids of the columns, in column order.
        :type datasets: Sequence[str]
        :return: The whole header row, the ``\\\\`` of LaTeX included.
        :rtype: str
        :raises ValueError: If first is not a non-empty string, or a
            dataset id is not one.
        """
        if not isinstance(first, str) or not first:
            self._error("[SupplementaryStyle.header] Input error: the "
                        "parameter first must be a non-empty string, "
                        f"not {first!r}.")
        return (r"\textbf{" + first + "} & "
                + " & ".join(r"\textbf{" + self.stream(d) + "}"
                             for d in datasets) + r"\\")

    def days(self, collected: Mapping[str, Any]) -> list[str]:
        """
        Days of one corpus in print order.

        :param collected: Day as the per-day block names it -> whatever
            was collected for it; only the keys are read.
        :type collected: Mapping[str, Any]
        :return: The declared days the corpus has, in the declared
            order, followed by the rest sorted alphabetically.
        :rtype: list[str]
        :raises TypeError: If collected is not a mapping.
        """
        if not isinstance(collected, Mapping):
            self._error("[SupplementaryStyle.days] Input error: the "
                        "parameter collected must be a mapping, not "
                        f"{type(collected).__name__}.", TypeError)
        return ([d for d in self.day_order if d in collected]
                + sorted(k for k in collected
                         if k not in self.day_order))

    # ----- floats ---------------------------------------------------- #
    def float_begin(self, wide: bool = True) -> str:
        """
        Opening line of a float of the supplement.

        :param wide: True for the full-width ``table*`` floats in
            ``\\footnotesize``, False for the two per-seed ``table``
            floats in ``\\scriptsize``, whose rows are longer.
        :type wide: bool
        :return: The whole opening line, column separation included.
        :rtype: str
        """
        if wide:
            return (r"\begin{table*}[ht]\centering\footnotesize"
                    r"\setlength{\tabcolsep}{" + self.column_sep + "}")
        return (r"\begin{table}[ht]\centering\scriptsize"
                r"\setlength{\tabcolsep}{" + self.seeds_column_sep
                + "}")

    def float_end(self, wide: bool = True) -> str:
        """
        Closing line of a float of the supplement.

        :param wide: True for the full-width floats, False for the two
            per-seed ones.
        :type wide: bool
        :return: ``\\end{table*}`` or ``\\end{table}``.
        :rtype: str
        """
        return r"\end{table*}" if wide else r"\end{table}"

    def __repr__(self) -> str:
        """
        Show the format and how many labels are known.

        :return: Text like "SupplementaryStyle(decimals=3,
            missing='--', streams=9, renamed=2)".
        :rtype: str
        """
        return (f"SupplementaryStyle(decimals={self.decimals}, "
                f"missing={self.missing!r}, "
                f"streams={len(self.dataset_shorts)}, "
                f"renamed={len(self.config_renames)})")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
