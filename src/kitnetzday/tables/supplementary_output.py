# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         src/kitnetzday/tables/supplementary_output.py             #
#   Version:      1.0.0                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-07                                                #
#   Description:  Writes the two artefacts of the supplement, the LaTeX     #
#                 tables and the JSON summary, and its console report.      #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import json
import logging
from pathlib import Path
from typing import Any

from kitnetzday.tables.supplementary import SupplementaryTables
from kitnetzday.tables.supplementary_summaries import SupplementarySummaries
from kitnetzday.utils.loggable import Loggable


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                MAIN CLASS                                 #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class SupplementaryOutput(Loggable):
    """
    The two files the supplement leaves behind, and what to print.

    The LaTeX file holds the fifteen floats ready to paste, and the
    JSON file holds the value of every cell plus the five summaries the
    body of the paper quotes. The console report names the two files and
    repeats the summaries, so that a run can be read without opening
    them.

    Note: the files are written directly and not through
    ``kitnetzday.io.Out``. That class prepends the provenance comments of a
    per-table artefact and writes one file per table, which would change
    the bytes of the submitted supplement.

    :param tables: The nine tables, already built.
    :type tables: SupplementaryTables
    :param summaries: The four summaries, or None to compute them from
        the inputs of the tables with the defaults.
    :type summaries: SupplementarySummaries | None
    :param logger: Logger created by the calling script, or None to stay
        silent.
    :type logger: logging.Logger | None
    :raises TypeError: If tables, summaries or logger are of the wrong
        type.
    """

    def __init__(self,
                 tables: SupplementaryTables,
                 summaries: SupplementarySummaries | None = None,
                 logger: logging.Logger | None = None) -> None:
        """
        Keep the tables and the summaries that are going to be written.

        :param tables: The nine tables, already built.
        :type tables: SupplementaryTables
        :param summaries: The four summaries, or None to compute them.
        :type summaries: SupplementarySummaries | None
        :param logger: Logger created by the calling script, or None.
        :type logger: logging.Logger | None
        :raises TypeError: If tables, summaries or logger are of the
            wrong type.
        """
        super().__init__(logger=logger)
        if not isinstance(tables, SupplementaryTables):
            self._error("[SupplementaryOutput.__init__] Input error: "
                        "the parameter tables must be a "
                        "SupplementaryTables, not "
                        f"{type(tables).__name__}.", TypeError)
        if summaries is not None and \
                not isinstance(summaries, SupplementarySummaries):
            self._error("[SupplementaryOutput.__init__] Input error: "
                        "the parameter summaries must be a "
                        "SupplementarySummaries or None, not "
                        f"{type(summaries).__name__}.", TypeError)
        self.tables = tables
        self.summaries = summaries or SupplementarySummaries(
            tables.data, logger=logger)

    def document(self, source: str | None = None) -> dict[str, Any]:
        """
        The JSON document: the values of the tables and the summaries.

        The provenance is the directory of the aggregated pass the
        numbers were read from.

        :param source: Text written as the provenance, or None to write
            the directory of the aggregated pass as it was given. The
            calling script passes a path relative to the package, so
            that the document does not depend on where it was run.
        :type source: str | None
        :return: The document, with the keys in the order the original
            inserted them.
        :rtype: dict[str, Any]
        """
        return self.summaries.document(
            self.tables.tables,
            str(self.tables.data.aggregate_dir) if source is None
            else source)

    def write(self,
              out_tex: str | Path,
              out_json: str | Path,
              source: str | None = None) -> tuple[Path, Path]:
        """
        Write the two artefacts of the supplement.

        The LaTeX goes out as one file with the fifteen floats and no
        trailing newline, and the JSON with indent 1 and the default
        ASCII escaping, exactly as the original wrote them.

        :param out_tex: Path of the ``.tex`` to write; its directory
            must exist.
        :type out_tex: str | Path
        :param out_json: Path of the ``.json`` to write; its directory
            must exist.
        :type out_json: str | Path
        :param source: Provenance text of the JSON document, or None
            for the directory of the aggregated pass as given.
        :type source: str | None
        :return: The two paths written, the LaTeX one first.
        :rtype: tuple[Path, Path]
        :raises OSError: If a file cannot be written.
        """
        tex_path = Path(out_tex)
        json_path = Path(out_json)
        tex_path.write_text(self.tables.tex_text(), encoding="utf-8")
        json_path.write_text(json.dumps(self.document(source),
                                        indent=1),
                             encoding="utf-8")
        return tex_path, json_path

    def report_lines(self,
                     out_tex: str | Path,
                     out_json: str | Path) -> list[str]:
        """
        The console report of one run, line by line.

        The script in ``bin/`` is the one that prints them, because
        library code does not print. The count of the first line is the
        number of LaTeX lines emitted before the caption pass.

        :param out_tex: Path of the ``.tex`` written.
        :type out_tex: str | Path
        :param out_json: Path of the ``.json`` written.
        :type out_json: str | Path
        :return: The two file lines followed by the four summaries.
        :rtype: list[str]
        :raises TypeError: If a summary that is quoted has no median,
            that is, when no cell contributed to it.
        """
        lines = [f"-> {out_tex} ({len(self.tables.lines)} lines)",
                 f"-> {out_json}"]
        lines.extend(self.summaries.report_lines())
        return lines

    def __repr__(self) -> str:
        """
        Show what is going to be written.

        :return: Text like "SupplementaryOutput(tables=8, lines=214,
            summaries=4)".
        :rtype: str
        """
        return (f"SupplementaryOutput(tables={len(self.tables.tables)}, "
                f"lines={len(self.tables.lines)}, summaries=4)")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
