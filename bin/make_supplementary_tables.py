# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         bin/make_supplementary_tables.py                          #
#   Version:      1.1.0                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-07                                                #
#   Description:  Writes fifteen LaTeX tables of the supplementary          #
#                 material and the JSON summary the body of the paper       #
#                 cites.                                                    #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from kitnetzday.naming import Naming
from kitnetzday.tables import (
                            SupplementaryInputs,
                            SupplementaryOutput,
                            SupplementaryStyle,
                            SupplementaryTables,
)
from kitnetzday.utils import LoggerFactory

# Name this script is recorded under in the log of the run.
GENERATOR: str = "bin/make_supplementary_tables.py"
# Names of the two artefacts inside the output directory.
TEX_NAME: str = "supplementary_tables.tex"
JSON_NAME: str = "supplementary_tables.json"
# Tables that are averages over the per-job records: when none of their
# cells holds a number, the per-job files were not found.
PER_JOB_TABLES: tuple[str, ...] = ("tab:s_wd", "tab:s_loc", "tab:s_kmd",
                                   "tab:s_mvd", "tab:s_fpr",
                                   "tab:s_alarm")
# What to tell the reader when the per-job records are missing.
NO_JOB_FILES: str = (
    "ERROR: no per-job metrics.json was found, so nothing was "
    "written.\nThese tables average fields of "
    "<results_root>/<job_id>/metrics.json over the seeds of each cell. "
    "Those files are shipped in data/jobs/; the full results tree is "
    "the directory the runner wrote (frozen/runner/worker.py "
    "--results-root).\nLooked in: {results_root}")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                AUXILIARY                                  #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """
    Read the command line.

    :param argv: Arguments to parse, or None to read ``sys.argv``.
    :type argv: Sequence[str] | None
    :return: The parsed arguments.
    :rtype: argparse.Namespace
    :raises SystemExit: If the command line is not valid.
    """
    parser = argparse.ArgumentParser(
        description="Tables of the supplementary material "
                    "(JISAS-D-26-02300).")
    parser.add_argument("--aggregate", default=None,
                        help="directory of the aggregate: runs.csv, "
                             "stats.json and aggregate/cells.csv; "
                             "default: the directory of paths.runs in "
                             "src/config.json")
    parser.add_argument("--results-root", default=None,
                        help="directory with one sub-directory per job "
                             "and its metrics.json; default: "
                             "paths.results_root")
    parser.add_argument("--out-tex", default=None,
                        help=f"default: <out_dir>/{TEX_NAME}")
    parser.add_argument("--out-json", default=None,
                        help=f"default: <out_dir>/{JSON_NAME}")
    return parser.parse_args(argv)


def _vocabulary(config: dict[str, Any],
                logger: logging.Logger | None = None) -> Naming:
    """
    Build the vocabulary of the study from the configuration file.

    :param config: Parsed ``src/config.json``.
    :type config: dict[str, Any]
    :param logger: Logger of this run, or None.
    :type logger: logging.Logger | None
    :return: The ids and their presentation order.
    :rtype: Naming
    :raises KeyError: If the configuration has no "study" section.
    """
    study = config["study"]
    return Naming(datasets=study["datasets"],
                  configurations=study["configurations"],
                  context_free=study["context_free_configurations"],
                  architectures=study["architectures"],
                  rope=config["analysis_plan"]["rope_half_width"],
                  logger=logger)


def _style(config: dict[str, Any],
           logger: logging.Logger | None = None) -> SupplementaryStyle:
    """
    Build the house style of the tables from the configuration file.

    :param config: Parsed ``src/config.json``.
    :type config: dict[str, Any]
    :param logger: Logger of this run, or None.
    :type logger: logging.Logger | None
    :return: The number format and the labels of the supplement.
    :rtype: SupplementaryStyle
    :raises KeyError: If the configuration has no "tables" section.
    """
    tables = config["tables"]
    return SupplementaryStyle(
        decimals=tables["decimals"], missing=tables["missing_marker"],
        column_sep=tables["column_separation_pt"], logger=logger)


def _relative(path: Path, root: Path) -> str:
    """
    Write a path relative to the package root when it lies inside it.

    :param path: Path to show.
    :type path: Path
    :param root: Root of this package.
    :type root: Path
    :return: The relative path with forward slashes, or the path as
        given when it is outside the package.
    :rtype: str
    """
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _has_job_records(tables: SupplementaryTables) -> bool:
    """
    Whether any cell of the per-job tables holds a number.

    :param tables: The tables, already built.
    :type tables: SupplementaryTables
    :return: False when every cell of every per-job table is empty,
        which is what a results root without job files produces.
    :rtype: bool
    """
    return any(value is not None
               for label in PER_JOB_TABLES
               for row in tables.tables.get(label, dict()).values()
               for value in row.values())


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                   MAIN                                    #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def main(argv: Sequence[str] | None = None) -> int:
    """
    Write fifteen supplementary tables and their JSON summary.

    Steps: load ``src/config.json``, build the run logger in ``logs/``,
    read the aggregated pass and every per-job ``metrics.json`` through
    ``SupplementaryInputs``, build the fifteen floats, and write the
    ``.tex`` ready to paste and the ``.json`` with the value of every
    cell plus five summaries, four of which the body of the paper
    quotes. No
    metric is computed here: the fields the aggregation left written
    are read and averaged over the two seeds of each cell, and the
    published means of ROC-AUC, F1 and AUPRC come from ``cells.csv``.

    Run it with: uv run bin/make_supplementary_tables.py

    The per-job ``metrics.json`` files travel in ``data/jobs/``. A run
    that finds none of them writes nothing and ends with exit code 1,
    instead of leaving tables full of missing markers.

    :param argv: Command line to parse, or None to read ``sys.argv``.
    :type argv: Sequence[str] | None
    :return: 0 when both files were written, 1 for an input that is
        missing or unreadable and for a pass without the per-job
        metrics, and 3 when a file cannot be written.
    :rtype: int
    :raises SystemExit: If the command line is not valid.
    """
    data = None
    written: tuple[Path, Path] = (Path(), Path())
    script = Path(__file__).resolve()
    root = script.parents[1]
    config = json.loads((root / "src" / "config.json").read_text(
        encoding="utf-8"))
    paths = config["paths"]
    args = _parse_args(argv)
    logger = LoggerFactory(log_dir=root / paths["logs"],
                           name=config["logging"]["name"],
                           level=config["logging"]["level"],
                           console=config["logging"]["console"]).build()
    aggregate = (Path(args.aggregate) if args.aggregate
                 else (root / paths["runs"]).parent)
    results_root = (Path(args.results_root) if args.results_root
                    else root / paths["results_root"])
    out_tex = (Path(args.out_tex) if args.out_tex
               else root / paths["out_dir"] / TEX_NAME)
    out_json = (Path(args.out_json) if args.out_json
                else root / paths["out_dir"] / JSON_NAME)
    try:
        data = SupplementaryInputs(aggregate, results_root,
                                   naming=_vocabulary(config, logger),
                                   logger=logger)
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}")
        return 1
    tables = SupplementaryTables(data=data,
                                 style=_style(config, logger),
                                 logger=logger)
    if not _has_job_records(tables):
        print(NO_JOB_FILES.format(results_root=results_root))
        return 1
    out = SupplementaryOutput(tables, logger=logger)
    try:
        out_tex.parent.mkdir(parents=True, exist_ok=True)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        written = out.write(out_tex, out_json,
                            source=_relative(aggregate, root))
    except OSError as error:
        print(f"ABORT: {error}")
        return 3
    for text in out.report_lines(_relative(written[0], root),
                                 _relative(written[1], root)):
        print(text)
    logger.info("[main] %s wrote %s and %s from %s.", GENERATOR,
                written[0], written[1], data)
    return 0


if __name__ == "__main__":
    sys.exit(main())


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
