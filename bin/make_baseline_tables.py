# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         bin/make_baseline_tables.py                               #
#   Version:      1.0.0                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-07                                                #
#   Modified:     2026-10-07                                                #
#   Description:  Writes the values of the two comparator tables: the       #
#                 reference KitNET on the KNAD captures and the Isolation   #
#                 Forest row of the main results.                           #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from kitnetzday.naming import Naming
from kitnetzday.tables import BaselineTables, SupplementaryInputs
from kitnetzday.utils import LoggerFactory

# Name this script is recorded under in the log of the run.
GENERATOR: str = "bin/make_baseline_tables.py"
# Names of the two artefacts inside the output directory.
JSON_NAME: str = "baselines.json"
MD_NAME: str = "baselines.md"
# What to tell the reader when the comparator jobs are missing.
NO_JOB_FILES: str = (
    "ERROR: no comparator job was found, so nothing was written.\n"
    "The two tables read the metrics.json of the T5-KITNET__* and "
    "T5-IF__* jobs under --results-root, which frozen/runner/"
    "baselines_t5.py produced. Those files are shipped in data/jobs/.\n"
    "Looked in: {results_root}")


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
        description="Comparator tables: reference KitNET and Isolation "
                    "Forest (JISAS-D-26-02300).")
    parser.add_argument("--aggregate", default=None,
                        help="directory of the aggregate: runs.csv, "
                             "stats.json and aggregate/cells.csv; "
                             "default: the directory of paths.runs in "
                             "src/config.json")
    parser.add_argument("--results-root", default=None,
                        help="directory with one sub-directory per job; "
                             "default: paths.results_root")
    parser.add_argument("--out-json", default=None,
                        help=f"default: <out_dir>/{JSON_NAME}")
    parser.add_argument("--out-md", default=None,
                        help=f"default: <out_dir>/{MD_NAME}")
    return parser.parse_args(argv)


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


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                   MAIN                                    #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def main(argv: Sequence[str] | None = None) -> int:
    """
    Write the values of the two comparator tables of the paper.

    Steps: load ``src/config.json``, build the run logger in ``logs/``,
    read the aggregated pass and the per-job ``metrics.json`` files,
    read the single run of the published KitNET implementation on the
    six KNAD captures and of the Isolation Forest on the nine streams,
    and write a JSON document and a Markdown preview. No metric is
    computed here.

    Run it with: uv run bin/make_baseline_tables.py

    :param argv: Command line to parse, or None to read ``sys.argv``.
    :type argv: Sequence[str] | None
    :return: 0 when both files were written, 1 for an input that is
        missing and for a results root without the comparator jobs, and
        3 when a file cannot be written.
    :rtype: int
    :raises SystemExit: If the command line is not valid.
    """
    data = None
    script = Path(__file__).resolve()
    root = script.parents[1]
    config = json.loads((root / "src" / "config.json").read_text(
        encoding="utf-8"))
    paths = config["paths"]
    study = config["study"]
    args = _parse_args(argv)
    logger = LoggerFactory(log_dir=root / paths["logs"],
                           name=config["logging"]["name"],
                           level=config["logging"]["level"],
                           console=config["logging"]["console"]).build()
    aggregate = (Path(args.aggregate) if args.aggregate
                 else (root / paths["runs"]).parent)
    results_root = (Path(args.results_root) if args.results_root
                    else root / paths["results_root"])
    out_json = (Path(args.out_json) if args.out_json
                else root / paths["out_dir"] / JSON_NAME)
    out_md = (Path(args.out_md) if args.out_md
              else root / paths["out_dir"] / MD_NAME)
    try:
        data = SupplementaryInputs(
            aggregate, results_root, naming=Naming(
                datasets=study["datasets"],
                configurations=study["configurations"],
                context_free=study["context_free_configurations"],
                architectures=study["architectures"],
                rope=config["analysis_plan"]["rope_half_width"],
                logger=logger), logger=logger)
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}")
        return 1
    tables = BaselineTables(data, logger=logger)
    if not tables.kitnet or not tables.forest:
        print(NO_JOB_FILES.format(results_root=results_root))
        return 1
    try:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        tables.write(out_json, out_md, _relative(aggregate, root))
    except OSError as error:
        print(f"ABORT: {error}")
        return 3
    print(tables.markdown())
    print(f"-> {_relative(out_json, root)}")
    print(f"-> {_relative(out_md, root)}")
    logger.info("[main] %s wrote %s and %s from %s.", GENERATOR, out_json,
                out_md, tables)
    return 0


if __name__ == "__main__":
    sys.exit(main())


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
