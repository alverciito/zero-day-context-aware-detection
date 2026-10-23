# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         bin/analyse_retrained_pair.py                             #
#   Version:      1.1.1                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Paired comparison of the retraining experiment: the       #
#                 windowed model against the per-record one, by cell.       #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from kitnetzday.audit import RetrainedPair, RetrainedPairReport
from kitnetzday.utils import LoggerFactory

# Name this script is recorded under in the log of the run.
GENERATOR: str = "bin/analyse_retrained_pair.py"
# Name of the result file inside the output directory.
JSON_NAME: str = "retrained_pair.json"
# What to tell the reader when no cell could be paired.
NOTHING_PAIRED: str = (
    "ERROR: no cell of the retraining experiment could be paired, so nothing "
    "was written.\nThe comparison reads primary.value of every "
    "T8-*/metrics.json under --results-root. Those files are shipped "
    "in data/jobs/; the full results tree is the directory the runner "
    "wrote (frozen/runner/worker.py --results-root).\n"
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
        description="Retraining experiment and training-budget ablation "
                    "(JISAS-D-26-02300). It trains nothing and uses no "
                    "GPU.")
    parser.add_argument("--json", default=None,
                        help=f"default: <out_dir>/{JSON_NAME}")
    parser.add_argument("--results-root", default=None,
                        help="directory with the T8-* jobs; default: "
                             "paths.results_root of src/config.json")
    parser.add_argument("--repo-root", default=None,
                        help="root against which the relative path of "
                             "every cell is written; default: the root "
                             "of this package")
    return parser.parse_args(argv)


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                   MAIN                                    #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def main(argv: Sequence[str] | None = None) -> int:
    """
    Pair the two variants of the retraining experiment and report.

    Steps: load ``src/config.json``, build the run logger in ``logs/``,
    read the primary metric of every ``T8-*/metrics.json``, pair the
    windowed variant against the per-record control over the cells they
    share (dataset, paradigm, seed, repetition), and write the JSON with
    the wins, the median, the Hodges-Lehmann estimate and its exact
    interval, the signed rank p value, the two paradigm halves and the
    per (dataset, paradigm) averages. The files are only read: nothing
    is trained and the results tree is never written to.

    Run it with: uv run bin/analyse_retrained_pair.py

    The per-job ``metrics.json`` files travel in ``data/jobs/``, so the
    default run reproduces the retraining experiment. A run in which no
    cell can be paired writes nothing and ends with exit code 1.

    :param argv: Command line to parse, or None to read ``sys.argv``.
    :type argv: Sequence[str] | None
    :return: 0 when the report was written, 1 when a metrics file lies
        outside the repository root or no cell could be paired, and 3
        when the file cannot be written.
    :rtype: int
    :raises SystemExit: If the command line is not valid.
    """
    report = dict()
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
    out_json = (Path(args.json) if args.json
                else root / paths["out_dir"] / JSON_NAME)
    results_root = Path(args.results_root
                        or root / paths["results_root"])
    pair = RetrainedPair(results_root=results_root,
                         repo_root=args.repo_root or root, logger=logger)
    try:
        report = pair.report()
    except ValueError as error:
        print(f"ERROR: {error}")
        return 1
    if not any(block.get("n_cells") for block
               in report["pairs"].values()):
        print(NOTHING_PAIRED.format(results_root=results_root))
        return 1
    try:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        pair.dump_json(report, out_json)
    except OSError as error:
        print(f"ABORT: {error}")
        return 3
    for text in RetrainedPairReport(logger=logger).render(report,
                                                          out_json):
        print(text)
    logger.info("[main] %s paired %d cell(s) into %s.", GENERATOR,
                len(pair.cells), out_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
