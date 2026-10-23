# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         bin/analyse_reconstruction.py                             #
#   Version:      1.1.0                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-07                                                #
#   Description:  Exploratory correlations between reconstruction           #
#                 quality and detection, with permutation p-values.         #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from kitnetzday.audit import ReconstructionCorrelation, write_result
from kitnetzday.utils import LoggerFactory

# Name this script is recorded under in the log of the run.
GENERATOR: str = "bin/analyse_reconstruction.py"
# Name of the result file inside the output directory.
JSON_NAME: str = "reconstruction_vs_detection.json"
# What to tell the reader when an input is missing.
MISSING_INPUT: str = (
    "This correlation needs two inputs: the result of the training "
    "audit (--audit; run bin/audit_training.py first, which writes "
    "{audit}) and the T1-*/metrics.json of the main grid under "
    "--results-root (shipped in data/jobs/).\n"
    "Looked in: {audit} | {results_root}")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                AUXILIARY                                  #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def _parse_args(argv: Sequence[str] | None = None,
                draws: int = 100000) -> argparse.Namespace:
    """
    Read the command line.

    :param argv: Arguments to parse, or None to read ``sys.argv``.
    :type argv: Sequence[str] | None
    :param draws: Default number of permutations, from the frozen plan
        in the configuration file.
    :type draws: int
    :return: The parsed arguments.
    :rtype: argparse.Namespace
    :raises SystemExit: If the command line is not valid.
    """
    parser = argparse.ArgumentParser(
        description="Exploratory correlations of the failure-case "
                    "analysis (JISAS-D-26-02300). It trains nothing "
                    "and uses no GPU.")
    parser.add_argument("--out", default=None,
                        help=f"default: <out_dir>/{JSON_NAME}")
    parser.add_argument("--audit", default=None,
                        help="result file of bin/audit_training.py; "
                             "default: paths.training_audit of "
                             "src/config.json")
    parser.add_argument("--results-root", default=None,
                        help="directory with the T1-* jobs; default: "
                             "paths.results_root")
    parser.add_argument("--draws", type=int, default=draws,
                        help="permutations per correlation; lowering "
                             "it is only good for a quick test, the "
                             "published figures use the plan value")
    return parser.parse_args(argv)


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                   MAIN                                    #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def main(argv: Sequence[str] | None = None) -> int:
    """
    Correlate reconstruction quality with detection, twice.

    Steps: load ``src/config.json``, build the run logger in ``logs/``,
    read the loss-to-floor ratios of the training audit and the two
    ROC-AUC values of every ``T1-*/metrics.json``, keep the variants
    that do leave the floor, de-duplicate the rows the MLP repeats, and
    write the Spearman rho of the raw error and of the fused detector
    with their within-group permutation p-values. Both analyses are
    exploratory and the result file says so.

    Run it with: uv run bin/audit_training.py, then
    uv run bin/analyse_reconstruction.py

    The audit result is the output of ``bin/audit_training.py`` and
    the job files travel in ``data/jobs/``, so the two commands above
    reproduce the two correlations of the failure-case analysis. A
    missing input
    ends the run with exit code 1 and the list of what is needed.

    :param argv: Command line to parse, or None to read ``sys.argv``.
    :type argv: Sequence[str] | None
    :return: 0 when the result was written, 1 when an input is missing
        or too few rows survive the filters, and 3 when the file cannot
        be written.
    :rtype: int
    :raises SystemExit: If the command line is not valid.
    """
    result = dict()
    script = Path(__file__).resolve()
    root = script.parents[1]
    config = json.loads((root / "src" / "config.json").read_text(
        encoding="utf-8"))
    paths = config["paths"]
    plan = config["analysis_plan"]
    args = _parse_args(argv, draws=plan["permutation_draws"])
    logger = LoggerFactory(log_dir=root / paths["logs"],
                           name=config["logging"]["name"],
                           level=config["logging"]["level"],
                           console=config["logging"]["console"]).build()
    out_json = (Path(args.out) if args.out
                else root / paths["out_dir"] / JSON_NAME)
    audit_path = Path(args.audit or root / paths["training_audit"])
    results_root = Path(args.results_root
                        or root / paths["results_root"])
    study = ReconstructionCorrelation(
        audit_path=audit_path, results_root=results_root,
        draws=args.draws, seed=plan["permutation_seed"], logger=logger)
    try:
        result = study.run()
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}")
        print(MISSING_INPUT.format(audit=audit_path,
                                   results_root=results_root))
        return 1
    try:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        write_result(result, out_json, logger=logger)
    except OSError as error:
        print(f"ABORT: {error}")
        return 3
    for text in study.summary_lines(result):
        print(text)
    print(f"-> {out_json}")
    logger.info("[main] %s correlated %d row(s) into %s.", GENERATOR,
                result["n"], out_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
