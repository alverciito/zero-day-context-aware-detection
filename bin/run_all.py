# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         bin/run_all.py                                            #
#   Version:      1.0.0                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-07                                                #
#   Modified:     2026-10-07                                                #
#   Description:  Runs the six entry points in order against the shipped    #
#                 data and reports which ones wrote their outputs.          #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import argparse
import importlib.util
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

# The entry points, in the order they have to run: the reconstruction
# study reads the result of the training audit.
STEPS: tuple[str, ...] = (
    "make_figures", "make_supplementary_tables", "make_baseline_tables",
    "audit_training", "analyse_retrained_pair", "analyse_reconstruction")
# Names of the result files of the last five, inside the output
# directory; they are the defaults of each script.
SUPPLEMENT_TEX: str = "supplementary_tables.tex"
SUPPLEMENT_JSON: str = "supplementary_tables.json"
BASELINES_JSON: str = "baselines.json"
BASELINES_MD: str = "baselines.md"
AUDIT_JSON: str = "training_audit.json"
PAIR_JSON: str = "retrained_pair.json"
RECONSTRUCTION_JSON: str = "reconstruction_vs_detection.json"


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                AUXILIARY                                  #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def _load_main(name: str) -> Callable[[Sequence[str]], int]:
    """
    Import one script of this directory and return its ``main``.

    :param name: File name of the script, without the extension.
    :type name: str
    :return: The ``main`` function of that script.
    :rtype: Callable[[Sequence[str]], int]
    :raises ImportError: If the script cannot be imported.
    """
    path = Path(__file__).resolve().with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"bin_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"[_load_main] Input error: the script {path} "
                          "must be importable, and it is not.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def _arguments(out_dir: Path, draws: int | None) -> dict[str, list[str]]:
    """
    Command line of every step for one output directory.

    :param out_dir: Directory every output is written to.
    :type out_dir: Path
    :param draws: Permutations of the reconstruction study, or None for
        the value of the frozen plan.
    :type draws: int | None
    :return: Script name -> its arguments.
    :rtype: dict[str, list[str]]
    """
    audit = str(out_dir / AUDIT_JSON)
    reconstruction = ["--audit", audit,
                      "--out", str(out_dir / RECONSTRUCTION_JSON)]
    if draws is not None:
        reconstruction += ["--draws", str(draws)]
    return {
        "make_figures": ["--out-dir", str(out_dir)],
        "make_supplementary_tables": [
            "--out-tex", str(out_dir / SUPPLEMENT_TEX),
            "--out-json", str(out_dir / SUPPLEMENT_JSON)],
        "make_baseline_tables": [
            "--out-json", str(out_dir / BASELINES_JSON),
            "--out-md", str(out_dir / BASELINES_MD)],
        "audit_training": ["--out-json", audit],
        "analyse_retrained_pair": ["--json", str(out_dir / PAIR_JSON)],
        "analyse_reconstruction": reconstruction}


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
        description="Runs every entry point of the package against the "
                    "shipped data (JISAS-D-26-02300). CPU only, about "
                    "one minute.")
    parser.add_argument("--out-dir", default=None,
                        help="directory for every output; default: "
                             "paths.out_dir of src/config.json")
    parser.add_argument("--draws", type=int, default=None,
                        help="permutations of the reconstruction "
                             "study; default: the value of the plan")
    return parser.parse_args(argv)


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                   MAIN                                    #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def main(argv: Sequence[str] | None = None) -> int:
    """
    Run the six entry points and say which ones succeeded.

    Steps: load ``src/config.json`` for the default output directory,
    call the ``main`` of every script of ``STEPS`` in order with its
    outputs under that directory, and print one status line per script.
    A step that fails does not stop the following ones, except that the
    reconstruction study cannot run without the audit result.

    Run it with: uv run bin/run_all.py

    :param argv: Command line to parse, or None to read ``sys.argv``.
    :type argv: Sequence[str] | None
    :return: 0 when every step returned 0, 1 otherwise.
    :rtype: int
    :raises SystemExit: If the command line is not valid.
    """
    codes: dict[str, int] = dict()
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "src" / "config.json").read_text(
        encoding="utf-8"))
    args = _parse_args(argv)
    out_dir = (Path(args.out_dir) if args.out_dir
               else root / config["paths"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    arguments = _arguments(out_dir, args.draws)
    for name in STEPS:
        print(f"\n===== bin/{name}.py =====")
        codes[name] = _load_main(name)(arguments[name])
    print("\n===== summary =====")
    for name in STEPS:
        print(f"{'ok' if codes[name] == 0 else 'FAILED':7s} "
              f"(exit {codes[name]})  bin/{name}.py")
    return 0 if all(code == 0 for code in codes.values()) else 1


if __name__ == "__main__":
    sys.exit(main())


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
