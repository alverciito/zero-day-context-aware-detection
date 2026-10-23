# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         bin/make_figures.py                                       #
#   Version:      1.1.3                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Writes outputs F1-F4 and T1-T3 from the frozen            #
#                 aggregate, plus the README, the table of the values       #
#                 used and the manifest.                                    #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import argparse
import datetime as dt
import hashlib
import json
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# The thread caps and the non-interactive backend must be in place before
# numpy and matplotlib are imported, so these statements sit between the
# import blocks.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "4")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from kitnetzday.figures import (  # noqa: E402
                             FailureCases,
                             Heatmap,
                             OracleFigure,
                             RanksFigure,
)
from kitnetzday.io import (  # noqa: E402
                        FIGURES_VERSION,
                        Inputs,
                        Out,
                        parse_f1_trivial_smoothed,
                        parse_fig7_csv,
                        read_timing_reference,
                        write_index,
                        write_manifest,
                        write_numbers,
)
from kitnetzday.naming import OUTPUT_IDS, Naming  # noqa: E402
from kitnetzday.tables import (  # noqa: E402
                            AblationsTable,
                            CostTable,
                            MainResultsTables,
)
from kitnetzday.utils import InputError, LoggerFactory  # noqa: E402

# Name this script is recorded under, in the provenance comment of every
# generated .tex, in README_figures.md and in figures.json.
GENERATOR: str = "bin/make_figures.py"
# Thread variables copied into the provenance.
THREAD_VARIABLES: tuple[str, ...] = ("OMP_NUM_THREADS",
                                     "OPENBLAS_NUM_THREADS",
                                     "MKL_NUM_THREADS")
# Where the guards of the pipeline that produced the results live. The
# note goes into figures.json so that a reader of the manifest knows why
# it has no "plan" and "env" blocks; see main().
PLAN_GUARDS: str = ("plan_guard / env_guard / versions live in "
                    "frozen/runner/metrics_rev.py, next to the code that "
                    "produced the results; this script records the "
                    "provenance of the kitnetzday package instead")
# Interpreter named in the reproduction command.
COMMAND_PREFIX: str = "uv run"
# What to tell the reader when the cost table has no row.
NO_TIMING: str = (
    "ERROR: the cost table (T3) has no row, so it was not written.\nIt "
    "is built from the jobs whose timing.json carries timing_ref = 1, "
    "together with their config.json and metrics.json, under "
    "--results-root. Those files are shipped in data/jobs/ (twelve "
    "reference runs on KNAD Active Wiretap, seed A).\n"
    "Looked in: {results_root}")
# Argument name -> flag, in the order the reproduction command lists them.
# The output directory is left out on purpose: it does not change any
# value, and with it two runs into different folders would not write the
# same files.
COMMAND_FLAGS: tuple[tuple[str, str], ...] = (
    ("aggregate", "--aggregate"), ("stats", "--stats"),
    ("cells", "--cells"), ("results_root", "--results-root"),
    ("only", "--only"))
_HASH_CHUNK: int = 1 << 22


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                AUXILIARY                                  #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def _now_iso() -> str:
    """
    Current local time to the second, as the rest of the pipeline writes.

    :return: Text like "2026-10-06T18:42:07", without a time zone.
    :rtype: str
    """
    return dt.datetime.now().isoformat(timespec="seconds")


def _sha256_file(path: Path) -> str:
    """
    Hash a file, reading it in 4 MiB chunks.

    :param path: Path of an existing, readable file.
    :type path: Path
    :return: The sha256 digest, 64 lower-case hexadecimal characters.
    :rtype: str
    :raises OSError: If the file cannot be read.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_provenance(root: Path) -> dict[str, Any]:
    """
    Identify the kitnetzday code of this run, module by module.

    The figures come out of a library, so what has to be pinned is the
    library: its version and the sha256 of every module that was
    imported while drawing.

    :param root: Root of this package, the parent of ``src``.
    :type root: Path
    :return: {"name", "version", "root", "modules_sha256", "guards"},
        with one hash per imported kitnetzday module, keyed by module name.
    :rtype: dict[str, Any]
    :raises OSError: If a module file cannot be read.
    """
    hashes = dict()
    path = None
    for name in sorted(n for n in sys.modules
                       if n == "kitnetzday" or n.startswith("kitnetzday.")):
        path = getattr(sys.modules[name], "__file__", None)
        if path is not None:
            hashes[name] = _sha256_file(Path(path))
    return {"name": "kitnetzday",
            "version": getattr(sys.modules.get("kitnetzday"),
                               "__version__", None),
            "root": "src/kitnetzday",
            "modules_sha256": hashes, "guards": PLAN_GUARDS}


def _library_versions() -> dict[str, str | None]:
    """
    Version of the interpreter and of the libraries kitnetzday imports.

    :return: Version per library, None for one that was not imported;
        the interpreter is included as "python".
    :rtype: dict[str, str | None]
    """
    versions: dict[str, str | None] = {"python": sys.version.split()[0]}
    module = None
    for name in ("numpy", "scipy", "matplotlib"):
        module = sys.modules.get(name)
        versions[name] = (None if module is None
                          else getattr(module, "__version__", None))
    return versions


def _build_command(args: argparse.Namespace,
                   with_notes: bool = True) -> str:
    """
    Reproduction command of this run, for the README and the table of
    values.

    The command is written relative to the root of the package, which
    is where it has to be run from.

    :param args: Parsed command line of this run.
    :type args: argparse.Namespace
    :param with_notes: Whether to repeat the ``--note`` arguments; the
        table of values uses the version without them.
    :type with_notes: bool
    :return: The command as one line of shell.
    :rtype: str
    """
    parts = [COMMAND_PREFIX, GENERATOR]
    value = None
    for name, flag in COMMAND_FLAGS:
        value = getattr(args, name, None)
        if value:
            parts += [flag, str(value)]
    if with_notes:
        for note in (getattr(args, "note", None) or list()):
            parts += ["--note", f'"{note}"']
    return " ".join(parts)


def _parse_args(argv: Sequence[str] | None = None,
                dpi: int = 300) -> argparse.Namespace:
    """
    Read the command line.

    :param argv: Arguments to parse, or None to read ``sys.argv``.
    :type argv: Sequence[str] | None
    :param dpi: Default resolution of the PNG previews, from the
        configuration file.
    :type dpi: int
    :return: The parsed arguments.
    :rtype: argparse.Namespace
    :raises SystemExit: If the command line is not valid.
    """
    parser = argparse.ArgumentParser(
        description="Figures and tables of the revised manuscript "
                    "(JISAS-D-26-02300). Every default comes from "
                    "src/config.json.")
    parser.add_argument("--aggregate", default=None,
                        help="aggregate.json; default: paths.aggregate")
    parser.add_argument("--cells", default=None,
                        help="cells.csv; default: paths.cells")
    parser.add_argument("--stats", default=None,
                        help="stats.json; default: paths.stats")
    parser.add_argument("--results-root", default=None,
                        help="directory with one sub-directory per job "
                             "(timing.json of the cost table); default: "
                             "paths.results_root")
    parser.add_argument("--out-dir", default=None,
                        help="default: paths.out_dir")
    parser.add_argument("--fig7-csv", default=None,
                        help="published values of the submitted Fig. 7; "
                             "default: paths.fig7_published")
    parser.add_argument("--fig7-trivial", default=None,
                        help="smoothed trivial F1 per stream of that "
                             "figure; default: paths.fig7_trivial_f1")
    parser.add_argument("--only", default=None,
                        help="subset: F1,F2,T1,T2,T3,F3,F4")
    parser.add_argument("--dpi", type=int, default=dpi)
    parser.add_argument("--note", action="append", default=list(),
                        help="free note copied to README_figures.md and "
                             "figures.json (repeatable)")
    return parser.parse_args(argv)


def _vocabulary(config: dict[str, Any],
                logger: logging.Logger | None = None) -> Naming:
    """
    Build the vocabulary of the study from the configuration file.

    :param config: Parsed ``src/config.json``.
    :type config: dict[str, Any]
    :param logger: Logger of this run, or None.
    :type logger: logging.Logger | None
    :return: The ids, their order and the ROPE of the frozen plan.
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


def _draw(inputs: Inputs,
          out: Out,
          only: set[str],
          config: dict[str, Any],
          args: argparse.Namespace,
          provenance: dict[str, Any],
          logger: logging.Logger | None = None) -> int:
    """
    Build the outputs that were asked for, in the published order.

    The three readers of the archived artefacts (the reference timings,
    the published Fig. 7 values and the trivial F1 of that figure) are
    only called when the output that needs them was asked for. A
    cost table without a single reference timing is not written: the
    caller reports it as an error.

    :param inputs: Reader of the aggregate, the cells and the statistics.
    :type inputs: Inputs
    :param out: Output directory and register of what it holds.
    :type out: Out
    :param only: Output ids to build, a subset of ``OUTPUT_IDS``.
    :type only: set[str]
    :param config: Parsed ``src/config.json``.
    :type config: dict[str, Any]
    :param args: Parsed command line, stored in the manifest.
    :type args: argparse.Namespace
    :param provenance: Provenance of this run, stored in the manifest.
    :type provenance: dict[str, Any]
    :param logger: Logger of this run, or None.
    :type logger: logging.Logger | None
    :return: How many reference timing jobs fed the cost table; 0 when
        the table was not asked for.
    :rtype: int
    :raises OSError: If an output file cannot be written.
    """
    style = config["figures"]
    rope = config["analysis_plan"]["rope_half_width"]
    fig7, n_values = dict(), 0
    f1_trivial = dict()
    timing_rows = (read_timing_reference(inputs.results_root,
                                         logger=logger)
                   if "T3" in only else list())
    if "F3" in only:
        fig7, n_values = parse_fig7_csv(
            inputs.paths["fig7_valores_publicados.csv"], logger=logger)
        f1_trivial = parse_f1_trivial_smoothed(
            inputs.paths["fig7_trivial_f1.csv"], logger=logger)
    if "F1" in only:
        Heatmap(size_inches=tuple(style["heatmap_size_inches"]),
                trivial_width_ratio=style["heatmap_width_ratios"][1],
                bottom=style["heatmap_bottom"],
                top=style["heatmap_top"],
                chance_level=style["chance_level"],
                font_size_small=style["font_size_small"],
                font_size_footnote=style["font_size_footnote"],
                logger=logger).build(inputs, out)
    if "F2" in only:
        RanksFigure(inputs=inputs, out=out, naming=inputs.naming,
                    rope=rope, logger=logger).build()
    if "T1" in only:
        MainResultsTables(inputs=inputs, out=out, logger=logger).build()
    if "T2" in only:
        AblationsTable(inputs=inputs, out=out, logger=logger).build()
    if "T3" in only and timing_rows:
        CostTable(inputs=inputs, out=out, timing_rows=timing_rows,
                  logger=logger).build()
    if "F3" in only:
        OracleFigure(naming=inputs.naming, logger=logger).build(
            inputs, out, fig7, n_values, f1_trivial)
    if "F4" in only:
        FailureCases(naming=inputs.naming,
                     chance_level=style["chance_level"],
                     font_size_small=style["font_size_small"],
                     font_size_footnote=style["font_size_footnote"],
                     logger=logger).build(inputs, out)
    write_numbers(inputs, out, logger=logger)
    write_index(inputs, out, timing_rows, args, provenance,
                logger=logger)
    write_manifest(inputs, out, provenance, args, timing_rows,
                   logger=logger)
    return len(timing_rows)


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                   MAIN                                    #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def main(argv: Sequence[str] | None = None) -> int:
    """
    Write the outputs F1-F4 and T1-T3 of the revised manuscript.

    Steps: load ``src/config.json``, build the run logger in ``logs/``,
    open the frozen aggregate through ``Inputs`` (which hashes every
    input and cross-checks ``cells.csv`` and ``stats.json`` against the
    aggregate), draw the outputs ``--only`` asks for, and close with
    ``numbers_used.md``, ``README_figures.md``, ``figures.json`` and
    ``SHA256SUMS``. No metric is computed here: every number is read as
    the aggregation and the statistics left it. The README of the
    package says which float of the paper each output corresponds to.

    Run it with: uv run bin/make_figures.py

    The plan guard, the environment guard and the version check
    (``plan_guard``, ``env_guard`` and ``versions``) live in
    ``metrics_rev.py``, which ships in ``frozen/runner/``,
    because they are part of the code that produced the results and
    they read the layout of the authors' repository: the commit that
    fixes the analysis plan, the lock file of the environment and the
    roots of that tree. This script records the provenance of the
    kitnetzday package instead: its version and the sha256 of every
    module imported while drawing. What makes a figure traceable here
    is the sha256 of every input, in ``figures.json#/inputs``.

    :param argv: Command line to parse, or None to read ``sys.argv``.
    :type argv: Sequence[str] | None
    :return: 0 when everything was written, 1 for an input that is
        missing or inconsistent and for a cost table without rows, 2
        for an unknown ``--only`` id, and 3 when the provenance or the
        output directory cannot be prepared.
    :rtype: int
    :raises SystemExit: If the command line is not valid.
    """
    inputs = None
    out = None
    provenance = dict()
    mismatch = list()
    n_timing = 0
    started = dt.datetime.now()
    script = Path(__file__).resolve()
    root = script.parents[1]
    config = json.loads((root / "src" / "config.json").read_text(
        encoding="utf-8"))
    paths = config["paths"]
    args = _parse_args(argv, dpi=config["figures"]["dpi"])
    logger = LoggerFactory(log_dir=root / paths["logs"],
                           name=config["logging"]["name"],
                           level=config["logging"]["level"],
                           console=config["logging"]["console"]).build()
    out_dir = (Path(args.out_dir) if args.out_dir
               else root / paths["out_dir"])
    try:
        provenance = {"figures_version": FIGURES_VERSION,
                      "figures_sha256": _sha256_file(script),
                      "figures_path": GENERATOR,
                      "kitnetzday": _package_provenance(root),
                      "versions": _library_versions(),
                      "matplotlib": matplotlib.__version__,
                      "threads_env": {k: os.environ.get(k)
                                      for k in THREAD_VARIABLES}}
    except OSError as error:
        print(f"ABORT: {error}")
        return 3
    try:
        inputs = Inputs(
            agg_path=args.aggregate or root / paths["aggregate"],
            stats_path=args.stats or root / paths["stats"],
            cells_path=args.cells or root / paths["cells"],
            results_root=(args.results_root
                          or root / paths["results_root"]),
            fig7_csv=args.fig7_csv or root / paths["fig7_published"],
            fig7_md=(args.fig7_trivial
                     or root / paths["fig7_trivial_f1"]),
            naming=_vocabulary(config, logger), logger=logger)
    except (InputError, ValueError) as error:
        print(f"ERROR: {error}")
        return 1
    try:
        out = Out(out_dir, args.dpi, _build_command(args),
                  f"{GENERATOR} sha256:"
                  f"{provenance['figures_sha256'][:12]}",
                  cmd_rows=_build_command(args, with_notes=False),
                  generator=GENERATOR, root=root, logger=logger)
    except (OSError, ValueError) as error:
        print(f"ABORT: {error}")
        return 3
    only = ({name.strip().upper() for name in args.only.split(",")}
            if args.only else set(OUTPUT_IDS))
    unknown = only - set(OUTPUT_IDS)
    if unknown:
        print(f"ERROR: unknown --only id: {sorted(unknown)} "
              f"(valid: {OUTPUT_IDS})")
        return 2
    logger.info(f"[main] Starting run with {inputs} and {out}.")
    n_timing = _draw(inputs, out, only, config, args, provenance, logger)
    print(f"[{_now_iso()}] outputs -> {out.shown(out_dir)} "
          f"({len(out.records)} outputs, {len(out.numbers)} values "
          "listed in numbers_used.md, "
          f"{(dt.datetime.now() - started).total_seconds():.1f} s; "
          f"{'PARTIAL INPUTS' if inputs.partial_inputs else 'FINAL'})")
    if inputs.consistency:
        mismatch = getattr(inputs, "cross_stats",
                           dict()).get("mismatch", list())
        print(f"ERROR: {len(inputs.consistency)} mismatches between the "
              "inputs (figures.json#/consistency); stats/aggregate "
              f"cross-check: {len(mismatch)} cells")
        for note in inputs.consistency[:5]:
            print(f"  - {note}")
        return 1
    if "T3" in only and not n_timing:
        print(NO_TIMING.format(results_root=inputs.results_root))
        return 1
    logger.info(f"[main] Wrote {len(out.records)} outputs and "
                f"{len(out.numbers)} rows of values to {out_dir}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
