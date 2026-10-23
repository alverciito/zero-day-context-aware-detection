# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         bin/audit_training.py                                     #
#   Version:      1.1.2                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Label-free audit of the training of the executed runs:    #
#                 the recorded loss of every autoencoder against its        #
#                 constant-predictor floor.                                 #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from kitnetzday.audit import (
                           METHOD,
                           VERDICTS,
                           TrainingAudit,
                           TrainingReport,
                           TrainingSummary,
                           build_result,
                           write_result,
)
from kitnetzday.utils import LoggerFactory
from kitnetzday.utils.formatting import fnum

# Name this script is recorded under, in the method block of the result
# file and in the log of the run.
GENERATOR: str = "bin/audit_training.py"
# Suffix of the Spanish note, written next to the result file.
NOTE_SUFFIX: str = ".md"
# How many warnings the console summary repeats.
WARNINGS_SHOWN: int = 10
# What to tell the reader when no job could be audited.
NOTHING_AUDITED: str = "ERROR: no job was audited, so nothing was written."
# What the audit reads and where each input comes from.
WHAT_IT_NEEDS: str = (
    "The audit needs, for every job, DONE, train_log.csv, config.json "
    "and fm_stats.json under --results-root (shipped in data/jobs/), "
    "and the floor of its stream, which comes either from the "
    "preprocessed streams under --cache-root (26 GB, not shipped; "
    "frozen/runner/pipelines.py builds them from the public datasets "
    "listed in data/streams.json) or from the vectors precomputed from "
    "them in --floor-vectors (shipped as data/floor_vectors.json).\n"
    "Looked in: {results_root} | {cache_root} | {vectors}")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                AUXILIARY                                  #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """
    Read the command line of the audit.

    :param argv: Arguments to parse, or None to read ``sys.argv``.
    :type argv: Sequence[str] | None
    :return: The parsed arguments.
    :rtype: argparse.Namespace
    :raises SystemExit: If the command line is not valid.
    """
    parser = argparse.ArgumentParser(
        description="Label-free training audit (JISAS-D-26-02300). It "
                    "uses no GPU and retrains nothing.")
    parser.add_argument("--results-root", default=None,
                        help="directory with one sub-directory per job; "
                             "default: paths.results_root of "
                             "src/config.json")
    parser.add_argument("--cache-root", default=None,
                        help="preprocessed streams, one sub-directory "
                             "per stream with its x.npy; default: "
                             "paths.stream_cache")
    parser.add_argument("--floor-vectors", default=None,
                        help="JSON with the floor vectors computed in "
                             "advance, used for the streams whose cache "
                             "is absent; default: paths.floor_vectors")
    parser.add_argument("--out-json", default=None,
                        help="default: paths.training_audit")
    parser.add_argument("--note", action="store_true",
                        help="also write the authors' note of the "
                             "audit (Markdown, in Spanish) next to the "
                             "result")
    parser.add_argument("--out-md", default=None,
                        help="path of that note; default: the result "
                             "file with the suffix .md")
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


def _method(audit: TrainingAudit, root: Path) -> dict[str, Any]:
    """
    Method block of the result file, with this run as its provenance.

    :param audit: Audit whose ``run`` has been called.
    :type audit: TrainingAudit
    :param root: Root of this package.
    :type root: Path
    :return: The method block, ready for ``build_result``.
    :rtype: dict[str, Any]
    :raises OSError: If the stored vectors cannot be hashed.
    """
    notes = dict(METHOD)
    sources = list(audit.floor.source.values())
    vectors = audit.floor.vectors_path
    notes["script"] = GENERATOR
    notes["interpreter"] = f"python {sys.version.split()[0]}"
    notes["floor_source"] = {
        "stream cache": sources.count("stream cache"),
        "stored": sources.count("stored"),
        "stored_file": (None if vectors is None or not vectors.exists()
                        else _relative(vectors, root)),
        "stored_sha256": (
            None if vectors is None or not vectors.exists()
            else hashlib.sha256(vectors.read_bytes()).hexdigest())}
    return notes


def _table(summary: TrainingSummary) -> list[str]:
    """
    The variant by verdict table, as plain text for the console.

    :param summary: Aggregates of the audit, already built.
    :type summary: TrainingSummary
    :return: One line per variant, with a header.
    :rtype: list[str]
    """
    lines = [f"{'variant':<16}{'jobs':>5}{'AEs':>6}{'trained':>9}"
             f"{'floor':>7}{'worse':>7}{'n/c':>5}{'median':>9}"
             f"{'datasets':>10}"]
    for name in summary.variants:
        row = summary.by_variant[name]
        counts = [row["counts"][v] for v in VERDICTS]
        trained = len(row["datasets_with_at_least_one_trained_ae"])
        lines.append(
            f"{name:<16}{row['n_jobs']:>5}{row['n_autoencoders']:>6}"
            f"{counts[0]:>9}{counts[1]:>7}{counts[2]:>7}{counts[3]:>5}"
            f"{fnum(row['median_ratio'], 4, '-'):>9}"
            f"{f'{trained}/{summary.n_datasets}':>10}")
    return lines


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                   MAIN                                    #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def main(argv: Sequence[str] | None = None) -> int:
    """
    Audit every finished job and write the result file.

    Steps: load ``src/config.json``, build the run logger in ``logs/``,
    audit every job directory that holds both ``DONE`` and
    ``train_log.csv``, tabulate the verdicts, write the JSON result and
    print the variant by verdict table. Nothing is trained, no GPU is
    used and no label is read: the loss comes from ``train_log.csv``,
    the feature groups from ``fm_stats.json``, the training stretch
    from ``config.json`` of each job, and the floor from the stream
    caches or, where those are absent, from the stored vectors.

    Run it with: uv run bin/audit_training.py

    With the files shipped in ``data/`` the floors come from
    ``data/floor_vectors.json``, which the authors computed from the
    preprocessed streams; the result file records that under
    ``method.floor_source``. A run that audits nothing writes nothing
    and ends with exit code 1 and the list of what is missing.

    :param argv: Command line to parse, or None to read ``sys.argv``.
    :type argv: Sequence[str] | None
    :return: 0 when the result was written, 1 when the results tree is
        missing, when no job could be audited or when the note cannot
        be written, and 3 when a file cannot be written.
    :rtype: int
    :raises SystemExit: If the command line is not valid.
    """
    audit = None
    note = ""
    script = Path(__file__).resolve()
    root = script.parents[1]
    config = json.loads((root / "src" / "config.json").read_text(
        encoding="utf-8"))
    paths = config["paths"]
    plan = config["analysis_plan"]
    args = _parse_args(argv)
    logger = LoggerFactory(log_dir=root / paths["logs"],
                           name=config["logging"]["name"],
                           level=config["logging"]["level"],
                           console=config["logging"]["console"]).build()
    results_root = Path(args.results_root
                        or root / paths["results_root"])
    cache_root = Path(args.cache_root or root / paths["stream_cache"])
    vectors = Path(args.floor_vectors or root / paths["floor_vectors"])
    out_json = (Path(args.out_json) if args.out_json
                else root / paths["training_audit"])
    out_md = (Path(args.out_md) if args.out_md
              else out_json.with_suffix(NOTE_SUFFIX))
    try:
        audit = TrainingAudit(
            results_root=results_root, cache_root=cache_root,
            vectors_path=vectors,
            trained_below=plan["floor_trained_below"],
            at_floor_below=plan["floor_at_floor_above"], logger=logger)
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}")
        print(WHAT_IT_NEEDS.format(results_root=results_root,
                                   cache_root=cache_root,
                                   vectors=vectors))
        return 1
    audit.run()
    if not audit.jobs:
        print(NOTHING_AUDITED)
        print(WHAT_IT_NEEDS.format(results_root=results_root,
                                   cache_root=cache_root,
                                   vectors=vectors))
        return 1
    summary = TrainingSummary(audit, logger=logger)
    summary.run()
    try:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        write_result(build_result(audit, summary, _method(audit, root),
                                  logger), out_json, logger=logger)
    except OSError as error:
        print(f"ABORT: {error}")
        return 3
    print(f"jobs audited: {len(audit.jobs)} | skipped: "
          f"{len(audit.skipped)} | autoencoders: {len(audit.flat)} | "
          f"warnings: {len(audit.warnings)}")
    for text in _table(summary):
        print(text)
    for text in audit.warnings[:WARNINGS_SHOWN]:
        print(f"warning: {text}")
    print(f"-> {_relative(out_json, root)}")
    if args.note:
        try:
            note = TrainingReport(summary, logger=logger).text()
        except (KeyError, ValueError) as error:
            print(f"ERROR: {_relative(out_json, root)} is written, but "
                  "the note needs every block of runs (its sections "
                  "4, 5 and 7 name the variants lstm and mlp and the "
                  f"stream ACI-IOT-2023): {type(error).__name__}: "
                  f"{error}.")
            return 1
        out_md.write_text(note, encoding="utf-8")
        print(f"-> {_relative(out_md, root)}")
    logger.info("[main] %s audited %d job(s) into %s.", GENERATOR,
                len(audit.jobs), out_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
