# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         src/kitnetzday/io/legacy.py                               #
#   Version:      1.0.2                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Readers of the artefacts of the original submission:      #
#                 Fig. 7, the trivial F1 audit and the timing jobs.         #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import csv
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from kitnetzday.naming import FIG7_DATASETS
from kitnetzday.utils.formatting import get

# Job id field "arT" / "arF" -> paradigm as the paper names it.
_PARADIGM = {"arT": "AR", "arF": "TSR"}


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                              MAIN FUNCTION                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def parse_fig7_csv(
        path: str | Path,
        fig7_datasets: Mapping[str, str] = FIG7_DATASETS,
        logger: logging.Logger | None = None,
) -> tuple[dict[tuple[str, str, str], dict[str, Any]], int]:
    """
    Read the published values of the submitted Fig. 7.

    Figure 7 of the submission is redrawn as Figure A.1 of the
    manuscript (Appendix A), as the results of the label-aware regime,
    so its bars must come from the values published back then
    and not from any new computation.
    The archived CSV holds one row per (dataset, variant, paradigm,
    metric) with the detector that was selected for that bar.

    Rows of unknown datasets and rows whose value does not parse are
    skipped silently: the CSV is an archived artefact and Figure A.1
    must be drawn with whatever of it is readable.

    Note: the detector is written once per key and overwritten by every
    row of that key, so a key whose two metric rows disagree on the
    detector keeps the detector of the last row read. The published CSV
    agrees on both rows.

    :param path: Path of ``fig7_valores_publicados.csv``; a missing
        file gives an empty result instead of an error.
    :type path: str | Path
    :param fig7_datasets: Dataset name as written in the CSV -> dataset
        id of the plan.
    :type fig7_datasets: Mapping[str, str]
    :param logger: Logger created by the calling script, or None to
        stay silent.
    :type logger: logging.Logger | None
    :return: A mapping (dataset_id, variant, paradigm) -> {"F1": v,
        "WindowDiff": v, "detector": d}, and how many rows were used.
    :rtype: tuple[dict[tuple[str, str, str], dict[str, Any]], int]
    """
    out: dict[tuple[str, str, str], dict[str, Any]] = dict()
    n = 0
    v = 0.0
    if not Path(path).exists():
        if logger is not None:
            logger.warning("[parse_fig7_csv] Input warning: the "
                           "published Fig. 7 values must exist, and "
                           f"they do not: {path}. An empty mapping is "
                           "returned.")
        return out, 0
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            x = fig7_datasets.get((row.get("dataset") or "").strip())
            if x is None:
                continue
            key = (x, (row.get("variante") or "").strip(),
                   (row.get("paradigma") or "").strip())
            try:
                v = float(row["valor"])
            except (KeyError, TypeError, ValueError):
                continue
            out.setdefault(key, dict())[
                (row.get("metrica") or "").strip()] = v
            out[key]["detector"] = (row.get("detector") or "").strip()
            n += 1
    return out, n


def parse_f1_trivial_smoothed(
        path: str | Path,
        fig7_datasets: Mapping[str, str] = FIG7_DATASETS,
        logger: logging.Logger | None = None,
) -> dict[str, float]:
    """
    Read the smoothed trivial F1 per dataset of the submitted Fig. 7.

    Figure A.1 of the manuscript draws, over the oracle bars of the
    submitted Fig. 7, the F1 a
    constant "everything is an attack" predictor reaches on the
    smoothed labels: twenty-four of the published bars do not beat it.
    Those numbers come from the audit written before the analysis plan
    was frozen, so they are labelled exploratory wherever they appear.

    The file is ``fig7_trivial_f1.csv``, with the columns ``dataset``
    and ``f1_trivial_smoothed``. A row of an unknown dataset, or whose
    value does not parse, is skipped.

    :param path: Path of ``fig7_trivial_f1.csv``; a missing file gives
        an empty result.
    :type path: str | Path
    :param fig7_datasets: Dataset name as written in the first column
        -> dataset id of the plan.
    :type fig7_datasets: Mapping[str, str]
    :param logger: Logger created by the calling script, or None to
        stay silent.
    :type logger: logging.Logger | None
    :return: Dataset id -> smoothed trivial F1 in [0, 1].
    :rtype: dict[str, float]
    """
    out: dict[str, float] = dict()
    if not Path(path).exists():
        if logger is not None:
            logger.warning("[parse_f1_trivial_smoothed] Input warning: "
                           "the trivial F1 table must exist, and it "
                           f"does not: {path}. An empty mapping is "
                           "returned.")
        return out
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            x = fig7_datasets.get((row.get("dataset") or "").strip())
            if x is None:
                continue
            try:
                out[x] = float(row["f1_trivial_smoothed"])
            except (KeyError, TypeError, ValueError):
                continue
    return out


def read_timing_reference(
        results_root: str | Path,
        logger: logging.Logger | None = None,
) -> list[dict[str, Any]]:
    """
    Collect the jobs that may be used in the cost table.

    The cost table of the paper may only quote runs that were measured
    in isolation, so it is built from the jobs whose ``timing.json``
    carries ``timing_ref = 1``. For each of them this reader gathers
    the wall clock, the device, the phase breakdown, the throughput,
    the memory peaks, the number of autoencoders and parameters from
    ``config.json``, and the post-processing total from
    ``metrics.json``.

    Exclusivity fails closed: a ``timing.json`` without
    ``concurrent_jobs_same_host`` with both ``start`` and ``end``
    (an older runner) is reported as unknown (None), never as
    exclusive. Concurrent runs are flagged and must not be used.

    Note: ``params_total`` stays None when no autoencoder publishes
    ``count_params``, so that the numpy denoising autoencoders print as
    missing and not as zero. Note too that a directory whose
    ``config.json`` or ``metrics.json`` cannot be read keeps the rest
    of its record, with those fields simply absent.

    :param results_root: Directory that holds one sub-directory per
        job; a path that is not a directory gives an empty list.
    :type results_root: str | Path
    :param logger: Logger created by the calling script, or None to
        stay silent.
    :type logger: logging.Logger | None
    :return: One record per reference job, in the alphabetical order of
        the job directories, with the keys "job_id", "variant",
        "paradigm", "wall_seconds", "device", "phases", "rows",
        "exec_rows_per_s", "gpu_peak_bytes", "rss_peak_bytes",
        "exclusive" and, when available, "exclusive_reason", "n_ae",
        "params_total" and "postproc_total_s".
    :rtype: list[dict[str, Any]]
    """
    rows: list[dict[str, Any]] = list()
    root = Path(results_root)
    t: Any = None
    conf: Any = None
    if not root.is_dir():
        if logger is not None:
            logger.warning("[read_timing_reference] Input warning: the "
                           "results root must be a directory, and it "
                           f"is not: {results_root}. An empty list is "
                           "returned.")
        return rows
    for d in sorted(p for p in root.iterdir()
                    if p.is_dir() and not p.name.startswith(".")):
        tp = d / "timing.json"
        if not tp.exists():
            continue
        try:
            t = json.loads(tp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if t.get("timing_ref") != 1:
            continue
        parts = d.name.split("__")
        variant = parts[2] if len(parts) > 2 else "?"
        ar = parts[3] if len(parts) > 3 else "?"
        rec = {"job_id": d.name, "variant": variant,
               "paradigm": _PARADIGM.get(ar, ar),
               "wall_seconds": t.get("wall_seconds"),
               "device": t.get("device"),
               "phases": t.get("phases") or dict(),
               "rows": t.get("rows"),
               "exec_rows_per_s": t.get("exec_rows_per_s"),
               "gpu_peak_bytes": max(
                   [v for v in (t.get("gpu_peak_bytes")
                                or dict()).values()
                    if isinstance(v, (int, float))] or [0]) or None,
               "rss_peak_bytes": t.get("rss_peak_bytes")}
        cj = t.get("concurrent_jobs_same_host")
        if not isinstance(cj, dict) or cj.get("start") is None \
                or cj.get("end") is None:
            rec["exclusive"] = None
            rec["exclusive_reason"] = ("timing.json without "
                                       "concurrent_jobs_same_host"
                                       "{start,end}")
        else:
            rec["exclusive"] = (cj.get("start") == 0
                                and cj.get("end") == 0)
            if not rec["exclusive"]:
                rec["exclusive_reason"] = (
                    "concurrent_jobs_same_host = "
                    f"{cj.get('start')}/{cj.get('end')}")
        cfg = d / "config.json"
        if cfg.exists():
            try:
                conf = json.loads(cfg.read_text(encoding="utf-8"))
                aes = get(conf, "hyperparameters_effective",
                          "autoencoders", default=list()) or list()
                pars = [a.get("count_params") for a in aes
                        if isinstance(a, dict)]
                pars = [p for p in pars
                        if isinstance(p, (int, float))
                        and not isinstance(p, bool)]
                rec["n_ae"] = len(aes)
                rec["params_total"] = (int(sum(pars)) if pars
                                       else None)
            except (OSError, ValueError):
                pass
        mj = d / "metrics.json"
        if mj.exists():
            try:
                rec["postproc_total_s"] = get(
                    json.loads(mj.read_text(encoding="utf-8")),
                    "timing", "total_s")
            except (OSError, ValueError):
                pass
        rows.append(rec)
    return rows


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
