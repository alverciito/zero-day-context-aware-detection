# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         tests/added_analyses/test_modified_encoders.py            #
#   Version:      1.1.0                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-22                                                #
#   Modified:     2026-10-22                                                #
#   Description:  Recomputes from the shipped files the post hoc experiment #
#                 with modified Conv1D and Conv2D encoders: the counts of   #
#                 the validation ratio and the entry rule (supplement,      #
#                 Table S72), the columns of Table S71, the two             #
#                 comparisons with the KitNET-style control and their       #
#                 within-block and dependence checks.                       #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import csv
import json
import math
from pathlib import Path
from typing import Any

import pytest

from kitnetzday.audit.paired_stats import PairedStats

# Root of the package and the folder of the experiment.
ROOT: Path = Path(__file__).resolve().parents[2]
DATA: Path = ROOT / "data"
FOLDER: Path = DATA / "modified_encoders"
JOBS: Path = FOLDER / "jobs"
DATASETS: tuple[str, ...] = tuple(f"D{i}" for i in range(1, 10))
# Stream identifiers in the job names, in the order D1 to D9.
STREAMS: tuple[str, ...] = (
    "knad-active_wiretap", "knad-fuzzing", "knad-os_scan",
    "knad-ssdp_flood", "knad-ssl_renegotiation", "knad-video_injection",
    "aci-iot-2023", "cic2017", "cic2018")
FAMILIES: tuple[str, ...] = ("conv1d", "conv2d")
# Configurations of data/aggregate/cells.csv: the unmodified encoder in
# its two modes, and the KitNET-style control.
UNMODIFIED: dict[str, tuple[str, str]] = {
    "conv1d": ("C1", "C2"), "conv2d": ("C3", "C4")}
CONTROL: str = "C10"
# Two-sided alpha of the exact interval with nine units (96.1 %).
ALPHA_9: float = 0.0390625
TOL: float = 1e-12
# Streams of the selection runs, marked with a dagger in both tables.
SELECTION: tuple[str, ...] = ("D1", "D7", "D9")
# The six packet captures, merged into one unit by the dependence rule.
KNAD: tuple[str, ...] = ("D1", "D2", "D3", "D4", "D5", "D6")
# Keys of the comparisons in data/modified_encoders/comparisons.json.
KEYS: dict[str, str] = {
    "conv1d": "Conv1Dmod_vs_original", "conv2d": "Conv2Dmod_vs_original"}

needs_data = pytest.mark.skipif(
    not JOBS.exists() or not (DATA / "aggregate" / "cells.csv").exists(),
    reason="the shipped data are not in data/")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                AUXILIARY                                  #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def _json(path: Path) -> Any:
    """
    Read a JSON file.

    :param path: File to read.
    :type path: Path
    :return: The parsed JSON.
    :rtype: Any
    """
    return json.loads(path.read_text(encoding="utf-8"))


def _job(stream: str, family: str, mode: str, seed: str) -> Path:
    """
    Folder of one run of the experiment.

    :param stream: Stream identifier of the job name.
    :type stream: str
    :param family: "conv1d" or "conv2d".
    :type family: str
    :param mode: "arF" (full-sequence) or "arT" (autoregressive).
    :type mode: str
    :param seed: "A", "B" or "C".
    :type seed: str
    :return: The folder, which must exist.
    :rtype: Path
    """
    found = sorted(JOBS.glob(
        f"T10-REP__{stream}__{family}_rep__{mode}__*__s{seed}__r1"))
    assert len(found) == 1, (stream, family, mode, seed, found)
    return found[0]


def _counts(family: str, seed: str, mode: str) -> dict[str, list[int]]:
    """
    Autoencoders below 0.90 of the validation ratio, per stream.

    An autoencoder is evaluable when the error of predicting the mean of
    its held-out records is finite and at least 1e-4; it counts when it
    is evaluable and its ratio is finite and below 0.90.

    :param family: "conv1d" or "conv2d".
    :type family: str
    :param seed: "A", "B" or "C".
    :type seed: str
    :param mode: "arF" or "arT".
    :type mode: str
    :return: Dataset -> [below 0.90, evaluable, below 0.90 whose error
        rises by 10 % or more with the window replaced].
    :rtype: dict[str, list[int]]
    """
    out: dict[str, list[int]] = dict()
    for dataset, stream in zip(DATASETS, STREAMS):
        report = _json(_job(stream, family, mode, seed)
                       / "convergence_report.json")
        below = evaluable = context = 0
        for ae in report["autoencoders"]:
            floor, ratio = ae["suelo"], ae["razon"]
            if floor is None or not math.isfinite(floor) or floor < 1e-4:
                continue
            evaluable += 1
            if ratio is None or not math.isfinite(ratio) or ratio >= 0.90:
                continue
            below += 1
            moved = ae.get("cociente_contexto_cambiado")
            if moved is not None and math.isfinite(moved) and moved >= 1.10:
                context += 1
        out[dataset] = [below, evaluable, context]
    return out


def _cells() -> dict[tuple[str, str], float]:
    """
    Mean ROC-AUC of the fused detector of the main experiment, per cell.

    :return: (dataset, configuration) -> ``auroc_FUS_mean``.
    :rtype: dict[tuple[str, str], float]
    """
    with (DATA / "aggregate" / "cells.csv").open(encoding="utf-8") as fh:
        return {(row["dataset_id"], row["config_id"]):
                float(row["auroc_FUS_mean"]) for row in csv.DictReader(fh)}


def _modified(family: str) -> dict[str, float]:
    """
    ROC-AUC of a modified encoder per stream: mean of its two modes and
    of seeds A and B, read from the per-run detection metrics.

    :param family: "conv1d" or "conv2d".
    :type family: str
    :return: Dataset -> mean ROC-AUC of the fused detector.
    :rtype: dict[str, float]
    """
    out: dict[str, float] = dict()
    for dataset, stream in zip(DATASETS, STREAMS):
        seeds = list()
        for seed in ("A", "B"):
            modes = [_json(_job(stream, family, mode, seed)
                           / "detection_metrics.json")["metricas"]
                     ["auroc_FUS"] for mode in ("arT", "arF")]
            seeds.append(sum(modes) / 2)
        out[dataset] = sum(seeds) / 2
    return out


def _ratios(family: str, seed: str, mode: str) -> list[float]:
    """
    Finite validation ratios of the evaluable autoencoders.

    :param family: "conv1d" or "conv2d".
    :type family: str
    :param seed: "A", "B" or "C".
    :type seed: str
    :param mode: "arF" or "arT".
    :type mode: str
    :return: The ratios, over the nine streams.
    :rtype: list[float]
    """
    out: list[float] = list()
    for stream in STREAMS:
        report = _json(_job(stream, family, mode, seed)
                       / "convergence_report.json")
        for ae in report["autoencoders"]:
            floor, ratio = ae["suelo"], ae["razon"]
            if floor is None or not math.isfinite(floor) or floor < 1e-4:
                continue
            if ratio is not None and math.isfinite(ratio):
                out.append(ratio)
    return out


def _median(values: list[float]) -> float:
    """
    Median of a list of numbers.

    :param values: The numbers; must not be empty.
    :type values: list[float]
    :return: The median.
    :rtype: float
    """
    ordered = sorted(values)
    half = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[half]
    return (ordered[half - 1] + ordered[half]) / 2


def _cell_column(column: str) -> dict[tuple[str, str], float]:
    """
    One column of data/aggregate/cells.csv, per cell.

    :param column: Name of the column.
    :type column: str
    :return: (dataset, configuration) -> value.
    :rtype: dict[tuple[str, str], float]
    """
    with (DATA / "aggregate" / "cells.csv").open(encoding="utf-8") as fh:
        return {(row["dataset_id"], row["config_id"]): float(row[column])
                for row in csv.DictReader(fh)}


def _per_run(family: str, key: str) -> dict[str, dict[str, float]]:
    """
    One detection metric of a modified encoder, per stream and seed,
    as the mean of its two modes.

    :param family: "conv1d" or "conv2d".
    :type family: str
    :param key: Key of ``metricas`` in detection_metrics.json.
    :type key: str
    :return: Dataset -> seed ("A" or "B") -> mean of the two modes.
    :rtype: dict[str, dict[str, float]]
    """
    out: dict[str, dict[str, float]] = dict()
    for dataset, stream in zip(DATASETS, STREAMS):
        out[dataset] = dict()
        for seed in ("A", "B"):
            modes = [_json(_job(stream, family, mode, seed)
                           / "detection_metrics.json")["metricas"][key]
                     for mode in ("arT", "arF")]
            out[dataset][seed] = sum(modes) / 2
    return out


def _differences(family: str) -> dict[str, float]:
    """
    Modified encoder minus the KitNET-style control, per stream.

    :param family: "conv1d" or "conv2d".
    :type family: str
    :return: Dataset -> difference of the mean ROC-AUC.
    :rtype: dict[str, float]
    """
    cells = _cells()
    modified = _modified(family)
    return {d: modified[d] - cells[(d, CONTROL)] for d in DATASETS}


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                   TESTS                                   #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
@needs_data
class TestModifiedEncoders:
    """The post hoc experiment, recomputed from the shipped files."""

    def test_runs_are_complete(self) -> None:
        """108 runs: two families, two modes, nine streams, three seeds."""
        jobs = sorted(p.name for p in JOBS.iterdir() if p.is_dir())
        assert len(jobs) == 108
        assert all((JOBS / j / "DONE").exists() for j in jobs)
        with_metrics = [j for j in jobs
                        if (JOBS / j / "detection_metrics.json").exists()]
        assert len(with_metrics) == 72
        assert not any(j.endswith("__sC__r1") for j in with_metrics)

    @pytest.mark.parametrize("family", FAMILIES)
    @pytest.mark.parametrize("seed", ("A", "B", "C"))
    def test_validation_ratio_counts(self, family: str, seed: str) -> None:
        """Table S72, column by column, from the held-out reports."""
        tables = _json(FOLDER / "supplement_tables.json")
        row = tables["convergencia"][f"{family}|{seed}"]
        full = _counts(family, seed, "arF")
        auto = _counts(family, seed, "arT")
        assert {d: v[:2] for d, v in full.items()} == \
            row["tsr_por_stream"]
        assert sum(v[0] for v in full.values()) == row["tsr_bajo"]
        assert sum(v[1] for v in full.values()) == row["tsr_evaluables"]
        assert sum(v[2] for v in full.values()) == row["contexto"]
        assert sum(1 for v in full.values() if v[0]) == \
            row["streams_con_alguno"]
        assert sum(v[0] for v in auto.values()) == row["ar_bajo"]
        assert sum(v[1] for v in auto.values()) == row["ar_evaluables"]
        assert sum(v[2] for v in auto.values()) == row["contexto_ar"]
        assert abs(_median(_ratios(family, seed, "arF"))
                   - row["mediana_razon_tsr"]) <= TOL

    @pytest.mark.parametrize("family", FAMILIES)
    def test_window_reliance_with_seeds_a_and_b(self, family: str) -> None:
        """Window replaced, seeds A and B pooled, as Section 35 gives."""
        pooled = _json(FOLDER / "supplement_tables.json")["contexto_ab"]
        context = below = 0
        for seed in ("A", "B"):
            full = _counts(family, seed, "arF")
            context += sum(v[2] for v in full.values())
            below += sum(v[0] for v in full.values())
        assert pooled[family] == [context, below]

    @pytest.mark.parametrize("family", FAMILIES)
    def test_entry_rule_with_seed_a(self, family: str) -> None:
        """Seed A: two thirds of 29 and seven of nine streams."""
        full = _counts(family, "A", "arF")
        below = sum(v[0] for v in full.values())
        evaluable = sum(v[1] for v in full.values())
        streams = sum(1 for v in full.values() if v[0])
        assert evaluable == 29
        assert 3 * below >= 2 * evaluable and streams >= 7

    @pytest.mark.parametrize("family", FAMILIES)
    def test_entry_rule_file(self, family: str) -> None:
        """entry_rule.json: its counts, its decision and its inputs."""
        rule = _json(FOLDER / "entry_rule.json")["familias"][family]
        full = _counts(family, "A", "arF")
        auto = _counts(family, "A", "arT")
        below = sum(v[0] for v in full.values())
        evaluable = sum(v[1] for v in full.values())
        streams = sum(1 for v in full.values() if v[0])
        tsr = rule["tsr"]
        assert (tsr["entrenados"], tsr["evaluables"]) == (below, evaluable)
        assert abs(tsr["fraccion_entrenados"] - below / evaluable) <= TOL
        assert tsr["streams_con_algun_entrenado"] == streams
        assert tsr["entrenados_que_dependen_del_contexto"] == \
            sum(v[2] for v in full.values())
        assert tsr["cumple_dos_tercios"] == (3 * below >= 2 * evaluable)
        assert tsr["cumple_streams"] == (streams >= 7)
        assert rule["entra"] == (tsr["cumple_dos_tercios"]
                                 and tsr["cumple_streams"])
        assert (rule["ar_descriptivo"]["entrenados"],
                rule["ar_descriptivo"]["evaluables"]) == \
            (sum(v[0] for v in auto.values()),
             sum(v[1] for v in auto.values()))
        outside = [v for d, v in full.items() if d not in SELECTION]
        rest = rule["fuera_de_los_streams_del_prevuelo"]
        assert (rest["entrenados"], rest["evaluables"],
                rest["streams_con_algun_entrenado"]) == \
            (sum(v[0] for v in outside), sum(v[1] for v in outside),
             sum(1 for v in outside if v[0]))
        for stream in STREAMS:
            for mode, part in (("arF", "tsr"), ("arT", "ar")):
                report = _json(_job(stream, family, mode, "A")
                               / "convergence_report.json")
                stored = rule["por_stream"][stream][part]
                assert stored["razon"] == \
                    [ae["razon"] for ae in report["autoencoders"]]
                assert stored["contexto"] == \
                    [ae.get("cociente_contexto_cambiado")
                     for ae in report["autoencoders"]]

    @pytest.mark.parametrize("family", FAMILIES)
    def test_roc_auc_per_stream(self, family: str) -> None:
        """Table S71: the modified, unmodified and control columns."""
        tables = _json(FOLDER / "supplement_tables.json")["auc"][family]
        cells = _cells()
        modified = _modified(family)
        first, second = UNMODIFIED[family]
        for dataset in DATASETS:
            row = tables[dataset]
            assert abs(modified[dataset] - row["modificado"]) <= TOL
            assert abs(cells[(dataset, CONTROL)] - row["control"]) <= TOL
            unmod = (cells[(dataset, first)] + cells[(dataset, second)]) / 2
            assert abs(unmod - row["sin_modificar"]) <= TOL

    @pytest.mark.parametrize("family", FAMILIES)
    def test_seeds_and_blocked_columns(self, family: str) -> None:
        """Table S71: each seed, and the within-block ROC-AUC columns."""
        tables = _json(FOLDER / "supplement_tables.json")["auc"][family]
        seeds = _per_run(family, "auroc_FUS")
        blocked = _per_run(family, "auc_loc_RAW")
        control = _cell_column("auc_loc_RAW_mean")
        for dataset in DATASETS:
            row = tables[dataset]
            for seed in ("A", "B"):
                assert abs(seeds[dataset][seed] - row[seed]) <= TOL
            mean = (blocked[dataset]["A"] + blocked[dataset]["B"]) / 2
            assert abs(mean - row["bloque_modificado"]) <= TOL
            assert abs(control[(dataset, CONTROL)]
                       - row["bloque_control"]) <= TOL

    def test_record_index_column(self) -> None:
        """Table S71, AUC_t: one value per stream, as in cells.csv."""
        stored = _json(FOLDER / "supplement_tables.json")["comparaciones"]
        reference = _cell_column("auc_t")
        for dataset, stream in zip(DATASETS, STREAMS):
            values = {round(_json(_job(stream, family, mode, seed)
                                  / "detection_metrics.json")
                            ["metricas"]["auc_t"], 12)
                      for family in FAMILIES for mode in ("arT", "arF")
                      for seed in ("A", "B")}
            assert len(values) == 1
            for family in FAMILIES:
                assert stored[family]["auc_t"][dataset] == sorted(values)
            assert abs(values.pop() - reference[(dataset, CONTROL)]) <= 1e-9

    def test_median_row(self) -> None:
        """Table S71, last row: the median of each column."""
        tables = _json(FOLDER / "supplement_tables.json")
        cells = _cells()
        auc_t = _cell_column("auc_t")
        blocked = _cell_column("auc_loc_RAW_mean")
        columns = [[auc_t[(d, CONTROL)] for d in DATASETS],
                   [cells[(d, CONTROL)] for d in DATASETS]]
        for family in FAMILIES:
            first, second = UNMODIFIED[family]
            modified = _modified(family)
            columns.append([(cells[(d, first)] + cells[(d, second)]) / 2
                            for d in DATASETS])
            columns.append([modified[d] for d in DATASETS])
        columns.append([blocked[(d, CONTROL)] for d in DATASETS])
        for family in FAMILIES:
            runs = _per_run(family, "auc_loc_RAW")
            columns.append([(runs[d]["A"] + runs[d]["B"]) / 2
                            for d in DATASETS])
        medians = tables["medianas_tabla_auc"]
        assert len(medians) == len(columns) == 9
        for column, stored in zip(columns, medians):
            assert abs(_median(column) - stored) <= 1e-9

    def test_comparisons_with_the_control(self) -> None:
        """Hodges-Lehmann, exact interval, exact p and Holm over two."""
        stats = PairedStats()
        cells = _cells()
        comparisons = _json(FOLDER / "comparisons.json")["e1"][
            "comparisons"]
        p_values = dict()
        for family, key in (("conv1d", "Conv1Dmod_vs_original"),
                            ("conv2d", "Conv2Dmod_vs_original")):
            modified = _modified(family)
            diffs = [modified[d] - cells[(d, CONTROL)] for d in DATASETS]
            stored = comparisons[key]
            assert abs(stats.hodges_lehmann(diffs) - stored["hl"]) <= TOL
            lower, upper, _ = stats.hl_interval(diffs, alpha=ALPHA_9)
            assert abs(lower - stored["hl_ci"]["lower"]) <= TOL
            assert abs(upper - stored["hl_ci"]["upper"]) <= TOL
            p_values[key] = stats.wilcoxon_exact(diffs)[1]
            assert abs(p_values[key] - stored["wilcoxon"]["p"]) <= TOL
            assert sum(1 for d in diffs if d > 0) == stored["k_pos"]
        ordered = sorted(p_values.items(), key=lambda item: item[1])
        holm, running = dict(), 0.0
        for rank, (key, p) in enumerate(ordered):
            running = max(running, min(1.0, (len(ordered) - rank) * p))
            holm[key] = running
        for key, value in holm.items():
            assert abs(value - comparisons[key]["p_holm"]) <= TOL

    @pytest.mark.parametrize("family", FAMILIES)
    def test_within_block_comparison(self, family: str) -> None:
        """Within-block check of each comparison: estimate, interval."""
        stats = PairedStats()
        stored = _json(FOLDER / "supplement_tables.json")[
            "comparaciones"][family]
        drift = _json(FOLDER / "comparisons.json")["e1"]["comparisons"][
            KEYS[family]]["robustness"]["drift"]
        runs = _per_run(family, "auc_loc_RAW")
        control = _cell_column("auc_loc_RAW_mean")
        diffs = [(runs[d]["A"] + runs[d]["B"]) / 2 - control[(d, CONTROL)]
                 for d in DATASETS]
        for dataset, value in zip(DATASETS, diffs):
            assert abs(value - drift["d_auc_loc_raw"][dataset]) <= TOL
        estimate = stats.hodges_lehmann(diffs)
        lower, upper, _ = stats.hl_interval(diffs, alpha=ALPHA_9)
        assert abs(estimate - stored["bloque_hl"]) <= TOL
        assert abs(estimate - drift["hl_auc_loc_raw"]) <= TOL
        for mine, theirs in zip((estimate, lower, upper),
                                stored["bloque_ci"]):
            assert abs(mine - theirs) <= TOL
        assert sum(1 for d in diffs if d > 0) == stored["bloque_k_pos"]

    @pytest.mark.parametrize("family", FAMILIES)
    def test_dependence_rule(self, family: str) -> None:
        """The six packet captures merged into one unit, and left out."""
        stats = PairedStats()
        stored = _json(FOLDER / "supplement_tables.json")[
            "comparaciones"][family]
        diffs = _differences(family)
        merged = [sum(diffs[d] for d in KNAD) / len(KNAD)] + \
            [diffs[d] for d in DATASETS if d not in KNAD]
        rest = [diffs[d] for d in DATASETS if d not in KNAD]
        for units, key in ((merged, "colapsado"), (rest, "sin_knad")):
            estimate, positive, count = stored[key]
            assert abs(stats.hodges_lehmann(units) - estimate) <= TOL
            assert sum(1 for d in units if d > 0) == positive
            assert len(units) == count

    @pytest.mark.parametrize("family", FAMILIES)
    def test_margin_count(self, family: str) -> None:
        """Streams above, below and within 0.01 of the control."""
        stored = _json(FOLDER / "supplement_tables.json")[
            "comparaciones"][family]["margen_0_01_control"]
        diffs = list(_differences(family).values())
        assert stored == [sum(1 for d in diffs if d > 0.01),
                          sum(1 for d in diffs if d < -0.01),
                          sum(1 for d in diffs if abs(d) <= 0.01)]
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
