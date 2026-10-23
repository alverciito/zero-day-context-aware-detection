# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         tests/added_analyses/test_added_analyses.py               #
#   Version:      1.2.1                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-08                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Checks that the analyses added after the analysis plan    #
#                 was fixed ship with their result file, and recomputes     #
#                 from the shipped data the part of them that the shipped   #
#                 data allow.                                               #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import csv
import json
import re
from pathlib import Path
from typing import Any

import pytest

from kitnetzday.audit.paired_stats import PairedStats

# Root of the package, its data and the two folders of the added analyses.
ROOT: Path = Path(__file__).resolve().parents[2]
DATA: Path = ROOT / "data"
INDEX: Path = ROOT / "added_analyses"
RESULTS: Path = DATA / "added_analyses"
# The seven analyses, one result file each.
NAMES: tuple[str, ...] = (
    "siete_flujos", "region_post_preludio", "score_bruto",
    "h1_referencias", "contencion", "knad_y_recuentos",
    "control_armonizado")
# The result files: those seven and the second file of one of them.
FILES: tuple[str, ...] = NAMES + ("knad_y_recuentos_recon",)
DATASETS: tuple[str, ...] = tuple(f"D{i}" for i in range(1, 10))
# Streams whose calibration window holds no attack records.
SEVEN: tuple[str, ...] = ("D1", "D2", "D3", "D4", "D5", "D6", "D8")
# Two-sided alpha of the exact interval: 96.1 % with nine units and
# 95.3 % with seven (1 - 0.9609375 and 1 - 0.953125).
ALPHA_9: float = 0.0390625
ALPHA_7: float = 0.046875
TOL: float = 1e-12

needs_data = pytest.mark.skipif(
    not RESULTS.exists() or not (DATA / "stats.json").exists(),
    reason="the shipped data are not in data/")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                AUXILIARY                                  #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def _result(name: str) -> dict[str, Any]:
    """
    Read the result file of one added analysis.

    :param name: Name of the analysis, one of ``NAMES``.
    :type name: str
    :return: The parsed JSON.
    :rtype: dict[str, Any]
    """
    return json.loads((RESULTS / f"{name}.json").read_text(
        encoding="utf-8"))


def _json(relative: str) -> Any:
    """
    Read a JSON file of ``data/``.

    :param relative: Path relative to ``data/``.
    :type relative: str
    :return: The parsed JSON.
    :rtype: Any
    """
    return json.loads((DATA / relative).read_text(encoding="utf-8"))


def _rows(relative: str) -> list[dict[str, str]]:
    """
    Read a CSV file of ``data/`` as a list of rows.

    :param relative: Path relative to ``data/``.
    :type relative: str
    :return: One dict per row, keyed by the header.
    :rtype: list[dict[str, str]]
    """
    with open(DATA / relative, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _metrics(job_id: str) -> dict[str, Any]:
    """
    Read the ``metrics.json`` of one run.

    :param job_id: Name of the run folder in ``data/jobs/``.
    :type job_id: str
    :return: The parsed JSON.
    :rtype: dict[str, Any]
    """
    return _json(f"jobs/{job_id}/metrics.json")


def _mean(values: list[float]) -> float:
    """
    Arithmetic mean of a non-empty list.

    :param values: The numbers to average.
    :type values: list[float]
    :return: Their mean.
    :rtype: float
    """
    return sum(values) / len(values)


def _confirmatory_differences() -> dict[str, dict[str, float]]:
    """
    Per-stream differences of the twelve confirmatory comparisons.

    :return: Name of the comparison in ``stats.json`` -> stream id ->
        paired difference in ROC-AUC of the fused detector.
    :rtype: dict[str, dict[str, float]]
    """
    stats = _json("stats.json")
    out: dict[str, dict[str, float]] = dict()
    for block in (stats["H1"], stats["H2"]["primary"],
                  stats["H3"]["family"], stats["H4"]):
        for name, entry in block["comparisons"].items():
            out[name] = entry["d"]
    return out


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                  TESTS                                    #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class TestPresence:
    """
    Every added analysis ships its result file, indexed by the README.
    """

    def test_folder_is_the_index(self) -> None:
        """
        ``added_analyses/`` holds its README, which names the result
        file of each of the seven analyses.
        """
        text = (INDEX / "README.md").read_text(encoding="utf-8")
        assert [p.name for p in INDEX.iterdir()
                if p.name != "__pycache__"] == ["README.md"]
        assert all(f"`{name}.json`" in text for name in FILES)

    @needs_data
    def test_result_files_are_the_ones_listed(self) -> None:
        """
        ``data/added_analyses/`` holds the eight result files and
        nothing else.
        """
        assert sorted(p.name for p in RESULTS.iterdir()) == sorted(
            f"{name}.json" for name in FILES)

    @needs_data
    @pytest.mark.parametrize("name", FILES)
    def test_result_file_is_shipped_and_parses(self, name: str) -> None:
        """
        The result file exists and is a non-empty JSON object.

        :param name: Name of the result file, without extension.
        :type name: str
        """
        assert isinstance(_result(name), dict) and _result(name)

    @needs_data
    @pytest.mark.parametrize("name", (
        "score_bruto", "h1_referencias", "region_post_preludio",
        "control_armonizado"))
    def test_result_identifies_its_script(self, name: str) -> None:
        """
        The result file gives the sha256 of the script that computed it.

        Four of the seven result files record it, as one digest of 64
        hexadecimal characters; it identifies the script in the
        repository of the authors.

        :param name: Name of the analysis.
        :type name: str
        """
        result = _result(name)
        block = result if "script_sha256" in result else result[
            "provenance"]
        assert re.fullmatch(r"[0-9a-f]{64}", block["script_sha256"])
        assert [k for k in block if k.startswith("script_sha256")] == [
            "script_sha256"]

    @needs_data
    @pytest.mark.parametrize("name", FILES)
    def test_paths_are_given_from_the_repository_root(
            self, name: str) -> None:
        """
        No path of the result file starts at the root of a file system.

        :param name: Name of the result file, without extension.
        :type name: str
        """
        text = (RESULTS / f"{name}.json").read_text(encoding="utf-8")
        assert not re.search(r'"/(?:home|tmp|mnt|usr)/', text)
        assert not re.search(r"/(?:home|mnt)/", text)


@needs_data
class TestSevenStreams:
    """
    ``siete_flujos``: recomputed from ``stats.json`` and ``runs.csv``.
    """

    def test_twelve_comparisons_on_seven_streams(self) -> None:
        """
        Hodges-Lehmann estimate, k+ and exact interval with n = 7.

        The differences of each comparison are those ``stats.json``
        stores, without ACI-IoT-2023 (D7) and CSE-CIC-IDS2018 (D9). With
        seven units there are 28 Walsh averages and the exact interval
        runs from the 3rd to the 26th, with 95.3 % coverage.
        """
        stats = PairedStats()
        source = _confirmatory_differences()
        rows = _result("siete_flujos")["calculo_1_doce_comparaciones"][
            "reducciones"]["n7_sin_ACI_ni_CIC2018"]
        assert len(rows) == 12
        for row in rows:
            d = [source[row["clave"]][s] for s in SEVEN]
            low, high, coverage = stats.hl_interval(d, alpha=ALPHA_7)
            assert row["unidades"] == list(SEVEN)
            assert stats.hodges_lehmann(d) == pytest.approx(
                row["hl"], abs=TOL)
            assert sum(1 for v in d if v > 0) == row["k_pos"]
            assert low == pytest.approx(row["ic"]["lower"], abs=TOL)
            assert high == pytest.approx(row["ic"]["upper"], abs=TOL)
            assert coverage == pytest.approx(0.953125)
            assert row["ic"]["coverage"] == pytest.approx(0.953125)

    def test_reductions_are_reported_without_p_values(self) -> None:
        """
        A reduction of the set of streams holds no p value.

        The plan fixes that the reductions are reported as estimate,
        counts and exact interval. The nine datasets keep the
        Holm-adjusted p value of the confirmatory test, which is the
        one of ``stats.json``.
        """
        source = {name: entry for block in (
            _json("stats.json")[f] for f in ("H1", "H4"))
            for name, entry in block["comparisons"].items()}
        blocks = _result("siete_flujos")["calculo_1_doce_comparaciones"][
            "reducciones"]
        assert len(blocks) == 6
        for name, rows in blocks.items():
            for row in rows:
                held = sorted(k for k in row if k.startswith(("p_", "holm")))
                if name != "n9":
                    assert held == list()
                elif row["clave"] in source:
                    assert row["p_holm"] == pytest.approx(
                        source[row["clave"]]["p_holm"], abs=TOL)
        compared = [entry["que"] for entry in _result("siete_flujos")[
            "regresion"]]
        assert len(compared) == 13
        assert not [text for text in compared if text.startswith("p ")]

    @pytest.mark.parametrize("stream", ("D7", "D9"))
    def test_in_sample_calibration_by_configuration(self,
                                                    stream: str) -> None:
        """
        Change of the fused detector, in-sample minus main calibration.

        From ``runs.csv``: rows of the main experiment, fused series,
        analyses ``main`` and ``S-INS``. Each configuration is the mean
        over its seeds of the per-seed difference; the replicates of the
        per-record MLP within a seed hold the same value.

        :param stream: Stream id, D7 (ACI-IoT-2023) or D9
            (CSE-CIC-IDS2018).
        :type stream: str
        """
        values: dict[tuple[str, str, str], dict[str, list[float]]] = dict()
        configs = _result("siete_flujos")[
            "calculo_2_calibracion_en_muestra"][stream]["configs"]
        for row in _rows("runs.csv"):
            if (row["dataset_id"] == stream and row["series"] == "FUS"
                    and row["in_S_star"] == "True"
                    and row["analysis"] in ("main", "S-INS")):
                key = (row["config_id"], row["seed"], row["analysis"])
                for metric in ("auroc", "f1", "fpr", "recall"):
                    values.setdefault(key, dict()).setdefault(
                        metric, list()).append(float(row[metric]))
        assert len(configs) == 11
        for config, entry in configs.items():
            for metric in ("auroc", "f1", "fpr", "recall"):
                per_seed = [
                    _mean(values[(config, seed, "S-INS")][metric])
                    - _mean(values[(config, seed, "main")][metric])
                    for seed in entry["semillas"]]
                assert _mean(per_seed) == pytest.approx(
                    entry[f"d_{metric}"], abs=TOL)


@needs_data
class TestRawScore:
    """
    The pre-specified tables on the raw score, from the shipped files.
    """

    @pytest.mark.parametrize("panel, column", (
        ("whole_segment", "auroc_RAW_mean"),
        ("blocks_10000", "auc_loc_RAW_mean")))
    def test_raw_score_per_cell(self, panel: str, column: str) -> None:
        """
        ROC-AUC of the raw score per cell equals ``cells.csv``.

        :param panel: Panel of the table in ``score_bruto.json``.
        :type panel: str
        :param column: Column of ``cells.csv`` with the same quantity.
        :type column: str
        """
        cells = _result("score_bruto")["results"]["point_1"][panel][
            "cells"]
        rows = _rows("aggregate/cells.csv")
        assert len(rows) == 99
        for row in rows:
            assert cells[row["config_id"]][row["dataset_id"]] == (
                pytest.approx(float(row[column]), abs=TOL))

    def test_knad_raw_and_fused(self) -> None:
        """
        ROC-AUC and EER of the raw score and of the fused detector on
        the six KNAD captures equal the ``metrics.json`` of each run,
        and each column is the mean of its runs.
        """
        captures = _result("knad_y_recuentos")[
            "parte1_knad_score_bruto"]["por_captura"]
        assert len(captures) == 6
        for capture in captures.values():
            for column in ("kitnet_ref", "control_dA", "mlp"):
                runs = capture[column]["runs"]
                for run in runs:
                    stored = _metrics(run["job_id"])["E"]
                    assert run["auc_raw"] == stored["RAW"]["auroc"]
                    assert run["eer_raw"] == stored["RAW"]["eer"]
                    assert run["auc_fus"] == stored["FUS"]["auroc"]
                    assert run["eer_fus"] == stored["FUS"]["eer"]
                for key in ("auc_raw", "eer_raw", "auc_fus", "eer_fus"):
                    assert capture[column][key] == pytest.approx(
                        _mean([run[key] for run in runs]), abs=TOL)


@needs_data
class TestRatioCountsOfTheMainExperiment:
    """
    ``knad_y_recuentos``, part 2, from the audit this package writes.
    """

    def test_counts_by_family(self) -> None:
        """
        Autoencoders per band of the ratio, main experiment only.

        ``data/reference/training_audit.json`` holds the ratio of every
        autoencoder of every audited run. Restricted to the runs of the
        main experiment (blocks ``T1-A-*`` and ``T1-B-*``), the counts
        below 0.9, in [0.9, 1.15), at 1.15 or above and not computable
        must be those of the result file, family by family.
        """
        verdict_key = {"trained": "lt_0.9", "at the floor": "0.9_to_1.15",
                       "worse than the floor": "ge_1.15",
                       "not computable": "nc"}
        counts: dict[str, dict[str, int]] = dict()
        runs: dict[str, int] = dict()
        expected = _result("knad_y_recuentos")[
            "parte2_recuentos_experimento_principal"][
            "experimento_principal"]
        for job in _json("reference/training_audit.json")["jobs"]:
            if job["block"].startswith(("T1-A", "T1-B")):
                runs[job["variant"]] = runs.get(job["variant"], 0) + 1
                for autoencoder in job["autoencoders"]:
                    key = verdict_key[autoencoder["verdict"]]
                    family = counts.setdefault(job["variant"], dict())
                    family[key] = family.get(key, 0) + 1
        assert sum(runs.values()) == expected["runs"] == 180
        for variant, entry in expected["por_familia"].items():
            assert runs[variant] == entry["runs"]
            for key in verdict_key.values():
                assert counts[variant].get(key, 0) == entry[key]


@needs_data
class TestRetrainingExperiment:
    """
    ``contencion``: the fused detector of the pair, from the run files.
    """

    @staticmethod
    def _key(job_id: str) -> str:
        """
        Stream key of the result file from the name of a run.

        :param job_id: Name of the run folder.
        :type job_id: str
        :return: For instance ``knad_fuzzing`` or ``aci``.
        :rtype: str
        """
        name = job_id.split("__")[1]
        return "aci" if name.startswith("aci") else name.replace("-", "_")

    def _pair(self, windowed: str, per_record: str, suffix: str,
              ) -> dict[str, list[float]]:
        """
        Windowed minus per-record ROC-AUC of the fused detector.

        :param windowed: Prefix of the runs of the windowed member.
        :type windowed: str
        :param per_record: Prefix of the runs of the per-record member.
        :type per_record: str
        :param suffix: Text the run name must contain, or an empty
            string for every run.
        :type suffix: str
        :return: Stream key -> one difference per matched pair of runs.
        :rtype: dict[str, list[float]]
        """
        out: dict[str, list[float]] = dict()
        for folder in sorted((DATA / "jobs").glob(f"{windowed}__*")):
            parts = folder.name.split("__")
            if suffix not in folder.name:
                continue
            twin = sorted((DATA / "jobs").glob(
                f"{per_record}__{parts[1]}__*__{'__'.join(parts[3:])}"))
            if len(twin) != 1 or not (folder / "metrics.json").exists() \
                    or not (twin[0] / "metrics.json").exists():
                continue
            out.setdefault(self._key(folder.name), list()).append(
                _metrics(folder.name)["primary"]["value"]
                - _metrics(twin[0].name)["primary"]["value"])
        return out

    def test_one_epoch_pair_over_nine_datasets(self) -> None:
        """
        One epoch: mean difference per dataset, estimate and interval.

        Each dataset is the mean of its four pairs (two seeds, two
        paradigms); the Hodges-Lehmann estimate and its exact 96.1 %
        interval over the nine datasets are recomputed with the routines
        of this package.
        """
        stats = PairedStats()
        stored = _result("contencion")["par_por_score"]["fus"][
            "one_epoch_dataset_level_n9"]
        pairs = self._pair("T8-CTX", "T8-BASE", "")
        d = {key: _mean(values) for key, values in pairs.items()}
        low, high, _ = stats.hl_interval(list(d.values()), alpha=ALPHA_9)
        assert all(len(values) == 4 for values in pairs.values())
        assert d == pytest.approx(stored["differences"], abs=TOL)
        assert stats.hodges_lehmann(list(d.values())) == pytest.approx(
            stored["hl"], abs=TOL)
        assert low == pytest.approx(stored["ci_lower"], abs=TOL)
        assert high == pytest.approx(stored["ci_upper"], abs=TOL)
        assert sum(1 for v in d.values() if v > 0) == stored["k_plus"] == 3

    def test_five_epoch_ablation_stream_by_stream(self) -> None:
        """
        One and five epochs, first seed, full-sequence paradigm.

        The difference of every stream of the table equals the one
        computed from the two ``metrics.json`` of its pair; eight
        streams have a five-epoch pair.
        """
        one = self._pair("T8-CTX", "T8-BASE", "__arF__")
        five = self._pair("T8-EP5", "T8-EP5B", "__arF__")
        table = _result("contencion")["por_flujo"]
        assert len(table) == 9 and len(five) == 8
        for row in table:
            first_seed = [
                _metrics(f.name)["primary"]["value"]
                - _metrics(f.name.replace("T8-CTX", "T8-BASE").replace(
                    "mlp_ctx_fix", "mlp_fix"))["primary"]["value"]
                for f in sorted((DATA / "jobs").glob("T8-CTX__*__arF__*sA*"))
                if self._key(f.name) == row["dataset_key"]]
            assert len(one[row["dataset_key"]]) == 2
            assert first_seed == [pytest.approx(row["fus"]["e1_diff"],
                                                abs=TOL)]
            if row["dataset_key"] in five:
                assert five[row["dataset_key"]] == [pytest.approx(
                    row["fus"]["e5_diff"], abs=TOL)]


@needs_data
class TestControlReferences:
    """
    ``h1_referencias``, ``control_armonizado`` and
    ``region_post_preludio`` against the values already shipped.
    """

    def test_h1_and_h4_as_run_equal_the_confirmatory_statistics(
            self) -> None:
        """
        The block "as run" of both tables is the one of ``stats.json``.
        """
        stats = _json("stats.json")
        result = _result("h1_referencias")
        for family in ("H1", "H4"):
            for name, entry in stats[family]["comparisons"].items():
                mine = result[family]["as_run"][name]
                assert mine["hl"] == pytest.approx(entry["hl"], abs=TOL)
                assert mine["k_pos"] == entry["k_pos"]
                assert mine["d"] == pytest.approx(entry["d"], abs=TOL)

    def test_control_with_one_stabiliser_equals_its_runs(self) -> None:
        """
        Block (c): the control repeated with a single stabiliser.

        Its ROC-AUC per dataset is the primary value of the nine runs
        ``T8-EPS__*`` of ``data/jobs/``, and the five H1 estimates of
        the block follow from them and from ``cells.csv``.
        """
        stats = PairedStats()
        result = _result("h1_referencias")
        cells = {(r["dataset_id"], r["config_id"]):
                 float(r["auroc_FUS_mean"])
                 for r in _rows("aggregate/cells.csv")}
        control: dict[str, float] = dict()
        families = {"Conv1D": ("C1", "C2"), "Conv2D": ("C3", "C4"),
                    "LSTM": ("C5", "C6"), "MHA": ("C7", "C8"),
                    "MLP": ("C9",)}
        for folder in sorted((DATA / "jobs").glob("T8-EPS__*")):
            metrics = _metrics(folder.name)
            control[metrics["job"]["dataset_id"]] = metrics[
                "primary"]["value"]
        assert control == pytest.approx(
            result["control_auroc_fus"]["same_stabiliser"]["per_dataset"],
            abs=TOL)
        for family, members in families.items():
            d = [_mean([cells[(s, c)] for c in members]) - control[s]
                 for s in DATASETS]
            assert stats.hodges_lehmann(d) == pytest.approx(
                result["H1"]["same_stabiliser"][
                    f"{family}_vs_original"]["hl"], abs=TOL)

    def test_whole_region_values_equal_the_cells(self) -> None:
        """
        ``region_post_preludio``: its whole-region values, which the
        restricted region is compared with, are those of ``cells.csv``.
        """
        full = _result("region_post_preludio")["cells"]["full"]
        for row in _rows("aggregate/cells.csv"):
            for series in ("FUS", "RAW"):
                assert full[series][row["dataset_id"]][
                    row["config_id"]] == pytest.approx(
                        float(row[f"auroc_{series}_mean"]), abs=TOL)

    def test_treatment_matched_control_starts_from_the_control(
            self) -> None:
        """
        ``control_armonizado``: the combination "as run" of its eight
        reproduces the control of the main results, stream by stream,
        and the summary holds the eight combinations for nine streams.
        """
        result = _result("control_armonizado")
        cells = {(r["dataset_id"], r["config_id"]):
                 float(r["auroc_FUS_mean"])
                 for r in _rows("aggregate/cells.csv")}
        assert sorted(result["streams"]) == list(DATASETS)
        for stream, entry in result["streams"].items():
            combos = [k for k in entry["cells"] if k.startswith("dA__")]
            assert len(combos) == 8
            assert entry["cells"][result["as_run_cell"]]["auroc"][
                "FUS"] == pytest.approx(cells[(stream, "C10")], abs=TOL)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
