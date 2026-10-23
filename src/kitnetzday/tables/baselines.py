# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         src/kitnetzday/tables/baselines.py                        #
#   Version:      1.0.0                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-07                                                #
#   Modified:     2026-10-07                                                #
#   Description:  The two comparator tables of the paper: the reference     #
#                 KitNET on the KNAD captures and the Isolation Forest.     #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import json
import logging
import statistics as st
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from kitnetzday.tables.supplementary_inputs import SupplementaryInputs
from kitnetzday.utils.formatting import get
from kitnetzday.utils.loggable import Loggable

# Job blocks of the two comparators, as prefixes of their directories.
KITNET_PREFIX: str = "T5-KITNET__"
FOREST_PREFIX: str = "T5-IF__"
# Configuration of the denoising-autoencoder ensemble of this work.
DA_CONFIG: str = "C10"
# The six packet-level captures, which are the streams the reference
# implementation was run on.
KNAD_DATASETS: tuple[str, ...] = ("D1", "D2", "D3", "D4", "D5", "D6")
# Decimals of the Markdown preview.
DECIMALS: int = 3


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                MAIN CLASS                                 #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class BaselineTables(Loggable):
    """
    The comparators that sit next to the eleven configurations.

    Two of the floats of the paper compare the configurations of the
    study with models that are not part of its grid. The first gives,
    for the six KNAD captures, the ROC-AUC and the equal error rate of
    the published KitNET implementation run under the label-free regime
    of this work, of the denoising-autoencoder ensemble of this work,
    and of the configuration with the highest mean ROC-AUC of each
    capture, which is an oracle selection and therefore an upper bound.
    The second is the Isolation Forest row of the main results.

    Nothing is computed here. The comparators are single runs, so their
    values are read as ``primary.value`` and ``E.FUS.eer`` of the
    ``metrics.json`` of their jobs; the values of the configurations
    come from ``SupplementaryInputs``, which already averaged them over
    the seeds.

    :param data: Inputs of the supplement, already read; they give the
        means of the configurations and the results root.
    :type data: SupplementaryInputs
    :param knad_datasets: Dataset ids of the captures of the first
        table, in row order; non-empty.
    :type knad_datasets: Sequence[str]
    :param da_config: Configuration id of the denoising-autoencoder
        ensemble of this work.
    :type da_config: str
    :param logger: Logger created by the calling script, or None to stay
        silent.
    :type logger: logging.Logger | None
    :raises TypeError: If data or logger are of the wrong type.
    :raises ValueError: If knad_datasets is empty or da_config is not a
        non-empty string.
    """

    def __init__(self,
                 data: SupplementaryInputs,
                 knad_datasets: Sequence[str] = KNAD_DATASETS,
                 da_config: str = DA_CONFIG,
                 logger: logging.Logger | None = None) -> None:
        """
        Check the settings and read the comparator jobs right away.

        :param data: Inputs of the supplement, already read.
        :type data: SupplementaryInputs
        :param knad_datasets: Dataset ids of the first table; non-empty.
        :type knad_datasets: Sequence[str]
        :param da_config: Configuration id of the ensemble of this work.
        :type da_config: str
        :param logger: Logger created by the calling script, or None.
        :type logger: logging.Logger | None
        :raises TypeError: If data or logger are of the wrong type.
        :raises ValueError: If knad_datasets is empty or da_config is
            not a non-empty string.
        """
        super().__init__(logger=logger)
        if not isinstance(data, SupplementaryInputs):
            self._error("[BaselineTables.__init__] Input error: the "
                        "parameter data must be a SupplementaryInputs, "
                        f"not {type(data).__name__}.", TypeError)
        if not knad_datasets:
            self._error("[BaselineTables.__init__] Input error: the "
                        "parameter knad_datasets must hold at least one "
                        "dataset id, and it is empty.")
        if not isinstance(da_config, str) or not da_config:
            self._error("[BaselineTables.__init__] Input error: the "
                        "parameter da_config must be a non-empty "
                        f"string, not {da_config!r}.")
        self.data = data
        self.knad_datasets = [str(d) for d in knad_datasets]
        self.da_config = da_config
        self.kitnet = self._read_jobs(KITNET_PREFIX)
        self.forest = self._read_jobs(FOREST_PREFIX)

    def _read_jobs(self, prefix: str) -> dict[str, dict[str, Any]]:
        """
        Read the single run of one comparator on every stream.

        A directory without ``metrics.json``, or whose file cannot be
        parsed, is skipped with a warning.

        :param prefix: Prefix of the job directories of the comparator.
        :type prefix: str
        :return: Dataset id -> {"job_id", "auroc", "eer"}.
        :rtype: dict[str, dict[str, Any]]
        """
        out: dict[str, dict[str, Any]] = dict()
        record: Any = None
        root = self.data.results_root
        if not root.is_dir():
            return out
        for path in sorted(root.glob(f"{prefix}*/metrics.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                self._warning("[BaselineTables._read_jobs] Input "
                              "warning: every metrics file must be "
                              f"readable JSON, and {path} is not "
                              f"({error}). The job is skipped.")
                continue
            out[get(record, "job", "dataset_id")] = {
                "job_id": path.parent.name,
                "auroc": get(record, "primary", "value"),
                "eer": get(record, "E", "FUS", "eer")}
        return out

    def best_config(self, dataset: str) -> str | None:
        """
        Configuration with the highest mean ROC-AUC of one stream.

        This is a selection on the evaluation segment, so whatever is
        reported for it is an upper bound and not an estimate. A tie is
        broken by comparing the configuration ids as text, as the
        summaries of the supplement do.

        :param dataset: Dataset id, such as "D1".
        :type dataset: str
        :return: The configuration id, or None when no configuration of
            the stream has a ROC-AUC.
        :rtype: str | None
        """
        scored = [(self.data.metrics_of(dataset, c).get("auroc"), c)
                  for c in self.data.naming.configurations
                  if self.data.metrics_of(dataset, c).get("auroc")
                  is not None]
        return max(scored)[1] if scored else None

    def knad_table(self) -> dict[str, Any]:
        """
        The comparison on the KNAD captures, with its row of means.

        :return: Dataset id -> {"dataset", "kitnet_reference",
            "kitnet_style_da", "best"}, each of the three with "auroc"
            and "eer", plus the key "mean" with the mean of every
            column over the captures that have a value.
        :rtype: dict[str, Any]
        """
        table: dict[str, Any] = dict()
        best: str | None = None
        for dataset in self.knad_datasets:
            best = self.best_config(dataset)
            table[dataset] = {
                "dataset": self.data.dataset_names.get(dataset, dataset),
                "kitnet_reference": {
                    k: self.kitnet.get(dataset, dict()).get(k)
                    for k in ("auroc", "eer", "job_id")},
                "kitnet_style_da": self._cell(dataset, self.da_config),
                "best": (dict() if best is None else dict(
                    self._cell(dataset, best), config=best,
                    name=self.data.config_label(best)))}
        table["mean"] = {
            column: {k: self.data.mean(
                [table[d][column].get(k) for d in self.knad_datasets])
                for k in ("auroc", "eer")}
            for column in ("kitnet_reference", "kitnet_style_da",
                           "best")}
        return table

    def forest_row(self) -> dict[str, Any]:
        """
        The Isolation Forest row of the main results, stream by stream.

        :return: {"by_dataset": dataset id -> {"job_id", "auroc",
            "eer"}, "mean", "median"}, the last two over the streams
            that have a value, None when there is none.
        :rtype: dict[str, Any]
        """
        values = [self.forest[d]["auroc"]
                  for d in self.data.naming.datasets
                  if self.forest.get(d, dict()).get("auroc") is not None]
        return {"by_dataset": {d: self.forest[d]
                               for d in self.data.naming.datasets
                               if d in self.forest},
                "mean": st.mean(values) if values else None,
                "median": st.median(values) if values else None}

    def document(self, source: str) -> dict[str, Any]:
        """
        The JSON document of the two comparator tables.

        :param source: Provenance text, the aggregated pass that was
            read.
        :type source: str
        :return: {"source", "knad_auc_eer", "isolation_forest"}.
        :rtype: dict[str, Any]
        """
        return {"source": source, "knad_auc_eer": self.knad_table(),
                "isolation_forest": self.forest_row()}

    def markdown(self) -> str:
        """
        Markdown preview of the two tables.

        :return: The text of the preview, ending with a newline.
        :rtype: str
        """
        knad = self.knad_table()
        forest = self.forest_row()
        lines = [
            "# Comparators", "",
            "## AUC and EER on the six KNAD captures", "",
            "| Capture | KitNET (reference) AUC | EER | KitNET-style dA "
            "AUC | EER | Best variant (oracle selection) | AUC | EER |",
            "|---|---|---|---|---|---|---|---|"]
        for dataset in self.knad_datasets:
            row = knad[dataset]
            lines.append("| " + " | ".join([
                str(row["dataset"]),
                self._number(row["kitnet_reference"].get("auroc")),
                self._number(row["kitnet_reference"].get("eer")),
                self._number(row["kitnet_style_da"].get("auroc")),
                self._number(row["kitnet_style_da"].get("eer")),
                str(row["best"].get("name", "--")),
                self._number(row["best"].get("auroc")),
                self._number(row["best"].get("eer"))]) + " |")
        mean = knad["mean"]
        lines.append("| **Mean** | " + " | ".join([
            self._number(mean["kitnet_reference"]["auroc"]),
            self._number(mean["kitnet_reference"]["eer"]),
            self._number(mean["kitnet_style_da"]["auroc"]),
            self._number(mean["kitnet_style_da"]["eer"]), "",
            self._number(mean["best"]["auroc"]),
            self._number(mean["best"]["eer"])]) + " |")
        lines += ["", "## Isolation Forest, ROC-AUC of the fused "
                  "detector (single run)", "", "| Stream | ROC-AUC |",
                  "|---|---|"]
        for dataset, row in forest["by_dataset"].items():
            lines.append(
                f"| {self.data.dataset_names.get(dataset, dataset)} | "
                f"{self._number(row.get('auroc'))} |")
        lines.append(f"| **Mean** | {self._number(forest['mean'])} |")
        lines.append(f"| **Median** | {self._number(forest['median'])} |")
        return "\n".join(lines) + "\n"

    def write(self,
              out_json: str | Path,
              out_md: str | Path,
              source: str) -> tuple[Path, Path]:
        """
        Write the JSON document and the Markdown preview.

        :param out_json: Path of the ``.json`` to write.
        :type out_json: str | Path
        :param out_md: Path of the ``.md`` to write.
        :type out_md: str | Path
        :param source: Provenance text of the JSON document.
        :type source: str
        :return: The two paths written, the JSON one first.
        :rtype: tuple[Path, Path]
        :raises OSError: If a file cannot be written.
        """
        json_path, md_path = Path(out_json), Path(out_md)
        json_path.write_text(json.dumps(self.document(source), indent=1),
                             encoding="utf-8")
        md_path.write_text(self.markdown(), encoding="utf-8")
        return json_path, md_path

    def _cell(self, dataset: str, config: str) -> dict[str, Any]:
        """
        Mean ROC-AUC and mean EER of one cell of the grid.

        :param dataset: Dataset id, such as "D1".
        :type dataset: str
        :param config: Configuration id, such as "C10".
        :type config: str
        :return: {"auroc", "eer"}, None where the cell has no value.
        :rtype: dict[str, Any]
        """
        metrics = self.data.metrics_of(dataset, config)
        return {"auroc": metrics.get("auroc"), "eer": metrics.get("eer")}

    def _number(self, value: Any) -> str:
        """
        Write one value of the preview.

        :param value: A number, or None.
        :type value: Any
        :return: The number with three decimals, or "--".
        :rtype: str
        """
        return "--" if value is None else f"{value:.{DECIMALS}f}"

    def __repr__(self) -> str:
        """
        Show how many comparator runs were read.

        :return: Text like "BaselineTables(kitnet=6, forest=9)".
        :rtype: str
        """
        return (f"BaselineTables(kitnet={len(self.kitnet)}, "
                f"forest={len(self.forest)})")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
