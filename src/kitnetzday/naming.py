# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         src/kitnetzday/naming.py                                  #
#   Version:      2.0.0                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Vocabulary of the study: dataset and configuration ids,   #
#                 their labels, and the presentation constants.             #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
from collections.abc import Iterable, Mapping, Sequence

from kitnetzday.utils.loggable import Loggable

# Default vocabulary of the analysis plan (plan section 14.1). The study
# values also live in src/config.json; the scripts in bin/ read that file
# and hand the values to the constructor, library code never reads it.
DATASET_IDS: list[str] = [f"D{i}" for i in range(1, 10)]
CONFIG_IDS: list[str] = [f"C{i}" for i in range(1, 12)]
# Plan section 10: the configurations without temporal context. Everything
# else in CONFIG_IDS carries temporal context (C1..C8 by default).
CONTEXT_FREE_CONFIGS: list[str] = ["C9", "C10", "C11"]
# Plan section 5.3: control configuration fixed by design.
CONTROL_CONFIG: str = "C10"
ARCHITECTURES: list[str] = ["Conv1D", "Conv2D", "LSTM", "MHA", "MLP"]
ARCH_CONFIGS: dict[str, tuple[str, ...]] = {
    "Conv1D": ("C1", "C2"), "Conv2D": ("C3", "C4"),
    "LSTM": ("C5", "C6"), "MHA": ("C7", "C8"), "MLP": ("C9",)}
# Plan section 10 (failure margin) and section 5.2 (region of practical
# equivalence); both are 0.01 ROC-AUC points.
FAILURE_MARGIN: float = 0.01
ROPE: float = 0.01
CONFIG_NAME_FALLBACK: dict[str, str] = {
    "C1": "Conv1D-AR", "C2": "Conv1D-TSR", "C3": "Conv2D-AR",
    "C4": "Conv2D-TSR", "C5": "LSTM-AR", "C6": "LSTM-TSR",
    "C7": "MHA-AR", "C8": "MHA-TSR", "C9": "MLP",
    "C10": "KitNET-style dA", "C11": "STAT"}
# Axis labels: the long name does not fit.
CONFIG_SHORT: dict[str, str] = {"C9": "MLP", "C10": "dA", "C11": "STAT"}
DATASET_SHORT: dict[str, str] = {
    "D1": "AW", "D2": "FUZZ", "D3": "OS", "D4": "SSDP", "D5": "SSL",
    "D6": "VI", "D7": "ACI", "D8": "CIC17", "D9": "CIC18"}
DATASET_LONG: dict[str, str] = {
    "D1": "KNAD Active Wiretap", "D2": "KNAD Fuzzing",
    "D3": "KNAD OS Scan", "D4": "KNAD SSDP Flood",
    "D5": "KNAD SSL Renegotiation", "D6": "KNAD Video Injection",
    "D7": "ACI-IoT-2023", "D8": "CIC-IDS2017", "D9": "CSE-CIC-IDS2018"}
# Submitted Fig. 7: name used in the archived CSV -> dataset id of the
# plan.
FIG7_DATASETS: dict[str, str] = {
    "ACTIVE-WIRETAP": "D1", "FUZZING": "D2", "OS-SCAN": "D3",
    "SSDP-FLOOD": "D4", "SSL-RENEGOTIATION": "D5",
    "VIDEO-INJECTION": "D6", "ACI-IOT-2023": "D7", "CIC2017": "D8",
    "CIC2018": "D9"}
# Submitted Fig. 7: order of the twelve bars inside every dataset group.
FIG7_ORDER: list[tuple[str, str]] = [
    ("MLP", "AR"), ("MLP", "TSR"), ("CONV1D", "AR"), ("CONV1D", "TSR"),
    ("CONV2D", "AR"), ("CONV2D", "TSR"), ("MHA", "AR"), ("MHA", "TSR"),
    ("LSTM", "AR"), ("LSTM", "TSR"), ("KitNET", "PbP"), ("STAT", "PbP")]

# Labels of the analysis plan (section 14.1) written verbatim into the
# LaTeX comments, the README and the table of the values used. They are
# output strings of the pipeline and the first word of each one stays in
# Spanish, as the analysis plan writes it; the README of the package
# explains them.
PLAN_TAGS: dict[str, str] = {
    "primary": "[CONFIRMATORIO plan §3.1]",
    "secondary": ("[SENSIBILIDAD plan §3.2 — métrica secundaria, sin "
                  "afirmaciones confirmatorias]"),
    "trivial": ("[CONFIRMATORIO plan §3.2 — referencias triviales, solo "
                "etiquetas]"),
    "h1": "[CONFIRMATORIO plan §5.3]",
    "omni": ("[CONFIRMATORIO plan §5.3 — omnibus descriptivo, no decide "
             "ninguna afirmacion]"),
    # Section 14.1 only admits CONFIRMATORIO / SENSIBILIDAD /
    # EXPLORATORIO (+ ORACULO at the end): the R1.10 presentation is not a
    # new label, it is the same confirmatory value shown without
    # selection.
    "r110": ("[CONFIRMATORIO plan §11 R1.10 — presentacion descriptiva, "
             "sin seleccion por resultados]"),
    "abl": "[SENSIBILIDAD plan §7]",
    "cost": ("[CONFIRMATORIO plan §9 — coste computacional, sin "
             "afirmaciones de rendimiento predictivo]"),
    "fail": "[EXPLORATORIO plan §10]",
    "o1": ("[ORACULO: as-submitted (O1): umbral y detector elegidos con "
           "etiquetas de evaluacion]"),
    "expl_preplan": ("[EXPLORATORIO] (audit made before the analysis "
                     "plan was fixed)"),
    # Only written when a run reads an aggregate that is not the final
    # one; with the shipped data no output carries it.
    "partial": "[PARTIAL INPUTS]"}

OUTPUT_IDS: list[str] = ["F1", "F2", "T1", "T2", "T3", "F3", "F4"]

# Style: no exotic colours, readable in black and white (greys, hatches
# and markers).
RC_PARAMS: dict[str, object] = {
    "figure.dpi": 110, "savefig.bbox": "tight", "font.size": 8.5,
    "axes.titlesize": 9.5, "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.grid": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "hatch.linewidth": 0.6,
    "lines.linewidth": 1.0, "patch.linewidth": 0.6}
GREY_FILL: str = "0.55"
GREY_LIGHT: str = "0.85"


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                MAIN CLASS                                 #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class Naming(Loggable):
    """
    Vocabulary of the study: which ids exist and how they are written.

    Every figure, table and report of the paper refers to a dataset by an
    id such as ``D8`` and to a configuration by an id such as ``C7``. The
    reader of the paper needs the long names instead ("CIC-IDS2017",
    "MHA-AR"), the axes of the figures only fit the short ones, and the
    code needs to know which configurations carry temporal context. This
    class keeps the four vocabularies in one place so that no module
    spells a label of its own.

    The defaults reproduce the submitted study (nine datasets, eleven
    configurations, five architectures). The scripts in ``bin/`` read
    ``src/config.json`` and pass the values here; library code never
    reads the configuration file.

    :param datasets: Dataset ids in presentation order, for example
        ``["D1", ..., "D9"]``; non-empty.
    :type datasets: Sequence[str]
    :param configurations: Configuration ids in presentation order, for
        example ``["C1", ..., "C11"]``; non-empty.
    :type configurations: Sequence[str]
    :param context_free: Configuration ids without temporal context
        (plan section 10). They must all appear in configurations.
    :type context_free: Iterable[str]
    :param architectures: Names of the autoencoder architectures.
    :type architectures: Sequence[str]
    :param arch_configs: Architecture name -> the configuration ids that
        use it, as a tuple in presentation order.
    :type arch_configs: Mapping[str, tuple[str, ...]]
    :param config_names: Configuration id -> long label used when the
        aggregate does not carry one.
    :type config_names: Mapping[str, str]
    :param config_short: Configuration id -> short axis label. Ids that
        are missing fall back to the long label.
    :type config_short: Mapping[str, str]
    :param dataset_short: Dataset id -> short axis label.
    :type dataset_short: Mapping[str, str]
    :param dataset_long: Dataset id -> long label for captions.
    :type dataset_long: Mapping[str, str]
    :param fig7_datasets: Dataset name as written in the archived CSV of
        the submitted Fig. 7 -> dataset id of the plan.
    :type fig7_datasets: Mapping[str, str]
    :param control: Configuration id of the control fixed by design
        (plan section 5.3); it must be one of configurations.
    :type control: str
    :param failure_margin: Margin in ROC-AUC points used by the failure
        cases of plan section 10; strictly positive.
    :type failure_margin: float
    :param rope: Half width of the region of practical equivalence in
        ROC-AUC points (plan section 5.2); strictly positive.
    :type rope: float
    :param logger: Logger created by the calling script, or None to stay
        silent.
    :type logger: logging.Logger | None
    :raises TypeError: If logger is not a logging.Logger or None.
    :raises ValueError: If datasets or configurations are empty, if
        control is not a configuration id, if any context-free id is not
        a configuration id, or if failure_margin or rope is not
        strictly positive.
    """

    def __init__(self,
                 datasets: Sequence[str] = tuple(DATASET_IDS),
                 configurations: Sequence[str] = tuple(CONFIG_IDS),
                 context_free: Iterable[str] = tuple(
                     CONTEXT_FREE_CONFIGS),
                 architectures: Sequence[str] = tuple(ARCHITECTURES),
                 arch_configs: Mapping[str, tuple[str, ...]] = (
                     ARCH_CONFIGS),
                 config_names: Mapping[str, str] = CONFIG_NAME_FALLBACK,
                 config_short: Mapping[str, str] = CONFIG_SHORT,
                 dataset_short: Mapping[str, str] = DATASET_SHORT,
                 dataset_long: Mapping[str, str] = DATASET_LONG,
                 fig7_datasets: Mapping[str, str] = FIG7_DATASETS,
                 control: str = CONTROL_CONFIG,
                 failure_margin: float = FAILURE_MARGIN,
                 rope: float = ROPE,
                 logger: logging.Logger | None = None) -> None:
        """
        Copy the vocabulary after checking that it is self-consistent.

        :param datasets: Dataset ids in presentation order; non-empty.
        :type datasets: Sequence[str]
        :param configurations: Configuration ids in order; non-empty.
        :type configurations: Sequence[str]
        :param context_free: Configuration ids without temporal context.
        :type context_free: Iterable[str]
        :param architectures: Architecture names.
        :type architectures: Sequence[str]
        :param arch_configs: Architecture -> configuration ids.
        :type arch_configs: Mapping[str, tuple[str, ...]]
        :param config_names: Configuration id -> long label.
        :type config_names: Mapping[str, str]
        :param config_short: Configuration id -> short label.
        :type config_short: Mapping[str, str]
        :param dataset_short: Dataset id -> short label.
        :type dataset_short: Mapping[str, str]
        :param dataset_long: Dataset id -> long label.
        :type dataset_long: Mapping[str, str]
        :param fig7_datasets: Fig. 7 CSV name -> dataset id.
        :type fig7_datasets: Mapping[str, str]
        :param control: Control configuration id.
        :type control: str
        :param failure_margin: Failure margin, strictly positive.
        :type failure_margin: float
        :param rope: Half width of the ROPE, strictly positive.
        :type rope: float
        :param logger: Logger created by the calling script, or None.
        :type logger: logging.Logger | None
        :raises TypeError: If logger is not a logging.Logger or None.
        :raises ValueError: If the vocabulary is empty or inconsistent.
        """
        super().__init__(logger=logger)
        unknown = list()
        if not datasets:
            self._error("[Naming.__init__] Input error: the parameter "
                        "datasets must hold at least one dataset id, not "
                        "an empty sequence.")
        if not configurations:
            self._error("[Naming.__init__] Input error: the parameter "
                        "configurations must hold at least one "
                        "configuration id, not an empty sequence.")
        self.datasets = [str(x) for x in datasets]
        self.configurations = [str(c) for c in configurations]
        self.context_free = [str(c) for c in context_free]
        unknown = [c for c in self.context_free
                   if c not in self.configurations]
        if unknown:
            self._error("[Naming.__init__] Input error: every id in the "
                        "parameter context_free must be a configuration "
                        f"id, and {unknown} is not in "
                        f"{self.configurations}.")
        if str(control) not in self.configurations:
            self._error("[Naming.__init__] Input error: the parameter "
                        "control must be one of the configuration ids "
                        f"{self.configurations}, not {control!r}.")
        if not float(failure_margin) > 0.0:
            self._error("[Naming.__init__] Input error: the parameter "
                        "failure_margin must be strictly positive (>0), "
                        f"not {failure_margin}.")
        if not float(rope) > 0.0:
            self._error("[Naming.__init__] Input error: the parameter "
                        f"rope must be strictly positive (>0), not "
                        f"{rope}.")
        self.architectures = [str(a) for a in architectures]
        self.arch_configs = {str(k): tuple(v)
                             for k, v in dict(arch_configs).items()}
        self.config_names = dict(config_names)
        self.config_shorts = dict(config_short)
        self.dataset_shorts = dict(dataset_short)
        self.dataset_longs = dict(dataset_long)
        self.fig7_datasets = dict(fig7_datasets)
        self.control = str(control)
        self.failure_margin = float(failure_margin)
        self.rope = float(rope)
        self.plan_tags = dict(PLAN_TAGS)

    # ----- datasets ------------------------------------------------- #
    def dataset_label(self, dataset_id: str) -> str:
        """
        Long name of a dataset, for captions and table rows.

        :param dataset_id: Dataset id such as "D8".
        :type dataset_id: str
        :return: The long name ("CIC-IDS2017"), or the id itself when
            the vocabulary does not know it.
        :rtype: str
        :raises TypeError: If dataset_id is not a string.
        """
        if not isinstance(dataset_id, str):
            self._error("[Naming.dataset_label] Input error: the "
                        "parameter dataset_id must be a string, not "
                        f"{type(dataset_id).__name__}.", TypeError)
        return self.dataset_longs.get(dataset_id, dataset_id)

    def dataset_short_label(self, dataset_id: str) -> str:
        """
        Short name of a dataset, for the axes of the figures.

        :param dataset_id: Dataset id such as "D8".
        :type dataset_id: str
        :return: The short label ("CIC17"), or the id itself when the
            vocabulary does not know it.
        :rtype: str
        :raises TypeError: If dataset_id is not a string.
        """
        if not isinstance(dataset_id, str):
            self._error("[Naming.dataset_short_label] Input error: the "
                        "parameter dataset_id must be a string, not "
                        f"{type(dataset_id).__name__}.", TypeError)
        return self.dataset_shorts.get(dataset_id, dataset_id)

    def fig7_dataset(self, csv_name: str) -> str | None:
        """
        Dataset id behind a dataset name of the submitted Fig. 7 CSV.

        The names in that CSV are upper case and hyphenated
        ("ACTIVE-WIRETAP"); surrounding blanks are stripped first, as
        the readers in ``kitnetzday.io.legacy`` do.

        :param csv_name: Dataset name as written in the archived CSV.
        :type csv_name: str
        :return: The dataset id, or None when the name is unknown.
        :rtype: str | None
        :raises TypeError: If csv_name is not a string.
        """
        if not isinstance(csv_name, str):
            self._error("[Naming.fig7_dataset] Input error: the "
                        "parameter csv_name must be a string, not "
                        f"{type(csv_name).__name__}.", TypeError)
        return self.fig7_datasets.get(csv_name.strip())

    # ----- configurations ------------------------------------------- #
    def config_label(self, config_id: str) -> str:
        """
        Long label of a configuration, for captions and table headers.

        :param config_id: Configuration id such as "C7".
        :type config_id: str
        :return: The long label ("MHA-AR"), or the id itself when the
            vocabulary does not know it.
        :rtype: str
        :raises TypeError: If config_id is not a string.
        """
        if not isinstance(config_id, str):
            self._error("[Naming.config_label] Input error: the "
                        "parameter config_id must be a string, not "
                        f"{type(config_id).__name__}.", TypeError)
        return self.config_names.get(config_id, config_id)

    def config_short(self, config_id: str) -> str:
        """
        Short label of a configuration, for the axes of the figures.

        Only the three context-free configurations have a short label;
        the rest fall back to the long one, which does fit.

        :param config_id: Configuration id such as "C10".
        :type config_id: str
        :return: The short label ("dA") or the long label.
        :rtype: str
        :raises TypeError: If config_id is not a string.
        """
        if not isinstance(config_id, str):
            self._error("[Naming.config_short] Input error: the "
                        "parameter config_id must be a string, not "
                        f"{type(config_id).__name__}.", TypeError)
        return self.config_shorts.get(config_id,
                                      self.config_label(config_id))

    def is_context_free(self, config_id: str) -> bool:
        """
        Whether a configuration works record by record, without context.

        :param config_id: Configuration id such as "C11".
        :type config_id: str
        :return: True for the context-free configurations of plan
            section 10 (C9, C10, C11 by default), False otherwise.
        :rtype: bool
        :raises TypeError: If config_id is not a string.
        """
        if not isinstance(config_id, str):
            self._error("[Naming.is_context_free] Input error: the "
                        "parameter config_id must be a string, not "
                        f"{type(config_id).__name__}.", TypeError)
        return config_id in self.context_free

    def context_configs(self) -> list[str]:
        """
        Configurations that do carry temporal context, in order.

        Note: the original script took them positionally, as the first
        eight entries of the configuration list; here they are the
        configurations that are not context-free, which is the same list
        (C1..C8) for the vocabulary of the paper.

        :return: Configuration ids with temporal context.
        :rtype: list[str]
        """
        return [c for c in self.configurations
                if c not in self.context_free]

    def architecture_configs(self, architecture: str) -> tuple[str, ...]:
        """
        Configuration ids that use one architecture.

        :param architecture: Architecture name such as "LSTM".
        :type architecture: str
        :return: The configuration ids, as a tuple in presentation
            order; empty when the architecture is unknown.
        :rtype: tuple[str, ...]
        :raises TypeError: If architecture is not a string.
        """
        if not isinstance(architecture, str):
            self._error("[Naming.architecture_configs] Input error: the "
                        "parameter architecture must be a string, not "
                        f"{type(architecture).__name__}.", TypeError)
        return self.arch_configs.get(architecture, tuple())

    def config_architecture(self, config_id: str) -> str | None:
        """
        Architecture behind a configuration, the reverse lookup.

        :param config_id: Configuration id such as "C5".
        :type config_id: str
        :return: The architecture name ("LSTM"), or None when no
            architecture claims that configuration (C10 and C11 by
            default, which are not trained networks).
        :rtype: str | None
        :raises TypeError: If config_id is not a string.
        """
        if not isinstance(config_id, str):
            self._error("[Naming.config_architecture] Input error: the "
                        "parameter config_id must be a string, not "
                        f"{type(config_id).__name__}.", TypeError)
        for architecture in self.architectures:
            if config_id in self.arch_configs.get(architecture, tuple()):
                return architecture
        return None

    # ----- plan labels ---------------------------------------------- #
    def plan_tag(self, key: str) -> str:
        """
        Label of the analysis plan written into the generated artefacts.

        The labels are output strings of the paper pipeline and are kept
        verbatim, Spanish included.

        :param key: Key of the label, one of "primary", "secondary",
            "trivial", "h1", "omni", "r110", "abl", "cost", "fail",
            "o1", "expl_preplan" or "partial".
        :type key: str
        :return: The label, for example "[CONFIRMATORIO plan §3.1]".
        :rtype: str
        :raises TypeError: If key is not a string.
        :raises KeyError: If key is not one of the known labels.
        """
        if not isinstance(key, str):
            self._error("[Naming.plan_tag] Input error: the parameter "
                        "key must be a string, not "
                        f"{type(key).__name__}.", TypeError)
        if key not in self.plan_tags:
            self._error("[Naming.plan_tag] Input error: the parameter "
                        "key must be one of "
                        f"{sorted(self.plan_tags)}, not {key!r}.",
                        KeyError)
        return self.plan_tags[key]

    def __repr__(self) -> str:
        """
        Show the size of the vocabulary and the control configuration.

        :return: Text like "Naming(datasets=9, configurations=11,
            architectures=5, control='C10')".
        :rtype: str
        """
        return (f"Naming(datasets={len(self.datasets)}, "
                f"configurations={len(self.configurations)}, "
                f"architectures={len(self.architectures)}, "
                f"control={self.control!r})")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
