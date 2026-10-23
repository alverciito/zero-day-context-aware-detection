# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         src/kitnetzday/audit/training_report.py                   #
#   Version:      1.0.1                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Spanish note of the training audit: the seven sections    #
#                 of tables and caveats, verbatim.                          #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import collections
import logging
from collections.abc import Mapping

from kitnetzday.audit.training_audit import (
    VERDICT_AT_FLOOR,
    VERDICT_TRAINED,
    VERDICT_UNDEFINED,
    VERDICT_WORSE,
)
from kitnetzday.audit.training_caveats import caveats, comma
from kitnetzday.audit.training_summary import (
    CONTRADICTION_LABELS,
    TrainingSummary,
    median_finite,
)
from kitnetzday.utils.formatting import fnum
from kitnetzday.utils.loggable import Loggable

# Date the archived note carries. It is an output string, so a re-run
# that is meant to reproduce the archived file keeps it.
GENERATED_ON: str = "2026-09-29"
# Dataset name as the runner writes it -> short column heading of the
# variant x dataset table. These are the runner names, not the D1..D9
# ids of kitnetzday.naming, which the audited runs do not use.
AUDIT_DATASET_SHORT: dict[str, str] = {
    "ACI-IOT-2023": "ACI", "CIC2017": "CIC17", "CIC2018": "CIC18",
    "KNAD:Active_Wiretap": "ActWire", "KNAD:Fuzzing": "Fuzz",
    "KNAD:OS_Scan": "OSScan", "KNAD:SSDP_Flood": "SSDP",
    "KNAD:SSL_Renegotiation": "SSLReneg",
    "KNAD:Video_Injection": "VideoInj"}
# Variants whose exceptions are read as marginal in section 4.
_MARGINAL: tuple[str, ...] = ("conv1d", "conv2d", "mha")
# The whole prose of the note, verbatim. Everything in Spanish: it is
# the note the authors wrote with the audit.
_INTRO: str = (
    "Generado el {date} sin GPU y sin reentrenar nada. Compara la "
    "perdida de entrenamiento ya registrada (`train_log.csv`, "
    "`loss_epoch_1`) con el **suelo del predictor constante** del grupo "
    "de features de cada autoencoder. `ratio = loss / floor`; `< 0,90` "
    "= entrenado, `[0,90, 1,15)` = en el suelo, `>= 1,15` = peor que el "
    "suelo.")
_NONE_WITH_ZERO: str = (
    "Con el criterio literal (`ratio < 0,90` en al menos un "
    "autoencoder de al menos un dataset) **ninguna variante queda con "
    "0/9**. El detalle, sin embargo, es muy asimetrico:")
_MARGINAL_PROSE: str = (
    "Los {n} casos de `conv1d`, `conv2d` y `mha` por debajo del suelo "
    "son **marginales** (ratio entre {lo} y {hi}, es decir justo al "
    "borde del corte 0,90) y todos menos uno caen en el MISMO grupo de "
    "features (`KNAD:Video_Injection`, AE 0, 89 features). Los 18 casos "
    "con ventana por debajo del suelo son:")
_DEGENERATE_PROSE: str = (
    "Un grupo es *degenerado* cuando su suelo es < 1e-3: tras el "
    "min-max del tramo de entrenamiento sus columnas quedan casi "
    "constantes (outliers que estiran el rango), el predictor constante "
    "ya casi no comete error y el cociente se dispara sin que eso diga "
    "nada sobre el aprendizaje. Son {deg} de {tot} AEs ({pct:.0f} %), y "
    "concentran {wd} de los {wt} veredictos *peor que el suelo*.")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                MAIN CLASS                                 #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class TrainingReport(Loggable):
    """
    The Spanish note the training audit can leave next to its result.

    The JSON result holds every number; this note is what a reader goes
    through: the global split of the verdicts, three tables (variant by
    verdict, dataset by verdict, and the median ratio of each cell),
    the variants that never got below their floor, the contradictions
    of the expected pattern, and ten caveats.

    Note: the note is in Spanish. It is the note the authors wrote
    with the audit and it is written only when the calling script asks
    for it; the result file and the console summary, which are in
    English, hold every number of it. Several of its sentences are
    fixed prose about the runs audited on 2026-09-29 ("todos menos uno
    caen en el MISMO grupo", "Los 18 casos con ventana"), so they
    describe those runs and not whatever tree is passed; the tables
    around them are recomputed.
    Note too that sections 5 and 7 name the variants "lstm" and "mlp"
    and the dataset "ACI-IOT-2023" and raise a KeyError when they were
    not run, as the audited script does.

    :param summary: Aggregates of the audit, with ``run`` already
        called on them.
    :type summary: TrainingSummary
    :param dataset_short: Dataset name of the runner -> short column
        heading; names that are missing print in full.
    :type dataset_short: Mapping[str, str]
    :param generated_on: Date written in the opening paragraph, as
        ISO-8601; the archived note carries 2026-09-29.
    :type generated_on: str
    :param logger: Logger created by the calling script, or None to stay
        silent.
    :type logger: logging.Logger | None
    :raises TypeError: If summary is not a TrainingSummary, or if logger
        is not a logging.Logger or None.
    :raises ValueError: If ``run`` was not called on the summary.
    """

    def __init__(self,
                 summary: TrainingSummary,
                 dataset_short: Mapping[str, str] = AUDIT_DATASET_SHORT,
                 generated_on: str = GENERATED_ON,
                 logger: logging.Logger | None = None) -> None:
        """
        Keep the aggregates and the two presentation settings.

        :param summary: Aggregates of the audit, already built.
        :type summary: TrainingSummary
        :param dataset_short: Dataset name -> short column heading.
        :type dataset_short: Mapping[str, str]
        :param generated_on: Date of the opening paragraph.
        :type generated_on: str
        :param logger: Logger created by the calling script, or None.
        :type logger: logging.Logger | None
        :raises TypeError: If summary is not a TrainingSummary, or if
            logger is not a logging.Logger or None.
        :raises ValueError: If ``run`` was not called on the summary.
        """
        super().__init__(logger=logger)
        if not isinstance(summary, TrainingSummary):
            self._error("[TrainingReport.__init__] Input error: the "
                        "parameter summary must be a TrainingSummary, "
                        f"not {type(summary).__name__}.", TypeError)
        if not summary.totals:
            self._error("[TrainingReport.__init__] Input error: the "
                        "parameter summary must have been built with "
                        "its run method, and its totals are empty.")
        self.summary = summary
        self.dataset_shorts = dict(dataset_short)
        self.generated_on = str(generated_on)

    def text(self) -> str:
        """
        The whole note as one text, ready to be written to disk.

        :return: The seven sections, newline separated.
        :rtype: str
        """
        return "\n".join(self.lines())

    def lines(self) -> list[str]:
        """
        The note line by line, in the order of the archived file.

        :return: Every line, the blank ones and the trailing one.
        :rtype: list[str]
        """
        return (self._overview() + self._variant_table()
                + self._dataset_table() + self._matrix_table()
                + self._never_section() + self._contradictions()
                + self._nondegenerate_table() + self._caveats())

    def _overview(self) -> list[str]:
        """
        Title, what the audit does, and the four global counts.

        :return: The lines of the opening block.
        :rtype: list[str]
        """
        total = self.summary.totals
        counts = self.summary.overall
        return [
            "# Auditoria retrospectiva del entrenamiento de las "
            "ejecuciones realizadas",
            "",
            _INTRO.format(date=self.generated_on),
            "",
            "- Directorios con `DONE` y `train_log.csv`: **"
            f"{total['job_dirs_with_DONE_and_train_log']}**.",
            f"- Jobs auditados: **{total['jobs_audited']}** "
            f"({total['autoencoders_audited']} autoencoders).",
            f"- Jobs saltados: **{total['jobs_skipped']}** (`original`, "
            "`stat` e `iforest`: `loss_epoch_1 = NA`).",
            f"- Reparto global: **{counts[VERDICT_TRAINED]} "
            f"entrenados**, **{counts[VERDICT_AT_FLOOR]} en el suelo**, "
            f"**{counts[VERDICT_WORSE]} peor que el suelo**, "
            f"{counts[VERDICT_UNDEFINED]} no calculables.",
            ""]

    def _variant_table(self) -> list[str]:
        """
        Section 1: one row per variant over every autoencoder.

        :return: The lines of the section, the TOTAL row included.
        :rtype: list[str]
        """
        out = ["## 1. Variante x veredicto (todos los autoencoders)",
               "",
               "| Variante | Jobs | AEs | Entrenado | En el suelo | "
               "Peor que el suelo | No calculable | Ratio mediano | "
               "Datasets con >= 1 AE entrenado |",
               "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        flat = self.summary.audit.flat
        nds = self.summary.n_datasets
        for name in self.summary.variants:
            row = self.summary.by_variant[name]
            counts = row["counts"]
            out.append(
                f"| `{name}` | {row['n_jobs']} | "
                f"{row['n_autoencoders']} | {counts[VERDICT_TRAINED]} | "
                f"{counts[VERDICT_AT_FLOOR]} | {counts[VERDICT_WORSE]} "
                f"| {counts[VERDICT_UNDEFINED]} | "
                f"{fnum(row['median_ratio'], 4, '-')} | "
                f"{len(row['datasets_with_at_least_one_trained_ae'])}"
                f"/{nds} |")
        counts = self.summary.overall
        out += [f"| **TOTAL** | {self.summary.totals['jobs_audited']} | "
                f"{len(flat)} | {counts[VERDICT_TRAINED]} | "
                f"{counts[VERDICT_AT_FLOOR]} | {counts[VERDICT_WORSE]} "
                f"| {counts[VERDICT_UNDEFINED]} | "
                f"{fnum(median_finite(r['ratio'] for r in flat), 4, '-')}"
                " | |", ""]
        return out

    def _dataset_table(self) -> list[str]:
        """
        Section 2: one row per dataset over every autoencoder.

        :return: The lines of the section.
        :rtype: list[str]
        """
        out = ["## 2. Dataset x veredicto",
               "",
               "| Dataset | AEs | Entrenado | En el suelo | Peor que el "
               "suelo | No calculable | Ratio mediano |",
               "|---|---:|---:|---:|---:|---:|---:|"]
        for name in self.summary.datasets:
            row = self.summary.by_dataset[name]
            counts = row["counts"]
            out.append(
                f"| {name} | {row['n_autoencoders']} | "
                f"{counts[VERDICT_TRAINED]} | "
                f"{counts[VERDICT_AT_FLOOR]} | {counts[VERDICT_WORSE]} "
                f"| {counts[VERDICT_UNDEFINED]} | "
                f"{fnum(row['median_ratio'], 4, '-')} |")
        out.append("")
        return out

    def _matrix_table(self) -> list[str]:
        """
        Section 3: median ratio of every variant by dataset cell.

        :return: The lines of the section, with the margins.
        :rtype: list[str]
        """
        datasets = self.summary.datasets
        flat = self.summary.audit.flat
        heads = " | ".join(self.dataset_shorts.get(d, d)
                           for d in datasets)
        out = ["## 3. Ratio mediano por variante y dataset",
               "",
               f"| Variante | {heads} | Todos |",
               "|---|" + "---:|" * (len(datasets) + 1)]
        for name in self.summary.variants:
            cells = [("-" if not self.summary.matrix[name].get(d)
                      else fnum(self.summary.matrix[name][d][
                          "median_ratio"], 4, "-"))
                     for d in datasets]
            out.append(f"| `{name}` | " + " | ".join(cells) + " | "
                       + fnum(self.summary.by_variant[name][
                           "median_ratio"], 4, "-") + " |")
        out += ["| **Todos** | " + " | ".join(
            fnum(self.summary.by_dataset[d]["median_ratio"], 4, "-")
            for d in datasets) + " | " + fnum(
                median_finite(r["ratio"] for r in flat), 4, "-") + " |",
                ""]
        return out

    def _never_section(self) -> list[str]:
        """
        Section 4: the variants with no trained autoencoder at all.

        When every variant has at least one, the section turns into
        the asymmetry per variant and the table of the exceptions.

        :return: The lines of the section.
        :rtype: list[str]
        """
        out = ["## 4. Variantes que no entrenaron en ningun dataset",
               ""]
        nds = self.summary.n_datasets
        if self.summary.never:
            out += ["Variantes con 0/9: " + ", ".join(
                f"`{v}`" for v in self.summary.never) + ".", ""]
            return out
        out += [_NONE_WITH_ZERO, ""]
        for name in self.summary.variants:
            row = self.summary.by_variant[name]
            out.append(
                f"- `{name}`: "
                f"{len(row['datasets_with_at_least_one_trained_ae'])}"
                f"/{nds} datasets con al menos un AE por debajo del "
                f"suelo ({row['counts'][VERDICT_TRAINED]} de "
                f"{row['n_autoencoders']} autoencoders).")
        marginal = [r for r in self.summary.trained
                    if r["variant"] in _MARGINAL]
        out += ["",
                _MARGINAL_PROSE.format(
                    n=len(marginal),
                    lo=comma(min(r["ratio"] for r in marginal)),
                    hi=comma(max(r["ratio"] for r in marginal))),
                "",
                "| Variante | Dataset | Job | AE | features | Suelo "
                "(MSE) | Loss (MSE) | Ratio |",
                "|---|---|---|---:|---:|---:|---:|---:|"]
        for r in self.summary.trained:
            out.append(f"| `{r['variant']}` | {r['dataset']} | "
                       f"`{r['job_id']}` | {r['ae_index']} | "
                       f"{r['n_features_group']} | "
                       f"{r['floor_mse']:.3e} | {r['loss_mse']:.3e} | "
                       f"{fnum(r['ratio'], 4, '-')} |")
        out.append("")
        return out

    def _contradictions(self) -> list[str]:
        """
        Section 5: the cases that go against the expected pattern.

        :return: The lines of the section, the table of kinds and the
            five readings that follow it.
        :rtype: list[str]
        """
        out = ["## 5. Casos que contradicen el patron esperado", ""]
        kinds = collections.Counter(c["kind"] for c
                                    in self.summary.contradictions)
        lstm = self.summary.by_variant["lstm"]
        mlp = self.summary.by_variant["mlp"]
        lstm_nd = self.summary.by_variant_nondegenerate["lstm"]
        mlp_nd = self.summary.by_variant_nondegenerate["mlp"]
        not_trained = (mlp_nd["n_autoencoders"]
                       - mlp_nd["counts"][VERDICT_TRAINED])
        out += ["| Tipo | Casos (AEs) |", "|---|---:|"]
        for kind, n in sorted(kinds.items(), key=lambda kv: -kv[1]):
            out.append(f"| {kind} | {n} |")
        out += ["",
                "- **`lstm` que no queda por debajo del suelo: "
                f"{kinds.get(CONTRADICTION_LABELS['lstm'], 0)} de "
                f"{lstm['n_autoencoders']} AEs** "
                f"({lstm['counts'][VERDICT_AT_FLOOR]} en el suelo, "
                f"{lstm['counts'][VERDICT_WORSE]} peor). Restringido a "
                "grupos de features no degenerados (suelo >= 1e-3) el "
                "cuadro se invierte: "
                f"{lstm_nd['counts'][VERDICT_TRAINED]}/"
                f"{lstm_nd['n_autoencoders']} entrenados y ratio "
                f"mediano {comma(lstm_nd['median_ratio'])}.",
                "- **`mlp` plano que no queda por debajo del suelo: "
                f"{kinds.get(CONTRADICTION_LABELS['mlp'], 0)} de "
                f"{mlp['n_autoencoders']} AEs.** En grupos no "
                f"degenerados son solo {not_trained} de "
                f"{mlp_nd['n_autoencoders']} (ratio mediano "
                f"{comma(mlp_nd['median_ratio'])})."]
        out += self._contradiction_readings(kinds)
        out.append("")
        return out

    def _contradiction_readings(self,
                                kinds: Mapping[str, int]) -> list[str]:
        """
        The last three readings of section 5.

        Note: the first two readings quote the smallest and the largest
        ratio of their group, so they raise a ValueError when no
        windowed autoencoder got below its floor, exactly as the
        audited script.

        :param kinds: How many autoencoders fall under each kind of
            contradiction.
        :type kinds: Mapping[str, int]
        :return: Three lines: the windowed variants below the floor,
            the MLP with context on CIC, and the undefined ratios.
        :rtype: list[str]
        :raises ValueError: If no windowed autoencoder is below its
            floor.
        """
        marginal = [r for r in self.summary.trained
                    if r["variant"] in _MARGINAL]
        ctx = [r for r in self.summary.trained
               if r["variant"] == "mlp_ctx"]
        cic = [r for r in self.summary.audit.flat
               if r["variant"] == "mlp_ctx"
               and r["dataset"] in ("CIC2017", "CIC2018")]
        undefined = [r for r in self.summary.audit.flat
                     if r["verdict"] == VERDICT_UNDEFINED]
        return [
            "- **Variantes con ventana por debajo del suelo: "
            f"{kinds.get('variante con ventana por debajo del suelo', 0)}"
            f" AEs** (tabla de la seccion 4). {len(marginal)} son "
            "`conv1d`/`conv2d`/`mha` con ratio entre "
            f"{comma(min(r['ratio'] for r in marginal))} y "
            f"{comma(max(r['ratio'] for r in marginal))}, todos menos "
            "uno sobre el mismo grupo (`KNAD:Video_Injection`, AE 0, 89 "
            "features): estan en el limite del corte, no son un "
            "entrenamiento claro.",
            f"- **`mlp_ctx` sobre CIC**: {len(ctx)} AEs con ratio "
            f"{comma(min(r['ratio'] for r in ctx))}-"
            f"{comma(max(r['ratio'] for r in ctx))}, la unica excepcion"
            " NO marginal entre las variantes con ventana. Los cinco "
            "tienen `ar = False` (TSR): son 5 de los "
            f"{len([r for r in cic if not r['ar']])} AEs TSR de "
            "`mlp_ctx` en `cic2017` y `cic2018`, mientras que los "
            f"{len([r for r in cic if r['ar']])} AEs `ar = True` (AR) de"
            " esos mismos dos datasets quedan todos en el suelo o peor. "
            "El tamano del grupo no separa los casos: los grupos que "
            "quedan por debajo del suelo tienen 8, 28, 31 y 34 features "
            "(fan_in aplanado 6.400-27.200) y en KNAD hay grupos de 8-15"
            " features (6.400-12.000) que no bajan del suelo.",
            f"- **{len(undefined)} AEs no calculables** (suelo "
            "exactamente 0, todas las columnas del grupo constantes en "
            "el tramo de entrenamiento): "
            + "; ".join(
                f"`{r['job_id']}` AE {r['ae_index']}, loss "
                + (f"{r['loss_mse']:.6g}" if r["loss_mse"] is not None
                   else "NA")
                for r in undefined) + "."]

    def _nondegenerate_table(self) -> list[str]:
        """
        Section 6: the variant table over the groups with a usable floor.

        :return: The lines of the section.
        :rtype: list[str]
        """
        deg = self.summary.degeneracy
        counts = self.summary.overall
        out = ["## 6. Tabla secundaria: solo grupos de features no "
               "degenerados",
               "",
               _DEGENERATE_PROSE.format(
                   deg=deg["n_degenerate"], tot=deg["n_total"],
                   pct=100.0 * deg["n_degenerate"] / deg["n_total"],
                   wd=deg["verdicts_in_degenerate_groups"][
                       VERDICT_WORSE],
                   wt=counts[VERDICT_WORSE]),
               "",
               "| Variante | AEs no degenerados | Entrenado | En el "
               "suelo | Peor que el suelo | Ratio mediano | Datasets "
               "con >= 1 AE entrenado |",
               "|---|---:|---:|---:|---:|---:|---:|"]
        for name in self.summary.variants:
            row = self.summary.by_variant_nondegenerate[name]
            kinds = row["counts"]
            out.append(
                f"| `{name}` | {row['n_autoencoders']} | "
                f"{kinds[VERDICT_TRAINED]} | {kinds[VERDICT_AT_FLOOR]} "
                f"| {kinds[VERDICT_WORSE]} | "
                f"{fnum(row['median_ratio'], 4, '-')} | "
                f"{len(row['datasets_with_at_least_one_trained_ae'])}"
                f"/{self.summary.n_datasets} |")
        out.append("")
        return out

    def _caveats(self) -> list[str]:
        """
        Section 7: what the audit does not prove, ten numbered caveats.

        The texts live in :mod:`kitnetzday.audit.training_caveats`, which
        fills in the two of them that quote numbers of the run.

        :return: The lines of the section and the closing pointer.
        :rtype: list[str]
        """
        return caveats(self.summary, logger=self.logger)

    def __repr__(self) -> str:
        """
        Show the date of the note and how many lines it holds.

        :return: Text like "TrainingReport(date='2026-09-29', l=96)".
        :rtype: str
        """
        return (f"TrainingReport(generated_on={self.generated_on!r}, "
                f"lines={len(self.lines())})")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
