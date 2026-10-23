# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         src/kitnetzday/audit/training_caveats.py                  #
#   Version:      1.0.3                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Section 7 of the training-audit note: the ten caveats     #
#                 that say what the audit does not prove.                   #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging

from kitnetzday.audit.training_summary import TrainingSummary

# The ten caveats, verbatim and in Spanish. The third and the fourth
# carry numbers of the run, so they are written as templates. The third
# is kept apart because it needs the dataset with the shortest training
# span.
CAVEATS: tuple[str, ...] = (
    "1. **Convencion de error.** Se compara MSE con MSE. "
    "`loss_epoch_1` es la perdida compilada de Keras y esa perdida es "
    "`MeanSquaredError` en las seis variantes con gradiente "
    "(multilayer.py:54, conv1d.py:67, conv2d.py:68, lstm.py:65, "
    "mha.py:69, networks_ctx.py:113). La distincion RMSE/MSE de "
    "`ERROR_FN` (networks_ctx.py) afecta al error POR FILA que "
    "va a `ae_errors.npy` y al score fusionado, no a la perdida "
    "auditada aqui. El suelo se calcula tambien como MSE.",
    "2. **`loss_epoch_1` es la MEDIA de la epoca, no la perdida "
    "final.** Con una sola epoca, un modelo que converge deprisa "
    "arrastra en esa media todo el tramo inicial sin entrenar, asi que "
    "los cocientes estan sesgados HACIA ARRIBA. Un veredicto "
    "*entrenado* es por tanto conservador; un *en el suelo* no "
    "distingue entre un modelo que nunca se movio y uno que aprendio "
    "muy al final de la epoca.",
    "4. **Grupos degenerados.** Ver seccion 6: el {pct:.0f} % de los "
    "AEs ({deg} de {tot}) tiene un suelo < 1e-3 y ahi el cociente no es "
    "informativo. La tabla principal los incluye; la secundaria los "
    "excluye.",
    "5. **No auditable: `original` y `stat` (89 jobs).** "
    "`train_log.csv` registra `loss_epoch_1 = NA` para las dos. `stat` "
    "(`StatisticalAnomaly`, stdev.py) es efectivamente deterministico, "
    "pero `original` (`DenoisingAutoencoder`, original_kitsune.py:"
    "78-115) **si hace descenso de gradiente** (SGD en numpy, fila a "
    "fila): simplemente no deja una perdida comparable y, ademas, "
    "normaliza con min/max ONLINE y +1e-6 en lugar de +1e-16, asi que "
    "su suelo no seria el mismo. Queda fuera de esta auditoria. "
    "`iforest` (1 job, T5-IF) tampoco tiene perdida.",
    "6. **Autoencoder de salida.** `train_log.csv` solo registra los "
    "AEs del mapa de features; el autoencoder de salida "
    "(`kitnet.py:182`, capa `stat` u `original`) no aparece y no se "
    "audita.",
    "7. **T4-E.** Los 4 jobs con gradiente de T4-E se auditaron con el "
    "tramo CONTAMINADO reconstruido en memoria (`x[start:start+B] = "
    "x[donor]`, run_job.py:578-584), verificando el bloque contra "
    "`regions.t4e_training_contamination.x_block_sha256_after`. Todos "
    "coinciden.",
    "8. **Objetivos AR y TSR.** El suelo es identico para las dos "
    "variantes porque el conjunto de objetivos es el mismo: "
    "`create_windowed_data` mapea `y = window[-1] = x[j]` y "
    "`create_windowed_data_ar` mapea `y = full_x[j+L] = x[j]`; lo que "
    "cambia es la ENTRADA, no el objetivo (sequential.py:14-70). El "
    "relleno circular de la primera ventana (ceros) afecta a la ENTRADA "
    "de las primeras L filas, no al objetivo.",
    "9. **Dropout.** `corruption_level = 0,0` en todos los jobs "
    "revisados, asi que la perdida de entrenamiento no lleva ruido de "
    "dropout.",
    "10. **No se re-ejecuto nada.** Ni entrenamiento, ni inferencia, ni "
    "GPU. Todo sale de `train_log.csv`, `fm_stats.json`, `config.json` "
    "y las caches de stream de solo lectura.")
CAVEAT_ROWS: str = (
    "3. **Rango de filas.** Se uso `regions.ae_train` de cada "
    "`config.json`, no un rango fijo: es [50.000, 200.000) en los 8 "
    "streams KNAD/CIC, pero [50.000, 72.264) en ACI-IOT-2023, cuyo "
    "corte de entrenamiento corregido es el dia 31-oct completo (FM "
    "50.000 + AE 22.264). Usar [50.000, 200.000) en ACI habria metido "
    "filas de evaluacion en el suelo. ACI es tambien el dataset con "
    "menos pasos de entrenamiento (22.264 frente a 150.000), lo que "
    "agrava el sesgo de la salvedad 2 y explica en parte su ratio "
    "mediano de {aci}.")
# Dataset of caveat 3: the one whose training span is the shortest.
ACI_DATASET: str = "ACI-IOT-2023"
# Line that closes the note.
POINTER: str = ("Tabla completa por job y por autoencoder, y bloque "
                "`method` con la formula exacta, en `training_audit.json`.")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                AUXILIARY                                  #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def comma(value: float | None, nd: int = 4) -> str:
    """
    Write a number for the prose, with the comma as decimal separator.

    The note is written in Spanish, so every number in a sentence
    carries a comma; the numbers inside the tables keep the dot of
    ``fnum``, as in the archived file.

    :param value: Number to write, or None for a missing one.
    :type value: float | None
    :param nd: Number of decimals; 4 in the note.
    :type nd: int
    :return: The number with a comma ("0,8931"), or "-".
    :rtype: str
    """
    if value is None:
        return "-"
    return (f"%.{nd}f" % value).replace(".", ",")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                              MAIN FUNCTION                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def caveats(summary: TrainingSummary,
            decimals: int = 4,
            logger: logging.Logger | None = None) -> list[str]:
    """
    Write the last section of the note: what the audit does not prove.

    The ten caveats are what keeps the audit honest. They state that the
    comparison is MSE against MSE; that the recorded loss is the mean of
    the single epoch and therefore biased upwards, which makes a verdict
    of "entrenado" conservative; which rows the floor is computed over
    and why the shortest dataset is the worst case; how many feature
    groups have a floor too small to divide by; which variants cannot be
    audited at all, the KitNET-style control among them; that the output
    autoencoder of the ensemble is not in the log; how the contaminated
    block was rebuilt; why the floor is the same for the two paradigms;
    that there was no dropout noise in the loss; and that nothing was
    re-executed.

    Two of them carry numbers of the run: the share of degenerate
    groups, and the median ratio of the dataset with the shortest
    training span.

    Note: the texts name files and line numbers of the code of the runs
    as the authors had it when the note was written, and counts of the
    audited runs ("89 jobs", "4 jobs T4-E", "1 job"), which are not
    recomputed. They describe the runs the note was written for.

    :param summary: Aggregates of the audit, with ``run`` already
        called on them; its degeneracy block and its dataset view are
        read.
    :type summary: TrainingSummary
    :param decimals: Decimals of the median ratio quoted in caveat 3;
        4 in the archived note, not negative.
    :type decimals: int
    :param logger: Logger created by the calling script, or None to stay
        silent.
    :type logger: logging.Logger | None
    :return: The lines of section 7, closed by the pointer to the result
        file and a final empty line.
    :rtype: list[str]
    :raises TypeError: If summary is not a TrainingSummary.
    :raises ValueError: If decimals is negative, if the summary has not
        been built, or if no autoencoder was audited.
    :raises KeyError: If the dataset of caveat 3 was not run.
    """
    out: list[str] = ["## 7. Salvedades", ""]
    median = 0.0
    if not isinstance(summary, TrainingSummary):
        message = ("[caveats] Input error: the parameter summary must "
                   f"be a TrainingSummary, not {type(summary).__name__}.")
        if logger is not None:
            logger.error(message)
        raise TypeError(message)
    if decimals < 0 or not summary.totals or not summary.degeneracy[
            "n_total"]:
        message = ("[caveats] Input error: the parameter summary must "
                   "be a built summary with at least one autoencoder "
                   f"and decimals must not be negative, not {decimals} "
                   f"and {summary!r}.")
        if logger is not None:
            logger.error(message)
        raise ValueError(message)
    degenerate = summary.degeneracy["n_degenerate"]
    total = summary.degeneracy["n_total"]
    median = summary.by_dataset[ACI_DATASET]["median_ratio"]
    out += [CAVEATS[0], CAVEATS[1],
            CAVEAT_ROWS.format(aci=comma(median, decimals)),
            CAVEATS[2].format(pct=100.0 * degenerate / total,
                              deg=degenerate, tot=total)]
    out += list(CAVEATS[3:])
    out += ["", POINTER, ""]
    return out


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
