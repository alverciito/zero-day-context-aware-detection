# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         src/kitnetzday/figures/failure_rule.py                    #
#   Version:      2.0.0                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Pre-specified selection of the failure cases of plan      #
#                 section 10, and its traceability rows.                    #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
from collections.abc import Mapping
from typing import Any

import numpy as np

from kitnetzday.io.inputs import Inputs
from kitnetzday.io.outputs import Out
from kitnetzday.naming import Naming
from kitnetzday.utils.formatting import get
from kitnetzday.utils.loggable import Loggable

# Pointers written into the rows of numbers_used.md. They name the file
# that holds the value.
STATS_SRC: str = "stats.json#/failure_cases"
FALLBACK_SRC: str = ("make_figures.py (rule of plan §10 applied to "
                     "aggregate.json#/value_sets/main)")
# Selection rules, as recorded in the notes of the figure. The margin is
# spelled out inside the text, in Spanish decimal notation.
RULE_MARGIN: str = "simple - context > 0,01 (§10)"
RULE_FALLBACK: str = ("ningun dataset supera el margen: el de menor (con "
                      "contexto - simple) (§10)")
FALLBACK_MARK: str = "ningun dataset supera"
FALLBACK_NOTE: str = ("Sin stats.json: la seleccion se aplica aqui con la "
                      "regla del plan §10 (sin recalcular ninguna "
                      "metrica: solo medias de familia sobre ROC-AUC ya "
                      "calculados).")
# Suffix of the row id -> key of the per-stream record, in writing order.
FAMILY_KEYS: tuple[tuple[str, str], ...] = (
    ("SIMPLE", "simple_mean"), ("CTX", "context_mean"),
    ("DIFF", "simple_minus_context"))


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                MAIN CLASS                                 #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class FailureRule(Loggable):
    """
    The failure-case rule of plan section 10, applied to the results.

    A stream is a failure case when the mean ROC-AUC of its
    context-free configurations exceeds the mean of its windowed ones
    by more than the failure margin. The rule was fixed before any
    result was seen, and ``stats.py`` applies it and stores the outcome
    in ``stats.json#/failure_cases``. This class reads that outcome
    when it is there and applies the same rule here when it is not, so
    that the figure can still be drawn from the aggregate alone. No
    metric is recomputed in either case: only means of ROC-AUC values
    that are already stored.

    Note: the texts of the two rules are the ones ``stats.py`` stores
    in ``stats.json`` and are kept verbatim, Spanish included, because
    ``is_fallback`` recognises the fallback rule by its text.
    RULE_MARGIN writes the margin as "0,01" whatever the value of the
    margin of the vocabulary is.

    :param naming: Vocabulary of the study, which carries the dataset
        order, which configurations are context-free and the failure
        margin of plan section 10. None builds the default vocabulary.
    :type naming: Naming | None
    :param logger: Logger created by the calling script, or None to
        stay silent.
    :type logger: logging.Logger | None
    :raises TypeError: If naming is not a Naming or None, or logger is
        not a logging.Logger or None.
    """

    def __init__(self,
                 naming: Naming | None = None,
                 logger: logging.Logger | None = None) -> None:
        """
        Store the vocabulary the rule is read against.

        :param naming: Vocabulary of the study, or None for the default.
        :type naming: Naming | None
        :param logger: Logger created by the calling script, or None.
        :type logger: logging.Logger | None
        :raises TypeError: If naming or logger has the wrong type.
        """
        super().__init__(logger=logger)
        if naming is not None and not isinstance(naming, Naming):
            self._error("[FailureRule.__init__] Input error: the "
                        "parameter naming must be a Naming or None, not "
                        f"{type(naming).__name__}.", TypeError)
        self.naming = naming if naming is not None else Naming(
            logger=logger)

    def select(self,
               inputs: Inputs) -> tuple[dict[str, Any], list[str], Any,
                                        str, list[str], list[str]]:
        """
        Read the stored selection, or apply the rule here.

        With ``stats.json`` present the three records (per-stream
        values, selected streams and the text of the rule) are taken as
        they were stored. Without it the family means are taken from
        the ROC-AUC values of the aggregate: for every stream the mean
        of the context-free configurations and the mean of the windowed
        ones, the stream being flagged when the difference exceeds the
        margin and every planned cell of both families has a value.
        When no stream is flagged, the one with the smallest difference
        is selected instead, as the plan pre-specifies.

        Note: a stream enters the record only when both families hold
        at least one value, the margin comparison is strict (>), and
        the cells without a value are listed context-free family first.

        :param inputs: Reader of the aggregate and of ``stats.json``.
        :type inputs: Inputs
        :return: The per-stream record, the selected stream ids, the
            selection rule as stored or applied, the source pointer,
            the missing-input notes and the free-text notes.
        :rtype: tuple[dict[str, Any], list[str], Any, str, list[str],
            list[str]]
        :raises TypeError: If inputs is not an Inputs.
        """
        missing = list()
        notes = list()
        margin = self.naming.failure_margin
        context = self.naming.context_configs()
        free = list(self.naming.context_free)
        if not isinstance(inputs, Inputs):
            self._error("[FailureRule.select] Input error: the "
                        "parameter inputs must be an Inputs, not "
                        f"{type(inputs).__name__}.", TypeError)
        per = get(inputs.stats or dict(), "failure_cases",
                  "per_dataset", default=None)
        selected = get(inputs.stats or dict(), "failure_cases",
                       "selected", default=None)
        rule = get(inputs.stats or dict(), "failure_cases",
                   "selection_rule", default=None)
        src = STATS_SRC
        if per is None:
            src = FALLBACK_SRC
            notes.append(FALLBACK_NOTE)
            per = dict()
            for x in self.naming.datasets:
                s = [inputs.auroc(x, c) for c in free
                     if inputs.auroc(x, c) is not None]
                k = [inputs.auroc(x, c) for c in context
                     if inputs.auroc(x, c) is not None]
                if s and k:
                    ms, mk = float(np.mean(s)), float(np.mean(k))
                    # A family that lacks the value of a planned cell
                    # (other than the cells the plan declares lost,
                    # section 4.5) cannot raise the flag: the mean
                    # would be of 1 of 3 against 2 of 8.
                    without = [f"{x}/{c}" for c in free + context
                               if inputs.auroc(x, c) is None
                               and not inputs.cell(x, c).get("lost")]
                    per[x] = {"simple_mean": ms, "context_mean": mk,
                              "simple_minus_context": ms - mk,
                              "n_simple": len(s), "n_context": len(k),
                              "cells_without_value": without,
                              "incomplete": bool(without),
                              "is_failure_case": (ms - mk) > margin
                              and not without}
                    if without:
                        missing.append(
                            f"{x}: {len(without)} planned cells of its "
                            "families have no value: the stream is not "
                            "flagged as a failure case even when the "
                            "difference exceeds the margin")
            selected = [x for x, r in per.items()
                        if r["is_failure_case"]]
            rule = RULE_MARGIN
            if not selected and per:
                selected = [min(per,
                                key=lambda z: per[z]["context_mean"]
                                - per[z]["simple_mean"])]
                rule = RULE_FALLBACK
        return per, selected or list(), rule, src, missing, notes

    def is_fallback(self, rule: Any = None) -> bool:
        """
        Whether the rule that was applied is the fallback one.

        The caption of the figure depends on it: with the fallback
        branch the marked streams do NOT meet the margin criterion and
        saying otherwise would be false.

        :param rule: Text of the rule, as stored or as applied.
        :type rule: Any
        :return: True when the text reports that no stream met the
            margin, False for any other text and for an empty one.
        :rtype: bool
        """
        return bool(rule) and FALLBACK_MARK in str(rule)

    def add_numbers(self,
                    inputs: Inputs,
                    out: Out,
                    per: Mapping[str, Any],
                    src: str,
                    ids: list[str],
                    missing: list[str]) -> None:
        """
        Register the three family values of every stream as rows.

        A stream with no record leaves a missing-input note instead of
        three rows, because one of its families holds no value.

        :param inputs: Reader, used only for the partial-inputs flag.
        :type inputs: Inputs
        :param out: Writer that collects the rows.
        :type out: Out
        :param per: Per-stream record returned by select.
        :type per: Mapping[str, Any]
        :param src: Source pointer returned by select.
        :type src: str
        :param ids: List of row ids, extended in place.
        :type ids: list[str]
        :param missing: List of missing-input notes, extended in place.
        :type missing: list[str]
        :raises TypeError: If inputs is not an Inputs, out is not an
            Out, or ids or missing is not a list.
        """
        if not isinstance(inputs, Inputs) or not isinstance(out, Out):
            self._error("[FailureRule.add_numbers] Input error: the "
                        "parameters inputs and out must be an Inputs "
                        f"and an Out, not {type(inputs).__name__} and "
                        f"{type(out).__name__}.", TypeError)
        if not isinstance(ids, list) or not isinstance(missing, list):
            self._error("[FailureRule.add_numbers] Input error: the "
                        "parameters ids and missing must be lists, not "
                        f"{type(ids).__name__} and "
                        f"{type(missing).__name__}.", TypeError)
        for x in self.naming.datasets:
            if x not in per:
                missing.append(f"{x}: no family means (a family "
                               "without values)")
                continue
            for key, k2 in FAMILY_KEYS:
                ids.append(out.add_number(
                    f"R.FAIL.{x}.{key}", per[x].get(k2),
                    f"{src}/per_dataset/{x}/{k2}",
                    self.naming.plan_tag("fail"), inputs.partial_inputs))

    def __repr__(self) -> str:
        """
        Show the size of the vocabulary and the failure margin.

        :return: Text like "FailureRule(datasets=9,
            failure_margin=0.01, context_free=3)".
        :rtype: str
        """
        return (f"FailureRule(datasets={len(self.naming.datasets)}, "
                f"failure_margin={self.naming.failure_margin}, "
                f"context_free={len(self.naming.context_free)})")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
