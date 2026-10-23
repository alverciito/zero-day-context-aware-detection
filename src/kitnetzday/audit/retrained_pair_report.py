# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         src/kitnetzday/audit/retrained_pair_report.py             #
#   Version:      1.1.1                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Printed text of the paired report of the retraining       #
#                 experiment, line by line.                                 #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from kitnetzday.audit.retrained_pair import PARADIGMS
from kitnetzday.utils.loggable import Loggable


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                MAIN CLASS                                 #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class RetrainedPairReport(Loggable):
    """
    Printed text of the report of ``RetrainedPair``, line by line.

    The lines follow, number by number, the standard output of the
    script the revision used, whose text was in Spanish: the wording
    is translated here and every figure is printed in the same order
    and with the same format. Keeping the text apart from the
    computation keeps ``print`` out of the library: the script in
    ``bin/`` is the only place that writes to the console.

    :param paradigms: Paradigm halves to print, each one as (label,
        value of the "ar" flag), in printing order; non-empty.
    :type paradigms: Sequence[tuple[str, bool]]
    :param logger: Logger created by the calling script, or None to stay
        silent.
    :type logger: logging.Logger | None
    :raises TypeError: If logger is not a logging.Logger or None, or if
        paradigms is not a sequence.
    :raises ValueError: If paradigms is empty.
    """

    def __init__(self,
                 paradigms: Sequence[tuple[str, bool]] = PARADIGMS,
                 logger: logging.Logger | None = None) -> None:
        """
        Check the paradigm halves and store them.

        :param paradigms: Halves as (label, "ar" flag); non-empty.
        :type paradigms: Sequence[tuple[str, bool]]
        :param logger: Logger created by the calling script, or None.
        :type logger: logging.Logger | None
        :raises TypeError: If paradigms is not a sequence.
        :raises ValueError: If paradigms is empty.
        """
        super().__init__(logger=logger)
        if isinstance(paradigms, (str, bytes)) or \
                not isinstance(paradigms, Sequence):
            self._error("[RetrainedPairReport.__init__] Input error: "
                        "the parameter paradigms must be a sequence of "
                        "(label, ar) pairs, not "
                        f"{type(paradigms).__name__}.", TypeError)
        if not paradigms:
            self._error("[RetrainedPairReport.__init__] Input error: "
                        "the parameter paradigms must hold at least one "
                        "half, and it is empty.")
        self.paradigms = tuple(paradigms)

    def render(self,
               report: Mapping[str, Any],
               json_path: str | Path | None = None) -> list[str]:
        """
        Text of the paired report, one element per printed line.

        A pair without paired cells prints its heading and a single
        line saying so.

        :param report: Report returned by ``RetrainedPair.report``.
        :type report: Mapping[str, Any]
        :param json_path: Path of the JSON that was written, appended as
            the closing pointer line, or None to leave it out.
        :type json_path: str | Path | None
        :return: The lines of the report, without trailing newlines.
        :rtype: list[str]
        :raises TypeError: If report is not a mapping.
        :raises ValueError: If report has no "pairs" section.
        """
        lines: list[str] = list()
        if not isinstance(report, Mapping):
            self._error("[RetrainedPairReport.render] Input error: the "
                        "parameter report must be a mapping, not "
                        f"{type(report).__name__}.", TypeError)
        if "pairs" not in report:
            self._error("[RetrainedPairReport.render] Input error: the "
                        "parameter report must hold a 'pairs' section, "
                        f"and it holds {sorted(report)}.")
        for windowed, blk in report["pairs"].items():
            lines.append("")
            lines.append(f"=== {windowed} vs control "
                         f"({blk['n_cells']} cells) ===")
            if not blk["n_cells"]:
                lines.append("  no paired cells")
                continue
            lines.extend(self._block_lines(blk))
        if json_path is not None:
            lines.append("")
            lines.append(f"-> {json_path}")
        return lines

    def __repr__(self) -> str:
        """
        Show the paradigm halves that will be printed.

        :return: Text like "RetrainedPairReport(paradigms=('TSR',
            'AR'))".
        :rtype: str
        """
        return ("RetrainedPairReport(paradigms="
                f"{tuple(label for label, _ in self.paradigms)})")

    def _block_lines(self, blk: Mapping[str, Any]) -> list[str]:
        """
        Printed lines of one non-empty block of statistics.

        The first line carries the headline numbers of the pair (wins,
        median, Hodges-Lehmann estimate with its interval and coverage,
        the signed rank p value and the range), then one line per
        paradigm half, one line per unit of analysis (the dataset is
        the unit of the plan), then the seed stability of the cells,
        and last one line per (dataset, paradigm) with its mean
        difference.

        Note: the p value is printed with ``:.4g``, so a pair whose
        differences are all exactly zero (p is None) fails here, as it
        did in the submitted script; that cannot happen with real
        ROC-AUC values.

        :param blk: Block of statistics of one pair, as
            ``RetrainedPair.report`` builds it, with at least one cell.
        :type blk: Mapping[str, Any]
        :return: The lines of the block, without trailing newlines.
        :rtype: list[str]
        """
        lines: list[str] = list()
        ci: Mapping[str, Any] = dict()
        wx: Mapping[str, Any] = dict()
        half: Mapping[str, Any] = dict()
        ci = blk["hl_ci"]
        wx = blk["wilcoxon"]
        lines.append(
            f"  window ahead in {blk['wins_ctx']}/"
            f"{blk['n_cells']} · median {blk['median']:+.4f} · "
            f"HL {blk['hl']:+.4f} [{ci['lower']:+.4f}, "
            f"{ci['upper']:+.4f}] coverage {ci['coverage']:.4f} · "
            f"Wilcoxon p={wx['p']:.4g} n={wx['n_eff']} · range "
            f"{blk['min']:+.4f} to {blk['max']:+.4f}")
        for label, _ in self.paradigms:
            if label in blk:
                half = blk[label]
                lines.append(
                    f"  {label}: {half['n']} cells · ahead in "
                    f"{half['wins_ctx']} · median "
                    f"{half['median']:+.4f} · HL {half['hl']:+.4f} "
                    f"[{half['hl_ci']['lower']:+.4f}, "
                    f"{half['hl_ci']['upper']:+.4f}]")
        for name, unit in blk.get("by_unit", dict()).items():
            lines.append(
                f"  by {name} (n={unit['n']}): ahead in "
                f"{unit['wins_ctx']} · HL {unit['hl']:+.4f} "
                f"[{unit['hl_ci']['lower']:+.4f}, "
                f"{unit['hl_ci']['upper']:+.4f}] coverage "
                f"{unit['hl_ci']['coverage']:.4f} · Wilcoxon "
                f"p={unit['wilcoxon']['p']:.4g}")
        lines.append("  (dataset, paradigm) groups with 2 seeds: "
                     f"{blk['n_groups_two_seeds']} · sign flips with "
                     f"the seed: {len(blk['sign_flips'])}")
        lines.append("  mean over seeds: window ahead in "
                     f"{blk['wins_ctx_dataset_ar_mean']}/"
                     f"{len(blk['per_dataset_ar_mean'])} (dataset, "
                     "paradigm) combinations")
        for x in blk["per_dataset_ar_mean"]:
            lines.append(f"    {x['dataset']:26s} "
                         f"{'AR ' if x['ar'] else 'TSR'} "
                         f"n={x['n_seeds']} mean delta "
                         f"{x['mean_delta']:+.4f}")
        return lines


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
