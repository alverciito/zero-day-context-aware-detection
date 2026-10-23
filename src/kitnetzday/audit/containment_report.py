# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         src/kitnetzday/audit/containment_report.py                #
#   Version:      2.0.0                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Printed text of the containment audit of the              #
#                 retraining experiment, line by line.                      #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from kitnetzday.audit.containment import (
                                       BUDGET_LABELS,
                                       BUDGET_ORDER,
                                       ContainmentAudit,
)
from kitnetzday.audit.convergence import (
                                       EVALUABLE_VERDICTS,
                                       VERDICT_CONSTANT,
                                       VERDICT_TRAINED,
)
from kitnetzday.utils.loggable import Loggable

# How many of the worst ratios are listed under each budget.
WORST_LISTED: int = 5


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                MAIN CLASS                                 #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class ContainmentReport(Loggable):
    """
    Printed text of the audit of ``ContainmentAudit``, line by line.

    Joining the lines with newlines and printing the result gives the
    audit, blank lines included. Keeping the text apart from the
    computation keeps ``print`` out of the library: a script in
    ``bin/`` is the only place that writes to the console.

    The text is in Spanish, as the authors wrote it.

    The audit is asked for its thresholds and its row filters, so that
    the printed counts cannot drift from the ones written to the JSON,
    and for its reader, so that an empty results root prints one single
    line instead of an audit of nothing.

    :param audit: Audit that produced the report being rendered.
    :type audit: ContainmentAudit
    :param worst_listed: How many of the worst ratios are listed under
        each budget; must be >= 0.
    :type worst_listed: int
    :param logger: Logger created by the calling script, or None to stay
        silent.
    :type logger: logging.Logger | None
    :raises TypeError: If logger is not a logging.Logger or None, if
        audit is not a ContainmentAudit, or if worst_listed is not an
        int.
    :raises ValueError: If worst_listed is negative.
    """

    def __init__(self,
                 audit: ContainmentAudit,
                 worst_listed: int = WORST_LISTED,
                 logger: logging.Logger | None = None) -> None:
        """
        Check the audit and the listing length, and store them.

        :param audit: Audit that produced the report.
        :type audit: ContainmentAudit
        :param worst_listed: How many worst ratios are listed, >= 0.
        :type worst_listed: int
        :param logger: Logger created by the calling script, or None.
        :type logger: logging.Logger | None
        :raises TypeError: If audit or worst_listed are of the wrong
            type.
        :raises ValueError: If worst_listed is negative.
        """
        super().__init__(logger=logger)
        if not isinstance(audit, ContainmentAudit):
            self._error("[ContainmentReport.__init__] Input error: the "
                        "parameter audit must be a ContainmentAudit, "
                        f"not {type(audit).__name__}.", TypeError)
        if isinstance(worst_listed, bool) or \
                not isinstance(worst_listed, int) or worst_listed < 0:
            self._error("[ContainmentReport.__init__] Input error: the "
                        "parameter worst_listed must be an int >= 0, "
                        f"not {worst_listed!r}.", TypeError)
        self.audit = audit
        self.worst_listed = int(worst_listed)

    def render(self,
               report: Mapping[str, Any],
               json_path: str | Path | None = None) -> list[str]:
        """
        Text of the audit, one element per printed line.

        The order is: first the health of the autoencoders, then the
        CG-2 test inside the job (the correct one) budget by budget,
        then the weaker test that pairs jobs with the rows it could not
        clear, and last the file written.

        :param report: Report returned by ``ContainmentAudit.report``.
        :type report: Mapping[str, Any]
        :param json_path: Path of the JSON that was written, announced
            in the closing line, or None to leave it out.
        :type json_path: str | Path | None
        :return: The lines of the audit, without trailing newlines.
        :rtype: list[str]
        :raises TypeError: If report is not a mapping.
        :raises ValueError: If report has no "rows" section.
        """
        lines: list[str] = list()
        tally: Mapping[str, int] = dict()
        n_eval = 0
        if not isinstance(report, Mapping):
            self._error("[ContainmentReport.render] Input error: the "
                        "parameter report must be a mapping, not "
                        f"{type(report).__name__}.", TypeError)
        if "rows" not in report:
            self._error("[ContainmentReport.render] Input error: the "
                        "parameter report must hold a 'rows' section, "
                        f"and it holds {sorted(report)}.")
        if not self.audit.reports.reports:
            return ["No hay ningun T8-*/convergence_report.json en "
                    f"{self.audit.reports.results_root}: nada que "
                    "comprobar."]
        tally = report["health_tally"]
        n_eval = sum(n for v, n in tally.items()
                     if v in EVALUABLE_VERDICTS)
        lines.append("Salud de los AE del bloque T8 (con el suelo del "
                     "predictor constante):")
        for verdict, n in sorted(tally.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {n:4d}  {verdict}")
        lines.append(f"  -> entrenados {tally.get(VERDICT_TRAINED, 0)}/"
                     f"{n_eval} de los AE evaluables "
                     f"({tally.get(VERDICT_CONSTANT, 0)} sin varianza "
                     "que aprender)")
        lines.append("")
        if report["within_job"]:
            for budget in BUDGET_ORDER:
                lines.extend(self._budget_lines(report["within_job"],
                                                budget))
        lines.append("CG-2 emparejando JOBS (mas debil, incluye "
                     f"varianza de inicializacion): {report['n_pairs']} "
                     f"pares, {report['n_violations']} violaciones "
                     f"(tolerancia {self.audit.tolerance:.0%}).")
        for row in report["rows"]:
            if row.get("status") == "OK":
                continue
            lines.append(self._row_line(row))
        if report["n_pairs"] and not report["n_violations"]:
            lines.append("  Todos los pares cumplen CG-2: el enventanado"
                         " nunca reconstruye peor que el control sin "
                         "ventana.")
        if json_path is not None:
            lines.append(f"  escrito {json_path}")
        return lines

    def __repr__(self) -> str:
        """
        Show the audit rendered and the length of the worst list.

        :return: Text like "ContainmentReport(audit=ContainmentAudit(
            ...), worst_listed=5)".
        :rtype: str
        """
        return (f"ContainmentReport(audit={self.audit!r}, "
                f"worst_listed={self.worst_listed})")

    def _budget_lines(self,
                      rows: list[dict[str, Any]],
                      budget: str) -> list[str]:
        """
        Printed lines of the CG-2 test inside the job, for one budget.

        The block closes with a blank line and with the worst ratios
        listed one per line so that a violation can be traced to its
        dataset, seed and autoencoder without opening the JSON.

        :param rows: Rows returned by ``ContainmentAudit.within_job``,
            every budget mixed.
        :type rows: list[dict[str, Any]]
        :param budget: Budget to print, "base" or "EP5".
        :type budget: str
        :return: The lines of that budget, empty when it has no rows.
        :rtype: list[str]
        """
        lines: list[str] = list()
        sub: list[dict[str, Any]] = list()
        ev: list[dict[str, Any]] = list()
        ratios: list[float] = list()
        wide: list[float] = list()
        narrow: list[float] = list()
        n_ok = 0
        sub, ev = self.audit.evaluable(rows, budget)
        if not sub:
            return lines
        lines.append("CG-2 DENTRO del job (el test correcto) - "
                     f"{BUDGET_LABELS[budget]}: {len(sub)} AE con ventana, "
                     "copia por contencion exacta en "
                     f"{sum(1 for r in sub if r['copy_exact'])}/"
                     f"{len(sub)}.")
        if ev:
            n_ok = len(self.audit.complying(ev))
            ratios = sorted(r["ratio"] for r in ev)
            wide = [r["ratio"] for r in ev if r["n_visible"]
                    and r["n_visible"] > self.audit.wide_visible]
            narrow = [r["ratio"] for r in ev if r["n_visible"]
                      and r["n_visible"] <= self.audit.wide_visible]
            lines.append("  evaluables (suelo >= "
                         f"{self.audit.floor_min:g}): {len(ev)} · "
                         f"cumplen {n_ok} · violan {len(ev) - n_ok}")
            lines.append("  ratio ventana/por-registro: mediana "
                         f"{self.audit.stats.median(ratios):.4f} · min "
                         f"{ratios[0]:.4f} · max {ratios[-1]:.4f}")
            lines.append("  la ventana MEJORA en "
                         f"{len(self.audit.improving(ev))}/{len(ev)} AE "
                         f"y empeora en {len(ev) - n_ok}")
            if wide and narrow:
                lines.append(
                    f"  por anchura: n_v > {self.audit.wide_visible} "
                    f"n={len(wide)} mediana "
                    f"{self.audit.stats.median(wide):.4f} · n_v <= "
                    f"{self.audit.wide_visible} n={len(narrow)} mediana "
                    f"{self.audit.stats.median(narrow):.4f}")
            for r in sorted(ev, key=lambda row: -row["ratio"]
                            )[:self.worst_listed]:
                lines.append(f"    peor: {r['dataset']} s{r['seed']} "
                             f"AE{r['ae']} n_v={r['n_visible']} "
                             f"ratio={r['ratio']:.3f}")
        lines.append("")
        return lines

    def _row_line(self, row: Mapping[str, Any]) -> str:
        """
        Printed line of one pair row that is not plain "OK".

        :param row: Row returned by ``ContainmentAudit.pair_rows``.
        :type row: Mapping[str, Any]
        :return: The line, with the ratio and the shape appended when
            the row carries them.
        :rtype: str
        """
        out = ""
        out = (f"  {row.get('status'):22s} {row.get('cell')} "
               f"AE{row.get('ae', '-')}")
        if "ratio_windowed_over_per_record" in row:
            out += (" ratio="
                    f"{row['ratio_windowed_over_per_record']:.4f}")
        if row.get("n_visible"):
            out += (f" n_v={row.get('n_visible')} "
                    f"n_h={row.get('n_hidden')}")
        return out


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
