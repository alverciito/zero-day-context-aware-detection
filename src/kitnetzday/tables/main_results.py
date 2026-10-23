# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         src/kitnetzday/tables/main_results.py                     #
#   Version:      1.0.3                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Main results table (T1), its one-page compact variant     #
#                 and the best configuration per stream (R1.10).            #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
from collections.abc import Mapping, Sequence
from typing import Any

from kitnetzday.io.inputs import Inputs
from kitnetzday.io.outputs import Out
from kitnetzday.utils.formatting import fnum, tex_escape
from kitnetzday.utils.loggable import Loggable

# Captions of the three tables. They are output strings: a change here
# changes the generated files and the reference outputs. The pieces
# are concatenated around the counts of the table.
_MAIN_CAPTION_A: str = (
    "Main results per dataset and configuration on the evaluation "
    "segment. ROC-AUC of the fused detector is the pre-specified "
    "primary metric; the dispersion column gives the observed range "
    "over the two seeds, and says \\emph{single run} for the "
    "deterministic cells. $F_1$ is calibrated with the label-free P99 rule "
    "on the calibration window (never with evaluation labels), and "
    "WindowDiff uses the standard boundary definition with the "
    "per-stream $k$ given in parentheses. Each dataset block starts with "
    "its trivial references: prevalence $p$, "
    "$F_{1,\\mathrm{trivial}}=2p/(1+p)$ and the row-index baseline "
    "AUC$_t$, which models have to beat as $\\max($AUC$_t$, 1 $-$ "
    "AUC$_t)$ since a detector returning the decreasing index is "
    "equally free; $\\ddagger$ marks a cell whose mean calibrated $F_1$, the "
    "value printed in the same row, is at or below "
    "$F_{1,\\mathrm{trivial}}$. The analysis plan also tracks a wider "
    "indicator, $F_1 \\leq F_{1,\\mathrm{trivial}} + 0.02$, which holds "
    "in at least one seed of ")
_MAIN_CAPTION_B: str = " of the "
_MAIN_CAPTION_C: str = " cells. Chance-level ROC-AUC is 0.5."
_COMPACT_CAPTION_A: str = (
    "ROC-AUC of the fused detector per configuration and dataset (mean "
    "over the two seeds), with the row-index baseline AUC$_t$, the "
    "symmetric reference a model has to beat, the prevalence and the "
    "number of seeds. Chance level is 0.5. Datasets: ")
_COMPACT_CAPTION_B: str = "."
# Plan section 11 (R1.10): this note is literal and mandatory; without
# it the table is not generated.
_BEST_NOTE: str = ("selected on evaluation results; descriptive, not an "
                   "estimate of expected performance")
_BEST_CAPTION_A: str = ("Best-performing configuration per dataset by "
                        "ROC-AUC of the fused detector: ")
_BEST_CAPTION_B: str = (
    ". The row-index baseline AUC$_t$ of each stream is repeated so that "
    "no winner is read without it, and the number of seeds behind each "
    "mean is given; with two seeds the ordering of the top "
    "configurations is inside the run-to-run noise reference of the "
    "analysis plan. This table selects on the evaluation segment and "
    "therefore supports no claim about a best architecture; the "
    "pre-specified comparisons are in Table~6 of the manuscript.")
# Notes stored in figures.json next to each table.
_MAIN_NOTES: tuple[str, ...] = (
    "Requires longtable; tab_main_results_compact.tex is a one-page "
    "alternative.",
    "F1 and WindowDiff are secondary (plan §3.2): they support no "
    "confirmatory claim.")
_COMPACT_NOTES: tuple[str, ...] = (
    "One-page alternative to tab_main_results (without F1 and "
    "WindowDiff).",)
_BEST_NOTES: tuple[str, ...] = (
    "Plan §11 R1.10 requires this table with the literal note; the note "
    "goes in the caption and as a LaTeX comment.",
    "Selection on the evaluation segment: not allowed in the abstract, "
    "the highlights or the contributions (plan §11).")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                MAIN CLASS                                 #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class MainResultsTables(Loggable):
    """
    Writer of the main results table of the manuscript (output T1).

    The class turns the aggregated results into three LaTeX artefacts
    and registers one row of ``numbers_used.md`` per printed value:
    ``tab_main_results``, a ``longtable`` with one block per dataset
    (its trivial references in a spanning row) and one row per
    configuration holding ROC-AUC of the fused detector, the across-seed
    dispersion, the calibrated $F_1$ and WindowDiff with its per-stream
    $k$; ``tab_main_results_compact``, the primary metric alone, so that
    it fits on one page; and ``tab_best_per_dataset``, the best
    configuration per stream asked for by comment R1.10, with the
    literal note of plan section 11 against the selection bias. Nothing
    is recomputed: every number is read through ``Inputs`` exactly as
    the aggregate stored it.

    :param inputs: Reader of the aggregated results, already loaded and
        cross-checked.
    :type inputs: Inputs
    :param out: Writer of the output directory, which also collects the
        rows of ``numbers_used.md``.
    :type out: Out
    :param logger: Logger created by the calling script, or None to stay
        silent.
    :type logger: logging.Logger | None
    :raises TypeError: If inputs is not an Inputs, if out is not an Out,
        or if logger is not a logging.Logger or None.
    """

    def __init__(self,
                 inputs: Inputs,
                 out: Out,
                 logger: logging.Logger | None = None) -> None:
        """
        Store the reader and the writer after checking their types.

        :param inputs: Reader of the aggregated results.
        :type inputs: Inputs
        :param out: Writer of the output directory.
        :type out: Out
        :param logger: Logger created by the calling script, or None.
        :type logger: logging.Logger | None
        :raises TypeError: If inputs is not an Inputs, if out is not an
            Out, or if logger is not a logging.Logger or None.
        """
        super().__init__(logger=logger)
        if not isinstance(inputs, Inputs):
            self._error("[MainResultsTables.__init__] Input error: the "
                        "parameter inputs must be an Inputs, not "
                        f"{type(inputs).__name__}.", TypeError)
        if not isinstance(out, Out):
            self._error("[MainResultsTables.__init__] Input error: the "
                        "parameter out must be an Out, not "
                        f"{type(out).__name__}.", TypeError)
        self.inputs = inputs
        self.out = out
        self.naming = inputs.naming
        self.datasets = list(self.naming.datasets)
        self.configurations = list(self.naming.configurations)
        self.tags = dict(self.naming.plan_tags)

    def build(self) -> bool:
        """
        Write the main results table, its compact variant and R1.10.

        Datasets and configurations are emitted in vocabulary order, and
        every printed value is registered as a row of values before it
        is written. The $\\ddagger$ mark compares the mean $F_1$ printed in
        the same row with the trivial $F_1$ of the dataset (strict,
        ``<=``), which is not the ``f1_le_trivial_FUS`` indicator of the
        aggregate: that one carries the +0.02 margin of plan section 3.2
        and is only counted in the caption.

        Note: a cell whose ROC-AUC is missing is listed in the "missing"
        notes but its row is still written, with the missing marker.

        :return: True once the three tables have been written.
        :rtype: bool
        :raises OSError: If one of the output files cannot be written.
        """
        ids = list()
        missing = list()
        notes = list()
        body = list()
        md = list()
        n_margin = 0
        caption, cap2 = "", ""
        tex, comp = list(), list()
        for x in self.datasets:
            ns = self.inputs.n_seeds(x)
            p, f1t, auct = (self.inputs.trivial(x, "prevalence"),
                            self.inputs.trivial(x, "f1_trivial"),
                            self.inputs.trivial(x, "auc_t"))
            for key, v, ptr in (("p", p, "prevalence"),
                                ("F1t", f1t, "f1_trivial"),
                                ("AUC_t", auct, "auc_t")):
                ids.append(self.out.add_number(
                    f"R.TRIV.{x}.{key}", v,
                    f"aggregate.json#/datasets/{x}/trivial/{ptr}",
                    self.tags["trivial"], self.inputs.partial_inputs))
            body.append("\\multicolumn{6}{l}{\\textit{%s} --- $n=%d$ "
                        "seed(s); prevalence $p=%s$; "
                        "$F_{1,\\mathrm{trivial}}=%s$; AUC$_t=%s$} \\\\"
                        % (tex_escape(self.inputs.dataset_label(x)), ns,
                           fnum(p), fnum(f1t), fnum(auct)))
            md.append(f"\n### {self.inputs.dataset_label(x)} (n={ns}; "
                      f"p={fnum(p)}; F1_triv={fnum(f1t)}; "
                      f"AUC_t={fnum(auct)})\n")
            md.append("| Config | ROC-AUC | dispersion | F1 | "
                      "WindowDiff (k) |")
            md.append("|---|---|---|---|---|")
            for c in self.configurations:
                cell = self.inputs.cell(x, c)
                a, a_ptr = self.inputs.auroc_src(x, c)
                f1 = self.inputs.cell_mean(x, c, "f1_FUS")
                wd = self.inputs.cell_mean(x, c, "wd_FUS")
                wdk = self.inputs.cell_mean(x, c, "wd_k_FUS")
                le_triv = self.inputs.cell_mean(
                    x, c, "f1_le_trivial_FUS")
                if a is None:
                    missing.append(
                        f"{x}/{c}: cell without a value "
                        f"({cell.get('lost_reason') or 'not run'})")
                dtex, dmd, dkind = self._dispersion(cell, ns)
                ids.append(self.out.add_number(
                    f"R.T1.{x}.{c}.AUROC_FUS", a, a_ptr,
                    "[CONFIRMATORIO plan §3.1 — valor por celda, "
                    "descriptivo, sin test]", self.inputs.partial_inputs))
                if dkind[0] == "SD":
                    ids.append(self.out.add_number(
                        f"R.T1.{x}.{c}.AUROC_SD", dkind[1],
                        f"aggregate.json#/cells/{x}/{c}"
                        "/auroc_FUS_sd_ddof1",
                        "[CONFIRMATORIO plan §3.1 — dispersion entre "
                        "semillas]", self.inputs.partial_inputs))
                elif dkind[0] == "range":
                    ids.append(self.out.add_number(
                        f"R.T1.{x}.{c}.AUROC_min",
                        cell.get("auroc_FUS_min"),
                        f"aggregate.json#/cells/{x}/{c}/auroc_FUS_min",
                        "[CONFIRMATORIO plan §3.1 — dispersion entre "
                        "semillas]", self.inputs.partial_inputs))
                    ids.append(self.out.add_number(
                        f"R.T1.{x}.{c}.AUROC_max",
                        cell.get("auroc_FUS_max"),
                        f"aggregate.json#/cells/{x}/{c}/auroc_FUS_max",
                        "[CONFIRMATORIO plan §3.1 — dispersion entre "
                        "semillas]", self.inputs.partial_inputs))
                ids.append(self.out.add_number(
                    f"R.T1.{x}.{c}.F1_FUS", f1,
                    f"aggregate.json#/cells/{x}/{c}/mean/f1_FUS",
                    self.tags["secondary"], self.inputs.partial_inputs))
                ids.append(self.out.add_number(
                    f"R.T1.{x}.{c}.WD_FUS", wd,
                    f"aggregate.json#/cells/{x}/{c}/mean/wd_FUS",
                    self.tags["secondary"], self.inputs.partial_inputs))
                ids.append(self.out.add_number(
                    f"R.T1.{x}.{c}.WDk_FUS", wdk,
                    f"aggregate.json#/cells/{x}/{c}/mean/wd_k_FUS",
                    "[CONFIRMATORIO plan §3.3 — k de WindowDiff, solo "
                    "etiquetas]", self.inputs.partial_inputs))
                f1_mark = ("$^{\\ddagger}$"
                           if (f1 is not None and f1t is not None
                               and f1 <= f1t) else "")
                if le_triv is not None and le_triv > 0.0:
                    n_margin += 1
                body.append(
                    "\\quad %s & %s & %s & %s%s & %s (%s) \\\\"
                    % (tex_escape(f"{c} {self.inputs.config_name(c)}"),
                       fnum(a), dtex, fnum(f1), f1_mark, fnum(wd),
                       fnum(wdk, 0)))
                md.append(
                    f"| {c} {self.inputs.config_name(c)} | {fnum(a)} | "
                    f"{dmd} | {fnum(f1)}"
                    f"{' (<= trivial)' if f1_mark else ''} | "
                    f"{fnum(wd)} ({fnum(wdk, 0)}) |")
            body.append("\\addlinespace")
        caption = (_MAIN_CAPTION_A + str(n_margin) + _MAIN_CAPTION_B
                   + str(len(self.datasets) * len(self.configurations))
                   + _MAIN_CAPTION_C)
        tex = ["% Requires \\usepackage{longtable} and "
               "\\usepackage{booktabs}.",
               "\\begingroup\\footnotesize",
               "\\begin{longtable}{@{}lrrrr@{}}",
               f"\\caption{{{caption}}}\\label{{tab:main-results}}\\\\",
               "\\toprule",
               "Configuration & ROC-AUC & Dispersion & $F_1$ & "
               "WindowDiff ($k$) \\\\",
               "\\midrule", "\\endfirsthead",
               "\\toprule",
               "Configuration & ROC-AUC & Dispersion & $F_1$ & "
               "WindowDiff ($k$) \\\\",
               "\\midrule", "\\endhead",
               "\\midrule \\multicolumn{5}{r}{\\textit{continued on the "
               "next page}} \\\\", "\\endfoot",
               "\\bottomrule", "\\endlastfoot"]
        tex += [b.replace("\\multicolumn{6}", "\\multicolumn{5}")
                for b in body]
        tex += ["\\end{longtable}", "\\endgroup"]
        self.out.save_table(
            "tab_main_results", "\n".join(tex),
            "# Main results (Markdown rendering)\n" + "\n".join(md), caption,
            "tab:main-results",
            self.tags["primary"] + " (ROC-AUC) / "
            + self.tags["secondary"] + " (F1, WindowDiff)",
            sorted(set(ids)), self.inputs.partial_inputs,
            {"out_id": "T1",
             "partial_inputs_reasons": self.inputs.partial_inputs_reasons,
             "missing": missing, "notes": notes + list(_MAIN_NOTES)})
        comp = ["% Requires \\usepackage{booktabs}.",
                "\\begin{table*}[t]", "  \\centering", "  \\footnotesize",
                "  \\begin{tabular}{@{}l" + "r" * len(self.datasets)
                + "@{}}", "  \\toprule",
                "  Configuration & "
                + " & ".join(self.naming.dataset_shorts[x]
                             for x in self.datasets) + " \\\\",
                "  \\midrule"]
        for c in self.configurations:
            cells = list()
            for x in self.datasets:
                a = self.inputs.auroc(x, c)
                cells.append(fnum(a, 3) if a is not None else "--")
            comp.append(
                "  " + tex_escape(f"{c} {self.inputs.config_name(c)}")
                + " & " + " & ".join(cells) + " \\\\")
        comp.append("  \\midrule")
        comp.append("  AUC$_t$ (row index) & "
                    + " & ".join(fnum(self.inputs.trivial(x, "auc_t"))
                                 for x in self.datasets) + " \\\\")
        # The reference a model has to beat is the symmetric one, because
        # a detector that returns the decreasing index is equally free. It
        # is the one the dagger mark and the body use.
        comp.append("  $\\max(\\mathrm{AUC}_t, 1-\\mathrm{AUC}_t)$ & "
                    + " & ".join(
                        fnum(max(self.inputs.trivial(x, "auc_t"),
                                 1.0 - self.inputs.trivial(x, "auc_t")))
                        for x in self.datasets) + " \\\\")
        comp.append("  Prevalence $p$ & "
                    + " & ".join(fnum(self.inputs.trivial(x,
                                                          "prevalence"))
                                 for x in self.datasets) + " \\\\")
        comp.append("  Seeds $n$ & "
                    + " & ".join(str(self.inputs.n_seeds(x))
                                 for x in self.datasets) + " \\\\")
        cap2 = (_COMPACT_CAPTION_A
                + ", ".join(f"{self.naming.dataset_shorts[x]} = "
                            f"{self.naming.dataset_longs[x]}"
                            for x in self.datasets)
                + _COMPACT_CAPTION_B)
        comp += ["  \\bottomrule", "  \\end{tabular}",
                 f"  \\caption{{{cap2}}}",
                 "  \\label{tab:main-results-compact}", "\\end{table*}"]
        self.out.save_table(
            "tab_main_results_compact", "\n".join(comp), None, cap2,
            "tab:main-results-compact", self.tags["primary"],
            sorted({i for i in ids
                    if ".AUROC_FUS" in i or i.startswith("R.TRIV")}),
            self.inputs.partial_inputs,
            {"out_id": "T1",
             "partial_inputs_reasons": self.inputs.partial_inputs_reasons,
             "missing": missing, "notes": list(_COMPACT_NOTES)})
        self.build_best_per_dataset(ids)
        return True

    def build_best_per_dataset(self,
                               ids_main: Sequence[str] = tuple()) -> bool:
        """
        Write the best configuration per stream asked for by R1.10.

        For every dataset the configuration with the largest ROC-AUC of
        the fused detector is reported, with the row-index baseline
        AUC$_t$ and the number of seeds behind the mean. Ties within
        1e-12 ROC-AUC points are all listed and marked "(tied)". The
        literal note of plan section 11 goes into the caption and into a
        LaTeX comment: without it the table would read as a claim about
        a best architecture.

        Note: the ids of the main table are accepted but are not used;
        the table collects its own ids, the ones written into its
        ``.tex``.

        :param ids_main: Ids of the values collected by the main table;
            accepted and ignored, any sequence of strings is valid.
        :type ids_main: Sequence[str]
        :return: True once the table has been written.
        :rtype: bool
        :raises OSError: If the output file cannot be written.
        """
        rows, ids, missing, md = list(), list(), list(), list()
        caption = ""
        tex = list()
        md = ["| Dataset | Best configuration | ROC-AUC | AUC_t | "
              "n seeds |", "|---|---|---|---|---|"]
        for x in self.datasets:
            vals = [(c, self.inputs.auroc(x, c))
                    for c in self.configurations]
            vals = [(c, v) for c, v in vals if v is not None]
            if not vals:
                missing.append(
                    f"{x}: no configuration with a ROC-AUC")
                continue
            best = max(vals, key=lambda z: z[1])
            tied = [c for c, v in vals if abs(v - best[1]) <= 1e-12]
            name = (self.inputs.config_name(best[0])
                    if len(tied) == 1
                    else ", ".join(tied) + " (tied)")
            auct = self.inputs.trivial(x, "auc_t")
            ids.append(self.out.add_number(
                f"R.R110.{x}.BEST.AUROC_FUS", best[1],
                self.inputs.auroc_ptr(x, best[0]), self.tags["r110"],
                self.inputs.partial_inputs))
            rows.append("%s & %s & %s & %s & %d \\\\"
                        % (tex_escape(self.inputs.dataset_label(x)),
                           tex_escape(f"{best[0]} {name}"),
                           fnum(best[1]), fnum(auct),
                           self.inputs.n_seeds(x)))
            md.append(f"| {self.inputs.dataset_label(x)} | {best[0]} "
                      f"{name} | {fnum(best[1])} | {fnum(auct)} | "
                      f"{self.inputs.n_seeds(x)} |")
        caption = _BEST_CAPTION_A + _BEST_NOTE + _BEST_CAPTION_B
        tex = ["% Requires \\usepackage{booktabs}.",
               "% Mandatory note of plan §11 R1.10 (literal): "
               + _BEST_NOTE,
               "\\begin{table}[t]", "  \\centering", "  \\footnotesize",
               "  \\begin{tabular}{@{}llrrr@{}}", "  \\toprule",
               "  Dataset & Best configuration & ROC-AUC & AUC$_t$ & "
               "Seeds \\\\", "  \\midrule"]
        tex += ["  " + r for r in rows] or [
            "  \\multicolumn{5}{c}{no data in the inputs} \\\\"]
        tex += ["  \\bottomrule", "  \\end{tabular}",
                f"  \\caption{{{caption}}}",
                "  \\label{tab:best-per-dataset}", "\\end{table}"]
        self.out.save_table(
            "tab_best_per_dataset", "\n".join(tex),
            "# Best configuration per dataset (Markdown rendering)\n"
            + "\n".join(md), caption, "tab:best-per-dataset",
            self.tags["r110"], sorted(set(ids)), self.inputs.partial_inputs,
            {"out_id": "T1",
             "partial_inputs_reasons": self.inputs.partial_inputs_reasons,
             "missing": missing, "notes": list(_BEST_NOTES)})
        return True

    def _dispersion(self,
                    cell: Mapping[str, Any],
                    n_seeds: int) -> tuple[str, str,
                                           tuple[str, float | None]]:
        """
        Across-seed dispersion of one cell, as plan section 7.1 words it.

        The standard deviation (ddof 1) is printed with three or more
        seeds, the observed range with exactly two, and nothing with a
        single one. Deterministic cells (``original`` and ``stat``: one
        run replicated for every seed) have no across-seed dispersion:
        printing "[0.322, 0.322]" as a range of two seeds would
        contradict the caption and misrepresent the replication of the
        C10 control that comment R1.2 asks about.

        Note: the observed number of non-imputed seeds lowers n_seeds
        only when it is stored as an ``int``, so a float (or a bool)
        does not.

        :param cell: Cell record of the aggregate, which may carry
            "deterministic", "auroc_FUS_sd_ddof1", "auroc_FUS_min",
            "auroc_FUS_max" and "auroc_FUS_n_seeds_noimp"; an empty
            mapping gives the single-run text.
        :type cell: Mapping[str, Any]
        :param n_seeds: Size of the analysis set of the dataset, the
            number of seeds behind the mean; zero or more.
        :type n_seeds: int
        :return: The LaTeX text of the dispersion column, the Markdown
            text of the preview, and a kind tag with its value: ("SD",
            sd), ("range", None), ("single", None) or ("single_det",
            None), all in ROC-AUC points.
        :rtype: tuple[str, str, tuple[str, float | None]]
        :raises TypeError: If cell is not a mapping or n_seeds is not an
            integer.
        """
        sd, mn, mx, n_obs = None, None, None, None
        n = 0
        if not isinstance(cell, Mapping):
            self._error("[MainResultsTables._dispersion] Input error: "
                        "the parameter cell must be a mapping, not "
                        f"{type(cell).__name__}.", TypeError)
        if isinstance(n_seeds, bool) or not isinstance(n_seeds, int):
            self._error("[MainResultsTables._dispersion] Input error: "
                        "the parameter n_seeds must be an integer, not "
                        f"{type(n_seeds).__name__}.", TypeError)
        n = n_seeds
        sd = cell.get("auroc_FUS_sd_ddof1")
        mn, mx = cell.get("auroc_FUS_min"), cell.get("auroc_FUS_max")
        if cell.get("deterministic"):
            return ("single run (deterministic)",
                    "1 run (deterministic)", ("single_det", None))
        n_obs = cell.get("auroc_FUS_n_seeds_noimp")
        if isinstance(n_obs, int) and n_obs >= 1:
            n = min(n, n_obs)
        if n >= 3 and sd is not None:
            return (f"$\\pm$ {float(sd):.3f}", f"±{float(sd):.3f}",
                    ("SD", float(sd)))
        if n == 2 and mn is not None and mx is not None:
            return (f"[{float(mn):.3f}, {float(mx):.3f}]",
                    f"[{float(mn):.3f}, {float(mx):.3f}]",
                    ("range", None))
        return "single run", "1 run", ("single", None)

    def __repr__(self) -> str:
        """
        Show the inputs, the output directory and the table sizes.

        :return: Text like "MainResultsTables(inputs=..., out=...,
            datasets=9, configurations=11)".
        :rtype: str
        """
        return (f"MainResultsTables(inputs={self.inputs!r}, "
                f"out={self.out!r}, datasets={len(self.datasets)}, "
                f"configurations={len(self.configurations)})")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
