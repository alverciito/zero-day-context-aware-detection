# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         src/kitnetzday/tables/supplementary_latex.py              #
#   Version:      1.0.2                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-13                                                #
#   Description:  LaTeX text of the supplementary tables: captions, cell    #
#                 escaping and moving the caption below the tabular.        #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
import re
from typing import Any

# Replacements of escape_cell, in this exact order. The first three undo
# the encoding damage of the source labels, the fourth turns a backslash of the
# source labels into a slash, and only then the three LaTeX specials get
# their backslash: that is why the backslash rule comes before them and
# is not applied again afterwards.
_CELL_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("\ufffd", "--"), ("\u2013", "--"), ("\u2014", "---"),
    ("\\", "/"), ("_", r"\_"), ("&", r"\&"), ("%", r"\%"))
# Floats, captions, labels and tabular ends, as the original regular
# expressions spelled them: the float pattern is non-greedy and matches
# both "table" and "table*".
_FLOAT_RE = re.compile(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", re.S)
_CAPTION_RE = re.compile(r"\\caption\{")
_LABEL_RE = re.compile(r"\s*\\label\{[^}]*\}")
_TABULAR_END_RE = re.compile(r"\\end\{tabular\*?\}")

# Captions of the fifteen floats, in academic English and already valid
# LaTeX. They are the text of the supplement: a change here is a change
# in the artefact. Some of them cite labels that the supplement defines
# outside these floats (tab:s_calib_attack, sec:s_calib_attack), so the
# fragment resolves once pasted there.
# The eight tables that share the configuration-by-stream grid, each
# with the label of its float and the metric key it prints.
GRID_TABLES: tuple[tuple[str, str, str], ...] = (
    ("F1-score of the fused detector at the calibrated threshold (99th "
     "percentile of the calibration segment), averaged over the two seeds. "
     "The trivial always-alarm F1 of each stream is in Table~1 of the "
     "manuscript.",
     "tab:s_f1", "f1"),
    ("WindowDiff of the fused detector at the calibrated threshold, under "
     "the standard definition given at the start of this section, with "
     "64-bit accumulators and $k$ equal to half the mean reference segment "
     "length. Lower is better.",
     "tab:s_wd", "wd"),
    ("Blocked ROC-AUC of the fused detector (fraction of concordant "
     "attack--benign pairs that fall inside the same block of 10{,}000 "
     "records), the drift control of Section~5.2 of the manuscript. A block "
     "contributes only if it holds records of both classes, which depends "
     "on the stream and not on the configuration; the blocks that do, out "
     "of the total, are {loc_blocks}.",
     "tab:s_loc", "auc_loc_fus"),
    ("Average precision of the fused detector on the evaluation segment, "
     "that is the step-wise area under its precision--recall curve. The "
     "trivial value is the attack prevalence of each stream.",
     "tab:s_auprc", "auprc"),
    ("ROC-AUC of the centroid detector (KMD) alone, at the same calibration "
     "and over the same evaluation segment as the fused detector. The fused "
     "score of Table~5 of the manuscript is the average of the robust "
     "z-scores of this detector and the next, fixed in advance; no detector "
     "is selected per configuration.",
     "tab:s_kmd", "auroc_kmd"),
    ("ROC-AUC of the distribution detector (MVD) alone. With "
     "Table~\\ref{tab:s_kmd} this gives the per-cell gap between the fixed "
     "fusion and each of its two members, which Section~4.4.1 of the "
     "manuscript promises and Section~6 summarises with two "
     "Hodges--Lehmann estimates.",
     "tab:s_mvd", "auroc_mvd"),
    ("False-positive rate of the fused detector at the calibrated "
     "threshold, the 99th percentile of the calibration segment. This is "
     "the operating point at which every F1 of Table~\\ref{tab:s_f1} is "
     "computed, and it is what a deployment bears: a stream with a high "
     "value here raises an alarm on that fraction of its benign records.",
     "tab:s_fpr", "fpr"),
    ("Alarm rate of the fused detector at the calibrated threshold: the "
     "fraction of evaluated records above it, benign and malicious "
     "together. The rule puts it near 1\\% on the calibration segment by "
     "construction, so its distance from that value on the evaluation "
     "segment measures how far the evaluated traffic has drifted from the "
     "calibrated one.",
     "tab:s_alarm", "alarm_rate"))
CAPTION_ATTACK: str = (
    "Recall per attack type at the calibrated threshold, averaged over the "
    "22 runs of the stream: two seeds for each of the eight context-aware "
    "configurations, four runs of the per-record MLP (its two paradigm runs "
    "per seed are identical) and one run of each of the two deterministic "
    "configurations. The column therefore weights the configurations "
    "unequally: it differs at the third decimal from the unweighted mean of "
    "the eleven columns of Tables~\\ref{tab:s_attack_cfg_a} "
    "and~\\ref{tab:s_attack_cfg_b} in 26 of the 38 rows, by at most 0.091 "
    "(Heartbleed, 11 records). ``Records'' is the number of records of that "
    "attack type in the evaluation segment, which is a property of the "
    "stream. Attack types come from the published labels of each stream; "
    "the six KNAD captures contain a single attack type each. "
    "FTP-BruteForce on CSE-CIC-IDS2018 has no row because none of its "
    "193{,}360 records is evaluated (Table~\\ref{tab:s_calib_attack}).")
# Recall per attack type and configuration: the caption takes the
# stream group, the column legend and the suffix of the label.
CAPTION_ATTACK_CONFIG: str = (
    "Recall per attack type and configuration at the calibrated threshold, "
    "on %s, averaged over the two seeds only (C10 and C11 are deterministic "
    "and have a single run). Columns: %s. Record counts per attack type are "
    "in Table~\\ref{tab:s_attack}, whose single column averages the 22 runs "
    "and is therefore not the unweighted mean of the eleven columns here. "
    "Every recall is obtained at the false-positive rate that "
    "Table~\\ref{tab:s_fpr} gives for the same stream and configuration at "
    "this threshold.")
# Those two tables: suffix of the label, slice of the stream order and
# the group as the caption names it.
ATTACK_CONFIG_GROUPS: tuple[tuple[str, int, int, str], ...] = (
    ("a", 0, 7, "the six packet-level captures and ACI-IoT-2023"),
    ("b", 7, 9, "CIC-IDS2017 and CSE-CIC-IDS2018"))
# Opening of those two floats, which are set smaller than the rest so
# that the eleven columns fit the page width.
ATTACK_CONFIG_BEGIN: str = (r"\begin{table*}[ht]\centering\scriptsize"
                            r"\setlength{\tabcolsep}{2.5pt}")
# The caption of the per-seed tables takes the stream group and the
# closing sentences of that table.
CAPTION_SEEDS: str = (
    "ROC-AUC of the fused detector per seed on %s. Each cell gives seed~A "
    "(1234) and seed~B (5678); the manuscript reports their mean. %s")
# Per-seed tables: suffix of the label, slice of the stream order, the
# group as the caption names it and the closing sentences of the
# caption. The split in two is typographic: with the nine columns of
# "0.996/0.994" the longest row goes over 130 characters and overflows
# the page width.
SEED_GROUPS: tuple[tuple[str, int, int, str, str], ...] = (
    ("a", 0, 6, "the six packet-level captures",
     "The seed-noise reference used throughout is $\\sqrt{2}$ times the "
     "between-seed standard deviation pooled over the nine stochastic "
     "configurations of the stream; the spread of one cell is not used as a "
     "threshold. The KitNET-style dA and STAT are deterministic and were "
     "run once per stream, so their cell repeats one value."),
    ("b", 6, 9, "the three flow corpora",
     "The seed-noise reference and the two deterministic rows are as in "
     "Table~\\ref{tab:s_seeds_a}."))
CAPTION_SENS: str = (
    "Pre-specified sensitivity analyses, by stream: median change over the "
    "eleven configurations. The calibration fraction is compared within its "
    "own experiment, whose 10\\% level uses the calibration objects of the "
    "main regime: the three fractions are scored on a common evaluation "
    "region that starts after the first 20\\% of the execution stream, so "
    "that they rank the same records, and that region is why the 10\\% "
    "level is not the main result. The threshold percentile cannot change "
    "ROC-AUC, which is threshold-independent, so only its effect on the "
    "calibrated F1 is reported.")
CAPTION_PERDAY: str = (
    "Results per day for the two CIC corpora, averaged over the 22 runs of "
    "the stream, as in Table~\\ref{tab:s_attack}. ``Records'' counts every "
    "evaluated record of the day; ``Attack'' and the three metrics are "
    "computed after setting aside the 800 records that follow each day "
    "change, so the attack column of CSE-CIC-IDS2018 adds up to 781 fewer "
    "records than the rows of that corpus in Table~\\ref{tab:s_attack}. "
    "ROC-AUC and AUC$_t$ are reported only for the days that are eligible "
    "under the pre-specified rule of the analysis plan: at least three "
    "attack episodes, with attack runs separated by fewer than 10{,}000 "
    "records merged into one episode, and at least 1{,}000 benign records, "
    "both counted on the records that remain after that guard. Monday of "
    "CIC-IDS2017 has no attack record. The other four rows with dashes are "
    "days of CSE-CIC-IDS2018 that hold both classes and a single episode; "
    "they carry 1{,}635{,}938 of the 1{,}689{,}364 attack records that the "
    "column holds for that corpus, so its per-day ROC-AUC rests on three "
    "days with 3.2\\% of them. For the ineligible days the table gives only "
    "the false-positive rate at the calibrated threshold; on 2018-02-14 it "
    "is 0.000 because no run raises an alarm on any evaluated record of "
    "that day, attack or benign (Section~\\ref{sec:s_calib_attack}). "
    "AUC$_t$ is the ROC-AUC of ranking the records of that day by their "
    "position in the stream; on Friday of CIC-IDS2017 it exceeds the mean "
    "ROC-AUC of the fused detector.")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                AUXILIARY                                  #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def _caption_end(body: str = "", start: int = 0) -> int:
    """
    Find where the argument of a ``\\caption`` command ends.

    The caption of a supplementary table is one long sentence that
    contains braces of its own (``10{,}000``, ``$\\Delta$F1``), so the
    closing brace cannot be found by searching for the first ``}``. The
    scanner walks the text keeping the brace depth, and skips the
    character after a backslash so that an escaped brace never changes
    the depth.

    Note: when the braces are unbalanced the scanner stops at the end of
    the float and the caller takes the rest of the float as the caption,
    which is what the original did.

    :param body: Whole text of one float environment.
    :type body: str
    :param start: Index just after the opening brace of the caption,
        that is, the end of the ``\\caption{`` match; 0 or greater.
    :type start: int
    :return: Index just after the closing brace of the caption, so that
        ``body[match.start():index]`` is the whole command.
    :rtype: int
    """
    i = start
    depth = 1
    while i < len(body) and depth:
        c = body[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    return i


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                              MAIN FUNCTION                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def escape_cell(s: Any = "") -> str:
    """
    Escape one dataset or attack-type label for a LaTeX table cell.

    The labels printed in the supplementary tables come from the
    published label files of the streams, not from the code, and they
    carry characters LaTeX would choke on.

    The attack-type labels of CIC-IDS2017 hold the short dash of
    cp1252 (0x96), which becomes U+FFFD when the file is read as UTF-8
    and printed as a diamond with a question mark ("Web Attack <?> Brute
    Force"). It is translated into the LaTeX short dash, which is what
    the original file meant to say.

    Note: this is not ``kitnetzday.utils.tex_escape``, and it must not be
    replaced by it. That function maps a backslash to
    ``\\textbackslash{}`` and also escapes ``#``, ``$``, ``{``, ``}``,
    ``~`` and ``^``; this one maps a backslash to a slash and leaves the
    rest alone, which is what produced the submitted tables.

    :param s: Value to escape; anything not a string is passed through
        ``str`` first.
    :type s: Any
    :return: The escaped text, ready to be pasted into a table cell.
    :rtype: str
    """
    out = str(s)
    for a, b in _CELL_REPLACEMENTS:
        out = out.replace(a, b)
    return out


def caption_below(text: str = "",
                  logger: logging.Logger | None = None) -> str:
    """
    Move the caption of every float below its tabular.

    The supplement sets the table caption under the table. Instead of
    touching the five places that emit a ``\\caption``, a single pass is
    made over the generated text: inside every ``table`` or ``table*``
    environment, the ``\\caption{...}`` command and the ``\\label{...}``
    that follows it are cut out and pasted after the last
    ``\\end{tabular}`` of that same float.

    The pass is idempotent: a float whose caption already sits after the
    tabular is left untouched, and so is a float with no caption or no
    tabular. Floats are rewritten from the last to the first so that the
    indices of the matches stay valid.

    Note: Elsevier asks for the table caption above the table; reverting
    means not calling this function. Note as well that the last
    ``\\end{tabular}`` of the float is used, so a float with two tabulars
    would get its caption after the second one.

    :param text: Whole LaTeX text of the supplementary tables, floats
        separated by blank lines.
    :type text: str
    :param logger: Logger created by the calling script, or None to stay
        silent.
    :type logger: logging.Logger | None
    :return: The same text with the captions moved below the tabulars.
    :rtype: str
    :raises TypeError: If text is not a string.
    """
    rewritten = list()
    message = ""
    body = ""
    if not isinstance(text, str):
        message = ("[caption_below] Input error: the parameter text must "
                   f"be a string, not {type(text).__name__}.")
        if logger is not None:
            logger.error(message)
        raise TypeError(message)
    for float_match in _FLOAT_RE.finditer(text):
        body = float_match.group(0)
        caption_match = _CAPTION_RE.search(body)
        if not caption_match:
            continue
        end_caption = _caption_end(body, caption_match.end())
        caption = body[caption_match.start():end_caption]
        label_match = _LABEL_RE.match(body[end_caption:])
        label = label_match.group(0) if label_match else ""
        end_all = end_caption + len(label)
        last_tabular = None
        for tabular_match in _TABULAR_END_RE.finditer(body):
            last_tabular = tabular_match
        if last_tabular is None or last_tabular.start() < end_all:
            # No tabular, or the caption already sits below it.
            continue
        rewritten.append(
            (float_match.start(), float_match.end(),
             body[:caption_match.start()].rstrip("\n") + "\n"
             + body[end_all:last_tabular.end()].lstrip("\n") + "\n"
             + caption + label.strip() + body[last_tabular.end():]))
    for start, end, new_body in reversed(rewritten):
        text = text[:start] + new_body + text[end:]
    return text


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
