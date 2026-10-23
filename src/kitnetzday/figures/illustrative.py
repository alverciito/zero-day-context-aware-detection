# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         src/kitnetzday/figures/illustrative.py                    #
#   Version:      1.1.1                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Illustrative figures of the manuscript and supplement,   #
#                 redrawn under the revised evaluation regime.              #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import LogLocator, NullLocator

from kitnetzday.figures.stream_context import (
                                            MAIN_QUANTILE,
                                            METRICS_API,
                                            StreamContext,
)
from kitnetzday.figures.stream_panels import BAND_RULE, StreamPanels
from kitnetzday.utils.loggable import Loggable

# Metrics attributes needed beyond the ones StreamContext needs.
CURVE_API: tuple[str, ...] = ("auroc", "clf_curve", "curve_metrics")
# Style of these figures. It is NOT the style of the main figures
# (kitnetzday.naming.RC_PARAMS): they are strips with their own type
# sizes. The font family is pinned and the fonts are embedded as
# TrueType (Type 42), not as Type 3.
RC_ILLUSTRATIVE: dict[str, object] = {
    "figure.dpi": 120, "savefig.bbox": "tight", "font.size": 9,
    "axes.titlesize": 8.5, "axes.labelsize": 9, "legend.fontsize": 8,
    "font.family": ["DejaVu Sans"], "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42, "ps.fonttype": 42}
# Runs the figures are drawn from: the submitted ones came from the
# oracle protocol of the submission, these from the revised runs.
ILLUSTRATIVE_JOBS: dict[str, str] = {
    "frame_conv1d_aci": ("T1-A-ACI__aci-iot-2023__conv1d__arT__"
                         "corrected.v1__L800__k4__sA__r1"),
    "frame_mha_ssdp": ("T1-A-KNAD__knad-ssdp_flood__mha__arF__"
                       "corrected.v1__L800__k4__sA__r1"),
    "dxt_conv1d_aw": ("T1-A-KNAD__knad-active_wiretap__conv1d__arT__"
                      "corrected.v1__L800__k4__sA__r1")}
# File name of every output, with the "rev_" prefix that tells it
# apart from the submitted figure it replaces.
FILE_NAMES: dict[str, str] = {
    "frame_conv1d_aci": "rev_frame_conv1d_aci.pdf",
    "roc_conv1d_aci": "rev_roc_conv1d_aci.pdf",
    "frame_mha_ssdp": "rev_frame_mha_ssdp.pdf",
    "dxt_conv1d_aw": "rev_dxt_conv1d_aw.pdf"}
# Titles, verbatim: frame_conv1d_aci is printed in Figure 2 of the
# manuscript, frame_mha_ssdp and dxt_conv1d_aw in Figures S1 and S2 of
# the supplement, and roc_conv1d_aci is printed in neither.
TITLES: dict[str, str] = {
    "frame_conv1d_aci": (
        "CONV1D-AR detection frame, ACI-IoT-2023\n(revised evaluation "
        "regime: chronological stream, causal detectors, label-free "
        "threshold)"),
    "roc_conv1d_aci": (
        "CONV1D-AR on ACI-IoT-2023 under the revised evaluation "
        "regime\n(evaluation segment only, original per-record labels, "
        "causal detectors)"),
    "frame_mha_ssdp": (
        "MHA-TSR detection frame, KNAD SSDP Flood\n(revised evaluation "
        "regime: causal detectors, label-free threshold)"),
    "dxt_conv1d_aw": (
        "Distribution detector score, CONV1D-AR on Active Wiretap\n"
        "(revised evaluation regime: causal filter, no future "
        "records)")}
X_LABEL: str = "record index in the stream"   # shared x axis
# Legend of a frame figure, in the order of its three patches.
FRAME_LEGEND: tuple[str, ...] = ("training (never evaluated)",
                                 "calibration (unlabelled)",
                                 "records labelled as attack")
# Panels of the distribution figure: series key -> row label.
DXT_PANELS: tuple[tuple[str, str], ...] = (
    ("MVD_raw", "(a) raw"), ("MVD_f", "(b) median filtered"))
# Logarithmic y axis of the distribution figure: quantiles of the
# positive values, floor relative to the top, and head room.
DXT_QUANTILES: tuple[float, float] = (0.02, 0.9995)
DXT_FLOOR: float = 1e-8
DXT_HEADROOM: float = 1.6
# Panels of the ROC figure: series key -> panel title.
ROC_PANELS: tuple[tuple[str, str], ...] = (
    ("MVD_f", "Distribution detector"),
    ("KMD_f", "Centroid detector"))
# Figure sizes in inches. Every figure is drawn at the width it is
# included at, so that the printed lettering stays at 7 pt or more: the
# first frame goes in the manuscript at 170 mm, and the MHA frame and
# the distribution figure in the supplementary material, whose text is
# 390 pt (137 mm) wide. With
# the tight bounding box the PDF comes out a little narrower than the
# canvas: these widths give about 456 pt and 380 to 400 pt.
FRAME_SIZE: tuple[float, float] = (7.2, 2.6)
FRAME_SIZE_SUPPLEMENT: tuple[float, float] = (6.1, 2.6)
DXT_SIZE: tuple[float, float] = (5.9, 4.2)
ROC_SIZE: tuple[float, float] = (5.25, 3.5)
# Type sizes of the pieces that do not take the style defaults.
DXT_LABEL_SIZE: float = 8.5
ROC_TITLE_SIZE: float = 8.5
ROC_LEGEND_SIZE: float = 7.5


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                MAIN CLASS                                 #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class IllustrativeFigures(Loggable):
    """
    The illustrative figures of the manuscript, redrawn as measured.

    The figures of the submission (two detection frames, the
    distribution-score panel and their ROC curves) came from the oracle
    protocol of the submission: centred filters reading fifty thousand
    future records, extended labels, metrics over the whole stream
    and, in the flow corpora, three hundred thousand benign copies
    prepended. This class redraws them from the revised runs through
    the frozen metrics module, so that what the reader sees is what the
    tables report: causal detectors, a fit without labels and the
    calibrated threshold. No new metric is computed; the fit and
    transform of that module are replayed on what the runner wrote.

    The metrics module is injected instead of imported: it is frozen
    code living outside the package, and only the script in ``bin/``
    knows where it is. That script must also select the non-interactive
    matplotlib backend ("Agg") first, as the original did. Building the
    class applies the style of these figures to ``rcParams``.

    :param metrics: Frozen metrics module, with the attributes of
        ``METRICS_API`` and ``CURVE_API``.
    :type metrics: ModuleType
    :param results_root: Directory with one sub-directory per job.
    :type results_root: str | Path
    :param out_dir: Directory the PDFs go to; it is created.
    :type out_dir: str | Path
    :param jobs: Figure key -> job id, keyed as ``ILLUSTRATIVE_JOBS``.
    :type jobs: Mapping[str, str]
    :param quantile: Quantile of the calibrated threshold drawn on the
        ROC curves, as the fit stores it ("0.99" in the paper).
    :type quantile: str
    :param panels: Drawing primitives; None builds them on out_dir.
    :type panels: StreamPanels | None
    :param rc_params: matplotlib settings applied on construction.
    :type rc_params: Mapping[str, object]
    :param logger: Logger of the calling script, or None to stay quiet.
    :type logger: logging.Logger | None
    :raises TypeError: If logger is not a logging.Logger or None, if
        metrics is not a module, or if panels is not a StreamPanels.
    :raises ValueError: If metrics lacks part of the API, if jobs is
        missing a key, or if quantile is not a non-empty string.
    :raises OSError: If the output directory cannot be created.
    """

    def __init__(self,
                 metrics: ModuleType,
                 results_root: str | Path,
                 out_dir: str | Path,
                 jobs: Mapping[str, str] = ILLUSTRATIVE_JOBS,
                 quantile: str = MAIN_QUANTILE,
                 panels: StreamPanels | None = None,
                 rc_params: Mapping[str, object] = RC_ILLUSTRATIVE,
                 logger: logging.Logger | None = None) -> None:
        """
        Check the settings, prepare the output and set the style.

        :param metrics: The frozen metrics module.
        :type metrics: ModuleType
        :param results_root: Directory with one sub-directory per job.
        :type results_root: str | Path
        :param out_dir: Directory the PDFs go to.
        :type out_dir: str | Path
        :param jobs: Figure key -> job id.
        :type jobs: Mapping[str, str]
        :param quantile: Quantile of the calibrated threshold.
        :type quantile: str
        :param panels: Drawing primitives, or None to build them.
        :type panels: StreamPanels | None
        :param rc_params: matplotlib settings applied on construction.
        :type rc_params: Mapping[str, object]
        :param logger: Logger created by the calling script, or None.
        :type logger: logging.Logger | None
        :raises TypeError: If a parameter has the wrong type.
        :raises ValueError: If the API or jobs are incomplete.
        :raises OSError: If the directory cannot be created.
        """
        super().__init__(logger=logger)
        missing = list()
        if not isinstance(metrics, ModuleType):
            self._error("[IllustrativeFigures.__init__] Input error: "
                        "the parameter metrics must be the frozen "
                        "metrics module, not "
                        f"{type(metrics).__name__}.", TypeError)
        missing = [a for a in METRICS_API + CURVE_API
                   if not hasattr(metrics, a)]
        if missing:
            self._error("[IllustrativeFigures.__init__] Input error: "
                        "the parameter metrics must provide "
                        f"{list(METRICS_API + CURVE_API)}, and it is "
                        f"missing {missing}.")
        missing = [k for k in ILLUSTRATIVE_JOBS if k not in jobs]
        if missing:
            self._error("[IllustrativeFigures.__init__] Input error: "
                        "the parameter jobs must hold the figure keys "
                        f"{sorted(ILLUSTRATIVE_JOBS)}, and it is "
                        f"missing {missing}.")
        if not isinstance(quantile, str) or not quantile:
            self._error("[IllustrativeFigures.__init__] Input error: "
                        "the parameter quantile must be a non-empty "
                        f"string, not {quantile!r}.")
        if panels is not None and not isinstance(panels, StreamPanels):
            self._error("[IllustrativeFigures.__init__] Input error: "
                        "the parameter panels must be a StreamPanels "
                        f"or None, not {type(panels).__name__}.",
                        TypeError)
        self.metrics = metrics
        self.results_root = Path(results_root)
        self.dir = Path(out_dir)
        self.jobs = dict(jobs)
        self.quantile = quantile
        self.rc_params = dict(rc_params)
        self.panels = panels or StreamPanels(self.dir,
                                             logger=self.logger)
        # The original script wrote the global rcParams at import
        # time; here they are set on construction. They stay active
        # while the figures are saved, which is when matplotlib reads
        # pdf.fonttype.
        plt.rcParams.update(self.rc_params)

    def context(self, key: str) -> StreamContext:
        """
        Prepare the run behind one figure.

        :param key: Figure key, as in ``ILLUSTRATIVE_JOBS``.
        :type key: str
        :return: The prepared run.
        :rtype: StreamContext
        :raises TypeError: If key is not a string.
        :raises KeyError: If key is not a known figure key.
        :raises FileNotFoundError: If the run is not in results_root.
        """
        if not isinstance(key, str):
            self._error("[IllustrativeFigures.context] Input error: "
                        "the parameter key must be a string, not "
                        f"{type(key).__name__}.", TypeError)
        if key not in self.jobs:
            self._error("[IllustrativeFigures.context] Input error: "
                        "the parameter key must be one of "
                        f"{sorted(self.jobs)}, not {key!r}.", KeyError)
        return StreamContext(self.jobs[key], self.results_root,
                             self.metrics, logger=self.logger)

    def frame(self,
              ctx: StreamContext,
              title: str,
              name: str,
              size: tuple[float, float] = FRAME_SIZE) -> Path:
        """
        Draw the detection frame of one run: its raw score stream.

        One dot per drawn record, with training, calibration and the
        attack episodes shaded behind: what the tables measure.

        :param ctx: Prepared run, as ``context`` returns it.
        :type ctx: StreamContext
        :param title: Title of the panel, in academic English.
        :type title: str
        :param name: File name inside the output directory.
        :type name: str
        :param size: Canvas (width, height) in inches, both strictly
            positive; ``FRAME_SIZE`` for the manuscript and
            ``FRAME_SIZE_SUPPLEMENT`` for the supplementary material.
        :type size: tuple[float, float]
        :return: The path written.
        :rtype: Path
        :raises TypeError: If ctx is not a StreamContext.
        :raises ValueError: If size is not a pair of positive numbers.
        :raises OSError: If the file cannot be written.
        """
        index = np.empty(0, dtype=np.int64)
        handles = list()
        self._require_context("frame", ctx)
        if len(size) != 2 or not all(float(v) > 0.0 for v in size):
            self._error("[IllustrativeFigures.frame] Input error: the "
                        "parameter size must be a pair (width, height) "
                        f"of strictly positive inches, not {size!r}.")
        index = self.panels.subsample(ctx.n_records)
        fig, ax = plt.subplots(figsize=(float(size[0]), float(size[1])))
        self.panels.mark_regions(ax, ctx)
        ax.plot(index, ctx.scores[index], ".", ms=0.6, color="black",
                alpha=0.45, rasterized=True)
        ax.set_xlabel(X_LABEL)
        ax.set_ylabel(r"$\log_{10}(1+s)$")
        ax.set_title(title)
        handles = [Rectangle((0, 0), 1, 1, fc="tab:blue", alpha=0.12),
                   Rectangle((0, 0), 1, 1, fc="tab:green", alpha=0.12),
                   Rectangle((0, 0), 1, 1, fc="tab:red", alpha=0.18)]
        ax.legend(handles, list(FRAME_LEGEND), loc="upper left",
                  framealpha=0.9, ncol=1)
        return self.panels.save(fig, name)

    def distribution(self,
                     ctx: StreamContext,
                     title: str,
                     name: str) -> Path:
        """
        Draw the distribution detector score, raw and median filtered.

        Two stacked panels on a shared record axis: the score as the
        detector produces it, and the same score after the causal
        median filter of the revised regime. Both use a logarithmic y
        axis cut at the quantiles of the positive values.

        :param ctx: Prepared run, as ``context`` returns it.
        :type ctx: StreamContext
        :param title: Title of the figure, placed on the upper panel.
        :type title: str
        :param name: File name inside the output directory.
        :type name: str
        :return: The path written.
        :rtype: Path
        :raises TypeError: If ctx is not a StreamContext.
        :raises OSError: If the file cannot be written.
        """
        index = np.empty(0, dtype=np.int64)
        values = np.empty(0)
        positive = np.empty(0)
        low = 0.0
        high = 0.0
        self._require_context("distribution", ctx)
        start = ctx.train_end
        spans = ctx.attack_spans()
        index = self.panels.subsample(
            len(ctx.series[DXT_PANELS[0][0]]))
        fig, axes = plt.subplots(2, 1, figsize=DXT_SIZE, sharex=True)
        for ax, (key, label) in zip(axes, DXT_PANELS):
            values = ctx.series[key][index]
            ax.plot(index + start, values, lw=0.4, color="black",
                    rasterized=True)
            ax.axvspan(ctx.calibration[0], ctx.calibration[1],
                       color="tab:green", alpha=0.14, lw=0)
            for a, b in spans:
                ax.axvspan(a, b, color="tab:red", alpha=0.12, lw=0)
            ax.set_ylabel(label + "\n$D(x,t)$",
                          fontsize=DXT_LABEL_SIZE)
            ax.set_xlim(start, ctx.n_records)
            positive = values[values > 0]
            if len(positive):
                low = float(np.quantile(positive, DXT_QUANTILES[0]))
                high = float(np.quantile(positive, DXT_QUANTILES[1]))
                ax.set_yscale("log")
                ax.set_ylim(max(low, high * DXT_FLOOR),
                            high * DXT_HEADROOM)
                ax.yaxis.set_major_locator(LogLocator(numticks=5))
                ax.yaxis.set_minor_locator(NullLocator())
        axes[1].set_xlabel(X_LABEL)
        axes[0].set_title(title)
        return self.panels.save(fig, name)

    def roc(self,
            ctx: StreamContext,
            title: str,
            name: str) -> dict[str, dict[str, float | None]]:
        """
        Draw the ROC curves of the two detectors on the evaluation.

        One panel per detector over the evaluation segment, with the
        original per-record labels, plus the point the calibrated
        threshold reaches: the curve is what the detector could do,
        the red dot what the label-free threshold does.

        Note: the ROC-AUC is formatted into the panel title with
        three decimals, so a run whose evaluation holds a single class,
        for which the metrics module reports no ROC-AUC, raises a
        TypeError while formatting; and the EER comes from a second
        call to ``curve_metrics``. Both are kept as submitted.

        :param ctx: Prepared run, as ``context`` returns it.
        :type ctx: StreamContext
        :param title: Title above both panels.
        :type title: str
        :param name: File name inside the output directory.
        :type name: str
        :return: Series key -> {"auroc": v, "eer": v} on the
            evaluation rows, with None where the metric is undefined.
        :rtype: dict[str, dict[str, float | None]]
        :raises TypeError: If ctx is not a StreamContext.
        :raises OSError: If the file cannot be written.
        """
        out: dict[str, dict[str, float | None]] = dict()
        values = np.empty(0)
        alarm = np.empty(0, dtype=bool)
        area: float | None = None
        self._require_context("roc", ctx)
        labels = ctx.labels[ctx.eval_rows]
        n_benign = int((~labels).sum()) or 1
        n_attack = int(labels.sum()) or 1
        fig, axes = plt.subplots(1, 2, figsize=ROC_SIZE,
                                 layout="constrained")
        for ax, (key, panel) in zip(axes, ROC_PANELS):
            values = ctx.eval_series(key)
            fps, tps, _ = self.metrics.clf_curve(values, labels)
            area = self.metrics.auroc(values, labels)
            out[key] = {"auroc": area,
                        "eer": self.metrics.curve_metrics(
                            values, labels).get("eer")}
            ax.plot(fps / fps[-1] if fps[-1] else fps,
                    tps / tps[-1] if tps[-1] else tps, lw=1.2,
                    color="black")
            ax.plot([0, 1], [0, 1], ls=":", lw=0.8, color="grey")
            alarm = values > ctx.threshold(key, self.quantile)
            ax.plot([float((alarm & ~labels).sum()) / n_benign],
                    [float((alarm & labels).sum()) / n_attack], "o",
                    ms=4, color="tab:red",
                    label="calibrated threshold (P99)")
            ax.set_title(f"{panel}\nROC-AUC = {area:.3f}",
                         fontsize=ROC_TITLE_SIZE)
            ax.set_xlabel("false positive rate")
            ax.set_ylabel("true positive rate")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.legend(loc="lower right", fontsize=ROC_LEGEND_SIZE)
        fig.suptitle(title, fontsize=ROC_TITLE_SIZE)
        # Room between the title above both panels and the panel
        # titles, which overlapped without it.
        fig.get_layout_engine().set(h_pad=0.08, hspace=0.10)
        self.panels.save(fig, name)
        return out

    def build_all(self) -> dict[str, Any]:
        """
        Build the four outputs and collect what they are made of.

        The runs are drawn in the order of the original script, which
        is the order of the keys of the summary: the ACI-IoT-2023
        frame and its curves, the SSDP frame, the Wiretap panel.

        :return: Summary of the run, one entry per output: the job
            id, the stream length N, the end T of training and, by
            output, the calibration interval, the evaluation row
            count, the error bands or the curve metrics.
        :rtype: dict[str, Any]
        :raises FileNotFoundError: If one of the three runs is missing.
        :raises OSError: If a file cannot be written.
        """
        out: dict[str, Any] = dict()
        centres = list()
        counts = list()
        aci = self.context("frame_conv1d_aci")
        self.frame(aci, TITLES["frame_conv1d_aci"],
                   FILE_NAMES["frame_conv1d_aci"])
        out["frame_conv1d_aci"] = {
            "job": self.jobs["frame_conv1d_aci"], "N": aci.n_records,
            "T": aci.train_end, "calib": list(aci.calibration),
            "n_E": int(len(aci.eval_rows))}
        out["roc_conv1d_aci"] = self.roc(
            aci, TITLES["roc_conv1d_aci"],
            FILE_NAMES["roc_conv1d_aci"])
        ssdp = self.context("frame_mha_ssdp")
        self.frame(ssdp, TITLES["frame_mha_ssdp"],
                   FILE_NAMES["frame_mha_ssdp"],
                   size=FRAME_SIZE_SUPPLEMENT)
        centres, counts = self.panels.bands(
            ssdp.scores[ssdp.eval_rows])
        out["frame_mha_ssdp"] = {
            "job": self.jobs["frame_mha_ssdp"], "N": ssdp.n_records,
            "T": ssdp.train_end, "band_centres_log10_1ps": centres,
            "band_counts": counts, "rule": BAND_RULE}
        wiretap = self.context("dxt_conv1d_aw")
        self.distribution(wiretap, TITLES["dxt_conv1d_aw"],
                          FILE_NAMES["dxt_conv1d_aw"])
        out["dxt_conv1d_aw"] = {
            "job": self.jobs["dxt_conv1d_aw"],
            "N": wiretap.n_records, "T": wiretap.train_end,
            "calib": list(wiretap.calibration)}
        return out

    def write_summary(self, path: str | Path, summary: Any) -> Path:
        """
        Write the summary of the run as JSON, indented by one space.

        :param path: Path of the summary file, for example
            ``figuras_ilustrativas.json``.
        :type path: str | Path
        :param summary: Structure to serialize, from ``build_all``.
        :type summary: Any
        :return: The path written.
        :rtype: Path
        :raises TypeError: If summary holds an unserializable value.
        :raises OSError: If the file cannot be written.
        """
        target = Path(path)
        target.write_text(json.dumps(summary, indent=1),
                          encoding="utf-8")
        return target

    def _require_context(self, method: str, ctx: Any) -> None:
        """
        Check that a prepared run really is one.

        :param method: Calling method, for the message.
        :type method: str
        :param ctx: Value received by the caller.
        :type ctx: Any
        :raises TypeError: If ctx is not a StreamContext.
        """
        if not isinstance(ctx, StreamContext):
            self._error(f"[IllustrativeFigures.{method}] Input error: "
                        "the parameter ctx must be a StreamContext, "
                        f"not {type(ctx).__name__}.", TypeError)

    def __repr__(self) -> str:
        """
        Show the output directory and the runs that are drawn.

        :return: Text like "IllustrativeFigures(dir='revised', jobs=3,
            quantile='0.99')".
        :rtype: str
        """
        return (f"IllustrativeFigures(dir={str(self.dir)!r}, "
                f"jobs={len(self.jobs)}, "
                f"quantile={self.quantile!r})")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
