# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         src/kitnetzday/figures/stream_panels.py                   #
#   Version:      1.0.1                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Drawing primitives the illustrative stream figures        #
#                 share: sampling, shading, error bands and saving.         #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from kitnetzday.figures.stream_context import StreamContext
from kitnetzday.utils.loggable import Loggable

# At most this many records are drawn per panel; beyond it the stream is
# sampled on an even grid, which is what keeps these figures at a few
# hundred kilobytes instead of a few hundred megabytes.
SUBSAMPLE_MAX: int = 60000
# Error bands of a frame figure: modes of a histogram of this many bins,
# at least one twentieth of the range apart, at most three of them.
BAND_BINS: int = 240
BAND_MAX: int = 3
BAND_SEPARATION: float = 20.0
# Rule written into the summary of the run, verbatim and in Spanish: it
# is an output string of the submitted pipeline. Note: it names 240 bins
# literally, so it stops describing the computation if band_bins is
# changed from its default.
BAND_RULE: str = ("modos del histograma de 240 bins de log10(1+s) en "
                  "E, separados >= 1/20 del rango")
# Shading of the regions of the stream, as submitted.
TRAIN_COLOUR: str = "tab:blue"
CALIB_COLOUR: str = "tab:green"
ATTACK_COLOUR: str = "tab:red"
TRAIN_ALPHA: float = 0.12
CALIB_ALPHA: float = 0.12
ATTACK_ALPHA: float = 0.18


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                MAIN CLASS                                 #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class StreamPanels(Loggable):
    """
    The primitives the three illustrative stream figures share.

    The figures of the revision draw streams of up to four million
    records on a seven-inch strip, so none of them is drawn record by
    record: they share one even sampling grid, one way of shading the
    training, calibration and attack regions, one rule for the error
    bands of a score distribution, and one way of writing the PDF.
    Keeping the four in one place is what makes the three figures
    comparable to each other.

    :param out_dir: Directory the PDFs go to; it is created.
    :type out_dir: str | Path
    :param subsample_max: Most records drawn per panel; positive.
    :type subsample_max: int
    :param band_bins: Bins of the histogram of the error bands;
        positive.
    :type band_bins: int
    :param logger: Logger created by the calling script, or None to
        stay silent.
    :type logger: logging.Logger | None
    :raises TypeError: If logger is not a logging.Logger or None.
    :raises ValueError: If subsample_max or band_bins is not a positive
        integer.
    :raises OSError: If the output directory cannot be created.
    """

    def __init__(self,
                 out_dir: str | Path,
                 subsample_max: int = SUBSAMPLE_MAX,
                 band_bins: int = BAND_BINS,
                 logger: logging.Logger | None = None) -> None:
        """
        Check the settings and create the output directory.

        :param out_dir: Directory the PDFs go to.
        :type out_dir: str | Path
        :param subsample_max: Most records drawn per panel; positive.
        :type subsample_max: int
        :param band_bins: Bins of the band histogram; positive.
        :type band_bins: int
        :param logger: Logger created by the calling script, or None.
        :type logger: logging.Logger | None
        :raises TypeError: If logger is not a logging.Logger or None.
        :raises ValueError: If a setting is not a positive integer.
        :raises OSError: If the directory cannot be created.
        """
        super().__init__(logger=logger)
        if isinstance(subsample_max, bool) or \
                not isinstance(subsample_max, int) or \
                subsample_max <= 0:
            self._error("[StreamPanels.__init__] Input error: the "
                        "parameter subsample_max must be a positive "
                        f"integer, not {subsample_max!r}.")
        if isinstance(band_bins, bool) or \
                not isinstance(band_bins, int) or band_bins <= 0:
            self._error("[StreamPanels.__init__] Input error: the "
                        "parameter band_bins must be a positive "
                        f"integer, not {band_bins!r}.")
        self.dir = Path(out_dir)
        self.subsample_max = subsample_max
        self.band_bins = band_bins
        self.dir.mkdir(parents=True, exist_ok=True)

    def subsample(self, n: int) -> np.ndarray:
        """
        Record indices to draw out of a panel of n records.

        Short panels are drawn whole; longer ones are sampled on an
        evenly spaced grid, truncated towards zero and deduplicated, so
        the grid stays sorted and never repeats a record.

        :param n: Records available in the panel; not negative.
        :type n: int
        :return: Indices to draw, sorted, of at most subsample_max
            entries.
        :rtype: np.ndarray
        :raises ValueError: If n is not a non-negative integer.
        """
        if isinstance(n, bool) or \
                not isinstance(n, (int, np.integer)) or n < 0:
            self._error("[StreamPanels.subsample] Input error: the "
                        "parameter n must be a non-negative integer "
                        f"number of records, not {n!r}.")
        if n <= self.subsample_max:
            return np.arange(n)
        return np.unique(np.linspace(0, n - 1, self.subsample_max)
                         .astype(np.int64))

    def mark_regions(self, ax: Axes, ctx: StreamContext) -> None:
        """
        Shade training, calibration and every attack episode.

        The reader has to be able to tell at a glance which part of the
        stream the numbers of the tables come from: the training
        segment is never evaluated, the calibration segment is seen
        without labels, and the attack episodes are what the detector
        is supposed to find. The x axis is set to the whole stream.

        :param ax: Axes of a frame figure.
        :type ax: Axes
        :param ctx: Prepared run, as ``StreamContext`` builds it.
        :type ctx: StreamContext
        :raises TypeError: If ctx is not a StreamContext.
        """
        if not isinstance(ctx, StreamContext):
            self._error("[StreamPanels.mark_regions] Input error: the "
                        "parameter ctx must be a StreamContext, not "
                        f"{type(ctx).__name__}.", TypeError)
        ax.axvspan(0, ctx.train_end, color=TRAIN_COLOUR,
                   alpha=TRAIN_ALPHA, lw=0,
                   label="training (not evaluated)")
        ax.axvspan(ctx.calibration[0], ctx.calibration[1],
                   color=CALIB_COLOUR, alpha=CALIB_ALPHA, lw=0,
                   label="calibration")
        for a, b in ctx.attack_spans():
            ax.axvspan(a, b, color=ATTACK_COLOUR, alpha=ATTACK_ALPHA,
                       lw=0)
        ax.set_xlim(0, ctx.n_records)

    def bands(self, values: np.ndarray) -> tuple[list[float],
                                                 list[int]]:
        """
        Centres and counts of the error bands of a distribution.

        The frame of a KNAD stream shows a few horizontal bands instead
        of a cloud, because the autoencoder reconstructs a handful of
        recurring record types. The bands are the tallest modes of a
        histogram of the scores, kept at least one twentieth of the
        range apart so that two bins of the same mode are not reported
        twice, and cut at three.

        Note: the modes are visited in the reverse of a stable
        ascending sort of the counts, so ties are broken by the last
        bin first; and the limit of three is checked after a candidate
        has been accepted or rejected. Both are kept as submitted.

        :param values: Scores of the evaluation rows, shape (n,), in
            log10(1 + s) units; non-empty.
        :type values: np.ndarray
        :return: The band centres in increasing order and their bin
            counts in the same order; at most three of each.
        :rtype: tuple[list[float], list[int]]
        :raises ValueError: If values is empty.
        """
        selected = list()
        counts = np.empty(0, dtype=np.int64)
        edges = np.empty(0)
        centres = np.empty(0)
        separation = 0.0
        if len(values) == 0:
            self._error("[StreamPanels.bands] Input error: the "
                        "parameter values must hold at least one "
                        "score, not an empty array.")
        counts, edges = np.histogram(values, bins=self.band_bins)
        centres = (edges[:-1] + edges[1:]) / 2
        separation = (values.max() - values.min()) / BAND_SEPARATION
        for i in np.argsort(counts)[::-1]:
            if counts[i] <= 0:
                continue
            if all(abs(centres[i] - centres[j]) >= separation
                   for j in selected):
                selected.append(i)
            if len(selected) == BAND_MAX:
                break
        return (sorted(float(centres[i]) for i in selected),
                [int(counts[i]) for i in
                 sorted(selected, key=lambda i: centres[i])])

    def save(self, fig: Figure, name: str) -> Path:
        """
        Write one figure as PDF into the output directory and close it.

        The illustrative figures are included in the manuscript by
        hand, so they get neither the PNG preview nor the LaTeX block
        of ``kitnetzday.io.Out``.

        :param fig: Finished matplotlib figure.
        :type fig: Figure
        :param name: File name inside the output directory; non-empty.
        :type name: str
        :return: The path written.
        :rtype: Path
        :raises ValueError: If name is not a non-empty string.
        :raises OSError: If the file cannot be written.
        """
        if not isinstance(name, str) or not name:
            self._error("[StreamPanels.save] Input error: the "
                        "parameter name must be a non-empty string, "
                        f"not {name!r}.")
        path = self.dir / name
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    def __repr__(self) -> str:
        """
        Show the output directory and the two sampling settings.

        :return: Text like "StreamPanels(dir='revised',
            subsample_max=60000, band_bins=240)".
        :rtype: str
        """
        return (f"StreamPanels(dir={str(self.dir)!r}, "
                f"subsample_max={self.subsample_max}, "
                f"band_bins={self.band_bins})")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
