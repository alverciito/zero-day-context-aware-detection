# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         src/kitnetzday/audit/paired_stats.py                      #
#   Version:      1.0.1                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Exact paired nonparametric statistics: median,            #
#                 Hodges-Lehmann estimate and Wilcoxon signed rank.         #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
import math
import numbers
from collections.abc import Iterable
from typing import Any

from kitnetzday.utils.loggable import Loggable

# Two-sided alpha of the Hodges-Lehmann interval: 1 - 0.9609375, which is
# the confidence level of the frozen analysis plan (config.json gives it
# rounded, as 0.961). The exact binary value is what the submitted
# numbers were computed with, so it is the default here.
HL_ALPHA: float = 0.0390625
# Up to this many non-zero differences the p value of the signed rank
# test is obtained by enumerating the 2 ** n sign patterns; above it, the
# normal approximation is used instead.
EXACT_MAX_N: int = 20


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                MAIN CLASS                                 #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class PairedStats(Loggable):
    """
    Paired statistics of one sample of differences, computed exactly.

    The retraining experiment compares the windowed model against
    the per-record one cell by cell, so every analysis is paired and the
    sample is small (a few dozen cells at most). With that many points a
    normal approximation is not credible, so this class works with the
    exact combinatorics instead: the Hodges-Lehmann estimate is the
    median of the Walsh averages, its interval comes from inverting the
    Wilcoxon signed rank statistic over the exact null distribution, and
    the p value enumerates the sign patterns while that is affordable.

    No value is dropped or winsorized, and ties keep their average rank,
    so the numbers are the ones the manuscript cites.

    :param alpha: Two-sided alpha of the Hodges-Lehmann interval, in
        (0, 1); 0.0390625 reproduces the submitted confidence level.
    :type alpha: float
    :param exact_max_n: Largest number of non-zero differences for which
        the exact enumeration of the 2 ** n sign patterns is used; above
        it the normal approximation takes over. Must be >= 0.
    :type exact_max_n: int
    :param logger: Logger created by the calling script, or None to stay
        silent.
    :type logger: logging.Logger | None
    :raises TypeError: If logger is not a logging.Logger or None, or if
        alpha or exact_max_n are not numbers.
    :raises ValueError: If alpha is not in (0, 1) or exact_max_n < 0.
    """

    def __init__(self,
                 alpha: float = HL_ALPHA,
                 exact_max_n: int = EXACT_MAX_N,
                 logger: logging.Logger | None = None) -> None:
        """
        Check the settings and store them.

        :param alpha: Two-sided alpha of the interval, in (0, 1).
        :type alpha: float
        :param exact_max_n: Threshold of the exact enumeration, >= 0.
        :type exact_max_n: int
        :param logger: Logger created by the calling script, or None.
        :type logger: logging.Logger | None
        :raises TypeError: If alpha or exact_max_n are not numbers.
        :raises ValueError: If alpha is not in (0, 1) or exact_max_n is
            negative.
        """
        super().__init__(logger=logger)
        if isinstance(alpha, bool) or not isinstance(alpha, numbers.Real):
            self._error("[PairedStats.__init__] Input error: the "
                        "parameter alpha must be a real number, not "
                        f"{type(alpha).__name__}.", TypeError)
        if not 0.0 < float(alpha) < 1.0:
            self._error("[PairedStats.__init__] Input error: the "
                        "parameter alpha must be in (0, 1), not "
                        f"{alpha}.")
        if isinstance(exact_max_n, bool) or \
                not isinstance(exact_max_n, int):
            self._error("[PairedStats.__init__] Input error: the "
                        "parameter exact_max_n must be an int, not "
                        f"{type(exact_max_n).__name__}.", TypeError)
        if exact_max_n < 0:
            self._error("[PairedStats.__init__] Input error: the "
                        "parameter exact_max_n must be >= 0, not "
                        f"{exact_max_n}.")
        self.alpha = float(alpha)
        self.exact_max_n = int(exact_max_n)

    def median(self, values: Any = tuple()) -> float | None:
        """
        Median of a sample, averaging the two middle values when even.

        :param values: Iterable of real numbers, in any order; an empty
            input gives None, which the reports print as a missing cell.
        :type values: Any
        :return: The median, or None when values is empty.
        :rtype: float | None
        :raises TypeError: If values is not an iterable of real numbers.
        """
        vals: list[float] = list()
        n = 0
        vals = sorted(self._as_floats("median", values))
        n = len(vals)
        if n == 0:
            return None
        if n % 2:
            return vals[n // 2]
        return (vals[n // 2 - 1] + vals[n // 2]) / 2.0

    def walsh(self, values: Any = tuple()) -> list[float]:
        """
        All the Walsh averages of a sample, sorted.

        The Walsh averages are the n (n + 1) / 2 pairwise means
        (x_i + x_j) / 2 with i <= j, the self-pairs included. Their
        median is the Hodges-Lehmann estimate and their order statistics
        give its interval, so both are built on this list.

        :param values: Iterable of real numbers, the paired differences.
        :type values: Any
        :return: The Walsh averages in increasing order, of length
            n (n + 1) / 2; empty when values is empty.
        :rtype: list[float]
        :raises TypeError: If values is not an iterable of real numbers.
        """
        vals: list[float] = list()
        vals = self._as_floats("walsh", values)
        return sorted((a + b) / 2.0
                      for i, a in enumerate(vals) for b in vals[i:])

    def hodges_lehmann(self, values: Any = tuple()) -> float:
        """
        Hodges-Lehmann estimate of the centre of a paired sample.

        It is the median of the Walsh averages: a robust location
        estimate that, unlike the plain median of the differences, uses
        the pairwise structure and is the estimator the Wilcoxon signed
        rank test inverts.

        :param values: Iterable of real numbers, the paired differences;
            must contain at least one value.
        :type values: Any
        :return: The estimate, in the units of the differences (ROC-AUC
            points in the paper).
        :rtype: float
        :raises TypeError: If values is not an iterable of real numbers.
        :raises ValueError: If values is empty. Note: the original
            script raised IndexError here instead, from indexing the
            empty list of Walsh averages; the callers of the paper never
            reach this branch, because they test for an empty sample
            first.
        """
        walsh: list[float] = list()
        out: float | None = None
        walsh = self.walsh(values)
        out = self.median(walsh)
        if out is None:
            self._error("[PairedStats.hodges_lehmann] Input error: the "
                        "parameter values must hold at least one "
                        "number, and it is empty.")
        return out

    def hl_interval(self,
                    values: Any = tuple(),
                    alpha: float | None = None,
                    ) -> tuple[float, float, float]:
        """
        Exact interval of the Hodges-Lehmann estimate.

        The interval is obtained by inverting the Wilcoxon signed rank
        test: the exact null distribution of W+ for n differences is
        built by convolution (a value either keeps its rank or does not,
        hence the doubling recursion), and the critical value c is the
        largest value of W+ with P(W+ <= c) <= alpha / 2. With the M
        Walsh averages sorted, the interval is [W_(c+1), W_(M-c)], that
        is the entries c and M - 1 - c counting from zero, and its
        coverage is 1 - 2 * P(W+ <= c). Because the distribution is
        discrete, the realised coverage is at least 1 - alpha and is
        returned alongside the bounds instead of being assumed.

        Note: an earlier version indexed the bounds with c + 1, one
        order statistic narrower on each side, while reporting the
        coverage of c. For n = 6 that labelled a 93.75 % interval as
        96.875 %. The bounds below are the corrected ones, the same the
        manuscript reports. Note also that when not even the most
        extreme order statistic reaches the requested coverage, which
        happens with a very small sample, the whole range of the Walsh
        averages is returned with coverage 1, and that the sample is
        used whole, zeros included, unlike the signed rank test below,
        which drops them.

        :param values: Iterable of real numbers, the paired
            differences; must contain at least one value.
        :type values: Any
        :param alpha: Two-sided alpha to use instead of the one of the
            instance, in (0, 1), or None to use the instance value.
        :type alpha: float | None
        :return: The lower bound, the upper bound and the realised
            coverage (1 - 2 * cumulative mass), all as floats.
        :rtype: tuple[float, float, float]
        :raises TypeError: If values is not an iterable of real numbers
            or alpha is not a number.
        :raises ValueError: If values is empty or alpha is not in
            (0, 1).
        """
        vals: list[float] = list()
        walsh: list[float] = list()
        counts: dict[int, int] = dict()
        use = 0.0
        total = 0
        cum = 0
        critical: int | None = None
        lo = 0.0
        hi = 0.0
        use = self.alpha if alpha is None else self._as_alpha(alpha)
        vals = self._as_floats("hl_interval", values)
        if not vals:
            self._error("[PairedStats.hl_interval] Input error: the "
                        "parameter values must hold at least one "
                        "number, and it is empty.")
        walsh = self.walsh(vals)
        counts = self._null_counts(len(vals))
        total = 2 ** len(vals)
        for s in sorted(counts):
            if (cum + counts[s]) / total > use / 2:
                break
            cum += counts[s]
            critical = s
        if critical is None:
            return walsh[0], walsh[-1], 1.0
        lo = walsh[critical] if critical < len(walsh) else walsh[0]
        hi = (walsh[len(walsh) - 1 - critical]
              if critical < len(walsh) else walsh[-1])
        return lo, hi, 1 - 2 * cum / total

    def wilcoxon_exact(self,
                       values: Any = tuple(),
                       exact_max_n: int | None = None,
                       ) -> tuple[float | None, float | None, int]:
        """
        Wilcoxon signed rank test of a paired sample, two-sided.

        Zero differences are dropped (Wilcoxon's own convention), the
        remaining ones are ranked by absolute value with average ranks
        for ties, and W+ is the sum of the ranks of the positive
        differences. For a small sample the p value is the exact
        fraction of the 2 ** n sign patterns at least as extreme as the
        observed one; above the threshold it comes from the normal
        approximation with the untied moments, mu = sum(r) / 2 and
        sd = sqrt(sum(r ** 2) / 4).

        Note: the exactness test counts a pattern once per side with
        ``s >= max(wp, total - wp) or s <= min(wp, total - wp)``, so a
        perfectly symmetric sample (W+ exactly half of the rank sum) has
        both sides met by the same patterns and the p value is capped at
        1 by the final ``min``; this is the behaviour of the submitted
        script and it is kept. Note also that the normal branch divides
        by sd without guarding sd = 0, which cannot happen once there is
        at least one non-zero difference.

        :param values: Iterable of real numbers, the paired
            differences; an all-zero or empty sample gives
            (None, None, 0).
        :type values: Any
        :param exact_max_n: Threshold of the exact enumeration to use
            instead of the one of the instance, >= 0, or None.
        :type exact_max_n: int | None
        :return: W+ (the rank sum of the positive differences), the
            two-sided p value capped at 1, and how many non-zero
            differences were used.
        :rtype: tuple[float | None, float | None, int]
        :raises TypeError: If values is not an iterable of real numbers
            or exact_max_n is not an int.
        :raises ValueError: If exact_max_n is negative.
        """
        vals: list[float] = list()
        nonzero: list[float] = list()
        ranks: list[float] = list()
        limit = 0
        n = 0
        wp = 0.0
        total = 0.0
        best = 0
        p = 0.0
        mu = 0.0
        sd = 0.0
        z = 0.0
        limit = (self.exact_max_n if exact_max_n is None
                 else self._as_limit(exact_max_n))
        vals = self._as_floats("wilcoxon_exact", values)
        nonzero = [x for x in vals if x != 0]
        n = len(nonzero)
        if not n:
            return None, None, 0
        ranks = self._abs_ranks(nonzero)
        wp = sum(r for r, x in zip(ranks, nonzero) if x > 0)
        total = sum(ranks)
        if n <= limit:
            for mask in range(2 ** n):
                s = sum(ranks[i] for i in range(n) if mask >> i & 1)
                if s >= max(wp, total - wp) or \
                        s <= min(wp, total - wp):
                    best += 1
            p = best / 2 ** n
        else:
            mu = total / 2.0
            sd = math.sqrt(sum(r * r for r in ranks) / 4.0)
            z = (wp - mu) / sd
            p = math.erfc(abs(z) / math.sqrt(2))
        return wp, min(1.0, p), n

    def __repr__(self) -> str:
        """
        Show the settings of the instance.

        :return: Text like "PairedStats(alpha=0.0390625,
            exact_max_n=20)".
        :rtype: str
        """
        return (f"PairedStats(alpha={self.alpha}, "
                f"exact_max_n={self.exact_max_n})")

    def _as_floats(self, method: str, values: Any) -> list[float]:
        """
        Read an input sample as a list of floats, keeping its order.

        :param method: Name of the calling method, for the message.
        :type method: str
        :param values: Iterable of real numbers; strings and bytes are
            rejected even though they are iterable.
        :type values: Any
        :return: The values as floats, in the order they were given.
        :rtype: list[float]
        :raises TypeError: If values is not an iterable, or holds a
            value that is not a real number (booleans included).
        """
        out: list[float] = list()
        if isinstance(values, (str, bytes)) or \
                not isinstance(values, Iterable):
            self._error(f"[PairedStats.{method}] Input error: the "
                        "parameter values must be an iterable of real "
                        f"numbers, not {type(values).__name__}.",
                        TypeError)
        for v in values:
            if isinstance(v, bool) or not isinstance(v, numbers.Real):
                self._error(f"[PairedStats.{method}] Input error: every "
                            "value must be a real number, not "
                            f"{type(v).__name__} ({v!r}).", TypeError)
            out.append(float(v))
        return out

    def _as_alpha(self, alpha: Any) -> float:
        """
        Validate a per-call alpha.

        :param alpha: Two-sided alpha, a real number in (0, 1).
        :type alpha: Any
        :return: The value as a float.
        :rtype: float
        :raises TypeError: If alpha is not a real number.
        :raises ValueError: If alpha is not in (0, 1).
        """
        if isinstance(alpha, bool) or not isinstance(alpha, numbers.Real):
            self._error("[PairedStats.hl_interval] Input error: the "
                        "parameter alpha must be a real number, not "
                        f"{type(alpha).__name__}.", TypeError)
        if not 0.0 < float(alpha) < 1.0:
            self._error("[PairedStats.hl_interval] Input error: the "
                        "parameter alpha must be in (0, 1), not "
                        f"{alpha}.")
        return float(alpha)

    def _as_limit(self, exact_max_n: Any) -> int:
        """
        Validate a per-call threshold of the exact enumeration.

        :param exact_max_n: Largest sample size enumerated, >= 0.
        :type exact_max_n: Any
        :return: The value as an int.
        :rtype: int
        :raises TypeError: If exact_max_n is not an int.
        :raises ValueError: If exact_max_n is negative.
        """
        if isinstance(exact_max_n, bool) or \
                not isinstance(exact_max_n, int):
            self._error("[PairedStats.wilcoxon_exact] Input error: the "
                        "parameter exact_max_n must be an int, not "
                        f"{type(exact_max_n).__name__}.", TypeError)
        if exact_max_n < 0:
            self._error("[PairedStats.wilcoxon_exact] Input error: the "
                        "parameter exact_max_n must be >= 0, not "
                        f"{exact_max_n}.")
        return int(exact_max_n)

    def _null_counts(self, n: int) -> dict[int, int]:
        """
        Exact null distribution of W+ for n differences, unnormalized.

        Under the null hypothesis every rank is added with probability
        one half, so the number of sign patterns reaching each value of
        W+ follows by convolving the ranks 1..n one at a time. The
        counts add up to 2 ** n.

        :param n: Number of non-zero differences, >= 0.
        :type n: int
        :return: Value of W+ -> how many of the 2 ** n sign patterns
            reach it.
        :rtype: dict[int, int]
        """
        counts: dict[int, int] = {0: 1}
        nxt: dict[int, int] = dict()
        for r in range(1, n + 1):
            nxt = dict(counts)
            for k, c in counts.items():
                nxt[k + r] = nxt.get(k + r, 0) + c
            counts = nxt
        return counts

    def _abs_ranks(self, values: list[float]) -> list[float]:
        """
        Ranks of the absolute values, ties sharing their average rank.

        The sort is the stable sort of Python, so equal absolute values
        keep the order they were given, and a group of ties receives the
        mean of the positions it occupies. The ranks are returned
        aligned with the input, not with the sorted order.

        :param values: Non-zero differences, in their original order.
        :type values: list[float]
        :return: Ranks in 1..n as floats, aligned with values.
        :rtype: list[float]
        """
        n = 0
        order: list[int] = list()
        ranks: list[float] = list()
        i = 0
        j = 0
        avg = 0.0
        n = len(values)
        order = sorted(range(n), key=lambda idx: abs(values[idx]))
        ranks = [0.0] * n
        while i < n:
            j = i
            while j + 1 < n and \
                    abs(values[order[j + 1]]) == abs(values[order[i]]):
                j += 1
            avg = (i + j + 2) / 2.0
            for t in range(i, j + 1):
                ranks[order[t]] = avg
            i = j + 1
        return ranks


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
