# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         tests/utils/test_formatting.py                            #
#   Version:      1.0.1                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Unit tests for the number, LaTeX and ranking helpers.     #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging

import numpy as np
import pytest

from kitnetzday.utils.formatting import (
                                      fnum,
                                      get,
                                      ids_comment,
                                      ranks_avg,
                                      tex_escape,
)


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                MAIN CLASS                                 #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class TestFnum:
    """
    Tests for the number written into a cell of a table of the paper.
    """

    def test_analytical_three_decimals(self) -> None:
        """
        Hand-computed case for 0.8734, 1/3 and 1.5.

        By hand: with three decimals 0.8734 is cut after the third one
        and the fourth, a 4, rounds down, so it prints "0.873". One
        third is 0.333333..., so it prints "0.333". With no decimals
        Python rounds half to even, so 1.5 prints "2" and 0.5 prints
        "0": that is why no table of the paper uses nd = 0.
        """
        assert fnum(0.8734) == "0.873"
        assert fnum(1.0 / 3.0) == "0.333"
        assert fnum(1.5, nd=0) == "2"
        assert fnum(0.5, nd=0) == "0"

    def test_missing_values_print_the_marker(self) -> None:
        """
        None, NaN, the infinities and unparsable text give the marker.
        """
        assert fnum(None) == "--"
        assert fnum(float("nan")) == "--"
        assert fnum(float("inf")) == "--"
        assert fnum(float("-inf")) == "--"
        assert fnum("not a number") == "--"
        assert fnum(np.array([1.0, 2.0])) == "--"

    def test_numeric_text_and_scalars_do_parse(self) -> None:
        """
        A numeric string and a numpy scalar are numbers.

        A one-element array is deliberately not asserted here: whether
        ``float`` accepts it depends on the numpy version, and the only
        behaviour the paper relies on is that anything that does not
        parse prints the marker.
        """
        assert fnum("0.5") == "0.500"
        assert fnum(np.float64(1.5)) == "1.500"

    def test_custom_marker(self) -> None:
        """
        The marker is a parameter, so a text report can use its own.
        """
        assert fnum(None, na="n/a") == "n/a"


class TestTexEscape:
    """
    Tests for the escaping of labels that travel into LaTeX.
    """

    def test_analytical_special_characters(self) -> None:
        """
        Hand-computed case for "a_b & 50% {c} ~ ^ # $".

        By hand, one replacement per character class: "_" becomes
        "\\_", "&" becomes "\\&", "%" becomes "\\%", "{" and "}" become
        "\\{" and "\\}", "#" becomes "\\#", "$" becomes "\\$", and "~"
        and "^" become the two text commands, because they are active
        characters in LaTeX and an escape alone would not print them.
        """
        assert tex_escape("a_b & 50% {c} ~ ^ # $") == (
            r"a\_b \& 50\% \{c\} \textasciitilde{} "
            r"\textasciicircum{} \# \$")

    def test_backslash_keeps_the_submitted_behaviour(self) -> None:
        """
        A lone backslash prints as "\\textbackslash\\{\\}".

        The backslash is replaced first, by "\\textbackslash{}", and the
        later brace replacements then escape the two braces that
        replacement introduced. The result is pinned here instead of
        being fixed: no dataset or configuration label of the study
        holds a backslash, so changing it would only risk the artefacts
        that are already generated.
        """
        assert tex_escape("\\") == r"\textbackslash\{\}"

    def test_non_strings_pass_through_str(self) -> None:
        """
        Anything that is not a string is written with str() first.
        """
        assert tex_escape(7) == "7"
        assert tex_escape(None) == "None"
        assert tex_escape() == ""


class TestGet:
    """
    Tests for the tolerant walk over a chain of dictionary keys.
    """

    def test_analytical_walk(self) -> None:
        """
        Hand-computed case over {"cells": {"D1": {"C7": {"auroc": .5}}}}.

        By hand: the four keys are followed in order and the value at
        the end is returned. Stopping one key short returns the branch
        itself, which is a dictionary.
        """
        tree = {"cells": {"D1": {"C7": {"auroc": 0.5}}}}
        assert get(tree, "cells", "D1", "C7", "auroc") == 0.5
        assert get(tree, "cells", "D1", "C7") == {"auroc": 0.5}

    def test_missing_branches_give_the_default(self) -> None:
        """
        A missing key, a stored null and a leaf never raise.
        """
        assert get({"a": {"b": 1}}, "a", "z") is None
        assert get({"a": {"b": None}}, "a", "b", default=5) == 5
        assert get({"a": 1}, "a", "b") is None
        assert get(None, "a", default="x") == "x"

    def test_empty_path_returns_the_object(self) -> None:
        """
        With no keys the object comes back, None giving the default.
        """
        assert get({"a": 1}) == {"a": 1}
        assert get(None, default=3) == 3


class TestIdsComment:
    """
    Tests for the LaTeX comment that lists the value ids.
    """

    def test_analytical_short_and_truncated_lists(self) -> None:
        """
        Hand-computed case for 2 ids and for 14 ids with k = 12.

        By hand: with 2 ids the comment carries the count and both ids.
        With 14 ids and k = 12 the first twelve are listed, N0 to N11,
        and the tail announces the 14 - 12 = 2 that are left out and
        where the whole list can be read.
        """
        assert ids_comment(["A1", "A2"], "fig") == \
            "% numbers_used.md ids (2): A1, A2"
        comment = ids_comment([f"N{i}" for i in range(14)], "fig")
        assert comment.startswith(
            "% numbers_used.md ids (14): N0, N1, ")
        assert "N11 ... (+2;" in comment
        assert "figures.json#/outputs/fig/numbers_ids" in comment
        assert "N12" not in comment

    def test_empty_list(self) -> None:
        """
        An output that uses no row of numbers_used.md says so.
        """
        assert ids_comment() == "% numbers_used.md ids: (none)"


class TestRanksAvg:
    """
    Tests for the average ranks that feed the Friedman omnibus.
    """

    def test_analytical_ties_share_their_average_rank(self) -> None:
        """
        Hand-computed case for the values [3, 1, 4, 1, 5].

        By hand: sorted, the five values are 1, 1, 3, 4, 5 and they
        occupy the positions 1, 2, 3, 4, 5. The two 1s occupy the
        positions 1 and 2, so each of them takes their mean,
        (1 + 2) / 2 = 1.5. The 3 is alone in position 3, the 4 in
        position 4 and the 5 in position 5. Read back in the order they
        were given, the ranks are [3, 1.5, 4, 1.5, 5], and they add up
        to 15 = n (n + 1) / 2 = 5 * 6 / 2, as the ranks of any row must.
        """
        ranks = ranks_avg([3, 1, 4, 1, 5])
        assert list(ranks) == [3.0, 1.5, 4.0, 1.5, 5.0]
        assert ranks.sum() == pytest.approx(15.0)

    def test_analytical_all_equal(self) -> None:
        """
        Hand-computed case for four equal values.

        By hand: the four values form a single group of ties occupying
        the positions 1, 2, 3 and 4, so every one of them takes
        (1 + 2 + 3 + 4) / 4 = 2.5, which is also the mean of the first
        and the last position, (1 + 4) / 2. They add up to
        10 = 4 * 5 / 2.
        """
        ranks = ranks_avg([2, 2, 2, 2])
        assert list(ranks) == [2.5, 2.5, 2.5, 2.5]
        assert ranks.sum() == pytest.approx(10.0)

    def test_single_value_and_empty_input(self) -> None:
        """
        One value gets rank 1; an empty row warns and gives nothing.
        """
        assert list(ranks_avg([7])) == [1.0]
        assert ranks_avg(tuple()).shape == (0,)

    def test_empty_input_warns_through_the_logger(
            self,
            caplog: pytest.LogCaptureFixture) -> None:
        """
        An empty row is reported as a warning, not as an error.

        :param caplog: pytest fixture that captures log records.
        :type caplog: pytest.LogCaptureFixture
        """
        with caplog.at_level(logging.WARNING, logger="test"):
            ranks_avg(list(), logger=logging.getLogger("test"))
        assert "[ranks_avg] Input warning" in caplog.records[-1].message
        assert "is empty" in caplog.records[-1].message

    def test_nan_becomes_its_own_largest_rank(self) -> None:
        """
        Hand-computed case for [1.0, NaN, 2.0].

        By hand: NaN never compares equal to anything and sorts to the
        end, so the sorted order is 1.0, 2.0, NaN over the positions 1,
        2 and 3. No group of ties forms, so the ranks read back are
        [1, 3, 2]: the NaN takes the largest rank instead of being
        flagged. The statistics of the paper therefore have to keep NaN
        out themselves, which is what this test documents.
        """
        assert list(ranks_avg([1.0, float("nan"), 2.0])) == \
            [1.0, 3.0, 2.0]

    def test_two_dimensional_input_raises(self) -> None:
        """
        A matrix is not a row of values and is refused with its shape.
        """
        with pytest.raises(ValueError, match="one-dimensional"):
            ranks_avg([[1.0, 2.0], [3.0, 4.0]])

    def test_unconvertible_input_raises_and_logs(
            self,
            caplog: pytest.LogCaptureFixture) -> None:
        """
        Text that is not a number raises ValueError and is logged.

        :param caplog: pytest fixture that captures log records.
        :type caplog: pytest.LogCaptureFixture
        """
        with caplog.at_level(logging.ERROR, logger="test"):
            with pytest.raises(ValueError,
                               match="could not be converted"):
                ranks_avg(["a", "b"], logger=logging.getLogger("test"))
        assert "[ranks_avg] Input error" in caplog.records[-1].message


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
