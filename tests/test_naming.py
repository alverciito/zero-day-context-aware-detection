# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         tests/test_naming.py                                      #
#   Version:      1.0.1                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Unit tests for the vocabulary queries of the study.       #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import pytest

from kitnetzday.naming import Naming


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                MAIN CLASS                                 #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class TestDatasetQueries:
    """
    Tests for the three ways a stream is written.
    """

    def test_long_and_short_labels(self) -> None:
        """
        An id has a caption label and an axis label, and no more.
        """
        naming = Naming()
        assert naming.dataset_label("D8") == "CIC-IDS2017"
        assert naming.dataset_label("D1") == "KNAD Active Wiretap"
        assert naming.dataset_short_label("D8") == "CIC17"
        assert naming.dataset_short_label("D2") == "FUZZ"

    def test_unknown_id_falls_back_to_itself(self) -> None:
        """
        An id the vocabulary does not know prints as the id.

        This is what keeps a new stream visible in a figure instead of
        silently disappearing from it.
        """
        naming = Naming()
        assert naming.dataset_label("D99") == "D99"
        assert naming.dataset_short_label("D99") == "D99"

    def test_fig7_names_are_stripped_before_the_lookup(self) -> None:
        """
        The dataset names of the submitted Fig. 7 CSV map to plan ids.

        The names in that CSV are upper case and hyphenated, and the
        reader may hand them over with blanks around them, so they are
        stripped first.
        """
        naming = Naming()
        assert naming.fig7_dataset("  CIC2017 ") == "D8"
        assert naming.fig7_dataset("ACTIVE-WIRETAP") == "D1"
        assert naming.fig7_dataset("NOT-A-STREAM") is None

    def test_non_string_ids(self) -> None:
        """
        An id that is not a string is refused by type, not guessed.
        """
        naming = Naming()
        for method in (naming.dataset_label,
                       naming.dataset_short_label,
                       naming.fig7_dataset):
            with pytest.raises(TypeError, match="must be a string"):
                method(8)


class TestConfigurationQueries:
    """
    Tests for the labels and the properties of a configuration.
    """

    def test_long_and_short_labels(self) -> None:
        """
        Only the three context-free ids have a short label.

        The other eight print their long label on an axis, because it
        already fits: "MHA-AR" is as short as a short label would be.
        """
        naming = Naming()
        assert naming.config_label("C7") == "MHA-AR"
        assert naming.config_label("C10") == "KitNET-style dA"
        assert naming.config_short("C10") == "dA"
        assert naming.config_short("C11") == "STAT"
        assert naming.config_short("C7") == "MHA-AR"
        assert naming.config_label("C99") == "C99"

    def test_context_free_configurations(self) -> None:
        """
        The three per-record configurations are the context-free ones.

        Everything else carries temporal context, so the eight
        remaining ids are exactly C1 to C8.
        """
        naming = Naming()
        assert naming.is_context_free("C9") is True
        assert naming.is_context_free("C10") is True
        assert naming.is_context_free("C11") is True
        assert naming.is_context_free("C1") is False
        assert naming.context_configs() == [f"C{i}"
                                            for i in range(1, 9)]

    def test_architecture_lookup_both_ways(self) -> None:
        """
        Hand-checked map between architectures and configurations.

        By hand: the four windowed architectures take two
        configurations each, one autoregressive and one
        time-series, so Conv1D holds C1 and C2, Conv2D C3 and C4, LSTM
        C5 and C6 and MHA C7 and C8; the MLP holds only C9, because
        with one record there is no difference between the two
        paradigms. That accounts for nine of the eleven ids: C10 and
        C11 are not trained networks and belong to no architecture.
        """
        naming = Naming()
        assert naming.architecture_configs("LSTM") == ("C5", "C6")
        assert naming.architecture_configs("MLP") == ("C9",)
        assert naming.architecture_configs("ResNet") == tuple()
        assert naming.config_architecture("C5") == "LSTM"
        assert naming.config_architecture("C9") == "MLP"
        assert naming.config_architecture("C10") is None
        assert naming.config_architecture("C11") is None
        assert sum(len(naming.architecture_configs(a))
                   for a in naming.architectures) == 9

    def test_non_string_ids(self) -> None:
        """
        A configuration id that is not a string is refused by type.
        """
        naming = Naming()
        for method in (naming.config_label, naming.config_short,
                       naming.is_context_free,
                       naming.config_architecture,
                       naming.architecture_configs):
            with pytest.raises(TypeError, match="must be a string"):
                method(7)


class TestPlanTags:
    """
    Tests for the labels of the analysis plan written into artefacts.
    """

    def test_known_tags(self) -> None:
        """
        The tags are returned verbatim, Spanish and section included.
        """
        naming = Naming()
        assert naming.plan_tag("primary") == \
            "[CONFIRMATORIO plan §3.1]"
        assert naming.plan_tag("fail") == "[EXPLORATORIO plan §10]"
        assert naming.plan_tag("partial") == "[PARTIAL INPUTS]"

    def test_unknown_tag(self) -> None:
        """
        An unknown key is a KeyError that lists the keys there are.
        """
        naming = Naming()
        with pytest.raises(KeyError, match="must be one of"):
            naming.plan_tag("made up")
        with pytest.raises(TypeError, match="must be a string"):
            naming.plan_tag(1)


class TestVocabulary:
    """
    Tests for the consistency the vocabulary checks at construction.
    """

    def test_defaults_are_the_submitted_study(self) -> None:
        """
        Nine streams, eleven configurations, five architectures.
        """
        naming = Naming()
        assert len(naming.datasets) == 9
        assert len(naming.configurations) == 11
        assert len(naming.architectures) == 5
        assert naming.control == "C10"
        assert naming.rope == 0.01
        assert naming.failure_margin == 0.01
        assert repr(naming) == ("Naming(datasets=9, configurations=11, "
                                "architectures=5, control='C10')")

    def test_a_smaller_vocabulary_is_allowed(self) -> None:
        """
        A subset of the study is a valid vocabulary of its own.

        This is what lets a partial run be tabulated: the ids are
        passed in by the script in bin/, read from src/config.json.
        """
        naming = Naming(datasets=["D1"], configurations=["C1", "C9"],
                        context_free=["C9"], control="C1")
        assert naming.datasets == ["D1"]
        assert naming.context_configs() == ["C1"]
        assert naming.is_context_free("C9") is True

    def test_empty_and_inconsistent_vocabularies(self) -> None:
        """
        An empty or self-contradicting vocabulary is refused.
        """
        with pytest.raises(ValueError, match="at least one dataset"):
            Naming(datasets=list())
        with pytest.raises(ValueError,
                           match="at least one .*configuration"):
            Naming(configurations=list())
        with pytest.raises(ValueError, match="must be a configuration"):
            Naming(context_free=["Z1"])
        with pytest.raises(ValueError, match="control must be one of"):
            Naming(control="Z1")

    def test_margins_must_be_positive(self) -> None:
        """
        A margin of zero would make every comparison equivalent.
        """
        with pytest.raises(ValueError, match="rope must be strictly"):
            Naming(rope=0.0)
        with pytest.raises(ValueError,
                           match="failure_margin must be strictly"):
            Naming(failure_margin=-0.01)


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
