# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         src/kitnetzday/io/__init__.py                             #
#   Version:      2.0.0                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Reading of the aggregated results and of the original     #
#                 artefacts, and writing of the outputs and reports.        #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
from kitnetzday.io.inputs import Inputs
from kitnetzday.io.legacy import (
                               parse_f1_trivial_smoothed,
                               parse_fig7_csv,
                               read_timing_reference,
)
from kitnetzday.io.outputs import Out
from kitnetzday.io.reports import (
                               FIGURES_VERSION,
                               IMPLEMENTATION_NOTES,
                               SCHEMA,
                               write_index,
                               write_manifest,
                               write_numbers,
)

__all__ = ["FIGURES_VERSION", "IMPLEMENTATION_NOTES", "Inputs", "Out",
           "SCHEMA", "parse_f1_trivial_smoothed", "parse_fig7_csv",
           "read_timing_reference", "write_index", "write_manifest",
           "write_numbers"]


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
