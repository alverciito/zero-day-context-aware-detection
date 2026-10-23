# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         src/kitnetzday/tables/__init__.py                         #
#   Version:      1.0.0                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-07                                                #
#   Description:  LaTeX tables of the manuscript and of its supplementary   #
#                 material.                                                 #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
from kitnetzday.tables.ablations import AblationsTable
from kitnetzday.tables.baselines import BaselineTables
from kitnetzday.tables.cost import CostTable
from kitnetzday.tables.main_results import MainResultsTables
from kitnetzday.tables.supplementary import SupplementaryTables
from kitnetzday.tables.supplementary_inputs import (
                                                 SUPPL_CONFIG_RENAME,
                                                 SUPPL_DATASET_SHORT,
                                                 SupplementaryInputs,
)
from kitnetzday.tables.supplementary_latex import caption_below, escape_cell
from kitnetzday.tables.supplementary_output import SupplementaryOutput
from kitnetzday.tables.supplementary_style import SupplementaryStyle
from kitnetzday.tables.supplementary_summaries import (
                                                    CALIBRATION_NOTE,
                                                    SupplementarySummaries,
)

__all__ = ["AblationsTable", "BaselineTables", "CALIBRATION_NOTE",
           "CostTable",
           "MainResultsTables", "SUPPL_CONFIG_RENAME",
           "SUPPL_DATASET_SHORT", "SupplementaryInputs",
           "SupplementaryOutput", "SupplementaryStyle",
           "SupplementarySummaries", "SupplementaryTables",
           "caption_below", "escape_cell"]


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
