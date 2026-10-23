# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         src/kitnetzday/figures/__init__.py                        #
#   Version:      1.0.1                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Figures of the revised manuscript, one module per         #
#                 figure of the analysis plan.                              #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
from kitnetzday.figures.contrasts import ContrastsPanel
from kitnetzday.figures.failure_rule import FailureRule
from kitnetzday.figures.failures import FailureCases
from kitnetzday.figures.h1_numbers import add_h1_numbers
from kitnetzday.figures.heatmap import Heatmap
from kitnetzday.figures.illustrative import IllustrativeFigures
from kitnetzday.figures.oracle import OracleFigure
from kitnetzday.figures.ranks import RanksFigure
from kitnetzday.figures.ranks_caption import mean_ranks_caption

__all__ = ["ContrastsPanel", "FailureCases", "FailureRule", "Heatmap",
           "IllustrativeFigures", "OracleFigure", "RanksFigure",
           "add_h1_numbers", "mean_ranks_caption"]


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
