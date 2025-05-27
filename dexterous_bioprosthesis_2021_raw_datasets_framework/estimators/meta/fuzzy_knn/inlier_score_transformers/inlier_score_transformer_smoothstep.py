from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.fuzzy_knn.inlier_score_transformers.inlier_score_transformer import (
    InlierScoreTransformer,
)
from numpy import copy
import numpy as np


class InlierScoreTransformerSmoothstep(InlierScoreTransformer):

    def __init__(self):
        super().__init__()

    def transform(self, inlier_score_matrix):

        transformed = 3.0 * inlier_score_matrix**2 - 2.0 * inlier_score_matrix**3

        return transformed
