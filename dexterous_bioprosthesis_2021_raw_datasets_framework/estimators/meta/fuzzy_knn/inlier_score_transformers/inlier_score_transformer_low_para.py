from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.fuzzy_knn.inlier_score_transformers.inlier_score_transformer import (
    InlierScoreTransformer,
)
from numpy import copy
import numpy as np


class InlierScoreTransformerLowPara(InlierScoreTransformer):

    def __init__(self, k=0.5, p=5):
        super().__init__()
        self.k = k
        self.p = p

    def transform(self, inlier_score_matrix):

        above_half = inlier_score_matrix > 0.5
        below_half = ~above_half

        transformed = np.empty_like(inlier_score_matrix)
        transformed[above_half] = inlier_score_matrix[above_half]
        transformed[below_half] = inlier_score_matrix[below_half] * (
            inlier_score_matrix[below_half] / 0.5
        ) ** (self.p * self.k)

        return transformed
