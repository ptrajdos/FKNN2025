from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.fuzzy_knn.inlier_score_transformers.inlier_score_transformer import (
    InlierScoreTransformer,
)
from numpy import copy
import numpy as np


class InlierScoreTransformerScaledSigmoid(InlierScoreTransformer):

    def __init__(self, k=10):
        super().__init__()
        self.k = k
        

    def transform(self, inlier_score_matrix):

        base = 1 / (1 + np.exp(-self.k * (inlier_score_matrix - 0.5)))
        min_val = 1 / (1 + np.exp(self.k / 2))
        max_val = 1 / (1 + np.exp(-self.k / 2))
        transformed =  (base - min_val) / (max_val - min_val)
        return transformed
