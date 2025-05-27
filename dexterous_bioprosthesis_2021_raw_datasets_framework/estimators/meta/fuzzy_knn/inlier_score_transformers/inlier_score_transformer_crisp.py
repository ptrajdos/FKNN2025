from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.fuzzy_knn.inlier_score_transformers.inlier_score_transformer import (
    InlierScoreTransformer,
)
from numpy import copy
import numpy as np


class InlierScoreTransformerCrisp(InlierScoreTransformer):

    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold  = threshold

    def transform(self, inlier_score_matrix):

        transformed = (inlier_score_matrix > self.threshold).astype(inlier_score_matrix.dtype)
        zero_rows = np.all(transformed == 0, axis=1)
        row_indices = np.where(zero_rows)[0]
        transformed[row_indices, :] = 1.0

        return transformed
