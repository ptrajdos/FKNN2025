from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.fuzzy_knn.inlier_score_transformers.inlier_score_transformer import (
    InlierScoreTransformer,
)
from numpy import copy


class InlierScoreTransformerSimple(InlierScoreTransformer):

    def transform(self, inlier_score_matrix):
        return copy(inlier_score_matrix)
