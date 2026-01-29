from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.fuzzy_knn.inlier_score_transformers.inlier_score_transformer_crisp import InlierScoreTransformerCrisp
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.fuzzy_knn.inlier_score_transformers.inlier_score_transformer_smoothstep import (
    InlierScoreTransformerSmoothstep,
)
from tests.estimators.meta.fuzzy_knn.inlier_score_transformers.inlier_score_transformer_test import (
    InlierScoreTransformerTest,
)


class InlierScoreTransformerCrispTest(InlierScoreTransformerTest):

    __test__ = True

    def get_transformers(self):
        return {
            "Base": InlierScoreTransformerCrisp(),
            "Th0": InlierScoreTransformerCrisp(threshold=0),
            "Th1": InlierScoreTransformerCrisp(threshold=1),
            "Th05": InlierScoreTransformerCrisp(threshold=0.5),
        }
