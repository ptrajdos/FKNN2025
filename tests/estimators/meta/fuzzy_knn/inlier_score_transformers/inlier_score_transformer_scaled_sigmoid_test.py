from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.fuzzy_knn.inlier_score_transformers.inlier_score_transformer_scaled_sigmoid import (
    InlierScoreTransformerScaledSigmoid,
)
from tests.estimators.meta.fuzzy_knn.inlier_score_transformers.inlier_score_transformer_test import (
    InlierScoreTransformerTest,
)


class InlierScoreTransformerScaledSigmoidTest(InlierScoreTransformerTest):

    __test__ = True

    def get_transformers(self):
        return {
            "Base": InlierScoreTransformerScaledSigmoid(),
            "K10": InlierScoreTransformerScaledSigmoid(k=10),
            "K20": InlierScoreTransformerScaledSigmoid(k=20),
            "K1": InlierScoreTransformerScaledSigmoid(k=1),
        }
