from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.fuzzy_knn.inlier_score_transformers.inlier_score_transformer_smoothstep import (
    InlierScoreTransformerSmoothstep,
)
from tests.estimators.meta.fuzzy_knn.inlier_score_transformers.inlier_score_transformer_test import (
    InlierScoreTransformerTest,
)


class InlierScoreTransformerSmoothstepTest(InlierScoreTransformerTest):

    __test__ = True

    def get_transformers(self):
        return {
            "Base": InlierScoreTransformerSmoothstep(),
        }
