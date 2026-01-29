from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.fuzzy_knn.inlier_score_transformers.inlier_score_transformer_low_para import (
    InlierScoreTransformerLowPara,
)
from tests.estimators.meta.fuzzy_knn.inlier_score_transformers.inlier_score_transformer_test import (
    InlierScoreTransformerTest,
)


class InlierScoreTransformerLowParaTest(InlierScoreTransformerTest):

    __test__ = True

    def get_transformers(self):
        return {
            "Base": InlierScoreTransformerLowPara(),
            "K_09": InlierScoreTransformerLowPara(k=0.9),
            "P_10": InlierScoreTransformerLowPara(p=10),
            "K01_P20": InlierScoreTransformerLowPara(k=0.1, p=20),
        }
