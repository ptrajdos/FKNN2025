from sklearn.metrics import pairwise_distances
from tests.estimators.meta.fuzzy_knn.similarity_calc.similarity_calc_test import (
    SimilarityCalcTest,
)
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.fuzzy_knn.similarity_calc.similarity_calc_exp import (
    SimilarityCalcExp,
)


class SimilarityCalcExpTest(SimilarityCalcTest):

    __test__ = True

    def get_similarity_calcs(self) -> dict:
        """
        Returns a list of similarity calculators to be tested.
        """

        return {
            "SimilarityCalcExp_default": SimilarityCalcExp(),
            "SimilarityCalcExp_gamma": SimilarityCalcExp(gamma=0.5),
            "SimilarityCalcExp_pw_dist": SimilarityCalcExp(
                pairwise_distances_func=pairwise_distances,
                pairwise_distances_kwargs={"metric": "euclidean"},
            ),
            "SimilarityCalcExp_cityblock": SimilarityCalcExp(
                pairwise_distances_kwargs={"metric": "cityblock"}
            ),
        }
