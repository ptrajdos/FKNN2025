from sklearn.metrics import pairwise_distances
from tests.estimators.meta.fuzzy_knn.similarity_calc.similarity_calc_test import (
    SimilarityCalcTest,
)
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.fuzzy_knn.similarity_calc.similarity_calc_rbf import (
    SimilarityCalcRBF,
)


class SimilarityCalcRBFTest(SimilarityCalcTest):

    __test__ = True

    def get_similarity_calcs(self) -> dict:
        """
        Returns a list of similarity calculators to be tested.
        """

        return {
            "SimilarityCalcRBF_default": SimilarityCalcRBF(),
            "SimilarityCalcRBF_gamma": SimilarityCalcRBF(gamma=0.5),
            "SimilarityCalcRBF_pw_dist": SimilarityCalcRBF(
                pairwise_distances_func=pairwise_distances,
                pairwise_distances_kwargs={"metric": "euclidean"},
            ),
            "SimilarityCalcRBF_cityblock": SimilarityCalcRBF(
                pairwise_distances_kwargs={"metric": "cityblock"}
            ),
        }
