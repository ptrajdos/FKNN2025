from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.fuzzy_knn.fuzzy_norms.t_norms.product_norm import ProductNorm
from tests.estimators.meta.fuzzy_knn.fuzzy_norms.t_norms.t_norm_test import TNormTest


class ProductNormTest(TNormTest):

    __test__ = True
    def get_t_norms(self) -> dict:
        """
        Returns a list of t-norms to be tested.
        """

        return {
            "ProductNorm": ProductNorm(),
        }
    