
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.fuzzy_knn.fuzzy_norms.t_norms.t_norm import TNorm


class ProductNorm(TNorm):
    """
    Product t-norm.
    """

    
    def t_norm(self, a, b):

        return a * b