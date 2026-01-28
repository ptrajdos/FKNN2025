import numpy as np
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.fuzzy_knn.similarity_calc.asimilarity_calc import (
    ASimilarityCalc,
)


class SimilarityCalcInv(ASimilarityCalc):
    """
    Class for calculating similarity using an inverse distance-related weights.
    """


    def pairwise_similarity(self, X, Y=None):
        """
        Compute the pairwise similarity between samples using the RBF kernel.

        Parameters
        ----------
        X : array-like, shape (n_samples_a, n_features)
            The first input data.
        Y : array-like, shape (n_samples_b, n_features), optional
            The second input data. If None, the pairwise similarity is computed within X.

        Returns
        -------
        array-like, shape (n_samples_a, n_samples_b)
            The pairwise similarity matrix.
        """
        distances = self.pairwise_distances_func_(
            X=X, Y=Y, **self.pairwise_distances_kwargs_
        )
        return 1.0 / (1.0 + distances)
