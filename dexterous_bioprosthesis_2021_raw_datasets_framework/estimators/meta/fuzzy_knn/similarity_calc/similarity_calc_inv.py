import numpy as np
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.fuzzy_knn.similarity_calc.asimilarity_calc import (
    ASimilarityCalc,
)


class SimilarityCalcInv(ASimilarityCalc):
    """
    Class for calculating similarity using an inverse distance-related weights.
    """

    def __init__(
        self, pairwise_distances_func=None, pairwise_distances_kwargs=None
    ):
        """
        Initialize the SimilarityCalcInv with a distance function and its parameters.

        Parameters
        ----------
        pairwise_distances_func : callable, optional
            A function to compute pairwise distances. If None, a default function is used.
        pairwise_distances_kwargs : dict, optional
            Additional keyword arguments for the distance function.
        """
        super().__init__(pairwise_distances_func, pairwise_distances_kwargs)

    def fit(self, X):
        super().fit(X)
        self.pairwise_distances_func_ = self._get_effective_pairwise_distances_func()
        self.pairwise_distances_kwargs_ = (
            self._get_effective_pairwise_distances_kwargs()
        )

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
