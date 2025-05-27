from sklearn.metrics import pairwise_distances
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.fuzzy_knn.similarity_calc.similarity_calc import (
    SimilarityCalc,
)


class ASimilarityCalc(SimilarityCalc):

    def __init__(self, pairwise_distances_func=None, pairwise_distances_kwargs=None):
        """
        Initialize the ASimilarityCalc with a distance function and its parameters.

        Parameters
        ----------
        pairwise_distances_func : callable, optional
            A function to compute pairwise distances. If None, a default function is used.
        pairwise_distances_kwargs : dict, optional
            Additional keyword arguments for the distance function.
        """
        self.pairwise_distances_func = pairwise_distances_func
        self.pairwise_distances_kwargs = pairwise_distances_kwargs

    def _get_effective_pairwise_distances_func(self):
        """
        Get the effective pairwise distance function, either the default or the one provided.
        """
        if self.pairwise_distances_func is None:
            return pairwise_distances
        else:
            return self.pairwise_distances_func

    def _get_effective_pairwise_distances_kwargs(self):
        """
        Get the effective keyword arguments for the pairwise distance function.
        """
        if self.pairwise_distances_kwargs is None:
            return {}
        else:
            return self.pairwise_distances_kwargs
