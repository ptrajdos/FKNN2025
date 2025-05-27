import abc


class SimilarityCalc:

    @abc.abstractmethod
    def fit(self, X):
        """
        Fit the similarity calculator to the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.
        """

    @abc.abstractmethod
    def pairwise_similarity(self, X, Y=None):
        """
        Compute the pairwise similarity between samples.

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
