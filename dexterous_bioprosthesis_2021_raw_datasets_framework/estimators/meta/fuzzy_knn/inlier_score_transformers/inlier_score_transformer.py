import abc


class InlierScoreTransformer(abc.ABC):

    @abc.abstractmethod
    def transform(self, inlier_score_matrix):
        """
        Transform inlier score matrix

        Arguments:
        ----------

        inlier_score_matrix: (n_samples, n_channels)

        Reutrns:
        ---------

        Transformed matrix (n_samples, n_channels)
        """
