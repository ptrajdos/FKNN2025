import unittest

import numpy as np


class SimilarityCalcTest(unittest.TestCase):

    __test__ = False

    @classmethod
    def setUpClass(cls):
        if not cls.__test__:
            raise unittest.SkipTest("Skipping")

    def get_similarity_calcs(self) -> dict:
        """
        Returns a list of similarity calculators to be tested.
        """
        raise unittest.SkipTest("Skipping")

    def get_test_data(self) -> list:
        """
        Returns test data for the similarity calculators.
        """
        rc_pairs = [(10, 20), (30, 40), (50, 60), (2, 10), (10, 1), (1, 10)]
        data = []
        for r, c in rc_pairs:
            data.append((np.random.random((r, c)), np.random.random((r, c))))

        return data

    def check_similarity_matrix(self, matrix):
        """
        Check the properties of the similarity matrix.
        """
        self.assertIsNotNone(matrix, "Similarity matrix is None")
        self.assertTrue(
            np.all(np.isfinite(matrix)), "Similarity matrix contains NaN or Inf values"
        )
        self.assertTrue(
            np.all(matrix >= 0), "Similarity matrix contains negative values"
        )
        self.assertTrue(
            np.all(matrix <= 1), "Similarity matrix contains values greater than 1"
        )

    def test_pairwise_similarity(self):
        """
        Test the pairwise similarity calculation.
        """
        for name, calc in self.get_similarity_calcs().items():
            with self.subTest(name=name):
                for X, Y in self.get_test_data():
                    # Fit the calculator
                    calc.fit(X)

                    self_similarity = calc.pairwise_similarity(X)
                    self.check_similarity_matrix(self_similarity)
                    self.assertEqual(
                        self_similarity.shape,
                        (X.shape[0], X.shape[0]),
                        "Shape of self similarity matrix is wrong",
                    )
                    self.assertTrue(
                        np.allclose(self_similarity, self_similarity.T),
                        "Self similarity matrix is not symmetric",
                    )
                    self.assertTrue(
                        np.all(np.diag(self_similarity) == 1),
                        "Self similarity matrix diagonal is not 1",
                    )

                    similarity = calc.pairwise_similarity(X, Y)
                    self.check_similarity_matrix(similarity)
                    self.assertEqual(
                        similarity.shape,
                        (X.shape[0], Y.shape[0]),
                        "Shape of similarity matrix is wrong",
                    )
