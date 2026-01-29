import unittest
import numpy as np


class InlierScoreTransformerTest(unittest.TestCase):

    __test__ = False

    @classmethod
    def setUpClass(cls):
        if not cls.__test__:
            raise unittest.SkipTest("Skipping")

    def get_transformers(self) -> dict:
        """
        Returns a list of T-norms to be tested.
        """
        raise unittest.SkipTest("Skipping")

    def get_test_data(self) -> list:
        """
        Returns test data for the T-norms.
        """
        rc_pairs = [(10, 20), (30, 40), (50, 60), (2, 10), (10, 1), (1, 10)]
        data = []
        for r, c in rc_pairs:
            data.append(
                    np.random.random((r, c)),
            )
            data.append(
                np.zeros((r, c)),
            )
            data.append(
                np.ones((r, c)),
            )

        return data

    def test_transform(self):
        for transformer_name, transformer in self.get_transformers().items():
            with self.subTest(transformer_name=transformer_name):
                for X in self.get_test_data():

                    transformed = transformer.transform(X)

                    self.assertIsNotNone(transformed, "Transformed is None")
                    self.assertTrue(
                        isinstance(transformed, np.ndarray),
                        "Transformed is not a numpy array",
                    )
                    self.assertTrue(
                        transformed.shape == X.shape,
                        "Transformed shape does not match input shape",
                    )
                    self.assertFalse(
                        np.any(np.isnan(transformed)), "Transformed contains NaN values"
                    )
                    self.assertFalse(
                        np.any(np.isinf(transformed)),
                        "Transformed contains infinite values",
                    )
                    self.assertTrue(
                        np.all(transformed >= 0), "Transformed contains negative values"
                    )
                    self.assertTrue(
                        np.all(transformed <= 1),
                        "Transformed contains values greater than 1",
                    )
