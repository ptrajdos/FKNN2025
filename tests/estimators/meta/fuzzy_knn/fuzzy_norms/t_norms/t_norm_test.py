import unittest
import numpy as np


class TNormTest(unittest.TestCase):
    """
    Base class for testing  T-norms.
    """

    __test__ = False

    @classmethod
    def setUpClass(cls):
        if not cls.__test__:
            raise unittest.SkipTest("Skipping")

    def get_t_norms(self) -> dict:
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
                (
                    np.random.random((r, c)),
                    np.random.random((r, c)),
                    np.random.random((r, c)), 
                    np.random.random((r, c)),
                )
            )

        return data

    def test_t_norm(self):
        """
        Check the properties of the T-norm.
        """

        for t_norm_name, t_norm_obj in self.get_t_norms().items():
            with self.subTest(t_norm=t_norm_name):
                for X, Y, Z, W in self.get_test_data():
                    # Check the properties of the T-norm
                    xy_result = t_norm_obj.t_norm(X, Y)
                    yx_result = t_norm_obj.t_norm(Y, X)
                    yxz_result = t_norm_obj.t_norm(Y, Z)

                    xyz_result = t_norm_obj.t_norm(xy_result, Z)
                    yxz_result = t_norm_obj.t_norm(yx_result, Z)

                    self.assertIsNotNone(
                        xy_result, f"T-norm result is None: {t_norm_name}"
                    )
                    self.assertTrue(
                        np.all(np.isfinite(xy_result)),
                        f"T-norm result contains NaN or Inf values: {t_norm_name}",
                    )
                    self.assertTrue(
                        np.all(xy_result >= 0),
                        f"T-norm result contains negative values: {t_norm_name}",
                    )
                    self.assertTrue(
                        np.all(xy_result <= 1),
                        f"T-norm result contains values greater than 1: {t_norm_name}",
                    )

                    self.assertIsNotNone(
                        yx_result, f"T-norm result is None: {t_norm_name}"
                    )
                    self.assertTrue(
                        np.all(np.isfinite(yx_result)),
                        f"T-norm result contains NaN or Inf values: {t_norm_name}",
                    )
                    self.assertTrue(
                        np.all(yx_result >= 0),
                        f"T-norm result contains negative values: {t_norm_name}",
                    )

                    self.assertTrue(
                        np.allclose(xy_result, yx_result),
                        f"T-norm result is not Commutative: {t_norm_name}",
                    )

                    ones = np.ones_like(X)
                    one_result = t_norm_obj.t_norm(X, ones)
                    self.assertTrue(
                        np.all(np.isclose(one_result, X)),
                        f"T-norm result is not Idempotent: {t_norm_name}",
                    )

                    self.assertTrue(
                        np.all(np.isclose(xyz_result, yxz_result)),
                        f"T-norm result is not Associative: {t_norm_name}",
                    )

                    # Check Monotonicity
                    zw_result = t_norm_obj.t_norm(Z, W)
                    z_greater = np.greater(Z, X)
                    w_greater = np.greater(W, Y)
                    cummulative = np.logical_and(z_greater, w_greater)
                    self.assertTrue(np.all( xy_result[cummulative] <= zw_result[cummulative]), f"T-norm result is not Monotonic: {t_norm_name}")

