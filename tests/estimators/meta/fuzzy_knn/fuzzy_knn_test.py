import unittest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split


from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.fuzzy_knn.fuzzy_knn import (
    FuzzyKNN,
)

from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.fuzzy_knn.inlier_score_transformers.inlier_score_transformer import InlierScoreTransformer
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.fuzzy_knn.inlier_score_transformers.inlier_score_transformer_crisp import InlierScoreTransformerCrisp
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.fuzzy_knn.inlier_score_transformers.inlier_score_transformer_smoothstep import (
    InlierScoreTransformerSmoothstep,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_creators.raw_signals_creator_sines import (
    RawSignalsCreatorSines,
)
from dexterous_bioprosthesis_2021_raw_datasets_framework.set_creators.np_signal_extractors.np_signal_extractor_mav import (
    NpSignalExtractorMav,
)
from dexterous_bioprosthesis_2021_raw_datasets_framework.set_creators.np_signal_extractors.np_signal_extractor_ssc import (
    NpSignalExtractorSsc,
)
from dexterous_bioprosthesis_2021_raw_datasets_framework.set_creators.set_creator_dwt import (
    SetCreatorDWT,
)


class FuzzyKNNTest(unittest.TestCase):

    def get_estimators(self, channel_features) -> dict:

        return {
            "Default": FuzzyKNN(channel_features=channel_features),
            "3NN": FuzzyKNN(n_neighbors=3, channel_features=channel_features),
            "NoneNeigh": FuzzyKNN(n_neighbors=None, channel_features=channel_features),
            "smoothstep": FuzzyKNN(
                inlier_score_transformer=InlierScoreTransformerSmoothstep(),
                channel_features=channel_features,
            ),
            "crisp_05": FuzzyKNN(
                inlier_score_transformer=InlierScoreTransformerCrisp(), channel_features=channel_features
            ),
            "crisp_0": FuzzyKNN(
                inlier_score_transformer=InlierScoreTransformerCrisp(threshold=0.0),
                channel_features=channel_features,
            ),
            "crisp_1": FuzzyKNN(
                inlier_score_transformer=InlierScoreTransformerCrisp(threshold=1.0),
                channel_features=channel_features,
            ),
        }

    def get_extractor(self):
        extractor = SetCreatorDWT(
            num_levels=2,
            wavelet_name="db6",
            extractors=[
                NpSignalExtractorMav(),
                NpSignalExtractorSsc(),
            ],
        )
        return extractor

    def test_estimator_sines(self):
        n_classes = 4
        sines_creator = RawSignalsCreatorSines(
            column_number=4, set_size=50, class_indices=[*range(n_classes)]
        )

        train_set = sines_creator.get_set()
        test_set = sines_creator.get_set()

        set_creator = self.get_extractor()

        X_train, y_train, _ = set_creator.fit_transform(train_set)
        X_test, y_test, _ = set_creator.transform(test_set)

        channel_indices = set_creator.get_channel_attribs_indices()

        for clf_name, clf in self.get_estimators(
            channel_features=channel_indices
        ).items():

            with self.subTest(name=clf_name):
                clf.fit(X_train, y_train)
                probas = clf.predict_proba(X_test)
                self.assertIsNotNone(probas, f"Probabilites are None: {clf}")
                self.assertFalse(
                    np.isnan(probas).any(), f"NaNs in probability predictions: {clf}"
                )
                self.assertTrue(
                    probas.shape[0] == len(X_test),
                    f"Different number of objects in prediction: {clf}",
                )
                self.assertTrue(
                    probas.shape[1] == n_classes,
                    f"Wrong number of classes in proba prediction: {clf}",
                )

                self.assertTrue(np.all(probas >= 0), f"Negative probabilities: {clf}")
                self.assertTrue(
                    np.all(probas <= 1), f"Probabilities bigger than one: {clf}"
                )

                prob_sums = np.sum(probas, axis=1)
                self.assertTrue(
                    np.allclose(
                        prob_sums, np.asanyarray([1 for _ in range(X_test.shape[0])])
                    ),
                    f"Not all sums close to one: {clf}",
                )

                y_pred = clf.predict(X_test)
                self.assertIsNotNone(y_pred, f"Prediction is None: {clf}")
                self.assertFalse(np.isnan(y_pred).any(), f"NaNs in prediction: {clf}")
                self.assertTrue(
                    len(y_pred) == len(X_test),
                    f"Different number of objects in prediction: {clf}",
                )

                u_classes = np.unique(y_test)
                u_pred = np.unique(y_pred)
                self.assertTrue(
                    np.isin(u_pred, u_classes).all(),
                    f"Not all classes are predicted: {clf}",
                )

    def test_iris(self):

        iris = load_iris()
        X = iris.data
        y = iris.target

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=42
        )

        channel_indices = [[0], [1], [2], [3]]

        for clf_name, clf in self.get_estimators(
            channel_features=channel_indices
        ).items():
            with self.subTest(name=clf_name):
                clf.fit(X_train, y_train)

                probas = clf.predict_proba(X_test)
                self.assertIsNotNone(probas, f"Probabilites are None: {clf}")
                self.assertFalse(
                    np.isnan(probas).any(), f"NaNs in probability predictions: {clf}"
                )
                self.assertTrue(
                    probas.shape[0] == len(X_test),
                    f"Different number of objects in prediction: {clf}",
                )
                self.assertTrue(
                    probas.shape[1] == len(np.unique(y_test)),
                    f"Wrong number of classes in proba prediction: {clf}",
                )

                self.assertTrue(np.all(probas >= 0), f"Negative probabilities: {clf}")
                self.assertTrue(
                    np.all(probas <= 1), f"Probabilities bigger than one: {clf}"
                )

                prob_sums = np.sum(probas, axis=1)
                self.assertTrue(
                    np.allclose(
                        prob_sums, np.asanyarray([1 for _ in range(X_test.shape[0])])
                    ),
                    f"Not all sums close to one: {clf}",
                )

                y_pred = clf.predict(X_test)
                self.assertIsNotNone(y_pred, f"Prediction is None: {clf}")
                self.assertFalse(np.isnan(y_pred).any(), f"NaNs in prediction: {clf}")
                self.assertTrue(
                    len(y_pred) == len(X_test),
                    f"Different number of objects in prediction: {clf}",
                )
                kappa_val = cohen_kappa_score(y_test, y_pred)
                self.assertTrue(
                    kappa_val >= 0,
                    f"Kappa score is negative: {kappa_val} for {clf}",
                )


if __name__ == "__main__":
    unittest.main()
