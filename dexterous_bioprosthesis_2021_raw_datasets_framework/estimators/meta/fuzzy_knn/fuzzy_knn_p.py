import itertools
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, check_is_fitted
from sklearn.calibration import LabelEncoder
from sklearn.dummy import check_random_state
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.fuzzy_knn.fuzzy_norms.t_norms.product_norm import (
    ProductNorm,
)
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.fuzzy_knn.inlier_score_transformers.inlier_score_transformer_simple import InlierScoreTransformerSimple
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.fuzzy_knn.similarity_calc.similarity_calc_exp import SimilarityCalcExp
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.fuzzy_knn.similarity_calc.similarity_calc_rbf import (
    SimilarityCalcRBF,
)
from dexterous_bioprosthesis_2021_raw_datasets_framework.preprocessing.select_attributes_transformer import (
    SelectAttributesTransformer,
)

from pt_outlier_probability.outlier_probability_estimator import (
    OutlierProbabilityEstimator,
)

from dexterous_bioprosthesis_2021_raw_datasets_framework.preprocessing.select_attributes_transformer import (
    SelectAttributesTransformer,
)

class FuzzyKNNP(BaseEstimator, ClassifierMixin):
    """
    Fuzzy K-Nearest Neighbors classifier.
    """

    def __init__(
        self,
        n_neighbors=5,
        outlier_detector_prototype=None,
        random_state=0,
        channel_features=None,
        similarity_calc=None,
        t_norm=None,
        inlier_score_transformer=None,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.channel_features = channel_features
        self.outlier_detector_prototype = outlier_detector_prototype
        self.random_state = random_state
        self.similarity_calc = similarity_calc
        self.t_norm = t_norm
        self.inlier_score_transformer = inlier_score_transformer


    def _preprocess_X_y(self, X, y):
        """
        If some class(es) has only one object, then append another copy of this object.
        This is to perform proper stratification in datasest with
        """
        uniq, counts = np.unique(y, return_counts=True)

        one_instance_labels = uniq[counts == 1]
        select_inndices = [True if i in one_instance_labels else False for i in y]

        Xc = np.append(X, X[select_inndices], axis=0)
        yc = np.append(y, y[select_inndices])

        return Xc, yc

    def _select_channel_group_features(self, channel_group_indices):
        """
        Selects features for channel group

        Attributes:
        -----------

        channel_group_indices -- iterable containing indices of channels constitutes the group.

        Returns:
        --------
        List containing combined features for each channel in the group.
        """

        all_selected_features = []

        for ch_idx in channel_group_indices:
            all_selected_features += self.channel_features[ch_idx]

        return all_selected_features

    def _prepare_base_outlier_detectors(self, X, y):
        """
        Prepares base outlier detectors.
        One base outlier detector per channel.
        """
        n_channels = len(self.channel_features)

        effective_outlier_detector_prototype = (
            self.outlier_detector_prototype
            if self.outlier_detector_prototype is not None
            else IsolationForest()
        )

        self.outlier_detectors_ = np.zeros(n_channels, dtype=object)
        y_o = np.ones_like(y, dtype=np.int64)
        # create outlier detectors
        for channel_id in range(n_channels):
            column_indices = self._select_channel_group_features([channel_id])
            base_detector = Pipeline(
                [
                    (
                        "trans",
                        SelectAttributesTransformer(column_indices=column_indices),
                    ),
                    (
                        "classifier",
                        OutlierProbabilityEstimator(
                            clone(effective_outlier_detector_prototype)
                        ),
                    ),
                ]
            )
            base_detector.fit(X, y_o)
            self.outlier_detectors_[channel_id] = base_detector

    def _create_base_classifier(self, X, y):
        y_train = self.label_encoder_.transform(y)
        self.class_matrix_ = np.eye(len(self.classes_))[y_train]

        self.similarity_calc_ = (
            self.similarity_calc
            if self.similarity_calc is not None
            else SimilarityCalcExp()
        )

        self.similarity_calc_.fit(X)

    def _set_effective_t_norm(self):
        """
        Returns the effective t-norm function.
        """
        if self.t_norm is None:
            self.t_norm_ = ProductNorm()
        else:
            self.t_norm_ = self.t_norm

    def _set_effective_inlier_score_transformer(self):
        """
        """
        if self.inlier_score_transformer is None:
            self.inlier_score_transformer_ = InlierScoreTransformerSimple()
        else:
            self.inlier_score_transformer_ = self.inlier_score_transformer


    def fit(self, X, y):
        self.random_state_ = check_random_state(self.random_state)
        self._set_effective_t_norm()
        self._set_effective_inlier_score_transformer()

        Xp, yp = self._preprocess_X_y(X, y)
        self.classes_ = np.unique(yp)

        self.label_encoder_ = LabelEncoder()
        self.label_encoder_.fit(yp)

        self._prepare_base_outlier_detectors(Xp, yp)

        self._create_base_classifier(Xp, yp)

        self.X_ = Xp
        self.y_ = yp


        return self

    def predict(self, X):

        proba = self.predict_proba(X)

        y_pred_org = np.argmax(proba, axis=1)
        y_pred = self.label_encoder_.inverse_transform(y_pred_org)
        return y_pred

    def _create_similarity_matrix(self, X):
        """
        Create similarity matrix for the input data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        Returns
        -------
        array-like, shape (n_X_samples, n_X_fit_samples, n_channels)
            The similarity matrix for each channel.
        """
        n_channels = len(self.channel_features)
        sim_matrix = np.zeros((X.shape[0], self.X_.shape[0], n_channels))

        for channel_id in range(n_channels):
            column_indices = self._select_channel_group_features([channel_id])
            sim_matrix[..., channel_id] = self.similarity_calc_.pairwise_similarity(
                X=X[:, column_indices], Y=self.X_[:, column_indices]
            )
        
        return sim_matrix
    
    def _get_inlier_score_matrix(self, X):
        """
        Create inlier score matrix for the input data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        Returns
        -------
        array-like, shape (n_X_samples, n_channels)
            The inlier score matrix for each channel.
        """
        n_channels = len(self.channel_features)
        inlier_score_matrix = np.zeros((X.shape[0], n_channels))

        # probability of not being an outlier!
        for channel_id in range(n_channels):
            inlier_score_matrix[..., channel_id] = self.outlier_detectors_[channel_id].predict_proba(X)[:, 1]

        transformed_inlier_score_matrix = self.inlier_score_transformer_.transform(inlier_score_matrix)


        return transformed_inlier_score_matrix
    
    def _combine_channel_predictions(self, inference_matrix):
        """
        Combine channel predictions.

        Parameters
        ----------
        inference_matrix : array-like, shape (n_X_samples, n_X_fit_samples, n_channels, n_classes)
            The inference matrix for each channel.

        Returns
        -------
        array-like, shape (n_X_samples, n_X_fit_samples, n_classes)
            The predictions averaged over channel supports.
        """
        combined_predictions = np.prod(inference_matrix, axis=2)
        overall_sim = np.sum(combined_predictions, axis=2, keepdims=False)
        
        if self.n_neighbors is not None:
           
            top_indices = np.argsort(overall_sim, axis=1)[:, -self.n_neighbors :]
            mask = np.zeros_like(overall_sim, dtype=bool)
            np.put_along_axis(mask, top_indices, True, axis=1)
            combined_predictions[~mask]= 0.0
        

        return combined_predictions
    
    def _get_class_predictions(self, channel_combined_predictions):
        """
        Get class predictions

        Parameters
        ----------
        channel_combined_predictions : array-like, shape (n_X_samples, n_X_fit_samples, n_classes)
            The predictions averaged over channel supports.
        Returns
        -------
        -------
        array-like, shape (n_X_samples, n_classes)
            The class predictions
        """
        class_predictions = np.sum(channel_combined_predictions, axis=1)
        class_predictions /= np.sum(class_predictions, axis=1, keepdims=True)
        return class_predictions

    def predict_proba(self, X):
        check_is_fitted(
            self,
            (
                "classes_",
                "X_",
                "y_",
            ),
        )
        
        
        sim_matrix = self._create_similarity_matrix(X)
        sim_class_matrix = self.t_norm_.t_norm(
            sim_matrix[..., np.newaxis],
            self.class_matrix_[np.newaxis, :, np.newaxis, :],
        )
    
        inlier_score_matrix = self._get_inlier_score_matrix(X)

        
        inference_matrix = sim_class_matrix ** inlier_score_matrix[:, np.newaxis, :, np.newaxis]

        channel_combined_predictions = self._combine_channel_predictions(inference_matrix)
        class_predictions = self._get_class_predictions(channel_combined_predictions)

        return class_predictions
