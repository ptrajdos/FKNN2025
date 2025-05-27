from sklearn.base import BaseEstimator, ClassifierMixin

import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.outlier.outlier_detector_combiner import OutlierDetectorCombiner


class OutlierDetectorCombinerMin(OutlierDetectorCombiner):
    
    def __init__(self, outlier_detectors) -> None:
        super().__init__(outlier_detectors)

    def fit(self, X, y=None):
        raise NotImplementedError("Fitting operation is not implemented for this class")
    
    def fit_predict(self,X, y=None):
        raise NotImplementedError("Fitting operation is not implemented for this class")
    
    def _combine_predictions_labels(self, base_predictions):
        """
        For an object return minimum value returned by one of predictors.
        If 0 means that object is an outlier,
             then one prediction saying outlier forces the sample to be classified as outlier.
        """
        return np.min(base_predictions, axis=1)
    
    def _combine_predictions_soft(self, base_predictions):
        """
        Combines soft predictions of multiple outlier detectors

        Arguments:
        ----------
        base_predictions -- numpy array of size (n_objects, 2, n_detectors)

        
        Returns:
        --------
        numpy array of size (n_objects,) containing predictions in range [0 -- outlier, 1 non-outlier]
        """
        n_objects = base_predictions.shape[0]
        combined = np.zeros((n_objects,2))

        combined[:,1] =  np.min(base_predictions[:,1], axis=1)
        combined[:,[0]] = 1.0 - combined[:,[1]]

        return combined