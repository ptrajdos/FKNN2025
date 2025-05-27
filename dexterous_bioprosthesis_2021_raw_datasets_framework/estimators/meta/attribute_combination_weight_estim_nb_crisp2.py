import numpy as np
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.attribute_combination_weight_estim_nb import (
    AttributeCombinationWeightEstimNB,
    attribute_weights_former_mean1,
)


class AttributeCombinationWeightEstimNBCrisp2(AttributeCombinationWeightEstimNB):
    def __init__(
        self,
        model_prototype=None,
        outlier_detector_prototype=None,
        random_state=0,
        channel_features=None,
        channel_combination_generator=None,
        channel_combination_generator_options=None,
        attribute_weights_former=attribute_weights_former_mean1,
        threshold=0.5,
    ):
        super().__init__(
            model_prototype,
            outlier_detector_prototype,
            random_state,
            channel_features,
            channel_combination_generator,
            channel_combination_generator_options,
            attribute_weights_former,
        )
        self.threshold = threshold



    def transform_attribute_weights(self, weights):

        transformed = (weights > self.threshold).astype(weights.dtype)
        zero_rows = np.all(transformed == 0, axis=2)  

        variant_idx, row_idx = np.where(zero_rows)

        transformed[variant_idx, row_idx, :] = 1.0

        return transformed