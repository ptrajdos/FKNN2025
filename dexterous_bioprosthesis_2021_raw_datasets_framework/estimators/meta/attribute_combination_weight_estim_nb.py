from __future__ import annotations
import itertools
import random
import numpy as np
from collections import defaultdict
from sklearn.base import BaseEstimator, ClassifierMixin, check_is_fitted
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state
from sklearn.base import clone
from dexterous_bioprosthesis_2021_raw_datasets.preprocessing.select_attributes_transformer import (
    SelectAttributesTransformer,
)
from kernelnb.estimators.estimatornb import EstimatorNB
from pt_outlier_probability.outlier_probability_estimator import (
    OutlierProbabilityEstimator,
)


def channel_group_gen(indices, group_sizes=[2]):
    result_list = []
    for group_size in group_sizes:
        result_list += [k for k in itertools.combinations(indices, group_size)]

    return result_list


def channel_group_filter_all(group_list):
    for group in group_list:
        yield group


def channel_group_filter_even_odd(group_list):
    for group in group_list:
        if len(group) == 1:
            yield group
            continue

        if len(group) == 2:
            pair = (group[0] % 2, group[1] % 2)
            if pair == (0, 1) or pair == (1, 0):
                yield group
                continue


def channel_group_gen_even_odd(indices, group_sizes=[2]):
    """
    Any sEMG + MMG combination
    """
    orig_results = channel_group_gen(indices, group_sizes)
    result_list = [*channel_group_filter_even_odd(orig_results)]

    return result_list


def channel_group_filter_even_odd2(group_list):
    for group in group_list:
        if len(group) == 1:
            yield group
            continue

        if len(group) == 2:
            if group[0] % 2 == 0 and group[1] % 2 == 1 and (group[0] + 1) == group[1]:
                yield group
                continue


def channel_group_gen_even_odd2(indices, group_sizes=[2]):
    """
    SEMG + MMG combination from single sensor.
    """
    orig_results = channel_group_gen(indices, group_sizes)
    result_list = [*channel_group_filter_even_odd2(orig_results)]

    return result_list


def channel_group_filter_even_odd3(group_list):
    for group in group_list:
        if len(group) == 1:
            yield group
            continue

        if len(group) == 2:
            pair = (group[0] % 2, group[1] % 2)
            not_one_step = (group[0] + 1) != group[1]
            if (pair == (0, 1) and not_one_step) or pair == (1, 0):
                yield group
                continue


def channel_group_gen_even_odd3(indices, group_sizes=[2]):
    """
    SEMG + MMG combination not a from single sensor.
    """
    orig_results = channel_group_gen(indices, group_sizes)
    result_list = [*channel_group_filter_even_odd3(orig_results)]

    return result_list



def channel_group_filter_even_and_odd(group_list):
    for group in group_list:
        if len(group) == 1:
            yield group
            continue

        all_even = all(group[i] % 2 == 0 for i in range(len(group)))
        all_odd = all(group[i] % 2 == 1 for i in range(len(group)))
        if all_even or all_odd:
            yield group
            continue


def channel_group_gen_even_and_odd(indices, group_sizes=[2]):
    """
    Any combination of sEMG and MMG only
    """
    orig_results = channel_group_gen(indices, group_sizes)
    result_list = [*channel_group_filter_even_and_odd(orig_results)]

    return result_list


def channel_group_filter_even(group_list):
    for group in group_list:
        all_even = all(group[i] % 2 == 0 for i in range(len(group)))
        if all_even:
            yield group
            continue

def shuffled_generator(lst):
    random.shuffle(lst)
    for item in lst:
        yield item

def channel_group_filter_even_odd_unique(group_list):
    seen_channels = set()
    for group in  shuffled_generator(group_list):
        if len(group) == 1:
            yield group
            continue

        if len(group) == 2:
            pair = (group[0] % 2, group[1] % 2)
            if (pair == (0, 1) or pair == (1, 0)) and group[0] not in seen_channels and group[1] not in seen_channels:
                seen_channels.update(group)
                yield group
                continue
                
def channel_group_gen_even_odd_unique(indices, group_sizes=[2]):
    """
    Random sEMG + MMG combination. Match the size of single sensor combinations
    """
    orig_results = channel_group_gen(indices, group_sizes)
    result_list = [*channel_group_filter_even_odd_unique(orig_results)]

    return result_list

def channel_group_filter_unique_pairs(group_list):
    seen_channels = set()
    for group in  shuffled_generator(group_list):
        if len(group) == 1:
            yield group
            continue

        if len(group) == 2:
            if group[0] not in seen_channels and group[1] not in seen_channels:
                seen_channels.update(group)
                yield group
                continue
                
def channel_group_gen_unique_pairs(indices, group_sizes=[2]):
    """
    Any combination. Match the size of single sensor combinations
    """
    orig_results = channel_group_gen(indices, group_sizes)
    result_list = [*channel_group_filter_unique_pairs(orig_results)]

    return result_list

def channel_group_gen_even(indices, group_sizes=[2]):
    """
    MMG combinations
    """
    orig_results = channel_group_gen(indices, group_sizes)
    result_list = [*channel_group_filter_even(orig_results)]

    return result_list


def channel_group_filter_odd(group_list):
    for group in group_list:
        all_odd = all(group[i] % 2 == 1 for i in range(len(group)))
        if all_odd:
            yield group
            continue


def channel_group_gen_odd(indices, group_sizes=[2]):
    """
    EMG combinations
    """
    orig_results = channel_group_gen(indices, group_sizes)
    result_list = [*channel_group_filter_odd(orig_results)]

    return result_list

def channel_group_filter_size(group_list, size=1):
    for group in group_list:
        if len(group) == size:
            yield group

def apply_filters(group_list, filters):
    result = group_list
    for f in filters:
        result = f(result)
    return result


def attribute_weights_former_mean1(
    estim: AttributeCombinationWeightEstimNB, combination_weights: np.ndarray
) -> np.ndarray:
    """

    Arguments:
    ----------

    param: AttributeCombinationWeightEstimNB:  estim -- estimator that contains channel features and channel combinations
    param: combination_weights -- weights for each channel combination (n_samples, n_combinations)

    Returns:
    Weight for each channel-related attribute (n_samples, n_attributes)

    """
    n_channels = len(estim.channel_features)
    n_samples = combination_weights.shape[0]
    channel_weights = np.zeros((n_samples, n_channels))
    n_attributes = estim._get_n_attribs()
    n_combinations = combination_weights.shape[1]

    for sample_idx in range(n_samples):
        for channel_combination_idx, channel_combination in enumerate(
            estim.channel_combinations_
        ):
            for channel_idx in channel_combination:
                channel_weights[sample_idx, channel_idx] += combination_weights[
                    sample_idx, channel_combination_idx
                ]

    channel_weights = channel_weights / n_combinations

    attribute_weights = np.zeros((1, n_samples, n_attributes))

    for channel_idx in range(n_channels):
        column_indices = estim._select_channel_group_features([channel_idx])
        attribute_weights[[0], :, column_indices] = channel_weights[:, channel_idx]

    return attribute_weights

def attribute_weights_former_mean2(
    estim: AttributeCombinationWeightEstimNB, combination_weights: np.ndarray
) -> np.ndarray:
    """

    Arguments:
    ----------

    param: AttributeCombinationWeightEstimNB:  estim -- estimator that contains channel features and channel combinations
    param: combination_weights -- weights for each channel combination (n_samples, n_combinations)

    Returns:
    Weight for each channel-related attribute (n_samples, n_attributes)

    """
    n_channels = len(estim.channel_features)
    n_samples = combination_weights.shape[0]
    channel_weights = np.zeros((n_samples, n_channels))
    n_attributes = estim._get_n_attribs()
    n_combinations = combination_weights.shape[1]

    normalization_factor = 0 
    for channel_combination in estim.channel_combinations_:
        normalization_factor += 1.0/len(channel_combination)

    for sample_idx in range(n_samples):
        for channel_combination_idx, channel_combination in enumerate(
            estim.channel_combinations_
        ):
            combination_len  = len(channel_combination)

            for channel_idx in channel_combination:
                channel_weights[sample_idx, channel_idx] += combination_weights[
                    sample_idx, channel_combination_idx
                ]/combination_len

    channel_weights = channel_weights / normalization_factor

    attribute_weights = np.zeros((1, n_samples, n_attributes))

    for channel_idx in range(n_channels):
        column_indices = estim._select_channel_group_features([channel_idx])
        attribute_weights[[0], :, column_indices] = channel_weights[:, channel_idx]

    return attribute_weights


def attribute_weights_former_proba1(
    estim: AttributeCombinationWeightEstimNB, combination_weights: np.ndarray
) -> np.ndarray:
    """

    Arguments:
    ----------

    param: AttributeCombinationWeightEstimNB:  estim -- estimator that contains channel features and channel combinations
    param: combination_weights -- weights for each channel combination (n_samples, n_combinations)

    Returns:
    Weight for each channel-related attribute (n_samples, n_attributes)

    """
    n_channels = len(estim.channel_features)
    n_samples = combination_weights.shape[0]
    channel_weights = np.zeros((n_samples, n_channels))
    n_attributes = estim._get_n_attribs()
    n_combinations = combination_weights.shape[1]

    def create_channel_map_len(channel_combinations, combination_len=1):
        channel_map = defaultdict(list)
        for channel_combination_idx, channel_combination in enumerate(
            channel_combinations
        ):
            if len(channel_combination) == combination_len:
                for channel_idx in channel_combination:
                    channel_map[channel_idx].append(channel_combination_idx)

        return channel_map

    # channel_idx -> column__idx
    single_channel_map = create_channel_map_len(
        estim.channel_combinations_, combination_len=1
    )

    # channel_idx -> columns__idx
    double_channel_map = create_channel_map_len(
        estim.channel_combinations_, combination_len=2
    )

    channel_weights = np.zeros((n_samples, n_channels))
    for channel_idx in range(n_channels):
        channel_response = combination_weights[:, single_channel_map[channel_idx]]
        joint_response = combination_weights[:, double_channel_map[channel_idx]]

        if channel_response.shape[1] == 0:
            channel_weights[:, channel_idx] = np.ones(n_samples, dtype=np.float64)
            continue

        if joint_response.shape[1] == 0:
            channel_weights[:, channel_idx] = channel_response.ravel()
            continue

        numerator = channel_response.ravel() * np.prod(joint_response, axis=1).ravel()
        denominator = np.prod(
            np.multiply(channel_response, joint_response)
            + np.multiply(1 - channel_response, 1 - joint_response),
            axis=1,
        )

        channel_weights[:, channel_idx] = np.nan_to_num(
            np.divide(numerator, denominator), nan=0.0
        )

    attribute_weights = np.zeros((1, n_samples, n_attributes))

    for channel_idx in range(n_channels):
        column_indices = estim._select_channel_group_features([channel_idx])
        attribute_weights[[0], :, column_indices] = channel_weights[:, channel_idx]

    return attribute_weights

def attribute_weights_former_att1(
    estim: AttributeCombinationWeightEstimNB, combination_weights: np.ndarray
) -> np.ndarray:
    """

    Arguments:
    ----------

    param: AttributeCombinationWeightEstimNB:  estim -- estimator that contains channel features and channel combinations
    param: combination_weights -- weights for each channel combination (n_samples, n_combinations)

    Returns:
    Weight for each channel-related attribute (n_samples, n_attributes)

    """
    n_channels = len(estim.channel_features)
    n_samples = combination_weights.shape[0]
    channel_weights = np.zeros((n_samples, n_channels))
    n_attributes = estim._get_n_attribs()
    n_combinations = combination_weights.shape[1]

    def create_channel_map_len(channel_combinations, combination_len=1):
        channel_map = defaultdict(list)
        for channel_combination_idx, channel_combination in enumerate(
            channel_combinations
        ):
            if len(channel_combination) == combination_len:
                for channel_idx in channel_combination:
                    channel_map[channel_idx].append(channel_combination_idx)

        return channel_map
    
    def get_comp_channel(channel_idx, n_channels):
        comp = channel_idx + 1 if channel_idx % 2 == 0 else channel_idx - 1
        return comp if 0 <= comp < n_channels else None

    
    single_channel_map = create_channel_map_len(
        estim.channel_combinations_, combination_len=1
    )

    def smooth_score(x: np.ndarray, K: np.ndarray, p: float = 5) -> np.ndarray:
        """
        Compute the smooth scoring function for arrays of x and K.
        
        Parameters:
            x (np.ndarray): Array of input values in [0, 1]
            K (np.ndarray): Array of attenuation coefficients in [0, 1], same shape as x or broadcastable
            p (float): Controls steepness of attenuation below 0.5 (default is 6)
        
        Returns:
            np.ndarray: Array of scores in [0, 1]
        """
        x = np.asarray(x)
        K = np.asarray(K)
        
        # Ensure broadcast compatibility
        assert x.shape == K.shape or K.shape == () or K.shape == (1,), "Shapes must be compatible"

        # Apply piecewise function
        above_half = x > 0.5
        below_half = ~above_half

        result = np.empty_like(x)
        result[above_half] = x[above_half]
        result[below_half] = x[below_half] * (x[below_half] / 0.5) ** (p * (K[below_half]))

        return result

    for channel_idx in range(n_channels):
        channel_response = combination_weights[:, single_channel_map[channel_idx]]
        comp_channel_idx = get_comp_channel(channel_idx, n_channels)

        if comp_channel_idx is None:
            channel_weights[:, channel_idx] = channel_response.ravel()
            continue

        comp_channel_response = combination_weights[:, single_channel_map[comp_channel_idx]]
        channel_weights[:, channel_idx] = smooth_score(
            channel_response.ravel(), comp_channel_response.ravel()
        )
    
    attribute_weights = np.zeros((1, n_samples, n_attributes))

    for channel_idx in range(n_channels):
        column_indices = estim._select_channel_group_features([channel_idx])
        attribute_weights[[0], :, column_indices] = channel_weights[:, channel_idx]

    return attribute_weights

def attribute_weights_former_att2(
    estim: AttributeCombinationWeightEstimNB, combination_weights: np.ndarray
) -> np.ndarray:
    """

    Arguments:
    ----------

    param: AttributeCombinationWeightEstimNB:  estim -- estimator that contains channel features and channel combinations
    param: combination_weights -- weights for each channel combination (n_samples, n_combinations)

    Returns:
    Weight for each channel-related attribute (n_samples, n_attributes)

    """
    n_channels = len(estim.channel_features)
    n_samples = combination_weights.shape[0]
    channel_weights = np.zeros((n_samples, n_channels))
    n_attributes = estim._get_n_attribs()
    n_combinations = combination_weights.shape[1]

    def create_channel_map_len(channel_combinations, combination_len=1):
        channel_map = defaultdict(list)
        for channel_combination_idx, channel_combination in enumerate(
            channel_combinations
        ):
            if len(channel_combination) == combination_len:
                for channel_idx in channel_combination:
                    channel_map[channel_idx].append(channel_combination_idx)

        return channel_map
    
    def get_comp_channel(channel_idx, n_channels):
        if channel_idx % 2 == 0:
            odd_numbers = [i for i in range(1, n_channels, 2)]
            return random.choice(odd_numbers)
        else:
            even_numbers = [i for i in range(0, n_channels, 2)]
            return random.choice(even_numbers)

    
    single_channel_map = create_channel_map_len(
        estim.channel_combinations_, combination_len=1
    )

    def smooth_score(x: np.ndarray, K: np.ndarray, p: float = 5) -> np.ndarray:
        """
        Compute the smooth scoring function for arrays of x and K.
        
        Parameters:
            x (np.ndarray): Array of input values in [0, 1]
            K (np.ndarray): Array of attenuation coefficients in [0, 1], same shape as x or broadcastable
            p (float): Controls steepness of attenuation below 0.5 (default is 6)
        
        Returns:
            np.ndarray: Array of scores in [0, 1]
        """
        x = np.asarray(x)
        K = np.asarray(K)
        
        # Ensure broadcast compatibility
        assert x.shape == K.shape or K.shape == () or K.shape == (1,), "Shapes must be compatible"

        # Apply piecewise function
        above_half = x > 0.5
        below_half = ~above_half

        result = np.empty_like(x)
        result[above_half] = x[above_half]
        result[below_half] = x[below_half] * (x[below_half] / 0.5) ** (p * (K[below_half]))

        return result

    for channel_idx in range(n_channels):
        channel_response = combination_weights[:, single_channel_map[channel_idx]]
        comp_channel_idx = get_comp_channel(channel_idx, n_channels)

        if comp_channel_idx is None:
            channel_weights[:, channel_idx] = channel_response.ravel()
            continue

        comp_channel_response = combination_weights[:, single_channel_map[comp_channel_idx]]
        channel_weights[:, channel_idx] = smooth_score(
            channel_response.ravel(), comp_channel_response.ravel()
        )
    
    attribute_weights = np.zeros((1, n_samples, n_attributes))

    for channel_idx in range(n_channels):
        column_indices = estim._select_channel_group_features([channel_idx])
        attribute_weights[[0], :, column_indices] = channel_weights[:, channel_idx]

    return attribute_weights

def attribute_weights_former_att3(
    estim: AttributeCombinationWeightEstimNB, combination_weights: np.ndarray
) -> np.ndarray:
    """

    Arguments:
    ----------

    param: AttributeCombinationWeightEstimNB:  estim -- estimator that contains channel features and channel combinations
    param: combination_weights -- weights for each channel combination (n_samples, n_combinations)

    Returns:
    Weight for each channel-related attribute (n_samples, n_attributes)

    """
    n_channels = len(estim.channel_features)
    n_samples = combination_weights.shape[0]
    channel_weights = np.zeros((n_samples, n_channels))
    n_attributes = estim._get_n_attribs()
    n_combinations = combination_weights.shape[1]

    def create_channel_map_len(channel_combinations, combination_len=1):
        channel_map = defaultdict(list)
        for channel_combination_idx, channel_combination in enumerate(
            channel_combinations
        ):
            if len(channel_combination) == combination_len:
                for channel_idx in channel_combination:
                    channel_map[channel_idx].append(channel_combination_idx)

        return channel_map
    
    def get_comp_channel(channel_idx, n_channels):
        numbers = [i for i in range(n_channels) if i != channel_idx]
        return random.choice(numbers)

    
    single_channel_map = create_channel_map_len(
        estim.channel_combinations_, combination_len=1
    )

    def smooth_score(x: np.ndarray, K: np.ndarray, p: float = 5) -> np.ndarray:
        """
        Compute the smooth scoring function for arrays of x and K.
        
        Parameters:
            x (np.ndarray): Array of input values in [0, 1]
            K (np.ndarray): Array of attenuation coefficients in [0, 1], same shape as x or broadcastable
            p (float): Controls steepness of attenuation below 0.5 (default is 6)
        
        Returns:
            np.ndarray: Array of scores in [0, 1]
        """
        x = np.asarray(x)
        K = np.asarray(K)
        
        # Ensure broadcast compatibility
        assert x.shape == K.shape or K.shape == () or K.shape == (1,), "Shapes must be compatible"

        # Apply piecewise function
        above_half = x > 0.5
        below_half = ~above_half

        result = np.empty_like(x)
        result[above_half] = x[above_half]
        result[below_half] = x[below_half] * (x[below_half] / 0.5) ** (p * (K[below_half]))

        return result

    for channel_idx in range(n_channels):
        channel_response = combination_weights[:, single_channel_map[channel_idx]]
        comp_channel_idx = get_comp_channel(channel_idx, n_channels)

        if comp_channel_idx is None:
            channel_weights[:, channel_idx] = channel_response.ravel()
            continue

        comp_channel_response = combination_weights[:, single_channel_map[comp_channel_idx]]
        channel_weights[:, channel_idx] = smooth_score(
            channel_response.ravel(), comp_channel_response.ravel()
        )
    
    attribute_weights = np.zeros((1, n_samples, n_attributes))

    for channel_idx in range(n_channels):
        column_indices = estim._select_channel_group_features([channel_idx])
        attribute_weights[[0], :, column_indices] = channel_weights[:, channel_idx]

    return attribute_weights

def attribute_weights_former_att4(
    estim: AttributeCombinationWeightEstimNB, combination_weights: np.ndarray
) -> np.ndarray:
    """

    Arguments:
    ----------

    param: AttributeCombinationWeightEstimNB:  estim -- estimator that contains channel features and channel combinations
    param: combination_weights -- weights for each channel combination (n_samples, n_combinations)

    Returns:
    Weight for each channel-related attribute (n_samples, n_attributes)

    """
    n_channels = len(estim.channel_features)
    n_samples = combination_weights.shape[0]
    channel_weights = np.zeros((n_samples, n_channels))
    n_attributes = estim._get_n_attribs()
    n_combinations = combination_weights.shape[1]

    def create_channel_map_len(channel_combinations, combination_len=1):
        channel_map = defaultdict(list)
        for channel_combination_idx, channel_combination in enumerate(
            channel_combinations
        ):
            if len(channel_combination) == combination_len:
                for channel_idx in channel_combination:
                    channel_map[channel_idx].append(channel_combination_idx)

        return channel_map
    
    def get_comp_channel(channel_idx, n_channels):
        numbers = [i for i in range(n_channels) if i != channel_idx]
        return random.choice(numbers)

    
    single_channel_map = create_channel_map_len(
        estim.channel_combinations_, combination_len=1
    )

    def smooth_score(x: np.ndarray, K: np.ndarray, p: float = 5) -> np.ndarray:
        """
        Compute the smooth scoring function for arrays of x and K.
        
        Parameters:
            x (np.ndarray): Array of input values in [0, 1]
            K (np.ndarray): Array of attenuation coefficients in [0, 1], same shape as x or broadcastable
            p (float): Controls steepness of attenuation below 0.5 (default is 6)
        
        Returns:
            np.ndarray: Array of scores in [0, 1]
        """
        x = np.asarray(x)
        K = np.asarray(K)
        
        # Ensure broadcast compatibility
        assert x.shape == K.shape or K.shape == () or K.shape == (1,), "Shapes must be compatible"

        # Apply piecewise function
        above_half = x > 0.5
        below_half = ~above_half

        result = np.empty_like(x)
        result[above_half] = x[above_half]
        result[below_half] = x[below_half] * (x[below_half] / 0.5) ** (p * (K[below_half]))

        return result

    for channel_idx in range(n_channels):
        channel_response = combination_weights[:, single_channel_map[channel_idx]]
        comp_channel_idx = get_comp_channel(channel_idx, n_channels)

        if comp_channel_idx is None:
            channel_weights[:, channel_idx] = channel_response.ravel()
            continue

        comp_channel_response = 0.5 * np.ones((n_samples, 1))
        channel_weights[:, channel_idx] = smooth_score(
            channel_response.ravel(), comp_channel_response.ravel()
        )
    
    attribute_weights = np.zeros((1, n_samples, n_attributes))

    for channel_idx in range(n_channels):
        column_indices = estim._select_channel_group_features([channel_idx])
        attribute_weights[[0], :, column_indices] = channel_weights[:, channel_idx]

    return attribute_weights


def attribute_weights_former_filter1(
    estim: AttributeCombinationWeightEstimNB,
    combination_weights: np.ndarray,
    filters=None,
) -> np.ndarray:
    """

    Arguments:
    ----------

    param: AttributeCombinationWeightEstimNB:  estim -- estimator that contains channel features and channel combinations
    param: combination_weights -- weights for each channel combination (n_samples, n_combinations)
    param: filters -- list of filters to apply to the channel combinations

    Returns:
    Weight for each channel-related attribute (n_samples, n_attributes)

    """
    n_channels = len(estim.channel_features)
    n_samples = combination_weights.shape[0]
    channel_weights = np.zeros((n_samples, n_channels))
    n_attributes = estim._get_n_attribs()
    n_combinations = combination_weights.shape[1]

    if filters is None:
        filters = [channel_group_filter_all]

    n_filters = len(filters)

    attribute_weights = np.zeros((n_filters, n_samples, n_attributes))
    channel_combination_map = {
        comb: idx for idx, comb in enumerate(estim.channel_combinations_)
    }
    for filter_idx in range(n_filters):
        channel_weights = np.zeros((n_samples, n_channels))

        filtered_combinations = [*filters[filter_idx](estim.channel_combinations_)]
        n_filtered_combinations = len(filtered_combinations)

        filtered_column_indices = [
            channel_combination_map[comb] for comb in filtered_combinations
        ]

        for filtered_combination_column_idx, filtered_combination in zip(
            filtered_column_indices, filtered_combinations
        ):
            for channel_idx in filtered_combination:
                channel_weights[:, channel_idx] += combination_weights[
                    :, filtered_combination_column_idx
                ]

        channel_weights = channel_weights / n_combinations
        # assert np.all(channel_weights >= 0), "Negative channel weights"
        # assert np.all(
        #     channel_weights <= 1
        # ), "Channel weights greater than 1"
        # assert np.all(np.isfinite(channel_weights)), "Channel weights are not finite"
        # assert np.all(np.isnan(channel_weights) == False), "Channel weights are NaN"

        for channel_idx in range(n_channels):
            column_indices = estim._select_channel_group_features([channel_idx])
            attribute_weights[filter_idx, :, column_indices] = channel_weights[
                :, channel_idx
            ]

    return attribute_weights


class AttributeCombinationWeightEstimNB(BaseEstimator, ClassifierMixin):

    def __init__(
        self,
        model_prototype=None,
        outlier_detector_prototype=None,
        random_state=0,
        channel_features=None,
        channel_combination_generator=None,
        channel_combination_generator_options=None,
        attribute_weights_former=attribute_weights_former_mean1,
    ) -> None:
        """
        #Meta-classifier that uses attribute weights to perform classification.
        It uses base classifier and outlier detectors to estimate attribute weights.
        The method accepts transformed attributes not raw signals

        Arguments:

        ----------

        model_prototype -- base classifier prototype for the ensemble

        outlier_detector_prototype -- prototype for outlier detector

        random_state -- seed for the random generator

        channel_features -- a list that contains channel specific features

        channel_combination_generator -- Function that generates channel combinations or None

        channel_combination_generator_options -- Options for the channel generator

        """
        super().__init__()

        self.model_prototype = model_prototype
        self.outlier_detector_prototype = outlier_detector_prototype

        self.random_state = random_state
        self.channel_features = channel_features
        self.channel_combination_generator = channel_combination_generator
        self.channel_combination_generator_options = (
            channel_combination_generator_options
        )
        self.attribute_weights_former = attribute_weights_former

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

        effective_channel_combination_generator_options = (
            self.channel_combination_generator_options
            if self.channel_combination_generator_options is not None
            else {"group_sizes": [2]}
        )
        effective_channel_combination_generator = (
            self.channel_combination_generator
            if self.channel_combination_generator is not None
            else channel_group_gen
        )

        self.channel_combinations_ = effective_channel_combination_generator(
            range(n_channels), **effective_channel_combination_generator_options
        )

        n_combinations = len(self.channel_combinations_)

        self.outlier_detectors_ = np.zeros(n_combinations, dtype=object)
        y_o = np.ones_like(y, dtype=np.int64)
        # create outlier detectors
        for channel_combination_id, channel_combination in enumerate(
            self.channel_combinations_
        ):
            column_indices = self._select_channel_group_features(channel_combination)
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
            self.outlier_detectors_[channel_combination_id] = base_detector

    def _create_base_classifier(self, X, y):
        """
        Trains base classifier
        """

        self.base_classifier_ = (
            clone(self.model_prototype)
            if self.model_prototype is not None
            else EstimatorNB()
        )

        y_train = self.label_encoder_.transform(y)
        self.base_classifier_.fit(X, y_train)

    def fit(self, X, y=None):
        self.random_state_ = check_random_state(self.random_state)

        Xp, yp = self._preprocess_X_y(X, y)
        self.classes_ = np.unique(yp)

        self.label_encoder_ = LabelEncoder()
        self.label_encoder_.fit(yp)

        self._prepare_base_outlier_detectors(Xp, yp)

        self._create_base_classifier(Xp, yp)

        return self

    def predict(self, X, y=None):

        check_is_fitted(
            self,
            (
                "classes_",
                "base_classifier_",
                "outlier_detectors_",
                "random_state_",
                "label_encoder_",
                "channel_combinations_",
            ),
        )

        soft_predictions = self.predict_proba(X)
        class_idxs = np.argmax(soft_predictions, axis=1)
        return self.label_encoder_.inverse_transform(class_idxs)

    def transform_attribute_weights(self, weights):
        """
        Transforms the weights if needed.
        This version just copy them.
        Placeholder for use in subclasses.

        Arguments:
        ----------

        param:weights:np.ndarray -- weights (n_variants ,n_samples, n_attributes)


        Returns:
        --------

        transformed weights (n_variants, n_samples, n_attributes)
        """
        return weights

    def form_attribute_weights(self, combination_weights: np.ndarray) -> np.ndarray:
        """

        Arguments:
        ----------

        param: combination_weights -- weights for each channel combination (n_samples, n_combinations)

        Returns:
        Weight for each channel-related attribute (n_samples, n_attributes)

        """
        attribute_weights = self.attribute_weights_former(self, combination_weights)

        return attribute_weights

    def _get_n_attribs(self):
        n_attribs = 0
        for ch_feats in self.channel_features:
            n_attribs += len(ch_feats)

        return n_attribs

    def _combine_probas(self, probas_variants):
        """
        Combines probabilities from different variants.
        This version just returns the first one.
        Placeholder for use in subclasses.

        Arguments:
        ----------

        param:probas_variants:np.ndarray -- probabilities for each variant (n_variants, n_samples, n_classes)

        Returns:
        --------

        combined probabilities (n_samples, n_classes)
        """

        return np.median(probas_variants, axis=0)

    def predict_proba(self, X, y=None):

        check_is_fitted(
            self,
            (
                "classes_",
                "base_classifier_",
                "outlier_detectors_",
                "random_state_",
                "label_encoder_",
                "channel_combinations_",
            ),
        )

        n_objects = X.shape[0]
        n_attributes = self._get_n_attribs()
        assert n_attributes == X.shape[1]

        n_channels = len(self.channel_features)
        n_combinations = len(self.channel_combinations_)

        attribute_weights_variants = np.zeros((n_objects, n_attributes))

        combination_weights = np.zeros((n_objects, n_combinations))

        for channel_combination_id, channel_combination in enumerate(
            self.channel_combinations_
        ):
            # (n_samples, 1)
            inlier_scores = self.outlier_detectors_[
                channel_combination_id
            ].predict_proba(X)[:, 1, np.newaxis]
            combination_weights[:, channel_combination_id] = inlier_scores.ravel()

        n_classes = len(self.classes_)
        attribute_weights_variants = self.form_attribute_weights(combination_weights)

        attribute_weights_variants = self.transform_attribute_weights(
            attribute_weights_variants
        )
        n_variants = attribute_weights_variants.shape[0]

        probas_variants = np.zeros((n_variants, n_objects, n_classes))

        for variant_idx in range(n_variants):
            for sample_idx in range(n_objects):
                probas_variants[variant_idx, sample_idx, :] = (
                    self.base_classifier_._predict_proba(
                        X[sample_idx : sample_idx + 1, :],
                        attribute_weights_variants[variant_idx, sample_idx, :],
                    )
                )
        probas = self._combine_probas(probas_variants)
        normalized_probas = probas / probas.sum(axis=1, keepdims=True)

        return normalized_probas
