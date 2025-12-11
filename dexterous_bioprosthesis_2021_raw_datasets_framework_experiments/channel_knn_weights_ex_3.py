import datetime
from enum import Enum
import logging
import os
import string
import warnings
from sklearn.base import check_is_fitted
from weightedknn.estimators.weightedknn import WeightedKNNClassifier


from results_storage.results_storage import ResultsStorage
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.attribute_sel_nb.attribute_weight_estim_nb_hard import (
    AttributeWeightEstimNBHard,
)
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.attribute_weight_estim_nb import (
    AttributeWeightEstimNB,
)
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.channel_combination_classifier_outliers_fast2 import (
    ChannelCombinationClassifierOutliersFast2,
)
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.channel_combination_classifier_outliers_fast2s import (
    ChannelCombinationClassifierOutliersFast2S,
)
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.des.des_p_outlier_full import (
    DespOutlierFull,
)
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.des.des_p_outlier_full2 import (
    DespOutlierFull2,
)
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.fuzzy_knn.fuzzy_knn import (
    FuzzyKNN,
)
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.fuzzy_knn.inlier_score_transformers.inlier_score_transformer_crisp import (
    InlierScoreTransformerCrisp,
)
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.fuzzy_knn.inlier_score_transformers.inlier_score_transformer_low_para import (
    InlierScoreTransformerLowPara,
)
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.fuzzy_knn.inlier_score_transformers.inlier_score_transformer_scaled_sigmoid import (
    InlierScoreTransformerScaledSigmoid,
)
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.outlier.outlier_detector_combiner_mean import (
    OutlierDetectorCombinerMean,
)
from dexterous_bioprosthesis_2021_raw_datasets_framework.tools.stats_tools import (
    p_val_matrix_to_vec,
    p_val_vec_to_matrix,
)

import pandas as pd

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    make_scorer,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM
from statsmodels.stats.multitest import multipletests
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.parameter_selection.gridsearchcv_oneclass2 import (
    GridSearchCVOneClass2,
)
from dexterous_bioprosthesis_2021_raw_datasets_framework.preprocessing.one_class.outlier_generators.outlier_generator_uniform import (
    OutlierGeneratorUniform,
)
from dexterous_bioprosthesis_2021_raw_datasets_framework.raw_signals.raw_signals_io import (
    read_signals_from_dirs,
)
from dexterous_bioprosthesis_2021_raw_datasets_framework.raw_signals_filters.raw_signals_filter_channel_idx import (
    RawSignalsFilterChannelIdx,
)
from dexterous_bioprosthesis_2021_raw_datasets_framework.raw_signals_spoilers.raw_signals_spoiler_multiple import (
    RawSignalsSpoilerMultiple,
)
from dexterous_bioprosthesis_2021_raw_datasets_framework.raw_signals_spoilers.raw_signals_spoiler_sine import (
    RawSignalsSpoilerSine,
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


from dexterous_bioprosthesis_2021_raw_datasets_framework.raw_signals_spoilers.raw_signals_spoiler_damper import (
    RawSignalsSpoilerDamper,
)
from dexterous_bioprosthesis_2021_raw_datasets_framework.raw_signals_spoilers.raw_signals_spoiler_gauss import (
    RawSignalsSpoilerGauss,
)
from dexterous_bioprosthesis_2021_raw_datasets_framework.raw_signals_spoilers.raw_signals_spoiler_multiple import (
    RawSignalsSpoilerMultiple,
)
from dexterous_bioprosthesis_2021_raw_datasets_framework.raw_signals_spoilers.raw_signals_spoiler_cubicclipper import (
    RawSignalsSpoilerCubicClipper,
)
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.fuzzy_knn.inlier_score_transformers.inlier_score_transformer_smoothstep import (
    InlierScoreTransformerSmoothstep,
)


from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from dexterous_bioprosthesis_2021_raw_datasets_framework.tools.warnings import (
    warn_with_traceback,
)

from dexterous_bioprosthesis_2021_raw_datasets_framework_experiments import settings

from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)

from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
import pickle
from sklearn.metrics import accuracy_score

from dexterous_bioprosthesis_2021_raw_datasets_framework_experiments.tools import logger
from dexterous_bioprosthesis_2021_raw_datasets_framework.tools.progressparallel import (
    ProgressParallel,
)
from joblib import delayed
from copy import deepcopy


from scipy.stats import wilcoxon
import seaborn as sns


import random
from scipy.stats import rankdata

# Plot line colors and markers
from cycler import cycler


from sklearn.model_selection._search import _estimator_has
from sklearn.utils._available_if import available_if
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OutputCodeClassifier

N_INTERNAL_SPLITS = 4


class GridSearchCVPP(GridSearchCV):
    @available_if(_estimator_has("_predict_proba"))
    def _predict_proba(self, X, weights=None):
        check_is_fitted(self)
        return self.best_estimator_._predict_proba(X, weights)


class PlotConfigurer:

    def __init__(self) -> None:
        self.is_configured = False

    def configure_plots(self):
        if not self.is_configured:
            # print("Configuring plot")
            dcc = plt.rcParams["axes.prop_cycle"]
            mcc = cycler(
                marker=[
                    "o",
                    "s",
                    "D",
                    "^",
                    "v",
                    ">",
                    "<",
                    "p",
                    "*",
                    "x",
                    # "h",
                    # "H",
                    # "|",
                    # "_",
                ]
            )
            cc = cycler(
                color=[
                    "r",
                    "g",
                ]
            )

            lcc = cycler(
                linestyle=[
                    "-",
                    "--",
                    "-.",
                    ":",  # Default styles
                    (0, (1, 1)),  # Densely dotted
                    (0, (3, 1, 1, 1)),  # Short dash-dot
                    (0, (5, 1)),  # Loosely dashed
                    (0, (5, 5)),  # Medium dashed
                    (0, (8, 3, 2, 3)),  # Dash-dot-dot
                    (0, (10, 2, 2, 2)),  # Long dash, short dot
                    (0, (15, 5, 5, 2)),  # Complex pattern
                ]
            )
            c = lcc * (dcc + mcc)

            plt.rc("axes", prop_cycle=c)
            # print('Params set', plt.rcParams['axes.prop_cycle'])
            self.is_configured = True


configurer = PlotConfigurer()


def wavelet_extractor2(wavelet_level=2):
    extractor = SetCreatorDWT(
        num_levels=wavelet_level,
        wavelet_name="db6",
        extractors=[
            NpSignalExtractorMav(),
            NpSignalExtractorSsc(),
        ],
    )
    return extractor


def create_extractors():

    extractors_dict = {
        "DWT": wavelet_extractor2(),
    }

    return extractors_dict


def generate_tuned_wknn():

    params = {"n_neighbors": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]}

    knn_est = WeightedKNNClassifier(
        algorithm="brute",
        weights="distance",
    )

    bac_scorer = make_scorer(balanced_accuracy_score)
    cv = StratifiedKFold(n_splits=N_INTERNAL_SPLITS, shuffle=True, random_state=0)
    gs = GridSearchCVPP(estimator=knn_est, param_grid=params, scoring=bac_scorer, cv=cv)
    return gs


def generate_classifiers():
    classifiers = {
        "WKNN": generate_tuned_wknn(),
    }
    return classifiers


def warn_unknown_labels(y_true, y_pred):
    true_set = set(y_true)
    pred_set = set(y_pred)
    diffset = pred_set.difference(true_set)
    if len(diffset) > 0:
        warnings.warn("Diffset: {}".format(diffset))


def acc_m(y_true, y_pred, labels=None, average=None, zero_division=None):
    return accuracy_score(y_true, y_pred)


def bac_m(y_true, y_pred, labels=None, average=None, zero_division=None):
    return balanced_accuracy_score(y_true, y_pred)


def kappa_m(y_true, y_pred, labels=None, average=None, zero_division=None):
    return cohen_kappa_score(y_true, y_pred)


def f1_score_m(y_true, y_pred, labels=None, average=None, zero_division=None):
    return f1_score(y_true, y_pred, average="micro")


def generate_metrics():
    metrics = {
        "ACC": acc_m,
        "BAC": bac_m,
        "Kappa": kappa_m,
        "F1": f1_score_m,
    }
    return metrics


def generate_base(
    base_classifier, channel_features, group_sizes=[2], outlier_detector_prototype=None
):

    return Pipeline(
        [("scaler", RobustScaler()), ("classifier", deepcopy(base_classifier))]
    )


def generate_fknn(
    base_classifier, channel_features, group_sizes=[2], outlier_detector_prototype=None
):
    """
    Proposed Fuzzy KNN.
    """
    pipeline = Pipeline(
        [
            ("scaler", RobustScaler()),
            (
                "estimator",
                FuzzyKNN(
                    outlier_detector_prototype=deepcopy(outlier_detector_prototype),
                    channel_features=channel_features,
                    random_state=0,
                    n_neighbors=5,
                ),
            ),
        ]
    )
    params = {
        "estimator__n_neighbors": [None, *range(1, 25, 2)],
        "estimator__inlier_score_transformer": [
            None,
            InlierScoreTransformerLowPara(),
            InlierScoreTransformerScaledSigmoid(),
            InlierScoreTransformerSmoothstep(),
        ],
    }
    bac_scorer = make_scorer(balanced_accuracy_score)
    cv = StratifiedKFold(n_splits=N_INTERNAL_SPLITS, shuffle=True, random_state=0)
    gs = GridSearchCV(estimator=pipeline, param_grid=params, scoring=bac_scorer, cv=cv)
    return gs


def generate_fknn_crisp(
    base_classifier, channel_features, group_sizes=[2], outlier_detector_prototype=None
):
    """
    Proposed Fuzzy KNN.
    """
    pipeline = Pipeline(
        [
            ("scaler", RobustScaler()),
            (
                "estimator",
                FuzzyKNN(
                    outlier_detector_prototype=deepcopy(outlier_detector_prototype),
                    channel_features=channel_features,
                    random_state=0,
                    n_neighbors=5,
                ),
            ),
        ]
    )
    params = {
        "estimator__n_neighbors": [None, *range(1, 25, 2)],
        "estimator__inlier_score_transformer": [
            InlierScoreTransformerCrisp(),
        ],
    }
    bac_scorer = make_scorer(balanced_accuracy_score)
    cv = StratifiedKFold(n_splits=N_INTERNAL_SPLITS, shuffle=True, random_state=0)
    gs = GridSearchCV(estimator=pipeline, param_grid=params, scoring=bac_scorer, cv=cv)
    return gs


def generate_desp_outlier_full(
    base_classifier, channel_features, group_sizes=[2], outlier_detector_prototype=None
):

    single_channel_ensemble = ChannelCombinationClassifierOutliersFast2(
        outlier_detector_prototype=deepcopy(outlier_detector_prototype),
        channel_combination_generator_options={"group_sizes": [1]},
        es_arguments={"k": 3, "random_state": 0},
        es_class=DespOutlierFull,
        model_prototype=deepcopy(base_classifier),
        channel_features=channel_features,
        partial_train=False,
    )

    return single_channel_ensemble


def generate_desp_outlier_full_soft_mean(
    base_classifier, channel_features, group_sizes=[2], outlier_detector_prototype=None
):

    single_channel_ensemble = ChannelCombinationClassifierOutliersFast2S(
        outlier_detector_prototype=deepcopy(outlier_detector_prototype),
        channel_combination_generator_options={"group_sizes": [1]},
        es_arguments={
            "k": 3,
            "mode": "weighting",
            "random_state": 0,
        },  # Important for weighted combination!
        es_class=DespOutlierFull2,
        model_prototype=deepcopy(base_classifier),
        channel_features=channel_features,
        outlier_detector_combiner_class=OutlierDetectorCombinerMean,
        partial_train=False,
    )

    return single_channel_ensemble


def generate_d_nb_hard(
    base_classifier, channel_features, group_sizes=[2], outlier_detector_prototype=None
):
    pipeline = Pipeline(
        [
            (
                "estimator",
                AttributeWeightEstimNBHard(
                    model_prototype=deepcopy(base_classifier),
                    channel_features=channel_features,
                    outlier_detector_prototype=deepcopy(outlier_detector_prototype),
                    random_state=0,
                ),
            )
        ]
    )
    return pipeline


def generate_d_nb_soft(
    base_classifier, channel_features, group_sizes=[2], outlier_detector_prototype=None
):
    pipeline = Pipeline(
        [
            (
                "estimator",
                AttributeWeightEstimNB(
                    model_prototype=deepcopy(base_classifier),
                    channel_features=channel_features,
                    outlier_detector_prototype=deepcopy(outlier_detector_prototype),
                    random_state=0,
                ),
            )
        ]
    )
    return pipeline

def generate_random_forest(base_classifier, channel_features, group_sizes=[2], outlier_detector_prototype=None):
    
    params = {
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
    }

    rf_est = RandomForestClassifier(random_state=0)

    bac_scorer = make_scorer(balanced_accuracy_score)
    cv = StratifiedKFold(n_splits=N_INTERNAL_SPLITS, shuffle=True, random_state=0)
    gs = GridSearchCVPP(estimator=rf_est, param_grid=params, scoring=bac_scorer, cv=cv)
    return gs

def generate_ecoc(base_classifier, channel_features, group_sizes=[2], outlier_detector_prototype=None):
    params = {
        "estimator__max_depth": [None, 10, 20],
        "estimator__min_samples_split": [2, 5, 10],
    }
    base_rf = RandomForestClassifier(random_state=0)
    ecoc_est = OutputCodeClassifier(
        estimator=base_rf,
        code_size=2,
        random_state=0,
    )
    bac_scorer = make_scorer(balanced_accuracy_score)
    cv = StratifiedKFold(n_splits=N_INTERNAL_SPLITS, shuffle=True, random_state=0)
    gs = GridSearchCVPP(estimator=ecoc_est, param_grid=params, scoring=bac_scorer, cv=cv)
    return gs

# TODO uncomment
def generate_methods():
    methods = {
        "B": generate_base,
        "RF": generate_random_forest,
        "ECOC": generate_ecoc,
        "DO": generate_desp_outlier_full,  # FROM CLDD 2024 K=1
        "DOa": generate_desp_outlier_full_soft_mean,  # Soft Weighting K=1
        "AW": generate_d_nb_soft,  # From CORES 2025 soft version
        "AWc": generate_d_nb_hard,  # From CORES 2025 hard version!
        "FKNN": generate_fknn,
        "FKNNc": generate_fknn_crisp,
    }
    return methods


# TODO -- INFO name '0' only for compatibility.
def generate_spoiled_ch_fraction():
    spoiled_channels_fractions = {
        "0": None,  # Random chanell noise
    }
    return spoiled_channels_fractions


def generate_ocsvm():

    nu_list = [0.1 * (i + 1) for i in range(9)]
    nu_list.append(0.05)
    params = {
        "estimator__gamma": ["auto", "scale"],
        "estimator__nu": nu_list,
    }

    kappa_scorer = make_scorer(balanced_accuracy_score)
    generator = OutlierGeneratorUniform()

    pipeline = Pipeline([("scaler", RobustScaler()), ("estimator", OneClassSVM())])
    cv = StratifiedKFold(n_splits=N_INTERNAL_SPLITS, shuffle=True, random_state=0)
    gs = GridSearchCVOneClass2(
        pipeline,
        test_outlier_generator=generator,
        param_grid=params,
        scoring=kappa_scorer,
        cv=cv,
    )

    return gs


def generate_outlier_detectors():
    detectors = {
        "SVM": generate_ocsvm,
    }
    return detectors


def generate_spoiler_All(snr, channels_spoiled_frac):
    return RawSignalsSpoilerMultiple(
        spoilers=[
            RawSignalsSpoilerSine(
                snr=snr, channels_spoiled_frac=channels_spoiled_frac, frequency=50
            ),
            RawSignalsSpoilerDamper(
                snr=snr, channels_spoiled_frac=channels_spoiled_frac
            ),
            RawSignalsSpoilerCubicClipper(
                snr=snr, channels_spoiled_frac=channels_spoiled_frac
            ),
            RawSignalsSpoilerGauss(
                snr=snr, channels_spoiled_frac=channels_spoiled_frac
            ),
            RawSignalsSpoilerSine(
                snr=snr,
                channels_spoiled_frac=channels_spoiled_frac,
                frequency=1,
                freq_deviation=0.5,
            ),
        ],
        spoiler_relabalers=None,
    )


def generate_spoiler_50Hz(snr, channels_spoiled_frac):
    return RawSignalsSpoilerMultiple(
        spoilers=[
            RawSignalsSpoilerSine(
                snr=snr, channels_spoiled_frac=channels_spoiled_frac, frequency=50
            )
        ],
        spoiler_relabalers=None,
    )


def generate_spoiler_damper(snr, channels_spoiled_frac):
    return RawSignalsSpoilerMultiple(
        spoilers=[
            RawSignalsSpoilerDamper(
                snr=snr, channels_spoiled_frac=channels_spoiled_frac
            )
        ],
        spoiler_relabalers=None,
    )


def generate_spoiler_clipper(snr, channels_spoiled_frac):
    return RawSignalsSpoilerMultiple(
        spoilers=[
            RawSignalsSpoilerCubicClipper(
                snr=snr, channels_spoiled_frac=channels_spoiled_frac
            )
        ],
        spoiler_relabalers=None,
    )


def generate_spoiler_gauss(snr, channels_spoiled_frac):
    return RawSignalsSpoilerMultiple(
        spoilers=[
            RawSignalsSpoilerGauss(snr=snr, channels_spoiled_frac=channels_spoiled_frac)
        ],
        spoiler_relabalers=None,
    )


def generate_spoiler_baseline_wander(snr, channels_spoiled_frac):
    return RawSignalsSpoilerMultiple(
        spoilers=[
            RawSignalsSpoilerSine(
                snr=snr,
                channels_spoiled_frac=channels_spoiled_frac,
                frequency=1,
                freq_deviation=0.5,
            )
        ],
        spoiler_relabalers=None,
    )


def generate_spoilers_gens():
    spoilers = {"All": generate_spoiler_All}
    return spoilers


def get_snr_levels():
    return [12, 10, 6, 5, 4, 3, 2, 1, 0]


class Dims(Enum):
    FOLDS = "folds"
    METRICS = "metrics"
    EXTRACTORS = "extractors"
    CLASSIFIERS = "classifiers"
    METHODS = "methods"
    DETECTORS = "detectors"
    SPOILERS = "spoilers"
    SNR = "snr"
    CHANNEL_SPOIL_FRACTION = "channel_spoil_fraction"


def run_experiment(
    input_data_dir_list,
    output_directory,
    n_splits=10,
    n_repeats=4,
    random_state=0,
    n_jobs=-1,
    overwrite=True,
    n_channels=None,
    append=True,
    progress_log_handler=None,
    comment_str="",
):

    os.makedirs(output_directory, exist_ok=True)

    comment_file = os.path.join(output_directory, "comment.txt")
    with open(comment_file, "w") as f:
        f.write(comment_str)
        f.write("Start time: {}\n".format(datetime.datetime.now()))
        f.write("\n")
        f.write("Data sets:\n")
        for data_set in data_sets:
            f.write(data_set)
            f.write("\n")
        f.write("\n")
        f.write("Function Parameters:\n")
        f.write(f"n_splits: {n_splits}\n")
        f.write(f"n_repeats: {n_repeats}\n")
        f.write(f"random_state: {random_state}\n")
        f.write(f"n_jobs: {n_jobs}\n")
        f.write(f"overwrite: {overwrite}\n")
        f.write(f"n_channels: {n_channels}\n")
        f.write(f"append: {append}\n")
        f.write("\n")

    metrics = generate_metrics()
    n_metrics = len(metrics)

    extractors_dict = create_extractors()
    n_extr = len(extractors_dict)

    classifiers_dict = generate_classifiers()
    n_classifiers = len(classifiers_dict)

    methods = generate_methods()
    n_methods = len(methods)

    skf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    n_folds = skf.get_n_splits()

    detector_generators = generate_outlier_detectors()
    n_detector_generators = len(detector_generators)

    spoiler_generators = generate_spoilers_gens()
    n_spoiler_generators = len(spoiler_generators)

    snrs = get_snr_levels()
    n_snrs = len(snrs)

    channel_spoil_fractions = generate_spoiled_ch_fraction()
    n_channel_spoil_fraction = len(channel_spoil_fractions)

    coords = {
        Dims.METRICS.value: [k for k in metrics],
        Dims.EXTRACTORS.value: [k for k in extractors_dict],
        Dims.CLASSIFIERS.value: [k for k in classifiers_dict],
        Dims.METHODS.value: [k for k in methods],
        Dims.DETECTORS.value: [k for k in detector_generators],
        Dims.SPOILERS.value: [k for k in spoiler_generators],
        Dims.SNR.value: [k for k in snrs],
        Dims.CHANNEL_SPOIL_FRACTION.value: [k for k in channel_spoil_fractions],
        Dims.FOLDS.value: [k for k in range(n_folds)],
    }

    for in_dir in tqdm(input_data_dir_list, desc="Data sets"):

        # metrics x extractors x classifiers x detectors x spoilers x snr x ch_fraction x methods x folds
        results = np.zeros(
            (
                n_metrics,
                n_extr,
                n_classifiers,
                n_detector_generators,
                n_spoiler_generators,
                n_snrs,
                n_channel_spoil_fraction,
                n_methods,
                n_folds,
            )
        )

        set_name = os.path.basename(in_dir)

        result_file_path = os.path.join(output_directory, "{}.pickle".format(set_name))

        exists = os.path.isfile(result_file_path)

        if exists and not (overwrite):
            print("Skipping {} !".format(set_name))
            continue

        pre_set = read_signals_from_dirs(in_dir)
        raw_set = pre_set["accepted"]

        if n_channels is not None:
            n_set_channels = raw_set[0].to_numpy().shape[1]
            n_effective_channels = min((n_set_channels, n_channels))
            indices = [*range(n_effective_channels)]
            filter = RawSignalsFilterChannelIdx(indices)
            raw_set = filter.fit_transform(raw_set)

        y = np.asanyarray(raw_set.get_labels())
        num_labels = len(np.unique(y))

        results_storage = ResultsStorage.init_coords(coords=coords, name="Storage")
        if os.path.exists(result_file_path):
            with open(result_file_path, "rb") as fh:
                loaded_storage = pickle.load(fh)
            loaded_storage.name = "loaded"
            results_storage = ResultsStorage.merge_with_loaded(
                loaded_obj=loaded_storage, new_obj=results_storage
            )

        def compute(fold_idx, train_idx, test_idx):
            # For tracing warnings inside multiprocessing
            warnings.showwarning = warn_with_traceback

            raw_train = raw_set[train_idx]
            raw_test = raw_set[test_idx]

            fold_res = []

            for extractor_name in ResultsStorage.coords_need_recalc(
                results_storage, Dims.EXTRACTORS.value
            ):
                extractor = extractors_dict[extractor_name]

                X_train, y_train, _ = extractor.fit_transform(raw_train)
                channel_features = extractor.get_channel_attribs_indices()

                for base_classifier_name in ResultsStorage.coords_need_recalc(
                    results_storage, Dims.CLASSIFIERS.value
                ):
                    base_classifier = classifiers_dict[base_classifier_name]

                    for detector_generator_name in ResultsStorage.coords_need_recalc(
                        results_storage, Dims.DETECTORS.value
                    ):
                        outlier_detector = detector_generators[
                            detector_generator_name
                        ]()

                        for method_name in tqdm(
                            ResultsStorage.coords_need_recalc(
                                results_storage, Dims.METHODS.value
                            ),
                            leave=False,
                            total=n_methods,
                            desc="Methods, Fold {}".format(fold_idx),
                        ):
                            method_creator = methods[method_name]

                            method = method_creator(
                                base_classifier,
                                channel_features,
                                [1],
                                outlier_detector_prototype=outlier_detector,
                            )

                            method.fit(X_train, y_train)

                            for snr in tqdm(
                                ResultsStorage.coords_need_recalc(
                                    results_storage, Dims.SNR.value
                                ),
                                total=n_snrs,
                                leave=False,
                                desc="SNR, fold {}".format(fold_idx),
                            ):

                                for (
                                    channel_spoil_f_name
                                ) in ResultsStorage.coords_need_recalc(
                                    results_storage,
                                    Dims.CHANNEL_SPOIL_FRACTION.value,
                                ):

                                    spoiled_fraction = channel_spoil_fractions[
                                        channel_spoil_f_name
                                    ]

                                    for (
                                        spoiler_generator_name
                                    ) in ResultsStorage.coords_need_recalc(
                                        results_storage,
                                        Dims.SPOILERS.value,
                                    ):

                                        signal_spoiler = spoiler_generators[
                                            spoiler_generator_name
                                        ](
                                            snr=snr,
                                            channels_spoiled_frac=spoiled_fraction,
                                        )

                                        raw_spoiled_test = signal_spoiler.fit_transform(
                                            raw_test
                                        )
                                        raw_spoiled_test += raw_test

                                        X_test, y_test, _ = extractor.transform(
                                            raw_spoiled_test
                                        )

                                        # TODO no Oracle. Leave like that?
                                        try:
                                            y_pred = method.predict(X_test, y_test)
                                        except TypeError:
                                            y_pred = method.predict(X_test)
                                        except Exception as e:
                                            raise e

                                        y_gt = y_test

                                        for (
                                            metric_name
                                        ) in ResultsStorage.coords_need_recalc(
                                            results_storage, Dims.METRICS.value
                                        ):
                                            metric = metrics[metric_name]

                                            metric_value = metric(y_gt, y_pred)

                                            # metrics x extractors x classifiers x detectors x spoilers x snr x ch_fraction x methods x folds
                                            fold_res.append(
                                                (
                                                    metric_name,
                                                    extractor_name,
                                                    base_classifier_name,
                                                    detector_generator_name,
                                                    spoiler_generator_name,
                                                    snr,
                                                    channel_spoil_f_name,
                                                    method_name,
                                                    fold_idx,
                                                    metric_value,
                                                )
                                            )
            return fold_res

        results_list = ProgressParallel(
            n_jobs=n_jobs,
            desc=f"K-folds for set: {set_name}",
            total=skf.get_n_splits(),
            leave=False,
            file_handler=progress_log_handler,
        )(
            delayed(compute)(fold_idx, train_idx, test_idx)
            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(raw_set, y))
        )

        for result_sublist in results_list:
            # metrics x extractors x classifiers x detectors x spoilers x snr x ch_fraction x methods x folds
            for (
                metric_name,
                extractor_name,
                base_classifier_name,
                detector_generator_name,
                spoiler_generator_name,
                snr,
                channel_spoil_f_name,
                method_name,
                fold_idx,
                metric_value,
            ) in result_sublist:
                results_storage.loc[
                    {
                        Dims.METRICS.value: metric_name,
                        Dims.EXTRACTORS.value: extractor_name,
                        Dims.CLASSIFIERS.value: base_classifier_name,
                        Dims.DETECTORS.value: detector_generator_name,
                        Dims.SPOILERS.value: spoiler_generator_name,
                        Dims.SNR.value: snr,
                        Dims.CHANNEL_SPOIL_FRACTION.value: channel_spoil_f_name,
                        Dims.METHODS.value: method_name,
                        Dims.FOLDS.value: fold_idx,
                    }
                ] = metric_value

        logging.debug("Dumping results")
        with open(result_file_path, "wb") as fh:
            pickle.dump(results_storage, file=fh)


def analyze_results_2C(results_directory, output_directory, alpha=0.05):
    configurer.configure_plots()
    result_files = [f for f in os.listdir(results_directory) if f.endswith(".pickle")]

    for result_file in result_files:
        result_file_basename = os.path.splitext(result_file)[0]
        result_file_path = os.path.join(results_directory, result_file)

        results_holder = pickle.load(open(result_file_path, "rb"))

        method_names = results_holder[Dims.METHODS.value].values
        n_methods = len(method_names)

        pdf_file_path = os.path.join(
            output_directory, "{}_snr_m2.pdf".format(result_file_basename)
        )

        with PdfPages(pdf_file_path) as pdf:

            for metric_name in results_holder[Dims.METRICS.value].values:

                for extractor_name in results_holder[Dims.EXTRACTORS.value].values:

                    for classifier_name in results_holder[
                        Dims.CLASSIFIERS.value
                    ].values:

                        for outlier_detector_name in results_holder[
                            Dims.DETECTORS.value
                        ].values:

                            # metrics x extractors x classifiers x detectors x spoilers x snr x ch_fraction x methods x folds
                            sub_results = results_holder.loc[
                                {
                                    Dims.METRICS.value: metric_name,
                                    Dims.EXTRACTORS.value: extractor_name,
                                    Dims.CLASSIFIERS.value: classifier_name,
                                    Dims.DETECTORS.value: outlier_detector_name,
                                }
                            ].to_numpy()  # spoilers x snr x chan_frac x methods x folds

                            # methods x spoilers x snrs x  ch_frac x folds
                            sub_results = np.moveaxis(
                                sub_results, [0, 1, 2, 3, 4], [3, 0, 1, 2, 4]
                            )

                            sub_results = np.mean(
                                sub_results, axis=(0, 2)
                            )  # snr x methods x folds

                            df = pd.DataFrame(columns=["snr", "method", "value"])

                            for i, snr_value in enumerate(
                                results_holder[Dims.SNR.value].values
                            ):
                                for j, method_name in enumerate(method_names):
                                    for k in range(sub_results.shape[2]):
                                        new_row = pd.DataFrame(
                                            {
                                                "snr": snr_value,
                                                "method": method_name,
                                                "value": sub_results[i, j, k],
                                            },
                                            index=[0],
                                        )
                                        df = pd.concat(
                                            [new_row, df.loc[:]]
                                        ).reset_index(drop=True)
                            sns.set(style="whitegrid")
                            sns.boxplot(
                                x=df["snr"],
                                y=df["value"],
                                hue=df["method"],
                                palette="husl",
                            )
                            plt.title(
                                "{}, {}, {}, Od:{}".format(
                                    metric_name,
                                    extractor_name,
                                    classifier_name,
                                    outlier_detector_name,
                                )
                            )

                            pdf.savefig()
                            plt.close()


def analyze_results_2C_ranks(results_directory, output_directory, alpha=0.05):
    configurer.configure_plots()
    result_files = [f for f in os.listdir(results_directory) if f.endswith(".pickle")]
    n_files = len(result_files)

    # Init

    classifier_names = None
    n_classifiers = None
    metric_names = None
    n_metrics = None
    extractor_names = None
    n_extractors = None
    outlier_detector_names = None
    n_outlier_detectors = None
    spoiler_names = None
    n_spoilers = None
    snrs = None
    n_snrs = None
    spoiled_channels_fraction = None
    n_sp_channel_fraction = None
    method_names = None
    n_methods = None
    n_folds = None

    global_results = None

    for result_file_id, result_file in enumerate(result_files):
        result_file_basename = os.path.splitext(result_file)[0]
        result_file_path = os.path.join(results_directory, result_file)

        results_storage = pickle.load(open(result_file_path, "rb"))
        results = results_storage.transpose(
            Dims.METRICS.value,
            Dims.EXTRACTORS.value,
            Dims.CLASSIFIERS.value,
            Dims.DETECTORS.value,
            Dims.SPOILERS.value,
            Dims.SNR.value,
            Dims.CHANNEL_SPOIL_FRACTION.value,
            Dims.METHODS.value,
            Dims.FOLDS.value,
        )

        if global_results is None:

            classifier_names = results_storage[Dims.CLASSIFIERS.value].values
            n_classifiers = len(classifier_names)
            metric_names = results_storage[Dims.METRICS.value].values
            n_metrics = len(metric_names)
            extractor_names = results_storage[Dims.EXTRACTORS.value].values
            n_extractors = len(extractor_names)
            outlier_detector_names = results_storage[Dims.DETECTORS.value].values
            n_outlier_detectors = len(outlier_detector_names)
            spoiler_names = results_storage[Dims.SPOILERS.value].values
            n_spoilers = len(spoiler_names)
            snrs = results_storage[Dims.SNR.value].values
            n_snrs = len(snrs)
            spoiled_channels_fraction = results_storage[
                Dims.CHANNEL_SPOIL_FRACTION.value
            ].values
            n_sp_channel_fraction = len(spoiled_channels_fraction)
            method_names = results_storage[Dims.METHODS.value].values
            n_methods = len(method_names)

            # metrics x extractors x classifiers x detectors x spoilers x snr x ch_fraction x methods x folds

            n_folds = results.shape[-1]

            # n_files x metrics x extractors x classifiers x detectors x spoilers x snr x ch_fraction x methods x folds
            global_results = np.zeros(
                (
                    n_files,
                    n_metrics,
                    n_extractors,
                    n_classifiers,
                    n_outlier_detectors,
                    n_spoilers,
                    n_snrs,
                    n_sp_channel_fraction,
                    n_methods,
                    n_folds,
                )
            )

        global_results[result_file_id] = results

        pdf_file_path = os.path.join(
            output_directory, "{}_snr_m2_ranks.pdf".format("ALL")
        )
        report_file_path = os.path.join(
            output_directory, "{}_snr_m2_ranks.md".format("ALL")
        )
        report_file_handler = open(report_file_path, "w+")

        with PdfPages(pdf_file_path) as pdf:

            for metric_id, metric_name in enumerate(metric_names):
                print("# {}".format(metric_name), file=report_file_handler)

                for extractor_id, extractor_name in enumerate(extractor_names):
                    print("## {}".format(extractor_name), file=report_file_handler)

                    for classifier_id, classifier_name in enumerate(classifier_names):
                        print(
                            "### {}".format(classifier_name), file=report_file_handler
                        )

                        for outlier_detector_id, outlier_detector_name in enumerate(
                            outlier_detector_names
                        ):
                            print(
                                "#### {}".format(outlier_detector_name),
                                file=report_file_handler,
                            )

                            # files x metrics x extractors x classifiers x detectors x spoilers x snr x ch_fraction x methods x folds
                            # files 0 x spoilers 1 x snr 2 x ch_fraction 3 x methods 4 x folds 5
                            sub_results = global_results[
                                :,
                                metric_id,
                                extractor_id,
                                classifier_id,
                                outlier_detector_id,
                                :,
                                :,
                                :,
                            ]
                            # methods  0 x snrs 1 x ( files 2 x spoilers 3 x ch_frac 4 x folds 5)
                            sub_results_r = np.moveaxis(
                                sub_results, [0, 1, 2, 3, 4, 5], [2, 3, 1, 4, 0, 5]
                            )
                            sub_results = sub_results_r.reshape((n_methods, n_snrs, -1))

                            ranked_data = rankdata(sub_results, axis=0)
                            # methods, snrs
                            avg_ranks = np.mean(ranked_data, axis=-1)

                            for method_id, method_name in enumerate(method_names):

                                plt.plot(
                                    [int(i) for i in snrs],
                                    avg_ranks[method_id, :],
                                    label=method_name,
                                )
                                plt.grid(True, linestyle="--", alpha=0.7)

                            plt.title(
                                "{}, {}, {}, {}".format(
                                    metric_name,
                                    extractor_name,
                                    classifier_name,
                                    outlier_detector_name,
                                )
                            )
                            plt.xlabel("SNR")
                            plt.ylabel("Criterion avg rank")
                            plt.legend()
                            pdf.savefig()
                            plt.close()

                            # avg_ranks (methods, snrs)
                            mi = pd.MultiIndex.from_arrays(
                                [
                                    [
                                        "{}".format(metric_name)
                                        for _ in range(n_methods)
                                    ],
                                    [m for m in method_names],
                                ]
                            )
                            av_rnk_df = pd.DataFrame(
                                avg_ranks.T,
                                columns=mi,
                                index=[
                                    "Avg Rnk {}, snr:{}".format(a, si)
                                    for si, a in zip(snrs, string.ascii_letters)
                                ],
                            )

                            # methods  0 x snrs 1 x ( files 2 x spoilers 3 x ch_frac 4)
                            sub_results_snr = sub_results_r.reshape(
                                (n_methods, n_snrs, -1)
                            )
                            for snr_id, (snr_name, snr_letter) in enumerate(
                                zip(snrs, string.ascii_letters)
                            ):
                                # methods    x ( files  x spoilers )
                                values = sub_results_snr[:, snr_id]
                                p_vals = np.zeros((n_methods, n_methods))
                                for i in range(n_methods):
                                    for j in range(n_methods):
                                        if i == j:
                                            continue

                                        values_squared_diff = np.sqrt(
                                            np.sum((values[i, :] - values[j, :]) ** 2)
                                        )
                                        if values_squared_diff > 1e-4:
                                            with warnings.catch_warnings():  # Normal approximation
                                                warnings.simplefilter("ignore")
                                                p_vals[i, j] = wilcoxon(
                                                    values[i], values[j]
                                                ).pvalue  # mannwhitneyu(values[:,i], values[:,j]).pvalue
                                        else:
                                            p_vals[i, j] = 1.0

                                p_val_vec = p_val_matrix_to_vec(p_vals)

                                p_val_vec_corrected = multipletests(
                                    p_val_vec, method="hommel"
                                )
                                corr_p_val_matrix = p_val_vec_to_matrix(
                                    p_val_vec_corrected[1],
                                    n_methods,
                                    symmetrize=True,
                                )

                                s_test_outcome = []
                                for i in range(n_methods):
                                    out_a = []
                                    for j in range(n_methods):
                                        if (
                                            avg_ranks[i, snr_id] > avg_ranks[j, snr_id]
                                            and corr_p_val_matrix[i, j] < alpha
                                        ):
                                            out_a.append(j + 1)
                                    if len(out_a) == 0:
                                        s_test_outcome.append("--")
                                    else:
                                        s_test_outcome.append(
                                            "{\\scriptsize "
                                            + ",".join([str(x) for x in out_a])
                                            + "}"
                                        )
                                av_rnk_df.loc[
                                    "{} {}, snr:{}_T".format(
                                        "Avg Rnk", snr_letter, snr_name
                                    )
                                ] = s_test_outcome
                                av_rnk_df.sort_index(inplace=True)

                            av_rnk_df.style.format(precision=3, na_rep="").format_index(
                                escape="latex", axis=0
                            ).format_index(escape="latex", axis=1).to_latex(
                                report_file_handler, multicol_align="c"
                            )

        report_file_handler.close()


if __name__ == "__main__":

    np.random.seed(0)
    random.seed(0)

    data_path0B = os.path.join(settings.DATAPATH, "MK_10_03_2022")

    data_sets = [data_path0B]
    # data_sets = [ os.path.join( settings.DATAPATH, "tsnre_windowed","A{}_Force_Exp_low_windowed".format(i)) for i in range(1,10) ]

    output_directory = os.path.join(
        settings.EXPERIMENTS_RESULTS_PATH,
        "./results_channel_knn_weights_3/",
    )
    os.makedirs(output_directory, exist_ok=True)

    log_dir = os.path.dirname(settings.EXPERIMENTS_LOGS_PATH)
    log_file = os.path.splitext(os.path.basename(__file__))[0]
    logger(log_dir, log_file, enable_logging=False)
    warnings.showwarning = warn_with_traceback

    progress_log_path = os.path.join(output_directory, "progress.log")
    progress_log_handler = open(progress_log_path, "w")

    comment_str = """
    Experiment 3.
    """
    run_experiment(
        data_sets,
        output_directory,
        n_splits=10,
        n_repeats=4,
        random_state=0,
        n_jobs=-1,
        overwrite=True,
        n_channels=8,
        progress_log_handler=progress_log_handler,
        comment_str=comment_str,
    )

    analysis_functions = [
        analyze_results_2C,
        analyze_results_2C_ranks,
    ]
    alpha = 0.05

    ProgressParallel(
        backend="multiprocessing",
        n_jobs=-1,
        desc="Analysis",
        total=len(analysis_functions),
        leave=False,
    )(
        delayed(fun)(output_directory, output_directory, alpha)
        for fun in analysis_functions
    )
