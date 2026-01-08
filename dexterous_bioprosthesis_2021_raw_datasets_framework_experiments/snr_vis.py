import datetime
import logging
import os
import warnings




import numpy as np
import pandas as pd
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals_io import (
    read_signals_from_archive,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_spoilers.raw_signals_spoiler_multiple import (
    RawSignalsSpoilerMultiple,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_spoilers.raw_signals_spoiler_sine import (
    RawSignalsSpoilerSine,
)


from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_spoilers.raw_signals_spoiler_damper import (
    RawSignalsSpoilerDamper,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_spoilers.raw_signals_spoiler_gauss import (
    RawSignalsSpoilerGauss,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_spoilers.raw_signals_spoiler_multiple import (
    RawSignalsSpoilerMultiple,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_spoilers.raw_signals_spoiler_rappam import (
    RawSignalsSpoilerRappAM,
)


from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from dexterous_bioprosthesis_2021_raw_datasets_framework.tools.warnings import (
    warn_with_traceback,
)

from dexterous_bioprosthesis_2021_raw_datasets_framework_experiments import settings


from tqdm import tqdm

from dexterous_bioprosthesis_2021_raw_datasets_framework_experiments.tools import logger




import random

# Plot line colors and markers


from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_window_segmentation_fs import (
    RawSignalsFilterWindowSegmentationFS,
)

N_INTERNAL_SPLITS = 4


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
            RawSignalsSpoilerRappAM(
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
    spoilers = {
        "50Hz": generate_spoiler_50Hz,
        "Damper": generate_spoiler_damper,
        "Clipper": generate_spoiler_clipper,
        "Gauss": generate_spoiler_gauss,
        "BaselineWander": generate_spoiler_baseline_wander,
    }
    return spoilers


def get_snr_levels():
    return [4, 2, 0]


def run_experiment(
    datasets,
    output_directory,
    comment_str="",
):

    os.makedirs(output_directory, exist_ok=True)

    comment_file = os.path.join(output_directory, "comment.txt")
    with open(comment_file, "w") as f:
        f.write(comment_str)
        f.write("Start time: {}\n".format(datetime.datetime.now()))
        f.write("\n")

    spoiler_generators = generate_spoilers_gens()
    n_spoiler_generators = len(spoiler_generators)

    snrs = get_snr_levels()
    n_snrs = len(snrs)

    for experiment_name, archive_path, input_data_regex in tqdm(
        datasets, desc="Data sets"
    ):

        set_name = experiment_name

        result_file_path = os.path.join(output_directory, "{}.pdf".format(set_name))

        pre_set = read_signals_from_archive(
            archive_path=archive_path,
            filter_regex=input_data_regex,
        )

        raw_set = pre_set["accepted"]
        if len(raw_set) == 0:
            logging.debug(f"Skipping: {set_name}")
            continue
        raw_set = raw_set[0:1]
        filter = RawSignalsFilterWindowSegmentationFS(250, 125)
        filtered_set = filter.fit_transform(raw_set)

        with PdfPages(result_file_path) as pdf:

            fig, ax = plt.subplots(
                nrows=n_spoiler_generators,
                ncols=n_snrs,
                figsize=(4 * n_snrs, 4 * n_spoiler_generators),
                squeeze=False,
            )

            for col_idx, snr in enumerate(snrs):
                for row_idx, (spoiler_name, spoiler_gen) in enumerate(
                    spoiler_generators.items()
                ):
                    spoiler = spoiler_gen(snr=snr, channels_spoiled_frac=1.0)
                    spoiled_set = spoiler.fit_transform(filtered_set)

                    signal = filtered_set[0]
                    spoiled_signal = spoiled_set[0]

                    n_samples, n_channels = signal.to_numpy().shape
                    time_axis = np.linspace(
                        0, n_samples / signal.sample_rate, n_samples
                    )

                    ax[row_idx, col_idx].set_title(
                        "{} SNR={}".format(spoiler_name, snr)
                    )
                    ax[row_idx, col_idx].set_xlabel("Time [s]")
                    ax[row_idx, col_idx].set_ylabel("Amplitude")

                    ch_idx = 0

                    ax[row_idx, col_idx].plot(
                        time_axis,
                        signal.to_numpy()[:, ch_idx],
                        color="blue",
                        alpha=0.3,
                        label="orig",
                    )
                    ax[row_idx, col_idx].plot(
                        time_axis,
                        spoiled_signal.to_numpy()[:, ch_idx],
                        color="red",
                        alpha=0.3,
                        label="contaminated",
                    )

                    ax[row_idx, col_idx].grid(True, linestyle="--", alpha=0.7)

            handles, labels = ax[0, 0].get_legend_handles_labels()

            fig.legend(
                handles=handles, labels=labels, loc="upper right", title="Legend"
            )
            fig.suptitle("Signal contamination", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.98])
            pdf.savefig(fig)
            plt.close(fig)


if __name__ == "__main__":

    np.random.seed(0)
    random.seed(0)

    data_path0B = os.path.join(settings.DATAPATH, "MK_10_03_2022.zip")
    data_sets = []
    data_sets.append(("mk_10_03_2022", data_path0B, "./*"))

    tsnre_path = os.path.join(settings.DATAPATH, "tsnre_windowed.zip")
    # for i in range(1, 10):
    #     data_sets.append(
    #         (
    #             "A{}_Force_Exp_low_windowed".format(i),
    #             tsnre_path,
    #             ".*/A{}_Force_Exp_low_windowed/.*".format(i),
    #         )
    #     )

    subjects = list([*range(1, 12)])  # ATTENTION
    experiments = list([*range(1, 2)])  # up to 4
    labels = ["restimulus"]

    db_name = "db3"
    db_archive_path = os.path.join(settings.DATAPATH, f"{db_name}.zip")

    for experiment in experiments:
        for label in labels:
            for su in subjects:
                data_sets.append(
                    (
                        f"S{su}_E{experiment}_A1_{label}",
                        db_archive_path,
                        f".*/S{su}_E{experiment}_A1_{label}/.*",
                    )
                )

    output_directory = os.path.join(
        settings.EXPERIMENTS_RESULTS_PATH,
        "./results_vis/",
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
        comment_str=comment_str,
    )
