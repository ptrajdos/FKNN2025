# A Compound Classification System with Fuzzy Model Applied to the Noise-Tolerant Control of a Bionic Hand via EMG and MMG Signals Recognition

## System Requirements

Requires: __Python>=3.9.7__
All required packages may be installed via __pip__.
Tested on: __Ubuntu Linux 22.04__, __macOS  Sequoia 15.5__

## Setup
To download test data, create the virtual envirionment, and install required packages type:
```
make create_env
```
To clean virtual environment type:
```
make clean
```


The experimental results will be located in: __./experiments\_results__

## Experiments

To run experiments type:
```
make run_experiments
```

Results will be placed in directories: __./experiments\_results/results\_channel\_knn\_weights\_[1-3]__.

Directory structure:

  + *MK_10_03_2022.pickle* -- raw results (for a single set) as numpy arrays.
  + *MK_10_03_2022_snr_m2.pdf* -- trends in quality criteria (median, Q1, Q3) over all SNR levels.
  + *ALL_snr_m2_ranks.pdf* -- Average ranks plots for different SNR values.
  + *ALL_snr_m2_ranks.md* -- Average ranks tables and statistical tests for different SNR values, criteria, ensemble sizes. 