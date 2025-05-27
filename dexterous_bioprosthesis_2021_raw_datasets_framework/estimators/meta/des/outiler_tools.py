import numpy as np
import numba as nb

@nb.jit(nopython=True)
def mask_outliers(base_competences,outlier_indicators,tol=1E-3):
    """
    Mask classifiers for which the outlier_indicators is -1 (an outlier).
    For rows with no competent classifiers selected, all base classifiers are selected.

    Arguments:
    ----------

    base_competences: numpy array, (n_points, n_base_classifiers). Base competences to be masked


    outlier_indicators numpy array, (n_points, n_base_classifiers). Array contains predictions of outlier detectors

    tol:float -- tolerance for detecting zero sums

    Returns:
    ---------

    Masked competences, numpy array (n_points, n_base_classifiers)

    """

    new_competences = np.copy(base_competences)

    for r in np.arange(base_competences.shape[0]):
        for c in np.arange(base_competences.shape[1]):

            if outlier_indicators[r,c] == -1:
                new_competences[r,c] = 0

        r_sum = np.sum( new_competences[r,:])
        if r_sum < tol:
            for c in np.arange(base_competences.shape[1]):
                new_competences[r,c] = 1

    return new_competences