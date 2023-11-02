"""This module contains functions related to the Ryan-Joiner test

##### List of functions (alphabetical order) #####

## Functions WITH good TESTS ###
- order_statistic(sample_size, alpha="3/8", safe=False)

## Functions WITH some TESTS ###



## Functions WITHOUT tests ###

- rj_critical_value(n, alpha=0.05)
- rj_p_value(statistic, n)
- ryan_joiner(x_data, alpha=0.05, method="blom", weighted=False)

- rj_correlation_plot(axes, x_data, method="blom", weighted=False)
- rj_dist_plot(axes, x_data, method="blom", min=4, max=50, deleted=False, weighted=False)

- make_bar_plot(axes, df, n_samples, alpha_column_name=None, n_rep_name=None, tests_column_names=None, normal=True, safe=False)
- make_heatmap(axes, df, n_samples, alpha_column_name=None, n_rep_name=None, tests_column_names=None, normal=True, safe=False)
- normal_distribution_plot(axes, n_rep, seed=None, xinfo=[0.00001, 0.99999, 1000], loc=0.0, scale=1.0, safe=False)



##### List of CLASS (alphabetical order) #####

##### Dictionary of abbreviations #####



Author: Anderson Marcos Dias Canteli <andersonmdcanteli@gmail.com>

Last update: November 02, 2023

Last update: November 02, 2023



"""

##### IMPORTS #####

### Standard ###
from collections import namedtuple

### Third part ###
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy import interpolate

# import seaborn as sns

### self made ###
from paramcheckup import parameters, types, numbers

# from .utils import constants

##### CONSTANTS #####


##### CLASS #####

##### FUNCTIONS #####


def order_statistic(sample_size, alpha="3/8", safe=False):
    """This function estimates the normal statistical order (:math:`p_{i}`) using approximations [1]_.

    Parameters
    ----------
    sample_size : int
        The sample size. Must be equal or greater than `4`;
    alpha : str, optional
        A `str` with the alpha value that should be adopted (See details in the Notes section). The options are:
         * `"0"`;
         * `"3/8"` (default);
         * `"1/2"`;

    safe : bool, optional
        Whether to check the inputs before performing the calculations (`True`) or not (`False`, default). Useful for beginners to identify problems in data entry (may reduce algorithm execution time).

    Returns
    -------
    pi : :doc:`numpy array <numpy:reference/generated/numpy.array>`
        The estimated statistical order (:math:`p_{i}`)

    See Also
    --------
    ryan_joiner


    Notes
    -----

    The `alpha` parameter corresponds to the values studied by [1]_, which adopts the following equation to estimate the statistical order:

    .. math::

            p_{i} = \\frac{i - \\alpha}{n - 2 \\times \\alpha + 1}

    where :math:`n` is the sample size and :math:`i` is the ith observation.

    alpha = A is adoptedd in the implementations of the Ryan-Joiner test in Minitab and Statext software. Also, this option is  cited by [2]_ as an alternative.

    `alpha="3/8" is adopted in the implementations of the Ryan-Joiner test in Minitab and Statext software. This option is also cited as an alternative by [2]_.

    References
    ----------
    .. [1] BLOM, G. Statistical Estimates and Transformed Beta-Variables. New York: John Wiley and Sons, Inc, p. 71-72, 1958.

    .. [2]  RYAN, T. A., JOINER, B. L. Normal Probability Plots and Tests for Normality, Technical Report, Statistics Department, The Pennsylvania State University, 1976. Available at `www.additive-net.de <https://www.additive-net.de/de/component/jdownloads/send/70-support/236-normal-probability-plots-and-tests-for-normality-thomas-a-ryan-jr-bryan-l-joiner>`_. Access on: 22 Jul. 2023.


    Examples
    --------
    >>> from normtest import ryan_joiner
    >>> size = 10
    >>> pi = ryan_joiner.order_statistic(size)
    >>> print(pi)
    [0.06097561 0.15853659 0.25609756 0.35365854 0.45121951 0.54878049
    0.64634146 0.74390244 0.84146341 0.93902439]

    """
    if safe:
        parameters.param_options(
            option=alpha,
            param_options=["0", "3/8", "1/2"],
            param_name="alpha",
            func_name="order_statistic",
        )
        types.is_int(
            value=sample_size, param_name="sample_size", func_name="order_statistic"
        )
        numbers.is_greater_than(
            value=sample_size,
            lower=4,
            param_name="sample_size",
            func_name="order_statistic",
            inclusive=True,
        )

    i = np.arange(1, sample_size + 1)
    if alpha == "3/8":
        alpha = 3 / 8
    elif alpha == "0":
        alpha = 0
    else:
        alpha = 0.5

    return (i - alpha) / (sample_size - 2 * alpha + 1)
