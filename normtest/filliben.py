"""This module contains functions related to the Filliben test

##### List of functions (cte_alphabetical order) #####

## Functions WITH good TESTS ###


## Functions WITH some TESTS ###
- citation()
- _uniform_order_medians(sample_size, safe=False)
- _normal_order_medians(mi, safe=False)

## Functions WITHOUT tests ###



##### List of CLASS (alphabetical order) #####



Author: Anderson Marcos Dias Canteli <andersonmdcanteli@gmail.com>

Created : November 08, 2023

Last update: November 09, 2023
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
from paramcheckup import parameters, types, numbers, numpy_arrays
from normtest import bibmaker
from .utils import constants


##### DOCUMENTATION #####
from .utils import documentation as docs


#### CONSTANTS ####
Filliben1975 = "FILLIBEN, J. J. The Probability Plot Correlation Coefficient Test for Normality. Technometrics, v. 17, n. 1, p. 111-117, 1975."
Blom1958 = "BLOM, G. Statistical Estimates and Transformed Beta-Variables. New York: John Wiley and Sons, Inc, p. 71-72, 1958."

FILLIBEN_CRITICAL = {
    "n": (
        None,
        None,
        None,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        55,
        60,
        65,
        70,
        75,
        80,
        85,
        90,
        95,
        100,
    ),
    0.005: (
        None,
        None,
        None,
        0.867,
        0.813,
        0.803,
        0.818,
        0.828,
        0.841,
        0.851,
        0.860,
        0.868,
        0.875,
        0.882,
        0.888,
        0.894,
        0.899,
        0.903,
        0.907,
        0.909,
        0.912,
        0.914,
        0.918,
        0.922,
        0.926,
        0.928,
        0.930,
        0.932,
        0.934,
        0.937,
        0.938,
        0.939,
        0.939,
        0.940,
        0.941,
        0.943,
        0.945,
        0.947,
        0.948,
        0.949,
        0.949,
        0.950,
        0.951,
        0.953,
        0.954,
        0.955,
        0.956,
        0.956,
        0.957,
        0.957,
        0.959,
        0.962,
        0.965,
        0.967,
        0.969,
        0.971,
        0.973,
        0.974,
        0.976,
        0.977,
        0.979,
    ),
    0.01: (
        None,
        None,
        None,
        0.869,
        0.822,
        0.822,
        0.835,
        0.847,
        0.859,
        0.868,
        0.876,
        0.883,
        0.889,
        0.895,
        0.901,
        0.907,
        0.912,
        0.916,
        0.919,
        0.923,
        0.925,
        0.928,
        0.930,
        0.933,
        0.936,
        0.937,
        0.939,
        0.941,
        0.943,
        0.945,
        0.947,
        0.948,
        0.949,
        0.950,
        0.951,
        0.952,
        0.953,
        0.955,
        0.956,
        0.957,
        0.958,
        0.958,
        0.959,
        0.959,
        0.960,
        0.961,
        0.962,
        0.963,
        0.963,
        0.964,
        0.965,
        0.967,
        0.970,
        0.972,
        0.974,
        0.975,
        0.976,
        0.977,
        0.978,
        0.979,
        0.981,
    ),
    0.025: (
        None,
        None,
        None,
        0.872,
        0.845,
        0.855,
        0.868,
        0.876,
        0.886,
        0.893,
        0.900,
        0.906,
        0.912,
        0.917,
        0.921,
        0.925,
        0.928,
        0.931,
        0.934,
        0.937,
        0.939,
        0.942,
        0.944,
        0.947,
        0.949,
        0.950,
        0.952,
        0.953,
        0.955,
        0.956,
        0.957,
        0.958,
        0.959,
        0.960,
        0.960,
        0.961,
        0.962,
        0.962,
        0.964,
        0.965,
        0.966,
        0.967,
        0.967,
        0.967,
        0.968,
        0.969,
        0.969,
        0.970,
        0.970,
        0.971,
        0.972,
        0.974,
        0.976,
        0.977,
        0.978,
        0.979,
        0.980,
        0.981,
        0.982,
        0.983,
        0.984,
    ),
    0.05: (
        None,
        None,
        None,
        0.879,
        0.868,
        0.879,
        0.890,
        0.899,
        0.905,
        0.912,
        0.917,
        0.922,
        0.926,
        0.931,
        0.934,
        0.937,
        0.940,
        0.942,
        0.945,
        0.947,
        0.950,
        0.952,
        0.954,
        0.955,
        0.957,
        0.958,
        0.959,
        0.960,
        0.962,
        0.962,
        0.964,
        0.965,
        0.966,
        0.967,
        0.967,
        0.968,
        0.968,
        0.969,
        0.970,
        0.971,
        0.972,
        0.972,
        0.973,
        0.973,
        0.973,
        0.974,
        0.974,
        0.974,
        0.975,
        0.975,
        0.977,
        0.978,
        0.980,
        0.981,
        0.982,
        0.983,
        0.984,
        0.985,
        0.985,
        0.986,
        0.987,
    ),
    0.10: (
        None,
        None,
        None,
        0.891,
        0.894,
        0.902,
        0.911,
        0.916,
        0.924,
        0.929,
        0.934,
        0.938,
        0.941,
        0.944,
        0.947,
        0.950,
        0.952,
        0.954,
        0.956,
        0.958,
        0.960,
        0.961,
        0.962,
        0.964,
        0.965,
        0.966,
        0.967,
        0.968,
        0.969,
        0.969,
        0.970,
        0.971,
        0.972,
        0.973,
        0.973,
        0.974,
        0.974,
        0.975,
        0.975,
        0.976,
        0.977,
        0.977,
        0.978,
        0.978,
        0.978,
        0.978,
        0.979,
        0.979,
        0.980,
        0.980,
        0.981,
        0.982,
        0.983,
        0.984,
        0.985,
        0.986,
        0.987,
        0.987,
        0.988,
        0.989,
        0.989,
    ),
    0.25: (
        None,
        None,
        None,
        0.924,
        0.931,
        0.935,
        0.940,
        0.944,
        0.948,
        0.951,
        0.954,
        0.957,
        0.959,
        0.962,
        0.964,
        0.965,
        0.967,
        0.968,
        0.969,
        0.971,
        0.972,
        0.973,
        0.974,
        0.975,
        0.975,
        0.976,
        0.977,
        0.977,
        0.978,
        0.979,
        0.979,
        0.980,
        0.980,
        0.981,
        0.981,
        0.982,
        0.982,
        0.982,
        0.983,
        0.983,
        0.983,
        0.984,
        0.984,
        0.984,
        0.984,
        0.985,
        0.985,
        0.985,
        0.985,
        0.986,
        0.986,
        0.987,
        0.988,
        0.989,
        0.989,
        0.990,
        0.991,
        0.991,
        0.991,
        0.992,
        0.992,
    ),
    0.50: (
        None,
        None,
        None,
        0.966,
        0.958,
        0.960,
        0.962,
        0.965,
        0.967,
        0.968,
        0.970,
        0.972,
        0.973,
        0.975,
        0.976,
        0.977,
        0.978,
        0.979,
        0.979,
        0.980,
        0.981,
        0.981,
        0.982,
        0.983,
        0.983,
        0.984,
        0.984,
        0.984,
        0.985,
        0.985,
        0.986,
        0.986,
        0.986,
        0.987,
        0.987,
        0.987,
        0.987,
        0.988,
        0.988,
        0.988,
        0.988,
        0.989,
        0.989,
        0.989,
        0.989,
        0.989,
        0.990,
        0.990,
        0.990,
        0.990,
        0.990,
        0.991,
        0.991,
        0.992,
        0.993,
        0.993,
        0.993,
        0.994,
        0.994,
        0.994,
        0.994,
    ),
    0.75: (
        None,
        None,
        None,
        0.991,
        0.979,
        0.977,
        0.977,
        0.978,
        0.979,
        0.980,
        0.981,
        0.982,
        0.982,
        0.983,
        0.984,
        0.984,
        0.985,
        0.986,
        0.986,
        0.987,
        0.987,
        0.987,
        0.988,
        0.988,
        0.988,
        0.989,
        0.989,
        0.989,
        0.990,
        0.990,
        0.990,
        0.990,
        0.990,
        0.991,
        0.991,
        0.991,
        0.991,
        0.991,
        0.992,
        0.992,
        0.992,
        0.992,
        0.992,
        0.992,
        0.992,
        0.993,
        0.993,
        0.993,
        0.993,
        0.993,
        0.993,
        0.994,
        0.994,
        0.994,
        0.995,
        0.995,
        0.995,
        0.995,
        0.996,
        0.996,
        0.996,
    ),
    0.90: (
        None,
        None,
        None,
        0.999,
        0.992,
        0.988,
        0.986,
        0.986,
        0.986,
        0.987,
        0.987,
        0.988,
        0.988,
        0.988,
        0.989,
        0.989,
        0.989,
        0.990,
        0.990,
        0.990,
        0.991,
        0.991,
        0.991,
        0.991,
        0.992,
        0.992,
        0.992,
        0.992,
        0.992,
        0.992,
        0.993,
        0.993,
        0.993,
        0.993,
        0.993,
        0.993,
        0.994,
        0.994,
        0.994,
        0.994,
        0.994,
        0.994,
        0.994,
        0.994,
        0.994,
        0.994,
        0.995,
        0.995,
        0.995,
        0.995,
        0.995,
        0.995,
        0.995,
        0.996,
        0.996,
        0.996,
        0.996,
        0.997,
        0.997,
        0.997,
        0.997,
    ),
    0.95: (
        None,
        None,
        None,
        1,
        0.996,
        0.992,
        0.990,
        0.990,
        0.990,
        0.990,
        0.990,
        0.990,
        0.990,
        0.991,
        0.991,
        0.991,
        0.991,
        0.992,
        0.992,
        0.992,
        0.992,
        0.993,
        0.993,
        0.993,
        0.993,
        0.993,
        0.993,
        0.994,
        0.994,
        0.994,
        0.994,
        0.994,
        0.994,
        0.994,
        0.994,
        0.995,
        0.995,
        0.995,
        0.995,
        0.995,
        0.995,
        0.995,
        0.995,
        0.995,
        0.995,
        0.995,
        0.995,
        0.995,
        0.996,
        0.996,
        0.996,
        0.996,
        0.996,
        0.996,
        0.997,
        0.997,
        0.997,
        0.997,
        0.997,
        0.997,
        0.998,
    ),
    0.975: (
        None,
        None,
        None,
        1,
        0.998,
        0.995,
        0.993,
        0.992,
        0.992,
        0.992,
        0.992,
        0.992,
        0.992,
        0.993,
        0.993,
        0.993,
        0.993,
        0.993,
        0.993,
        0.993,
        0.994,
        0.994,
        0.994,
        0.994,
        0.994,
        0.994,
        0.994,
        0.995,
        0.995,
        0.995,
        0.995,
        0.995,
        0.995,
        0.995,
        0.995,
        0.995,
        0.996,
        0.996,
        0.996,
        0.996,
        0.996,
        0.996,
        0.996,
        0.996,
        0.996,
        0.996,
        0.996,
        0.996,
        0.996,
        0.996,
        0.996,
        0.997,
        0.997,
        0.997,
        0.997,
        0.997,
        0.997,
        0.997,
        0.998,
        0.998,
        0.998,
    ),
    0.99: (
        None,
        None,
        None,
        1,
        0.999,
        0.997,
        0.996,
        0.995,
        0.995,
        0.994,
        0.994,
        0.994,
        0.994,
        0.994,
        0.994,
        0.994,
        0.994,
        0.994,
        0.995,
        0.995,
        0.995,
        0.995,
        0.995,
        0.995,
        0.995,
        0.995,
        0.995,
        0.995,
        0.995,
        0.995,
        0.996,
        0.996,
        0.996,
        0.996,
        0.996,
        0.996,
        0.996,
        0.996,
        0.996,
        0.996,
        0.996,
        0.996,
        0.997,
        0.997,
        0.997,
        0.997,
        0.997,
        0.997,
        0.997,
        0.997,
        0.997,
        0.997,
        0.997,
        0.997,
        0.998,
        0.998,
        0.998,
        0.998,
        0.998,
        0.998,
        0.998,
    ),
    0.995: (
        None,
        None,
        None,
        1,
        1,
        0.998,
        0.997,
        0.996,
        0.996,
        0.995,
        0.995,
        0.995,
        0.995,
        0.995,
        0.995,
        0.995,
        0.995,
        0.995,
        0.995,
        0.995,
        0.995,
        0.996,
        0.996,
        0.996,
        0.996,
        0.996,
        0.996,
        0.996,
        0.996,
        0.996,
        0.996,
        0.996,
        0.996,
        0.996,
        0.996,
        0.997,
        0.997,
        0.997,
        0.997,
        0.997,
        0.997,
        0.997,
        0.997,
        0.997,
        0.997,
        0.997,
        0.997,
        0.997,
        0.997,
        0.997,
        0.997,
        0.997,
        0.998,
        0.998,
        0.998,
        0.998,
        0.998,
        0.998,
        0.998,
        0.998,
        0.998,
    ),
}

##### CLASS #####


##### FUNCTIONS #####


def citation(export=False):
    """This function returns the reference from Filliben's test, with the option to export the reference in `.bib` format.

    Parameters
    ----------
    export : bool
        Whether to export the reference as `Filliben1975.bib` file (`True`) or not (`False`, default);


    Returns
    -------
    reference : str
        The Filliben Test reference

    """
    reference = bibmaker.make_article(
        author="James J. Filliben",
        title="The Probability Plot Correlation Coefficient Test for Normality",
        journaltitle="Technometrics",
        year=1975,
        citekey="Filliben1975",
        date=None,
        volume="17",
        number="1",
        pages="111--117",
        doi="10.2307/1268008",
        month=2,
        export=export,
    )
    return reference


@docs.docstring_parameter(
    sample_size=docs.SAMPLE_SIZE["type"],
    samp_size_desc=docs.SAMPLE_SIZE["description"],
    safe=docs.SAFE["type"],
    safe_desc=docs.SAFE["description"],
    mi=docs.MI["type"],
    mi_desc=docs.MI["description"],
    fi_ref=Filliben1975,
)
def _uniform_order_medians(sample_size, safe=False):
    """This function estimates the uniform order statistic median (:math:`m_{{i}}`) used in the Filliben normality test [1]_.

    Parameters
    ----------
    {sample_size}
        {samp_size_desc}
    {safe}
        {safe_desc}

    Returns
    -------
    {mi}
        {mi_desc}

    See Also
    --------
    fi_test


    Notes
    -----
    The uniform order statistic median is estimated using:

    .. math::

            m_{{i}} = \\begin{{cases}}1-0.5^{{1/n}} & i = 1\\\ \\frac{{i-0.3175}}{{n+0.365}} & i = 2, 3,  \\ldots , n-1 \\\ 0.5^{{1/n}}& i=n \\end{{cases}}

    where :math:`n` is the sample size and :math:`i` is the ith observation.


    References
    ----------
    .. [1] {fi_ref}



    Examples
    --------
    >>> from normtest import filliben
    >>> uniform_order = filliben._uniform_order_medians(7)
    >>> print(uniform_order)
    array([0.09427634, 0.22844535, 0.36422267, 0.5       , 0.63577733,
           0.77155465, 0.90572366])
    """

    if safe:
        types.is_int(
            value=sample_size, param_name="sample_size", func_name="normal_medians"
        )
        numbers.is_greater_than(
            value=sample_size,
            lower=4,
            param_name="sample_size",
            func_name="normal_medians",
            inclusive=True,
        )

    i = np.arange(1, sample_size + 1)
    mi = (i - 0.3175) / (sample_size + 0.365)
    mi[0] = 1 - 0.5 ** (1 / sample_size)
    mi[-1] = 0.5 ** (1 / sample_size)

    return mi


@docs.docstring_parameter(
    mi=docs.MI["type"],
    mi_desc=docs.MI["description"],
    safe=docs.SAFE["type"],
    safe_desc=docs.SAFE["description"],
    zi=docs.ZI["type"],
    zi_desc=docs.ZI["description"],
)
def _normal_order_medians(mi, safe=False):
    """This function transforms the uniform order median to normal order median using the standard Normal distribution (:math:`z_{{i}}`).

    Parameters
    ----------
    {mi}
        {mi_desc}


    Returns
    -------
    {zi}
        {zi_desc}


    Notes
    -----
    The transformation to the standard Normal scale is done using the equation:

    .. math::

            z_{{i}} = \\phi^{{-1}} \\left(m_{{i}} \\right)

    where :math:`m_{{i}}` is the uniform statistical order and :math:`\\phi^{{-1}}` is the inverse of the standard Normal distribution. The transformation is performed using :doc:`stats.norm.ppf() <scipy:reference/generated/scipy.stats.norm>`.


    See Also
    --------
    fi_test


    Examples
    --------
    >>> from normtest import filliben
    >>> uniform_order = filliben._uniform_order_medians(7)
    >>> normal_order = filliben._normal_order_medians(uniform_order)
    >>> print(normal_order)
    [-1.31487275 -0.74397649 -0.3471943   0.          0.3471943   0.74397649
    1.31487275]


    """
    func_name = "_normal_order_medians"
    if safe:
        types.is_numpy(value=mi, param_name="mi", func_name=func_name)
        numpy_arrays.n_dimensions(
            arr=mi,
            param_name="mi",
            func_name=func_name,
            n_dimensions=1,
        )
        numpy_arrays.greater_than_n(
            array=mi,
            param_name="mi",
            func_name=func_name,
            minimum=4,
            inclusive=True,
        )

    normal_ordered = stats.norm.ppf(mi)

    return normal_ordered


@docs.docstring_parameter(
    x_data=docs.X_DATA["type"],
    x_data_desc=docs.X_DATA["description"],
    zi=docs.ZI["type"],
    zi_desc=docs.ZI["description"],
    safe=docs.SAFE["type"],
    safe_desc=docs.SAFE["description"],
    statistic=docs.STATISTIC["type"],
    statistic_desc=docs.STATISTIC["description"],
    fi_ref=Filliben1975,
)
def _statistic(x_data, zi, safe=False):
    """This function estimates the statistic of the Filliben normality test [1]_

    Parameters
    ----------
    {x_data}
        {x_data_desc}
    {zi}
        {zi_desc}
    {safe}
        {safe_desc}

    Returns
    -------
    statistic
        {statistic_desc}


    See Also
    --------
    fi_test


    Notes
    -----
    The test statistic (:math:`F_{{p}}`) is estimated through the correlation between the ordered data and the Normal statistical order:


    .. math::

            F_p = \\frac{{\\sum_{{i=1}}^n \\left(x_i - \\overline{{x}}\\right) \\left(z_i - \\overline{{z}}\\right)}}{{\\sqrt{{\\sum_{{i=1}}^n \\left( x_i - \\overline{{x}}\\right)^2 \\sum_{{i=1}}^n \\left( z_i - \\overline{{z}}\\right)^2}}}}

    where :math:`z_{{i}}` values are the z-score values of the corresponding experimental data (:math:`x_{{{{i}}}}`) value, and :math:`n` is the sample size.

    The correlation is estimated using :doc:`scipy.stats.pearsonr() <scipy:reference/generated/scipy.stats.pearsonr>`.

    References
    ----------
    .. [1] {fi_ref}


    Examples
    --------
    >>> from normtest import filliben
    >>> import numpy as np
    >>> x_data = np.array([6, 1, -4, 8, -2, 5, 0])
    >>> uniform_order = filliben._uniform_order_medians(x_data.size)
    >>> normal_order = filliben._normal_order_medians(uniform_order)
    >>> x_data = np.sort(x_data)
    >>> statistic = filliben._statistic(x_data, normal_order)
    >>> print(statistic)
    0.9854095718708367


    """
    if safe:
        func_name = "_statistic"
        types.is_numpy(value=x_data, param_name="x_data", func_name=func_name)
        numpy_arrays.greater_than_n(
            array=x_data,
            param_name="x_data",
            func_name=func_name,
            minimum=4,
            inclusive=True,
        )
        types.is_numpy(value=zi, param_name="zi", func_name=func_name)
        numpy_arrays.greater_than_n(
            array=zi, param_name="zi", func_name=func_name, minimum=4, inclusive=True
        )
        numpy_arrays.matching_size(
            array_a=x_data,
            param_name_a="x_data",
            array_b=zi,
            param_name_b="zi",
            func_name=func_name,
        )

    correl = stats.pearsonr(x_data, zi)[0]
    return correl


@docs.docstring_parameter(
    sample_size=docs.SAMPLE_SIZE["type"],
    sample_size_desc=docs.SAMPLE_SIZE["description"],
    alpha=docs.ALPHA["type"],
    alpha_desc=docs.ALPHA["description"],
    zi=docs.ZI["type"],
    zi_desc=docs.ZI["description"],
    safe=docs.SAFE["type"],
    safe_desc=docs.SAFE["description"],
    fi_ref=Filliben1975,
)
def _critical(sample_size, alpha=0.05, safe=False):
    """This function calculates the critical value for the Filliben normality test [1]_.

    Parameters
    ----------
    {sample_size}
        {sample_size_desc}
    {alpha}
        {alpha_desc}
    {safe}
        {safe_desc}

    Returns
    -------
    critical : float
        The critical value.


    References
    ----------
    .. [1] FILLIBEN, J. J. The Probability Plot Correlation Coefficient Test for Normality. Technometrics, 17(1), 111-117, (1975). Available at `doi.org/10.2307/1268008 <https://doi.org/10.2307/1268008>`_.


    """
    # if safe:
    #     types.is_int(value=sample_size, param_name="sample_size", func_name="critical_value")
    #     numbers.is_greater_than(value="sample_size", lower=3, param_name="sample_size", func_name="critical_value", inclusive=True)
    #     types.is_float(value=alpha, param_name="alpha", func_name="critical_value")
    #     numbers.is_between_a_and_b(value=alpha, a=0.005, b=0.995, param_name="alpha", func_name=critical_value, inclusive=True)

    if sample_size not in filliben_critical["n"]:
        print("sample size not found")
    if alpha not in list(filliben_critical.keys())[1:]:
        print("alpha not found")

    if sample_size > 100:
        print("Os resultados podem não ser confiaveis devido a erros de extrapolação.")

    return filliben_critical[alpha][sample_size]
