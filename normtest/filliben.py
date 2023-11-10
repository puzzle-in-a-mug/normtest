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
    "n": (None, None, None, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100),

    0.005 : (None, None, None, .867, .813, .803, .818, .828, .841, .851, .860, .868, .875, .882, .888, .894, .899, .903, .907, .909, .912, .914, .918, .922, .926, .928, .930, .932, .934, .937, .938, .939, .939, .940, .941, .943, .945, .947, .948, .949, .949, .950, .951, .953, .954, .955, .956, .956, .957, .957, .959, .962, .965, .967, .969, .971, .973, .974, .976, .977, .979),


    0.01 : ( None, None, None, .869, .822, .822, .835, .847, .859, .868, .876, .883, .889, .895, .901, .907, .912, .916, .919, .923, .925, .928, .930, .933, .936, .937, .939, .941, .943, .945, .947, .948, .949, .950, .951, .952, .953, .955, .956, .957, .958, .958, .959, .959, .960, .961, .962, .963, .963, .964, .965, .967, .970, .972, .974, .975, .976, .977, .978, .979, .981),

    0.025 : (None, None, None, .872, .845, .855, .868, .876, .886, .893, .900, .906, .912, .917, .921, .925, .928, .931, .934, .937, .939, .942, .944, .947, .949, .950, .952, .953, .955, .956, .957, .958, .959, .960, .960, .961, .962, .962, .964, .965, .966, .967, .967, .967, .968, .969, .969, .970, .970, .971, .972, .974, .976, .977, .978, .979, .980, .981, .982, .983, .984),

    0.05 : (None, None, None, .879, .868, .879, .890, .899, .905, .912, .917, .922, .926, .931, .934, .937, .940, .942, .945, .947, .950, .952, .954, .955, .957, .958, .959, .960, .962, .962, .964, .965, .966, .967, .967, .968, .968, .969, .970, .971, .972, .972, .973, .973, .973, .974, .974, .974, .975, .975, .977, .978, .980, .981, .982, .983, .984, .985, .985, .986, .987),

    0.10 : (None, None, None, .891, .894, .902, .911, .916, .924, .929, .934, .938, .941, .944, .947, .950, .952, .954, .956, .958, .960, .961, .962, .964, .965, .966, .967, .968, .969, .969, .970, .971, .972, .973, .973, .974, .974, .975, .975, .976, .977, .977, .978, .978, .978, .978, .979, .979, .980, .980, .981, .982, .983, .984, .985, .986, .987, .987, .988, .989, .989),


    0.25 : (None, None, None, .924, .931, .935, .940, .944, .948, .951, .954, .957, .959, .962, .964, .965, .967, .968, .969, .971, .972, .973, .974, .975, .975, .976, .977, .977, .978, .979, .979, .980, .980, .981, .981, .982, .982, .982, .983, .983, .983, .984, .984, .984, .984, .985, .985, .985, .985, .986, .986, .987, .988, .989, .989, .990, .991, .991, .991, .992, .992),


    0.50 : (None, None, None, .966, .958, .960, .962, .965, .967, .968, .970, .972, .973, .975, .976, .977, .978, .979, .979, .980, .981, .981, .982, .983, .983, .984, .984, .984, .985, .985, .986, .986, .986, .987, .987, .987, .987, .988, .988, .988, .988, .989, .989, .989, .989, .989, .990, .990, .990, .990, .990, .991, .991, .992, .993, .993, .993, .994, .994, .994, .994),

    0.75 : (None, None, None, .991, .979, .977, .977, .978, .979, .980, .981, .982, .982, .983, .984, .984, .985, .986, .986, .987, .987, .987, .988, .988, .988, .989, .989, .989, .990, .990, .990, .990, .990, .991, .991, .991, .991, .991, .992, .992, .992, .992, .992, .992, .992, .993, .993, .993, .993, .993, .993, .994, .994, .994, .995, .995, .995, .995, .996, .996, .996),

    0.90 : (None, None, None, .999, .992, .988, .986, .986, .986, .987, .987, .988, .988, .988, .989, .989, .989, .990, .990, .990, .991, .991, .991, .991, .992, .992, .992, .992, .992, .992, .993, .993, .993, .993, .993, .993, .994 , .994, .994, .994, .994, .994, .994, .994, .994, .994, .995, .995, .995, .995, .995, .995, .995, .996, .996, .996, .996, .997, .997, .997, .997),

    0.95 : (None, None, None, 1, .996, .992, .990 , .990, .990, .990, .990, .990, .990, .991, .991, .991, .991, .992, .992, .992, .992, .993, .993, .993, .993, .993, .993, .994, .994, .994, .994, .994, .994, .994, .994, .995, .995, .995, .995, .995, .995, .995, .995, .995, .995, .995, .995, .995, .996, .996, .996, .996, .996, .996, .997, .997, .997, .997, .997, .997, .998),


    0.975 : (None, None, None, 1, .998, .995, .993, .992, .992, .992, .992, .992, .992, .993, .993, .993, .993, .993, .993, .993, .994, .994, .994, .994, .994, .994, .994, .995, .995, .995, .995, .995, .995, .995, .995, .995, .996, .996, .996, .996, .996, .996, .996, .996, .996, .996, .996, .996, .996, .996, .996, .997, .997, .997, .997, .997, .997, .997, .998, .998, .998),

    0.99 : (None, None, None, 1, .999, .997, .996, .995, .995, .994, .994, .994, .994, .994, .994, .994, .994, .994, .995, .995, .995, .995, .995, .995, .995, .995, .995, .995, .995, .995, .996, .996, .996, .996, .996, .996, .996, .996, .996, .996, .996, .996, .997, .997, .997, .997, .997, .997, .997, .997, .997, .997, .997, .997, .998, .998, .998, .998, .998, .998, .998),

    0.995 : (None, None, None, 1, 1, .998, .997, .996, .996, .995, .995, .995, .995, .995, .995, .995, .995, .995, .995, .995, .995, .996, .996, .996, .996, .996, .996, .996, .996, .996, .996, .996, .996, .996, .996, .997, .997, .997, .997, .997, .997, .997, .997, .997, .997, .997, .997, .997, .997, .997, .997, .997, .998, .998, .998, .998, .998, .998, .998, .998, .998)
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
