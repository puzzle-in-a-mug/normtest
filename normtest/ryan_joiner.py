"""This module contains functions related to the Ryan-Joiner test

##### List of functions (cte_alphabetical order) #####

## Functions WITH good TESTS ###
- order_statistic(sample_size, cte_alpha="3/8", safe=False)
- citation(export=False)
- critical_value(sample_size, alpha=0.05, safe=False)

## Functions WITH some TESTS ###
- normal_order_statistic(x_data, weighted=False, cte_alpha="3/8", safe=False)


## Functions WITHOUT tests ###
- rj_critical_value(n, cte_alpha=0.05)
- rj_p_value(statistic, n)
- ryan_joiner(x_data, cte_alpha=0.05, method="blom", weighted=False)

- rj_correlation_plot(axes, x_data, method="blom", weighted=False)
- rj_dist_plot(axes, x_data, method="blom", min=4, max=50, deleted=False, weighted=False)

- make_bar_plot(axes, df, n_samples, cte_alpha_column_name=None, n_rep_name=None, tests_column_names=None, normal=True, safe=False)
- make_heatmap(axes, df, n_samples, cte_alpha_column_name=None, n_rep_name=None, tests_column_names=None, normal=True, safe=False)
- normal_distribution_plot(axes, n_rep, seed=None, xinfo=[0.00001, 0.99999, 1000], loc=0.0, scale=1.0, safe=False)



##### List of CLASS (cte_alphabetical order) #####

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
from paramcheckup import parameters, types, numbers, numpy_arrays
from . import bib

# from .utils import constants

##### CONSTANTS #####


##### CLASS #####

##### FUNCTIONS #####


def normal_order_statistic(x_data, weighted=False, cte_alpha="3/8", safe=False):
    """This function transforms the statistical order to the standard Normal distribution scale (:math:`b_{i}`).

    Parameters
    ----------
    x_data : :doc:`numpy array <numpy:reference/generated/numpy.array>`
        One dimension :doc:`numpy array <numpy:reference/generated/numpy.array>` with at least ``4`` observations.
    cte_alpha : str, optional
        A `str` with the `cte_alpha` value that should be adopted (see details in the Notes section). The options are:

        * `"0"`;
        * `"3/8"` (default);
        * `"1/2"`;

    weighted : bool, optional
        Whether to estimate the Normal order considering the repeats as its average (`True`) or not (`False`, default). Only has an effect if the dataset contains repeated values.
    safe : bool, optional
        Whether to check the inputs before performing the calculations (`True`) or not (`False`, default). Useful for beginners to identify problems in data entry (may reduce algorithm execution time).

    Returns
    -------
    bi : :doc:`numpy array <numpy:reference/generated/numpy.array>`
        The statistical order in the standard Normal distribution scale.


    Notes
    -----
    The transformation to the standard Normal scale is done using the equation:

    .. math::

            b_{i} = \\phi^{-1} \\left(p_{i} \\right)

    where :math:`p_i{}` is the normal statistical order and \\phi^{-1} is the inverse of the standard Normal distribution. The transformation is performed using :doc:`stats.norm.ppf() <scipy:reference/generated/scipy.stats.norm>`.

    The statistical order (:math:`p_{i}`) is estimated using :func:`order_statistic` function. See this function for details on parameter `cte_alpha`.

    Examples
    --------
    The first example uses `weighted=False`:

    >>> import numpy as np
    >>> from normtest import ryan_joiner
    >>> data = np.array([148, 148, 154, 158, 158, 160, 161, 162, 166, 170, 182, 195, 210])
    >>> result = ryan_joiner.normal_order_statistic(data, weighted=False)
    >>> print(result)
    [-1.67293739 -1.16188294 -0.84837993 -0.6020065  -0.38786869 -0.19032227
    0.          0.19032227  0.38786869  0.6020065   0.84837993  1.16188294
    1.67293739]

    The second example uses `weighted=True`, with the same data set:

    >>> result = ryan_joiner.normal_order_statistic(data, weighted=True)
    >>> print(result)
    [-1.37281032 -1.37281032 -0.84837993 -0.4921101  -0.4921101  -0.19032227
    0.          0.19032227  0.38786869  0.6020065   0.84837993  1.16188294
    1.67293739]


    Note that the results are only different for positions where we have repeated values. Using `weighted=True`, the normal statistical order is obtained with the average of the order statistic values.

    The results will be identical if the data set does not contain repeated values.

    """
    if safe:
        types.is_numpy(
            value=x_data, param_name="x_data", func_name="normal_order_statistic"
        )
        numpy_arrays.n_dimensions(
            arr=x_data,
            param_name="x_data",
            func_name="normal_order_statistic",
            n_dimensions=1,
        )
        numpy_arrays.greater_than_n(
            array=x_data,
            param_name="x_data",
            func_name="normal_order_statistic",
            minimum=4,
            inclusive=True,
        )
        types.is_bool(
            value=weighted, param_name="weighted", func_name="normal_order_statistic"
        )

    # ordering
    x_data = np.sort(x_data)
    if weighted:
        df = pd.DataFrame({"x_data": x_data})
        # getting mi values
        df["Rank"] = np.arange(1, df.shape[0] + 1)
        df["Ui"] = order_statistic(
            sample_size=x_data.size, cte_alpha=cte_alpha, safe=safe
        )
        df["Mi"] = df.groupby(["x_data"])["Ui"].transform("mean")
        normal_ordered = stats.norm.ppf(df["Mi"])
    else:
        ordered = order_statistic(
            sample_size=x_data.size, cte_alpha=cte_alpha, safe=safe
        )
        normal_ordered = stats.norm.ppf(ordered)

    return normal_ordered


def order_statistic(sample_size, cte_alpha="3/8", safe=False):
    """This function estimates the normal statistical order (:math:`p_{i}`) using approximations [1]_.

    Parameters
    ----------
    sample_size : int
        The sample size. Must be equal or greater than `4`;
    cte_alpha : str, optional
        A `str` with the `cte_alpha` value that should be adopted (see details in the Notes section). The options are:
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

    The `cte_alpha` parameter corresponds to the values studied by [1]_, which adopts the following equation to estimate the statistical order:

    .. math::

            p_{i} = \\frac{i - \\cte_alpha}{n - 2 \\times \\cte_alpha + 1}

    where :math:`n` is the sample size and :math:`i` is the ith observation.


    .. admonition:: Info

        `cte_alpha="3/8"` is adopted in the implementations of the Ryan-Joiner test in Minitab and Statext software. This option is also cited as an alternative by [2]_.

    References
    ----------
    .. [1] BLOM, G. Statistical Estimates and Transformed Beta-Variables. New York: John Wiley and Sons, Inc, p. 71-72, 1958.

    .. [2] RYAN, T. A., JOINER, B. L. Normal Probability Plots and Tests for Normality, Technical Report, Statistics Department, The Pennsylvania State University, 1976. Available at `www.additive-net.de <https://www.additive-net.de/de/component/jdownloads/send/70-support/236-normal-probability-plots-and-tests-for-normality-thomas-a-ryan-jr-bryan-l-joiner>`_. Access on: 22 Jul. 2023.


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
            option=cte_alpha,
            param_options=["0", "3/8", "1/2"],
            param_name="cte_alpha",
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
    if cte_alpha == "3/8":
        cte_alpha = 3 / 8
    elif cte_alpha == "0":
        cte_alpha = 0
    else:
        cte_alpha = 0.5

    return (i - cte_alpha) / (sample_size - 2 * cte_alpha + 1)


def statistic(x_data, zi, safe=False):
    """This function estimates the Ryan-Joiner test statistic [1]_.

    Parameters
    ----------
    x_data : :doc:`numpy array <numpy:reference/generated/numpy.array>`
        One dimension :doc:`numpy array <numpy:reference/generated/numpy.array>` with at least ``4`` observations (ordered in ascending order).
    bi : :doc:`numpy array <numpy:reference/generated/numpy.array>`
        The corresponding statistical order in the standard Normal distribution scale.
    safe : bool, optional
        Whether to check the inputs before performing the calculations (`True`) or not (`False`, default). Useful for beginners to identify problems in data entry (may reduce algorithm execution time).


    Returns
    -------
    statistic : float
        The test statistic


    Notes
    -----

    The test statistic (:math:`R_{p}`) is estimated through the correlation between the ordered data and the Normal statistical order:

    .. math::

            R_{p}=\\dfrac{\\sum_{i=1}^{n}x_{(i)}z_{(i)}}{\\sqrt{s^{2}(n-1)\\sum_{i=1}^{n}z_{(i)}^2}}

    where :math:`z_{(i)}` values are the z-score values of the corresponding experimental data (:math:`x_{({i)}}`) value, :math:`n` is the sample size and :math:`s^{2}` is the sample variance.

    The correlation is estimated using :doc:`scipy.stats.pearsonr() <scipy:reference/generated/scipy.stats.pearsonr>`.


    References
    ----------
    .. [1] RYAN, T. A., JOINER, B. L. Normal Probability Plots and Tests for Normality, Technical Report, Statistics Department, The Pennsylvania State University, 1976. Available at `www.additive-net.de <https://www.additive-net.de/de/component/jdownloads/send/70-support/236-normal-probability-plots-and-tests-for-normality-thomas-a-ryan-jr-bryan-l-joiner>`_. Access on: 22 Jul. 2023.


    Examples
    --------
    >>> from normtest import ryan_joiner
    >>> import numpy as np
    >>> x_data = np.array([148, 148, 154, 158, 158, 160, 161, 162, 166, 170, 182, 195, 210])
    >>> x_data = np.sort(x_data)
    >>> order_statistic = ryan_joiner.order_statistic(x_data.size)
    >>> normal_order = ryan_joiner.normal_order_statistic(x_data)
    >>> result = ryan_joiner.statistic(x_data, normal_order)
    >>> print(result)
    0.9225156050800545


    """
    func_name = "statistic"
    if safe:  # missing equal size test
        types.is_numpy(value=x_data, param_name="x_data", func_name=func_name)
        numpy_arrays.n_dimensions(
            arr=x_data,
            param_name="x_data",
            func_name=func_name,
            n_dimensions=1,
        )
        numpy_arrays.greater_than_n(
            array=x_data,
            param_name="x_data",
            func_name=func_name,
            minimum=4,
            inclusive=True,
        )
        types.is_numpy(value=zi, param_name="zi", func_name=func_name)
        numpy_arrays.n_dimensions(
            arr=zi,
            param_name="zi",
            func_name=func_name,
            n_dimensions=1,
        )
        numpy_arrays.greater_than_n(
            array=zi,
            param_name="zi",
            func_name=func_name,
            minimum=4,
            inclusive=True,
        )

    return stats.pearsonr(zi, x_data)[0]


def citation(export=False):
    """This function returns the reference from Ryan Joiner's test, with the option to export the reference in `.bib` format.

    Parameters
    ----------
    export : bool
        Whether to export the reference as `ryan-joiner.bib` file (`True`) or not (`False`, default);

    Returns
    -------
    reference : str
        The Ryan Joiner Test Reference


    """
    reference = bib.make_techreport(
        citekey="RyanJoiner1976",
        author="Thomas A. Ryan, Jr. and Brian L. Joiner",
        title="Normal Probability Plots and Tests for Normality",
        institution="The Pennsylvania State University, Statistics Department",
        year="1976",
        export=False,
    )
    if export:
        with open("ryan-joiner.bib", "w") as my_bib:
            my_bib.write(reference)
    return reference


def critical_value(sample_size, alpha=0.05, safe=False):
    """This function calculates the critical value of the Ryan-Joiner test [1]_.

    Parameters
    ----------
    sample_size : int
        The sample size. Must be greater than ``3``.
    alpha : float, optional
        The level of significance (:math:`\\alpha`). Must be ``0.01``, ``0.05`` (default) or ``0.10``.
    safe : bool, optional
        Whether to check the inputs before performing the calculations (`True`) or not (`False`, default). Useful for beginners to identify problems in data entry (may reduce algorithm execution time).

    Returns
    -------
    critical : float
        The critical value of the test.

    See Also
    --------
    pass

    Notes
    -----
    The critical values are calculated using [1]_ the following equations:

    .. math::

            R_{p;\\alpha=0.10}^{'} = 1.0071 - \\frac{0.1371}{\\sqrt{n}} - \\frac{0.3682}{n} + \\frac{0.7780}{n^{2}}

            R_{p;\\alpha=0.05}^{'} = 1.0063 - \\frac{0.1288}{\\sqrt{n}} - \\frac{0.6118}{n} + \\frac{1.3505}{n^{2}}

            R_{p;\\alpha=0.01}^{'} = 0.9963 - \\frac{0.0211}{\\sqrt{n}} - \\frac{1.4106}{n} + \\frac{3.1791}{n^{2}}

    where :math:`n` is the sample size.


    References
    ----------
    .. [1]  RYAN, T. A., JOINER, B. L. Normal Probability Plots and Tests for Normality, Technical Report, Statistics Department, The Pennsylvania State University, 1976. Available at `www.additive-net.de <https://www.additive-net.de/de/component/jdownloads/send/70-support/236-normal-probability-plots-and-tests-for-normality-thomas-a-ryan-jr-bryan-l-joiner>`_. Access on: 22 Jul. 2023.


    Examples
    --------
    >>> from normtest import ryan_joiner
    >>> critical = ryan_joiner.critical_value(10, alpha=0.05)
    >>> print(critical)
    0.9178948637370312

    """
    func_name = "critical_value"
    if safe:
        parameters.param_options(
            option=alpha,
            param_options=[0.01, 0.05, 0.10],
            param_name="alpha",
            func_name=func_name,
        )
        types.is_int(value=sample_size, param_name="sample_size", func_name=func_name)
        numbers.is_greater_than(
            value=sample_size,
            lower=4,
            param_name="sample_size",
            func_name=func_name,
            inclusive=True,
        )

    if alpha == 0.1:
        return (
            1.0071
            - (0.1371 / np.sqrt(sample_size))
            - (0.3682 / sample_size)
            + (0.7780 / sample_size**2)
        )
    elif alpha == 0.05:
        return (
            1.0063
            - (0.1288 / np.sqrt(sample_size))
            - (0.6118 / sample_size)
            + (1.3505 / sample_size**2)
        )
    else:  # alpha == 0.01:
        return (
            0.9963
            - (0.0211 / np.sqrt(sample_size))
            - (1.4106 / sample_size)
            + (3.1791 / sample_size**2)
        )


def p_value(statistic, sample_size, safe=False):
    """This function estimates the probability associated with the Ryan-Joiner Normality test.

    Parameters
    ----------
    statistic : float (positive)
        The Ryan-Joiner test statistics.
    sample_size : int
        The sample size. Must be equal or greater than `4`;
    safe : bool, optional
        Whether to check the inputs before performing the calculations (`True`) or not (`False`, default). Useful for beginners to identify problems in data entry (may reduce algorithm execution time).

    Returns
    -------
    p_value : float or str
        The probability of the test.

    See Also
    --------
    rj_critical_value
    ryan_joiner


    Notes
    -----
    The test probability is estimated through linear interpolation of the test statistic with critical values from the Ryan-Joiner test [1]_. The Interpolation is performed using the :doc:`scipy.interpolate.interp1d() <scipy:reference/generated/scipy.interpolate.interp1d>` function.

    * If the test statistic is greater than the critical value for :math:`\\alpha=0.10`, the result is always *"p > 0.100"*.
    * If the test statistic is lower than the critical value for :math:`\\alpha=0.01`, the result is always *"p < 0.010"*.



    References
    ----------
    .. [1]  RYAN, T. A., JOINER, B. L. Normal Probability Plots and Tests for Normality, Technical Report, Statistics Department, The Pennsylvania State University, 1976. Available at `www.additive-net.de <https://www.additive-net.de/de/component/jdownloads/send/70-support/236-normal-probability-plots-and-tests-for-normality-thomas-a-ryan-jr-bryan-l-joiner>`_. Access on: 22 Jul. 2023.


    Examples
    --------
    >>> from normtest import ryan_joiner
    >>> p_value = ryan_joiner.p_value(0.90, 10)
    >>> print(p_value)
    0.030930589077996555

    """
    func_name = "p_value"
    if safe:
        numbers.is_between_a_and_b(
            value=statistic,
            a=0,
            b=1,
            param_name="statistic",
            func_name=func_name,
            inclusive=False,
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

    alphas = np.array([0.10, 0.05, 0.01])
    criticals = np.array(
        [
            critical_value(sample_size=sample_size, alpha=alphas[0], safe=False),
            critical_value(sample_size=sample_size, alpha=alphas[1], safe=False),
            critical_value(sample_size=sample_size, alpha=alphas[2], safe=False),
        ]
    )
    f = interpolate.interp1d(criticals, alphas)
    if statistic > max(criticals):
        return "p > 0.100"
    elif statistic < min(criticals):
        return "p < 0.010"
    else:
        p_value = f(statistic)
        return p_value
