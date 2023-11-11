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
from copy import deepcopy


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
from .utils import critical_values, constants


##### DOCUMENTATION #####
from .utils import documentation as docs


#### CONSTANTS ####
Filliben1975 = "FILLIBEN, J. J. The Probability Plot Correlation Coefficient Test for Normality. Technometrics, v. 17, n. 1, p. 111-117, 1975."
Blom1958 = "BLOM, G. Statistical Estimates and Transformed Beta-Variables. New York: John Wiley and Sons, Inc, p. 71-72, 1958."


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
    safe=docs.SAFE["type"],
    safe_desc=docs.SAFE["description"],
    critical=docs.CRITICAL["type"],
    critical_desc=docs.CRITICAL["description"],
    fi_ref=Filliben1975,
)
def _critical_value(sample_size, alpha=0.05, safe=False):
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
    {critical}
        {critical_desc}


    References
    ----------
    .. [1] {fi_ref}


    Examples
    --------
    >>> from normtest import filliben
    >>> sample_size = 7
    >>> critical = filliben._critical_value(sample_size, alpha=0.05)
    >>> print(critical)
    0.899


    """
    func_name = "_critical_value"
    # making a copy from original critical values
    critical = deepcopy(critical_values.FILLIBEN_CRITICAL)

    if safe:
        types.is_int(value=sample_size, param_name="sample_size", func_name=func_name)
        numbers.is_greater_than(
            value=sample_size,
            lower=4,
            param_name="sample_size",
            func_name=func_name,
            inclusive=True,
        )
        types.is_float(value=alpha, param_name="alpha", func_name=func_name)
        numbers.is_between_a_and_b(
            value=alpha,
            a=0.005,
            b=0.995,
            param_name="alpha",
            func_name=func_name,
            inclusive=True,
        )
        parameters.param_options(
            option=alpha,
            param_options=list(critical.keys())[1:],
            param_name="alpha",
            func_name=func_name,
        )

    if sample_size not in critical["n"]:
        if sample_size < 100:
            constants.user_warning(
                "The Filliben critical value may not be accurate as it was obtained with linear interpolation."
            )
        else:
            constants.user_warning(
                "The Filliben critical value may not be accurate as it was obtained with linear *extrapolation*."
            )

    f = interpolate.interp1d(
        critical["n"][3:], critical[alpha][3:], fill_value="extrapolate"
    )

    return float(f(sample_size))


@docs.docstring_parameter(
    axes=docs.AXES["type"],
    axes_desc=docs.AXES["description"],
    statistic=docs.STATISTIC["type"],
    statistic_desc=docs.STATISTIC["description"],
    sample_size=docs.SAMPLE_SIZE["type"],
    sample_size_desc=docs.SAMPLE_SIZE["description"],
    safe=docs.SAFE["type"],
    safe_desc=docs.SAFE["description"],
    fi_ref=Filliben1975,
)
def dist_plot(axes, test=None, alphas=[0.10, 0.05, 0.01], safe=False):
    """This function generates axis with critical data from the Filliben Normality test [1]_.

    Parameters
    ----------
    {axes}
        {axes_desc}
    test : tuple (optional), with two elements:
        {statistic}
            {statistic_desc}
        {sample_size}
            {sample_size_desc}
    alphas : list of floats, optional
        The significance level (:math:`\\alpha`) to draw the critical lines. Default is `[0.10, 0.05, 0.01]`. It can be a combination of:

        * ``0.005``;
        * ``0.01``;
        * ``0.025``;
        * ``0.05``;
        * ``0.10``;
        * ``0.25``;
        * ``0.50``;
        * ``0.75``;
        * ``0.90``;
        * ``0.95``;
        * ``0.975``;
        * ``0.99``;
        * ``0.995``;

    {safe}
        {safe_desc}

    Returns
    -------
    {axes}
        {axes_desc}


    References
    ----------
    .. [1] {fi_ref}


    Examples
    --------
    >>> from normtest import filliben
    >>> import matplotlib.pyplot as plt
    >>> >>> fig, ax = plt.subplots(figsize=(6, 4))
    >>> filliben.dist_plot(axes=ax, test=(0.98538, 7))
    >>> # plt.savefig("filliben_paper.png")
    >>> plt.show()


    .. image:: img/filliben_paper.png
        :alt: Default critical chart for Filliben Normality test
        :align: center


    """
    # making a copy from original critical values
    critical = deepcopy(critical_values.FILLIBEN_CRITICAL)
    if safe:
        func_name = "dist_plot"
        types.is_subplots(value=axes, param_name="axes", func_name=func_name)

        if test is not None:
            types.is_tuple(value=test, param_name="test", func_name=func_name)
            numbers.is_float_or_int(
                value=test[0], param_name="test[0]", func_name=func_name
            )
            numbers.between_a_and_b(
                value=test[0],
                a=0,
                b=1,
                param_name="test[0]",
                func_name=func_name,
                inclusive=False,
            )
            types.is_int(value=test[1], param_name=test[1], func_name=func_name)
            numbers.is_greater_than(
                value=test[1],
                lower=3,
                param_name="test[1]",
                func_name=func_name,
                inclusive=True,
            )

        for alpha in alphas:
            parameters.param_options(
                option=alpha,
                param_options=list(critical.keys())[1:],
                param_name="alphas",
                func_name=func_name,
            )

    if test is not None:
        axes.scatter(test[1], test[0], c="r", label="statistic")

    for alpha in alphas:
        axes.scatter(critical["n"], critical[alpha], label=alpha)
    axes.set_xlabel("Sample size")
    axes.set_ylabel("Filliben critical values")
    axes.legend(loc=4)

    return axes
