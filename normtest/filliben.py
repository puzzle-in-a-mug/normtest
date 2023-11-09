"""This module contains functions related to the Filliben test

##### List of functions (cte_alphabetical order) #####

## Functions WITH good TESTS ###


## Functions WITH some TESTS ###
- citation()
- _uniform_order_medians(sample_size, safe=False)

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


def _uniform_order_medians(sample_size, safe=False):
    """This function estimates the uniform order statistic median (:math:`m_{i}`) used in the Filliben normality test [1]_.

    Parameters
    ----------
    sample_size : int
        The sample size (greater than ``3``);
    safe : bool, optional
        Whether to check the inputs before performing the calculations (*True*) or not (*False*, default). Useful for beginners to identify problems in data entry (may reduce algorithm execution time).

    Returns
    -------
    mi : :doc:`numpy array <numpy:reference/generated/numpy.array>`
        The estimated the uniform order statistic median (:math:`m_{i}`)

    See Also
    --------
    fi_test


    Notes
    -----
    The uniform order statistic median is estimated using:

    .. math::

            m_{i} = \\begin{cases}1-0.5^{1/n} & i = 1\\\ \\frac{i-0.3175}{n+0.365} & i = 2, 3,  \\ldots , n-1 \\\ 0.5^{1/n}& i=n \\end{cases}

    where :math:`n` is the sample size and :math:`i` is the ith observation.


    References
    ----------
    .. [1] FILLIBEN, J. J. The Probability Plot Correlation Coefficient Test for Normality. Technometrics, 17(1), 111-117, (1975). Available at `doi.org/10.2307/1268008 <https://doi.org/10.2307/1268008>`_.



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
            lower=3,
            param_name="sample_size",
            func_name="normal_medians",
            inclusive=True,
        )

    i = np.arange(1, sample_size + 1)
    mi = (i - 0.3175) / (sample_size + 0.365)
    mi[0] = 1 - 0.5 ** (1 / sample_size)
    mi[-1] = 0.5 ** (1 / sample_size)

    return mi
