"""This module contains functions related to the Filliben test

##### List of functions (cte_alphabetical order) #####

## Functions WITH good TESTS ###


## Functions WITH some TESTS ###
- citation()


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
