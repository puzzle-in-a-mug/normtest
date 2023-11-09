"""This module contains functions related to the Filliben test

##### List of functions (cte_alphabetical order) #####

## Functions WITH good TESTS ###


## Functions WITH some TESTS ###



## Functions WITHOUT tests ###



##### List of CLASS (alphabetical order) #####



Author: Anderson Marcos Dias Canteli <andersonmdcanteli@gmail.com>

Created : November 08, 2023

Last update: November 08, 2023
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
from .utils import constants


##### DOCUMENTATION #####
from .utils import documentation as docs

#### CONSTANTS ####
Filliben1975 = "FILLIBEN, J. J. The Probability Plot Correlation Coefficient Test for Normality. Technometrics, v. 17, n. 1, p. 111-117, 1975."
Blom1958 = "BLOM, G. Statistical Estimates and Transformed Beta-Variables. New York: John Wiley and Sons, Inc, p. 71-72, 1958."


##### CLASS #####


##### FUNCTIONS #####


def citation(export=False):
    """This function returns the reference from Ryan Joiner's test, with the option to export the reference in `.bib` format.

    Parameters
    ----------
    export : bool
        Whether to export the reference as `ryan-joiner.bib` file (`True`) or not (`False`, default);


    Returns
    -------
    reference : str
        The Ryan Joiner Test reference

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
