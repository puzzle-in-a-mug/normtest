"""

##### List of functions (alphabetical order) #####

## Functions WITH good TESTS ###


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
- ordered_statistics(n, method)


##### List of CLASS (alphabetical order) #####

##### Dictionary of abbreviations #####



Author: Anderson Marcos Dias Canteli <andersonmdcanteli@gmail.com>

Created: September 22, 2023.

Last update: October 04, 2023



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
import seaborn as sns

### home made ###
from .utils import checkers
from .utils import constants

##### CONSTANTS #####


##### CLASS #####

##### FUNCTIONS #####


## RYAN - JOINER TEST ##
