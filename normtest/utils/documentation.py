ALPHA = {
    "type": "alpha : float, optional",
    "description": "The level of significance (:math:`\\alpha`). Must be ``0.01``, ``0.05`` (default) or ``0.10``;",
}


WEIGHTED = {
    "type": "weighted : bool, optional",
    "description": "Whether to estimate the Normal order considering the repeats as its average (`True`) or not (`False`, default). Only has an effect if the dataset contains repeated values;",
}


SAFE = {
    "type": "safe : bool, optional",
    "description": "Whether to check the inputs before performing the calculations (`True`) or not (`False`, default). Useful for beginners to identify problems in data entry (may reduce algorithm execution time);",
}

SAMPLE_SIZE = {
    "type": "sample_size : int",
    "description": "The sample size. Must be equal or greater than ``4``;",
}

X_DATA = {
    "type": "x_data : :doc:`numpy array <numpy:reference/generated/numpy.array>`",
    "description": "One dimension :doc:`numpy array <numpy:reference/generated/numpy.array>` with at least ``4`` observations.",
}


CTE_ALPHA = {
    "type": "cte_alpha : str, optional",
    "description": """A `str` with the `cte_alpha` value that should be adopted (see details in the Notes section). The options are:

        * `"0"`;
        * `"3/8"` (default);
        * `"1/2"`;""",
}

# PARAM = {
#     "type":
#     "description":
# }
