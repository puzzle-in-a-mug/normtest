"""Tests if  ``uniform_order_medians`` is working as expected

--------------------------------------------------------------------------------
Command to run at the prompt:
    python -m unittest -v tests/filliben/test_uniform_order_medians.py

--------------------------------------------------------------------------------
"""

### GENERAL IMPORTS ###
import os
import unittest
import numpy as np

### FUNCTION IMPORT ###
from normtest.filliben import _uniform_order_medians

os.system("cls")


class Test_uniform_order_medians(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.filliben = np.array(
            [0.0943, 0.2284, 0.3642, 0.5, 0.6358, 0.7716, 0.9057]
        )  # Filliben example
        cls.sample_size = 10

    def test_outputs(self):
        result = _uniform_order_medians(self.sample_size, False)
        self.assertTrue(isinstance(result, np.ndarray), msg=f"not numpy type")

        result = _uniform_order_medians(sample_size=self.sample_size, safe=False)
        self.assertTrue(isinstance(result, np.ndarray), msg=f"not numpy type")

    def test_safe(self):
        sample_size = 7
        result = _uniform_order_medians(sample_size, safe=True)
        self.assertTrue(
            np.allclose(result, self.filliben, atol=4), msg="arrays does not match"
        )

    def test_pass(self):
        sample_size = 7
        result = _uniform_order_medians(sample_size)
        self.assertTrue(
            np.allclose(result, self.filliben, atol=4), msg="arrays does not match"
        )

    def test_small_n(self):
        n_values = [-5, 0, 3]
        for n in n_values:
            with self.assertRaises(
                ValueError,
                msg=f"Does not raised ValueError when sample size is small",
            ):
                result = _uniform_order_medians(n, safe=True)


if __name__ == "__main__":
    unittest.main()
