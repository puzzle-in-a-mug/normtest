"""Tests if  ``order_statistic`` is working as expected

--------------------------------------------------------------------------------
Command to run at the prompt:
    python -m unittest -v tests/ryan_joiner/test_order_statistic.py
    or
    python -m unittest -b tests/ryan_joiner/test_order_statistic.py

--------------------------------------------------------------------------------
"""

### GENERAL IMPORTS ###
import os
import unittest
import numpy as np
import random

### FUNCTION IMPORT ###
from normtest.ryan_joiner import order_statistic

os.system("cls")


class Test_order_statistic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n = random.randrange(4, 200)
        methods = ["0", "3/8", "1/2"]
        cls.method = random.sample(methods, 1)[0]
        cls.safe = True

    def test_input(self):
        result = order_statistic(self.n, self.method, self.safe)
        self.assertIsInstance(
            result,
            np.ndarray,
            msg=f"not a float when method={self.method} and n={self.n}",
        )

        result = order_statistic(
            sample_size=self.n, cte_alpha=self.method, safe=self.safe
        )
        self.assertIsInstance(
            result,
            np.ndarray,
            msg=f"not a float when method={self.method} and n={self.n}",
        )

    def test_outputs(self):
        result = order_statistic(self.n, self.method)
        self.assertIsInstance(
            result,
            np.ndarray,
            msg=f"not a float when method={self.method} and n={self.n}",
        )

    def test_safe(self):
        result = order_statistic(self.n, self.method, safe=True)
        self.assertIsInstance(
            result,
            np.ndarray,
            msg=f"not a float when method={self.method} and n={self.n}",
        )

    def test_3_8_odd(self):
        n = 9
        expected = np.array(
            [
                0.067568,
                0.175676,
                0.283784,
                0.391892,
                0.500000,
                0.608108,
                0.716216,
                0.824324,
                0.932432,
            ]
        )
        result = order_statistic(n, "3/8")
        for pair in zip(result, expected):
            self.assertAlmostEqual(
                pair[0], pair[1], places=5, msg=f"wrong statisitc order for 3/8"
            )

    def test_3_8_even(self):
        n = 10
        expected = np.array(
            [
                0.060976,
                0.158537,
                0.256098,
                0.353659,
                0.451220,
                0.548780,
                0.646341,
                0.743902,
                0.841463,
                0.939024,
            ]
        )
        result = order_statistic(n, "3/8")
        for pair in zip(result, expected):
            self.assertAlmostEqual(
                pair[0], pair[1], places=5, msg=f"wrong statisitc order for 3/8"
            )

    def test_0_odd(self):
        n = 9
        expected = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        result = order_statistic(n, "0")
        for pair in zip(result, expected):
            self.assertAlmostEqual(
                pair[0], pair[1], places=5, msg=f"wrong statisitc order for 0"
            )

    def test_0_even(self):
        n = 10
        expected = np.array(
            [
                0.090909,
                0.181818,
                0.272727,
                0.363636,
                0.454545,
                0.545455,
                0.636364,
                0.727273,
                0.818182,
                0.909091,
            ]
        )
        result = order_statistic(n, "0")
        for pair in zip(result, expected):
            self.assertAlmostEqual(
                pair[0], pair[1], places=5, msg=f"wrong statisitc order for 0"
            )

    def test_1_2_odd(self):
        n = 9
        expected = np.array(
            [
                0.055556,
                0.166667,
                0.277778,
                0.388889,
                0.500000,
                0.611111,
                0.722222,
                0.833333,
                0.944444,
            ]
        )
        result = order_statistic(n, "1/2")
        for pair in zip(result, expected):
            self.assertAlmostEqual(
                pair[0], pair[1], places=5, msg=f"wrong statisitc order for 1/2"
            )

    def test_1_2_even(self):
        n = 10
        expected = np.array(
            [
                0.050000,
                0.150000,
                0.250000,
                0.350000,
                0.450000,
                0.550000,
                0.650000,
                0.750000,
                0.850000,
                0.950000,
            ]
        )
        result = order_statistic(n, "1/2")
        for pair in zip(result, expected):
            self.assertAlmostEqual(
                pair[0], pair[1], places=5, msg=f"wrong statisitc order for 1/2"
            )

    def test_wrong_method(self):
        with self.assertRaises(
            ValueError,
            msg=f"Does not raised ValueError when method={None} and n={self.n}",
        ):
            result = order_statistic(self.n, None, safe=True)

    def test_n_not_int(self):
        n_values = [
            5.1,
            "5",
            [5],
            (6,),
        ]
        for n in n_values:
            with self.assertRaises(
                TypeError,
                msg=f"Does not raised ValueError when method={self.method} and n={n}",
            ):
                result = order_statistic(n, self.method, safe=True)

    def test_small_n(self):
        n_values = [-5, 0, 3]
        for n in n_values:
            with self.assertRaises(
                ValueError,
                msg=f"Does not raised ValueError when method={self.method} and n={n}",
            ):
                result = order_statistic(n, self.method, safe=True)


if __name__ == "__main__":
    unittest.main()
