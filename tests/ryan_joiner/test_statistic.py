"""Tests if  ``statistic`` is working as expected

--------------------------------------------------------------------------------
Command to run at the prompt:
    python -m unittest -v tests/ryan_joiner/test_statistic.py
    or
    python -m unittest -b tests/ryan_joiner/test_statistic.py

--------------------------------------------------------------------------------
"""

### GENERAL IMPORTS ###
import os
import unittest
import numpy as np
import random

### FUNCTION IMPORT ###
from normtest.ryan_joiner import statistic

os.system("cls")


class Test_statistic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x_data = np.array(
            [
                148,
                154,
                158,
                160,
                161,
                162,
                166,
                170,
                170,
                182,
                195,
            ]
        )
        cls.zi = np.array(
            [
                -1.59322,
                -1.06056,
                -0.72791,
                -0.46149,
                -0.22469,
                0.00000,
                0.22469,
                0.46149,
                0.72791,
                1.06056,
                1.59322,
            ]
        )
        cls.result = 0.9565

    def test_input(self):
        result = statistic(self.x_data, self.zi)
        self.assertIsInstance(
            result,
            float,
            msg=f"not a float output",
        )

        result = statistic(x_data=self.x_data, zi=self.zi)
        self.assertIsInstance(
            result,
            float,
            msg=f"not a float output",
        )

        self.assertAlmostEqual(result, self.result, places=3, msg="wrong statistic")


if __name__ == "__main__":
    unittest.main()
