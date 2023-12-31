"""Tests if  ``_p_value`` is working as expected

--------------------------------------------------------------------------------
Command to run at the prompt:
    python -m unittest -v tests/looney_gulledge/test__p_value.py
--------------------------------------------------------------------------------
"""

### GENERAL IMPORTS ###
import os
import unittest
import random

### FUNCTION IMPORT ###
from normtest.looney_gulledge import _p_value

os.system("cls")


class Test__p_value(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n = random.randrange(4, 100)
        cls.statistic = random.uniform(0.001, 0.999)

    def test_outputs(self):
        result = _p_value(self.statistic, self.n)
        self.assertIsInstance(
            result,
            (float, str),
            msg=f"not a float when statistic={self.statistic} and n={self.n}",
        )

        result = _p_value(statistic=self.statistic, sample_size=self.n)
        self.assertIsInstance(
            result,
            (float, str),
            msg=f"not a float when statistic={self.statistic} and n={self.n}",
        )

    def test_pass(self):
        result = _p_value(0.826, 5)
        self.assertEqual(result, 0.01, msg=f"wrong p-value")

        result = _p_value(0.918, 10)
        self.assertEqual(result, 0.05, msg=f"wrong p-value")

        result = _p_value(0.974, 22)
        self.assertEqual(result, 0.25, msg=f"wrong p-value")

        result = _p_value(0.400, 22)
        self.assertEqual(result, "p < 0.005", msg=f"wrong p-value")

        result = _p_value(0.998, 6)
        self.assertEqual(result, "p > 0.995", msg=f"wrong p-value")


if __name__ == "__main__":
    unittest.main()
