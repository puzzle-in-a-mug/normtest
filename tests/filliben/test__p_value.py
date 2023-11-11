"""Tests if  ``_p_value`` is working as expected

--------------------------------------------------------------------------------
Command to run at the prompt:
    python -m unittest -v tests/filliben/test__p_value.py
    or
    python -m unittest -b tests/filliben/test__p_value.py

--------------------------------------------------------------------------------
"""

### GENERAL IMPORTS ###
import os
import unittest
import random

### FUNCTION IMPORT ###
from normtest.filliben import _p_value

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

        result = _p_value(statistic=self.statistic, sample_size=self.n, safe=False)
        self.assertIsInstance(
            result,
            (float, str),
            msg=f"not a float when statistic={self.statistic} and n={self.n}",
        )

    def test_safe(self):
        result = _p_value(self.statistic, self.n, safe=True)
        self.assertIsInstance(
            result,
            (float, str),
            msg=f"not a float when statistic={self.statistic} and n={self.n}",
        )

    def test_pass(self):
        result = _p_value(0.879, 5)
        self.assertAlmostEqual(result, 0.05, places=2, msg=f"wrong p-value")

        result = _p_value(0.954, 10)
        self.assertAlmostEqual(result, 0.25, places=2, msg=f"wrong p-value")

        result = _p_value(0.918, 22)
        self.assertAlmostEqual(result, 0.005, places=2, msg=f"wrong p-value")

        result = _p_value(0.400, 22)
        self.assertEqual(result, "p < 0.005", msg=f"wrong p-value")

        result = _p_value(0.998, 6)
        self.assertEqual(result, "p > 0.995", msg=f"wrong p-value")

    def test_impossible_statistic(self):
        statistics = [-1, 0, 1, 3]
        for statistic in statistics:
            with self.assertRaises(
                ValueError,
                msg=f"Does not raised ValueError when statistic={statistic} and n={self.n}",
            ):
                result = _p_value(statistic, self.n, safe=True)


if __name__ == "__main__":
    unittest.main()
