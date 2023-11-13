"""Tests if  ``Filliben`` is working as expected

--------------------------------------------------------------------------------
Command to run at the prompt:
    python -m unittest -v tests/filliben/test_Filliben.py
--------------------------------------------------------------------------------
"""

### GENERAL IMPORTS ###
import os
import unittest
from scipy import stats
import random
import numpy as np

### CLASS IMPORT ###
from normtest import Filliben

os.system("cls")


class Test_init(unittest.TestCase):
    def test_default(self):
        teste = Filliben()
        self.assertTrue(teste.safe, msg="wrong safe")
        self.assertEqual(teste.alpha, 0.05, msg="wrong alpha")

    def test_changed(self):
        teste = Filliben(alpha=0.10, safe=False)
        self.assertFalse(teste.safe, msg="wrong safe")
        self.assertEqual(teste.alpha, 0.10, msg="wrong alpha")


class Test_fit_applied(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = stats.norm.rvs(size=random.randint(5, 30))
        alphas = [0.01, 0.05, 0.1]
        cls.alpha = random.sample(alphas, 1)[0]

    def test_not_applied(self):
        teste = Filliben()
        self.assertIsNone(teste.conclusion, msg="wrong conclusion")

    def test_applied(self):
        teste = Filliben()
        teste.fit(self.data)
        self.assertIsInstance(teste.conclusion, str, msg="wrong conclusion type")

    def test_safe(self):
        with self.assertRaises(
            TypeError,
            msg=f"Does not raised TypeError when alpha = 0",
        ):
            teste = Filliben(alpha=0)

        with self.assertRaises(
            TypeError,
            msg=f"Does not raised TypeError when alpha = 1",
        ):
            teste = Filliben(alpha=1)
        with self.assertRaises(
            ValueError,
            msg=f"Does not raised ValuError when alpha = 0.004",
        ):
            teste = Filliben(alpha=0.004)
        with self.assertRaises(
            ValueError,
            msg=f"Does not raised ValuError when alpha = 0.996",
        ):
            teste = Filliben(alpha=0.996)

        with self.assertRaises(
            TypeError,
            msg=f"Does not raised TypeError when safe=0.996",
        ):
            teste = Filliben(safe=0.996)


class Test_fit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = np.array([6, 1, -4, 8, -2, 5, 0])
        cls.alpha = 0.05

    def test_applied(self):
        teste = Filliben()
        teste.fit(self.data)
        self.assertAlmostEqual(
            teste.statistic, 0.98538, places=3, msg="wrong statistic"
        )
        self.assertAlmostEqual(teste.critical, 0.899, places=3, msg="wrong critical")
        self.assertEqual(teste.conclusion, "Fail to reject H₀", msg="wrong conclusion")
        self.assertEqual(len(teste.normality), 4, msg="wrong number of outputs")
        self.assertIsInstance(
            teste.normality.statistic, float, msg="wrong type for statistic"
        )
        self.assertIsInstance(
            teste.normality.critical, float, msg="wrong type for critical"
        )
        self.assertIsInstance(
            teste.normality.p_value, float, msg="wrong type for pvalor"
        )
        self.assertIsInstance(
            teste.normality.conclusion, str, msg="wrong type for conclusion"
        )

    def test_safe(self):
        data = [
            [148, 154, 158, 160, 161, 162, 166, 170, 182, 195, 236],
            (148, 154, 158, 160, 161, 162, 166, 170, 182, 195, 236),
            148,
            148.5,
            "148",
        ]
        for d in data:
            with self.assertRaises(
                TypeError,
                msg=f"Does not raised ValueError when type = {type(d).__name__}",
            ):
                teste = Filliben()
                teste.fit(d)

        data = np.array([[1, 2, 3, 4, 5]])
        with self.assertRaises(
            ValueError,
            msg=f"Does not raised ValueError when n dim is wrong",
        ):
            teste = Filliben()
            teste.fit(data)

        n_values = [
            np.array([1, 2, 3]),
            np.array([1, 2]),
            np.array([1]),
        ]
        for n in n_values:
            with self.assertRaises(
                ValueError,
                msg=f"Does not raised ValueError when sample size is small",
            ):
                teste = Filliben()
                teste.fit(data)


if __name__ == "__main__":
    unittest.main()
