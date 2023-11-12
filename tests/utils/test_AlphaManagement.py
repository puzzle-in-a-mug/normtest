"""Tests if  ``AlphaManagement`` is working as expected

--------------------------------------------------------------------------------
Command to run at the prompt:
    python -m unittest -v tests/utils/test_AlphaManagement.py
--------------------------------------------------------------------------------
"""

### GENERAL IMPORTS ###
import os
import unittest


### CLASS IMPORT ###
from normtest.utils.helpers import AlphaManagement

os.system("cls")


class Test_init(unittest.TestCase):
    def test_set_to_0_1(self):
        teste = AlphaManagement(0.1)
        self.assertEqual(teste.alpha, 0.1, msg="alpha is not 0.1")

    def test_default(self):
        teste = AlphaManagement()
        self.assertEqual(teste.alpha, 0.05, msg="default alpha is not 0.05")


class Test_set_alpha(unittest.TestCase):
    def test_set_to_0_1(self):
        teste = AlphaManagement()
        teste.set_alpha(0.1)
        self.assertEqual(teste.alpha, 0.1, msg="Did not changed alpha the 0.1")


class Test_get_alpha(unittest.TestCase):
    def test_default(self):
        teste = AlphaManagement()
        result = teste.get_alpha()
        self.assertEqual(result, 0.05, msg="The default alpha is not 0.05")

    def test_pt_br(self):
        teste = AlphaManagement()
        teste.set_alpha(0.01)
        result = teste.get_alpha()
        self.assertEqual(result, 0.01, msg="The get alpha returned wrong value")


if __name__ == "__main__":
    unittest.main()
