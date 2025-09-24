import unittest
import numpy as np
import os
from pathlib import Path
from lib.myvrplib.data_module import get_data_format

class TestReadingMethods(unittest.TestCase):

    def test_get_data_format(self):
        vrplib_data_path = Path(os.getcwd()) / 'tests/tests_data/inst_00234.mdvrp' 
        data_format = get_data_format(vrplib_data_path)

        self.assertEqual('vrplib', data_format)

        cordeau_data_path = Path(os.getcwd()) / 'tests/tests_data/p01' 
        data_format = get_data_format(cordeau_data_path)

        self.assertEqual('cordeau', data_format)

        wrong_data_path = Path(os.getcwd()) / 'tests/tests_data/wrong_cordeau' 

        self.assertRaises(ValueError, get_data_format, wrong_data_path)
