import unittest
import cvrptw
from cvrptw.initial_solutions.initial_solutions import neighbours, time_neighbours
from cvrptw.data_module import data

class NeighbourhoodTests(unittest.TestCase):
    """
    Tests for functions that calculate neighbours in initial_solutions.py.
    """
    # for el in data["time_window"]:
    #     print(el)
    # print(data["time_window"])

    def test_neighbours_function(self):
        depots = data["depots"]
        customer = 46
        calculated_nearest = neighbours(customer, depots)
        true_nearest = 42
        self.assertEqual(calculated_nearest[0], true_nearest)
        customer = 14
        calculated_nearest = neighbours(customer, depots)
        true_nearest = 28
        self.assertEqual(calculated_nearest[0], true_nearest)

    def test_time_neighbours_function(self):

    # print(f"Best: {data['time_window'][-11]}")
    # print(len(data["time_window"]))
    # print(data["time_window"][42])
        depots = data["depots"]
        customer = 9     # [80, 233]
        calculated_nearest = time_neighbours(customer, depots)
        true_nearest = 10
        self.assertEqual(calculated_nearest[0], true_nearest)


if __name__ == "__main__":
    unittest.main()
