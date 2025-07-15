import unittest
import cvrptw
from lib.initial_solutions.initial_solutions import neighbours, time_neighbours, nearest_neighbor_tw
from lib.myvrplib.dataset_readers import (
    read_cordeau_data,
)
from lib.myvrplib.vrpstates import CVRPTWState

class NeighbourhoodTests(unittest.TestCase):
    """
    Tests for functions that calculate neighbours in initial_solutions.py.
    """
    data = read_cordeau_data(
        "/home/pettepiero/tirocinio/dial-a-ride/data/c-mdvrptw/pr12"
    )
    state = CVRPTWState(dataset=data)

    def test_neighbours_function(self):
        # Some hand picked checks for this dataset
        customer = 8
        self.assertEqual(neighbours(self.state, customer)[0], 13)

        customer = 5
        self.assertEqual(neighbours(self.state, customer)[0], 7)

        customer = 23
        self.assertEqual(neighbours(self.state, customer)[0], 30)

    def test_time_neighbours_function(self):

        # Some hand picked checks for this dataset
        customer = 27
        self.assertEqual(time_neighbours(self.state, customer)[0], 19)

        customer = 78
        self.assertEqual(time_neighbours(self.state, customer)[0], 65)

    def test_nn_sol(self):
        sol = nearest_neighbor_tw(self.state)

if __name__ == "__main__":
    unittest.main()
