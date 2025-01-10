import unittest
import cvrptw
from cvrptw.route import Route
from cvrptw.data_module import data
from cvrptw.initial_solutions.initial_solutions import neighbours, time_neighbours, nearest_neighbor_tw

class RouteTests(unittest.TestCase):
    """
    Tests for functions of route.py.
    """

    print(data["time_window"])

    def test_calculate_planned_times(self):
        """
        Test the function that calculates the planned times of the route.
        """
        initial_solution = nearest_neighbor_tw()
        route = initial_solution.routes[1]
        print(route.customers_list)


if __name__ == "__main__":
    unittest.main()
