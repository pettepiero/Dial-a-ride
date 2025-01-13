import unittest
import cvrptw
from cvrptw.route import Route
from cvrptw.data_module import data
from cvrptw.initial_solutions.initial_solutions import neighbours, time_neighbours, nearest_neighbor_tw

class RouteTests(unittest.TestCase):
    """
    Tests for functions of route.py. Warning: use this with SEED = 1234 for reproducibility in
    data_module.py.
    """
    def test_calculate_planned_times(self):
        """
        Test the function that calculates the planned times of the route.
        """
        # initial_solution = nearest_neighbor_tw()

        # for route in initial_solution.routes:
        #     print(route.customers_list)
        #     print(f"{'Cust':<6} {'TW':<20} {'Plan':<15} {'Serv':<10} {'Edge'}")
        #     for idx, customer in enumerate(route.customers_list[:-1]):
        #         tw = f"[{data['time_window'][customer][0]}, {data['time_window'][customer][1]}]"
        #         plan = f"[{route.planned_windows[idx][0]:.2f}, {route.planned_windows[idx][1]:.2f}]"
        #         service = f"{data['service_time'][customer]:.1f}"
        #         edge = f"{data['edge_weight'][route.customers_list[idx]][route.customers_list[idx + 1]]:.2f}"

        #         print(f"{customer:<6} {tw:<20} {plan:<15} {service:<10} {edge}")

        #     print("\n")
        # print(data["edge_weight"][50][30])


if __name__ == "__main__":
    unittest.main()
