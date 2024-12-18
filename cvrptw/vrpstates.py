from data_module import data
from myvrplib import END_OF_DAY
from route import Route

import numpy as np


class CvrptwState:
    """
    Solution state for CVRPTW. It has two data members, routes and unassigned.
    Routes is a list of list of integers, where each inner list corresponds to
    a single route denoting the sequence of customers to be visited. A route
    does not contain the start and end depot. Times is a list of lists, containing
    the planned arrival times for each customer in the routes. The outer list
    corresponds to the routes, and the inner list corresponds to the customers of
    the route. Unassigned is a list of integers, each integer representing an
    unassigned customer.
    """

    def __init__(
        self,
        routes: list[Route],
        times: list,
        dataset: dict = data,
        unassigned: list = None,
    ):
        self.routes = routes
        self.times = times  # planned arrival times for each customer
        self.dataset = dataset
        self.unassigned = unassigned if unassigned is not None else []
        # DEBUG
        print(f"type(dataset) = {type(dataset)}")

        self.twc = self.generate_twc_matrix(dataset["time_window"], dataset["edge_weight"])
        self.qmax = self.get_qmax()
        self.dmax = self.get_dmax()
        self.norm_tw = (
            self.twc / self.dmax
        )  # Note: maybe use only norm_tw in the future?

    def __str__(self):
        return f"Routes: {[route.customers_list for route in self.routes]},\nUnassigned: {self.unassigned},\nTimes: {self.times}"

    def copy(self):
        return CvrptwState(
            [route.copy() for route in self.routes],  # Deep copy each Route
            self.times.copy(),
            self.dataset.copy(),
            self.unassigned.copy(),
        )

    def objective(self):
        """
        Computes the total route costs.
        """
        return sum(route.cost for route in self.routes)

    @property
    def cost(self):
        """
        Alias for objective method. Used for plotting.
        """
        return self.objective()

    def find_route(self, customer):
        """
        Return the route that contains the passed-in customer.
        """
        for route in self.routes:
            if customer in route.customers_list:
                return route
        print(f"Solution does not contain customer {customer}.")

    def find_index_in_route(self, customer, route: Route):
        """
        Return the index of the customer in the route.
        """
        if customer in route.customers_list:
            return route.customers_list.index(customer)

        raise ValueError(f"Given route does not contain customer {customer}.")

    def update_times(self):
        """
        Update the arrival times of each customer in the routes.
        """
        for route in self.routes:
            route.times = self.evaluate_times_of_route(route)
            route.latest_start_times = route.get_latest_times()
            route.earliest_start_times = route.get_earliest_times()

    def evaluate_times_of_route(self, route: Route):
        """
        Update the tuples of times of each customer in a given route.
        """
        route.earliest_start_times = route.get_earliest_times()
        route.latest_start_times = route.get_latest_times()
        route.times = list(zip(route.earliest_start_times, route.latest_start_times))

    def can_be_inserted_wang(self, customer, route: Route, mu: int) -> bool:
        """
        Check if the customer can be inserted at the given position in the route. Based on formula (15)
        of Wang et al (2024).
        Parameters:
            customer: int, the customer to be inserted. (c in the paper)
            route: Route, the route where the customer is to be inserted.
            mu: int, the position where the customer is to be inserted (i_mu, i_mu+1).

        """
        # NOTE: should we insert the service time too?
        if mu < 0 or mu >= len(route) - 1:
            # print("Error Trying to insert at the start or end of the route.")
            return False
        i_mu = route.customers_list[mu]
        i_mu_plus_1 = route.customers_list[mu + 1]
        est_mu = route.earliest_start_times[mu]
        tic = self.dataset["edge_weight"][i_mu][customer]
        a_c = self.dataset["time_window"][customer][0]

        lst_mu_plus_1 = route.latest_start_times[mu + 1]
        tci = self.dataset["edge_weight"][customer][i_mu_plus_1]
        b_c = self.dataset["time_window"][customer][1]
        # NOTE: service time?

        if max(est_mu + tic, a_c) <= min(lst_mu_plus_1 - tci, b_c):
            return True
        else:
            return False

    def generate_twc_matrix(self, time_windows: list, distances: list) -> list:
        """
        Generate the time window constraints matrix.
        """
        twc = np.zeros_like(distances)
        for i in range(distances.shape[0]):
            for j in range(distances.shape[0]):
                if i != j:
                    twc[i][j] = time_window_compatibility(
                        distances[i, j], time_windows[i], time_windows[j]
                    )
                else:
                    twc[i][j] = -np.inf
        return twc

    def print_state_dimensions(self):
        print(f"Number of routes: {len(self.routes)}")
        print(
            f"Length of routes: {[len(route.customers_list) for route in self.routes]}"
        )
        print(f"Dimensions of times: {[len(route) for route in self.times]}")
        print(f"Number of unassigned customers: {len(self.unassigned)}")
        print(
            f"Number of customers in routes: {sum(len(route.customers_list) for route in self.routes)}"
        )

    def get_dmax(self):
        """
        Get the maximum distance between any two customers.
        """
        return np.max(self.dataset["edge_weight"])

    def get_qmax(self):
        """
        Get the maximum demand of any customer.
        """
        return np.max(self.dataset["demand"])


# NOTE: maybe add time influence on cost of solution ?
# def route_cost(route):
#     distances = dataset["edge_weight"]
#     tour = [0] + route.customers_list + [0]

#     return sum(distances[tour[idx]][tour[idx + 1]] for idx in range(len(tour) - 1))


def time_window_compatibility(tij, twi: tuple, twj: tuple):
    """
    Time Window Compatibility (TWC) between a pair of vertices i and j. Based on
    'An adaptive large neighborhood search for the multi-depot dynamic vehicle
    routing problem with time windows' by Wang et al. (2024)
    """
    (ai, bi) = twi
    (aj, bj) = twj

    if bj > ai + tij:
        return -np.inf  # Incompatible time windows
    else:
        return min([bi + tij, bj]) - max([ai + tij, aj])
