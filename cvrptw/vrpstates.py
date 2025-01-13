from data_module import data
from myvrplib import END_OF_DAY, UNASSIGNED_PENALTY
from route import Route

import numpy as np


class CvrptwState:
    """
    UPDATE DESCRIPTION:
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
        dataset: dict = data,
        unassigned: list = None,
    ):
        self.routes = routes
        # self.times = times  # planned arrival times for each customer
        self.dataset = dataset
        self.unassigned = unassigned if unassigned is not None else []

        self.twc = self.generate_twc_matrix(dataset["time_window"], dataset["edge_weight"])
        self.qmax = self.get_qmax()
        self.dmax = self.get_dmax()
        self.norm_tw = (
            self.twc / self.dmax
        )  # Note: maybe use only norm_tw in the future?

    def __str__(self):
        return f"Routes: {[route.customers_list for route in self.routes]}, \nUnassigned: {self.unassigned}, \nTimes: {self.times}"

    def copy(self):
        return CvrptwState(
            [route.copy() for route in self.routes],  # Deep copy each Route
            self.dataset.copy(),
            self.unassigned.copy(),
        )

    def objective(self):
        """
        Computes the total route costs.
        """
        unassigned_penalty = UNASSIGNED_PENALTY * len(self.unassigned)
        return sum(route.cost for route in self.routes) + unassigned_penalty

    @property
    def cost(self):
        """
        Alias for objective method. Used for plotting.
        """
        return self.objective()

    def find_route(self, customer: int) -> tuple:
        """
        Return the route that contains the passed-in customer.
            Parameters:
                customer: int
                    The customer to find.
            Returns:
                tuple
                    The route that contains the customer and its index.
        """
        found = False
        for idx, route in enumerate(self.routes):
            if customer in route.customers_list:
                found = True
                return route, idx
        if not found:
            # raise ValueError(f"Customer {customer} not found in any route.")
            print(f"Customer {customer} not found in any route.")

    def find_index_in_route(self, customer, route: Route):
        """
        Return the index of the customer in the route.
        """
        if customer in route.customers_list:
            return route.customers_list.index(customer)

        raise ValueError(f"Given route does not contain customer {customer}.")

    def update_times_attributes_routes(self):
        """
        Update the start, end and planned times for each customer in the routes.
        """        
        for route in self.routes:
            est = route.get_earliest_times()
            lst = route.get_latest_times()
            route.start_times = list(zip(est, lst))
            # TODO udpate planned windows
            route.calculate_planned_times()

    def generate_twc_matrix(self, time_windows: list, distances: list, cordeau: bool = True) -> list:
        """
        Generate the time window compatability matrix matrix. If cordeau is True,
        the first row and column are set to -inf, as customer 0 is not considered
        in the matrix.
            Parameters:
                time_windows: list
                    List of time windows for each customer.
                distances: list
                    List of distances between each pair of customers.
                cordeau: bool
                    If True, the first row and column are set to -inf.
            Returns:
                list
                    Time window compatibility matrix.
        """
        start_idx = 1 if cordeau else 0
        twc = np.zeros_like(distances)
        for i in range(start_idx, distances.shape[0]):
            for j in range(start_idx, distances.shape[0]):
                if i != j:
                    twc[i][j] = time_window_compatibility(
                        distances[i, j], time_windows[i], time_windows[j]
                    )
                else:
                    twc[i][j] = -np.inf
        if cordeau:
            for i in range(distances.shape[0]):
                twc[i][0] = -np.inf
                twc[0][i] = -np.inf
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

    def n_served_customers(self):
        """
        Return the number of served customers.
        """
        return sum(len(route.customers_list[1:-1]) for route in self.routes)

    def served_customers(self):
        """
        Return the list of served customers.
        """
        return [customer for route in self.routes for customer in route.customers_list[1:-1]]


# NOTE: maybe add time influence on cost of solution ?
# def route_cost(route):
#     distances = dataset["edge_weight"]
#     tour = [0] + route.customers_list + [0]

#     return sum(distances[tour[idx]][tour[idx + 1]] for idx in range(len(tour) - 1))


def time_window_compatibility(tij: float, twi: tuple, twj: tuple) -> float:
    """
    Time Window Compatibility (TWC) between a pair of vertices i and j. Based on
    Wang et al. (2024). Returns the time window compatibility between two customers
    i and j, given their time windows and the travel time between them.
        Parameters:
            tij: float
                Travel time between the two customers.
            twi: tuple
                Time window of customer i.
            twj: tuple
                Time window of customer j.
        Returns:
            float
                Time window compatibility between the two customers.
    """
    (ai, bi) = twi
    (aj, bj) = twj

    if bj > ai + tij:
        return -np.inf  # Incompatible time windows
    else:
        return round(min([bi + tij, bj]) - max([ai + tij, aj]), 2)
