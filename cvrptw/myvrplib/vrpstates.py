from cvrptw.myvrplib.data_module import (
    data, 
    generate_dynamic_df, 
    dynamic_df_from_dict, 
    cost_matrix_from_coords,
    create_depots_dict
)
from cvrptw.myvrplib.myvrplib import END_OF_DAY, UNASSIGNED_PENALTY
from cvrptw.myvrplib.route import Route
import numpy as np
import pandas as pd


class CvrptwState:
    """
    Class representing the state of the CVRPTW problem.
    Attributes:
        routes: list
            List of routes in the state.
        dataset: dict
            Dictionary containing the dataset.
        unassigned: list
            List of unassigned customers.
        nodes_df: pd.DataFrame
            DataFrame containing the customers data.
        seed: int
            Seed for the random number generator.
        distances: np.ndarray
            Matrix of distances between each pair of customers.
        twc: np.ndarray
            Time window compatibility matrix.
        qmax: float
            Maximum demand of any customer.
        dmax: float
            Maximum distance between any two customers.
        norm_tw: np.ndarray
            Normalized time window compatibility matrix.
        n_vehicles: int
            Number of vehicles in the dataset.
        depots: dict
            Dictionary containing the depots information
        n_customers: int
            Number of customers in the dataset.
        vehicle_capacity: int
            Capacity of the vehicles in the dataset.
        current_time: int
            Current time of the simulation.
    """

    def __init__(
        self,
        routes: list[Route] = None,
        dataset: dict = data,
        unassigned: list = None,
        nodes_df: pd.DataFrame = None,
        current_time: int = 0,
        seed: int = 0,
    ):

        self.dataset = dataset
        self.seed = seed
        self.routes = routes if routes is not None else []

        if nodes_df is not None:
            self.nodes_df = nodes_df
        else:
            self.nodes_df = dynamic_df_from_dict(data, seed=seed)
        self.unassigned = unassigned if unassigned is not None else []
        # Initialize distances matrix
        full_coordinates = self.nodes_df[["x", "y"]].values
        for depot in dataset["depots"]:
            full_coordinates = np.append(
                full_coordinates, [dataset["node_coord"][depot]], axis=0
            )
        self.distances = cost_matrix_from_coords(coords=full_coordinates)
        # Initialize time window compatibility matrix
        full_times = self.nodes_df[["start_time", "end_time"]].values
        for depot in dataset["depots"]:
            full_times = np.append(full_times, [[0, END_OF_DAY]], axis=0)
        full_times = full_times.tolist()
        self.twc = self.generate_twc_matrix(
            full_times,
            self.distances,
        )
        self.current_time = current_time

        self.qmax = self.get_qmax()
        self.dmax = self.get_dmax()
        self.norm_tw = (
            self.twc / self.dmax
        )  # Note: maybe use only norm_tw in the future?
        self.n_vehicles = dataset["vehicles"]
        self.depots = create_depots_dict(dataset)
        self.n_customers = len(self.nodes_df) -1     # first line is not a customer
        self.vehicle_capacity = dataset["capacity"]

    def __str__(self):
        return f"Routes: {[route.customers_list for route in self.routes]}, \nUnassigned: {self.unassigned}"

    def copy(self):
        return CvrptwState(
            [route.copy() for route in self.routes],  # Deep copy each Route
            self.dataset.copy(),
            self.unassigned.copy(),
            self.nodes_df.copy(deep=True),
            self.current_time,
            seed=self.seed
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
                    The customer ID to find.
            Returns:
                tuple
                    The route that contains the customer and its index.
        """
        assert customer >= 0, f"Customer ID must be non-negative, got {customer}."
        assert customer > len(self.nodes_df), f"Customer ID must be less than the number of customers, got {customer}."

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
        assert route is not None, "Route must be provided."
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

    def generate_twc_matrix(self, time_windows: list, distances: np.ndarray, cordeau: bool = True) -> list:
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

    def get_dmax(self):
        """
        Get the maximum distance between any two customers.
        """
        # return np.max(self.dataset["edge_weight"])
        return np.max(self.distances)

    def get_qmax(self):
        """
        Get the maximum demand of any customer.
        """
        return self.nodes_df["demand"].max()

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
