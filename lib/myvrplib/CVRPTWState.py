from lib.myvrplib.data_module import (
    generate_dynamic_df, 
    dynamic_df_from_dict, 
    cost_matrix_from_coords,
    create_depots_dict
)
from lib.myvrplib.myvrplib import END_OF_DAY, UNASSIGNED_PENALTY, LOGGING_LEVEL
from lib.myvrplib.route import Route
from lib.myvrplib.CVRPState import CVRPState 
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)


class CvrptwState(CVRPState):
    """
    Class representing the state of the CVRPTW problem.

    Attributes
    ----------
    routes: list
        List of routes in the state.
    routes_cost: list
        List of costs of each route.
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
        dataset: dict,
        routes: list[Route] = None,
        routes_cost: list = None,
        given_unassigned: list = None,
        distances: np.ndarray = None,
        nodes_df: pd.DataFrame = None,
        current_time: int = 0,
        seed: int = 0,
    ):
        super().__init__(dataset=dataset, routes=routes, routes_cost=routes_cost,
                given_unassigned=given_unassigned, distances=distances, nodes_df=nodes_df,
                seed=seed)

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

        self.norm_tw = (
            self.twc / self.dmax
        )  # Note: maybe use only norm_tw in the future?

    def copy(self):
        return CvrptwState(
            dataset             = self.dataset.copy(),
            routes              = [route.copy() for route in self.routes],  # Deep copy each Route
            routes_cost         = self.routes_cost.copy(),
            given_unassigned    = self.unassigned.copy(),
            distances           = self.distances.copy(),
            nodes_df            = self.nodes_df.copy(deep=True),
            current_time        = self.current_time,
            seed                = self.seed
        )

    def update_times_attributes_routes(self, route_index: int):
        """
        Update the start, end and planned times for each customer in the routes.
        """        
        self.update_est_lst(route_index)
        # TODO udpate planned windows
        self.calculate_planned_times(route_index)

    def generate_twc_matrix(self, time_windows: list, distances: np.ndarray, cordeau: bool = True) -> list:
        """
        Generate the time window compatability matrix matrix. If cordeau is True,
        the first row and column are set to -inf, as customer 0 is not considered
        in the matrix.

        Parameters
        ----------
        time_windows: list
            List of time windows for each customer.
        distances: list
            List of distances between each pair of customers.
        cordeau: bool
            If True, the first row and column are set to -inf.

        Returns
        -------
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


    def update_est_lst(self, route_index: int):
        """
        Calculates vectors of the earliest (EST) and latest (LST) start times for each customer in the route.
        Based on equations (3b) and (13) of Wang et al. (2024).

        Parameters
        ----------
        route_index: int
            Index of the route.

        Returns
        -------
        None
        """
        est = []
        route = self.routes[route_index].customers_list
        df = self.nodes_df

        # If first element is a depot, then the earliest start time is 0
        if route[0] in self.depots["depots_indices"]:
            est.append(0)
        else:
            print(f"ERROR: first node {route[0]} in route {route_index} is not a depot, which are: {self.depots['depots_indices']}")
            # TODO: This will have to be changed for dynamic case
            raise AssertionError("First node in route is not a depot")

        # Implementation of formula 3b of Wang et al. (2024)
        for i in range(1, len(route)-1):
            current = route[i]
            prev = route[i-1]
            time = float(round(max(
                est[i - 1]
                + df.loc[prev, "service_time"].item()
                + self.distances[current][prev],
                df.loc[current, "start_time"].item(),
            ), 2))
            est.append(time)

        if len(est) != len(route):
            AssertionError("Error in calculating earliest start times")

        lst = [None] * len(route)
        lst[-1] = END_OF_DAY
        for i in reversed(range(len(route) - 1)):
            next = route[i + 1]
            current = route[i]
            time = round(
                min(
                    lst[i + 1]
                    - df.loc[current, "service_time"].item()
                    - self.distances[current][next],
                    df.loc[current, "end_time"].item()
                ),
                2,
            )
            lst[i] = float(time)

        if len(lst) != len(route):
            AssertionError("Error in calculating latest start times")

        self.routes[route_index].start_times = list(zip(est, lst))

    def calculate_planned_times(self, route_index:int):
        """
        Calculate the planned arrival and departure times for each customer in the route.
        """
        route = self.routes[route_index]
        df = self.nodes_df
        tw = []
        first_customer = route.customers_list[1]

        tw.append(
            [
                0,
                float(
                    round(
                        max(
                            0,
                            df.loc[first_customer, "start_time"].item()
                            - self.distances[0][first_customer]
                        ),
                        2,
                    )
                ),
            ]
        )

        last_departure = tw[0][1]
        last_customer = route.customers_list[0]
        for customer in route.customers_list[1:]:
            # Planned arrival time at customer idx
            # Planned departure time is the planned arrival time + service time
            arr = last_departure + self.distances[last_customer][customer]
            dep = arr + df.loc[customer, "service_time"].item()
            tw.append([float(round(arr, 2)), float(round(dep, 2))])
            last_departure = dep
            last_customer = customer
        self.routes[route_index].planned_windows = tw


def time_window_compatibility(tij: float, twi: tuple, twj: tuple) -> float:
    """
    Time Window Compatibility (TWC) between a pair of vertices i and j. Based on eq. (12) of
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
        return round(min([bi + tij, bj]) - max([ai + tij, aj]), 2)
    else:
        return -np.inf  # Incompatible time windows
