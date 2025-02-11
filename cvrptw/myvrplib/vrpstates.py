from cvrptw.myvrplib.data_module import (
    data,
    generate_dynamic_df,
    dynamic_df_from_dict,
    cost_matrix_from_coords,
    create_depots_dict,
    dynamic_extended_df,
    generate_twc_matrix,
    create_cust_nodes_mapping
)
from cvrptw.myvrplib.myvrplib import END_OF_DAY, UNASSIGNED_PENALTY, LOGGING_LEVEL
from cvrptw.myvrplib.route import Route
import numpy as np
import pandas as pd
import logging
from typing import Union

logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)


class CvrptwState:
    """
    Class representing the state of the CVRPTW problem.
    Attributes:
        routes: list
            List of routes in the state.
        routes_cost: list
            List of costs of each route.
        dataset: dict or pd.DataFrame
            Dictionary or pd.DataFrame containing the dataset.
        cust_to_nodes: dict
            Dictionary mapping customer IDs to node IDs.
        unassigned: list
            List of unassigned customers' IDs (note node IDs). 
            To get node IDs use mapping cust_to_nodes.
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
        n_vehicles: int,
        vehicle_capacity: int,
        routes: list[Route] = None,
        routes_cost: list = None,
        dataset: Union[dict, pd.DataFrame] = data,
        given_unassigned: list = None,
        distances: np.ndarray = None,
        current_time: int = 0,
        seed: int = 0,
    ):

        if isinstance(dataset, dict):
            self.dataset = dataset
            self.seed = seed
            self.nodes_df = dynamic_df_from_dict(dataset, seed=seed)
        elif isinstance(dataset, pd.DataFrame):
            self.nodes_df = dataset
            self.seed = seed
            self.depots = dataset.loc[dataset["service_time"] == 0, "id"].tolist()
        else:
            raise ValueError("Dataset must be a dictionary or a DataFrame.")
        
        self.routes = routes if routes is not None else []
        # Update self.nodes_df with the routes
        for idx, route in enumerate(self.routes):
            for customer in route.customers_list:
                self.nodes_df.loc[self.nodes_df["id"] == customer, "route"] = int(idx)

        # # Initialize distances matrix

        # Change data format for twc and distances computation
        self.dataset = self.nodes_df
        self.twc_format_nodes_df = dynamic_extended_df(self.nodes_df)
        self.cust_to_nodes = create_cust_nodes_mapping(self.twc_format_nodes_df)
        if distances is not None:
            self.distances = distances
        else:
            self.distances = cost_matrix_from_coords(
                coords=self.twc_format_nodes_df[["x", "y"]].values,
                cordeau=False
            )

        if routes_cost is not None:
            self.routes_cost = routes_cost
            logger.debug(f"Passed len(routes_cost): {len(self.routes_cost)}")
            logger.debug(f"passed routes_cost: {self.routes_cost}")
        else:
            self.routes_cost = np.array([self.route_cost_calculator(idx) for idx in range(len(self.routes))])
            logger.debug(f"Calculated len(routes_cost): {len(self.routes_cost)}")

        # self.routes_cost = routes_cost if routes_cost is not None else [self.route_cost_calculator(idx) for idx in range(len(self.routes))]
        # logger.debug(f"len(routes_cost): {len(self.routes_cost)}")
        # logger.debug(f"len(self.routes) = {len(self.routes)}")

        assert len(self.routes) == len(self.routes_cost), "Routes and routes_cost must have the same length."
        self.n_vehicles = n_vehicles

        self.depots = create_depots_dict(self.nodes_df, n_vehicles)
        if given_unassigned is not None:
            self.unassigned = given_unassigned
        else:
            unassigned = self.nodes_df[["id", "route"]]
            unassigned = unassigned.loc[pd.isna(unassigned["route"]), "id"].tolist()
            # filter out depots
            unassigned = [customer for customer in unassigned if customer not in self.depots["depots_indices"]]      

            self.unassigned = unassigned

        # Initialize time window compatibility matrix
        self.twc = generate_twc_matrix(
            self.twc_format_nodes_df[["start_time", "end_time"]].values.tolist(),
            self.distances,
            cordeau=False, #node indices start from 0 (cust indices start from 1 because of cordeau)
        )

        del self.twc_format_nodes_df

        self.current_time = current_time

        self.qmax = self.get_qmax()
        self.dmax = self.get_dmax()
        self.norm_tw = (
            self.twc / self.dmax
        )  # Note: maybe use only norm_tw in the future?

        self.n_vehicles = n_vehicles
        self.n_customers = self.nodes_df.loc[
            self.nodes_df["demand"] != 0
        ].count() 
        self.vehicle_capacity = vehicle_capacity

    def __str__(self):
        return  f"Planned routes: {[route.customers_list for route in self.routes]}, \
                \nUnassigned customer IDs:  {self.unassigned}"

    def copy(self):
        return CvrptwState(
            [route.copy() for route in self.routes],  # Deep copy each Route
            self.routes_cost.copy(),
            self.dataset.copy(),
            self.unassigned.copy(),
            self.distances.copy(),
            self.nodes_df.copy(deep=True),
            self.current_time,
            seed=self.seed
        )

    def route_cost_calculator(self, route_id: int):
        """
        Compute the cost of a route.
        """
        route = self.routes[route_id].customers_list
        cost = 0
        picked_up_customers = []
        for idx, customer in enumerate(route[:-1]):
            if customer not in picked_up_customers:
                start_node = self.cust_to_nodes[customer][0]
                picked_up_customers.append(customer)
            else:
                start_node = self.cust_to_nodes[customer][1]
            next_customer = route[idx + 1]
            if next_customer not in picked_up_customers:
                next_node = self.cust_to_nodes[next_customer][0]
                # add next_customer to picked_up_customers only
                # when it is considered the current customer, not now
            else:
                next_node = self.cust_to_nodes[next_customer][1]

            cost += self.distances[start_node][next_node]

        return round(cost, 2)

    def objective(self):
        """
        Computes the total route costs.
        """
        assert len(self.routes) == len(self.routes_cost), "Routes and routes_cost must have the same length."
        unassigned_penalty = UNASSIGNED_PENALTY * len(self.unassigned)
        return sum([self.routes_cost[idx] for idx in range(len(self.routes))]) + unassigned_penalty

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
        assert (
            customer < self.nodes_df.shape[0]
        ), f"Customer ID must be less than {self.nodes_df.shape[0]}, got {customer}."

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
        Return the tuple containing indices of the customer in the route.
        (pick up index, delivery index)
        """
        assert route is not None, "Route must be provided."
        if customer in route.customers_list:
            return [i for i, x in enumerate(route.customers_list) if x == customer]

        raise ValueError(f"Given route does not contain customer {customer}.")

    def update_times_attributes_routes(self, route_index: int):
        """
        Update the start, end and planned times for each customer in the routes.
        """        
        self.update_est_lst(route_index)
        # TODO udpate planned windows
        self.calculate_planned_times(route_index)

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
        served_customers = self.served_customers()
        
        return len(served_customers)

    def served_customers(self):
        """
        Return the list of served customers.
        """
        served_customers = set()
        for route in self.routes:
            for customer in route.customers_list[1:-1]:
                served_customers.add(customer)
        
        return list(served_customers)

        # TODO: Test this method

    def update_est_lst(self, route_index: int):
        """
        Calculates vectors of the earliest (EST) and latest (LST) start times for each NODE in the route.
        Based on equations (3b) and (13) of Wang et al. (2024).
        Parameters:
            route_index: int
                Index of the route.
        Returns:
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

    def update_unassigned_list(self):
        """
        Update the list of unassigned customers.
        """
        self.unassigned = self.nodes_df.loc[pd.isna(self.nodes_df["route"]), "id"].tolist()
        # filter out depots
        self.unassigned = [customer for customer in self.unassigned if customer not in self.depots["depots_indices"]]

    def compute_route_demand(self, route_idx: int) -> None:
        """
        Compute the demand of a route and store in the route object.
        Parameters:
            route_idx: int
                Index of the route.
        Returns:
            None
        """
        demand = 0
        for customer in self.routes[route_idx].customers_list:
            demand += self.nodes_df.loc[customer, "demand"].item()
        self.routes[route_idx].demand = demand
