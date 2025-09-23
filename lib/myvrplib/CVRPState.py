from lib.myvrplib.data_module import (
    generate_dynamic_df, 
    dynamic_df_from_dict, 
    cost_matrix_from_coords,
    create_depots_dict
)
from lib.myvrplib.myvrplib import UNASSIGNED_PENALTY, LOGGING_LEVEL
from lib.myvrplib.route import Route
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)

class CVRPState:
    """
    Class representing a generic CVRP problem state.
    
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
    #twc: np.ndarray
    #    Time window compatibility matrix.
    qmax: float
        Maximum demand of any customer.
    dmax: float
        Maximum distance between any two customers.
    #norm_tw: np.ndarray
    #    Normalized time window compatibility matrix.
    n_vehicles: int
        Number of vehicles in the dataset.
    depots: dict
        Dictionary containing the depots information
    n_customers: int
        Number of customers in the dataset.
    vehicle_capacity: int
        Capacity of the vehicles in the dataset.
    #current_time: int
    #    Current time of the simulation.
    """
    def __init__(
        self,
        dataset: dict,
        routes: list[Route] = None,
        routes_cost: list = None,
        given_unassigned: list = None,
        distances: np.ndarray = None,
        nodes_df: pd.DataFrame = None,
        seed: int = 0,
    ):

        self.dataset = dataset
        self.seed = seed
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        self.routes = routes if routes is not None else []
        if nodes_df is not None:
            self.nodes_df = nodes_df
        else:
            self.nodes_df = dynamic_df_from_dict(dataset, seed=seed)
        # Initialize distances matrix

        full_coordinates = self.nodes_df[["x", "y"]].values

        if distances is not None:
            self.distances = distances
        else:
            self.distances = cost_matrix_from_coords(coords=full_coordinates)

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

        self.depots = create_depots_dict(dataset)
        if given_unassigned is not None:
            self.unassigned = given_unassigned
        else:
            unassigned = self.nodes_df[["id", "route"]]
            unassigned = unassigned.loc[pd.isna(unassigned["route"]), "id"].tolist()
            # filter out depots
            unassigned = [customer for customer in unassigned if customer not in self.depots["depots_indices"]]      

            self.unassigned = unassigned

        self.qmax = self.get_qmax()
        self.dmax = self.get_dmax()
        self.n_vehicles = dataset["vehicles"]
        self.n_customers = len(self.nodes_df) -1     # first line is not a customer
        self.n_planned_customers = self.n_served_customers() 
        self.vehicle_capacity = dataset["capacity"]

    def __str__(self):
        return f"Routes: {[route.customers_list for route in self.routes]}, \nUnassigned: {self.unassigned}"

    def copy(self):
        return CVRPState(
            dataset             = self.dataset.copy(),
            routes              = [route.copy() for route in self.routes],  # Deep copy each Route
            routes_cost         = self.routes_cost.copy(),
            given_unassigned    = self.unassigned.copy(),
            distances           = self.distances.copy(),
            nodes_df            = self.nodes_df.copy(deep=True),
            seed                = self.seed
        )

    def route_cost_calculator(self, route_id: int):
        """
        Compute the cost of a route.
        """
        route = self.routes[route_id].customers_list
        cost = 0
        for idx, customer in enumerate(route[:-1]):
            next_customer = route[idx + 1]
            logger.debug(f"Route: {route}")
            cost += self.distances[customer][next_customer]

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
        Return the index of the customer in the route.
        """
        assert route is not None, "Route must be provided."
        if customer in route.customers_list:
            return route.customers_list.index(customer)

        raise ValueError(f"Given route does not contain customer {customer}.")

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

        # TODO: Test this method

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

        Parameters
        ----------
        route_idx: int
            Index of the route.

        Returns
        -------
        None
        """
        demand = 0
        for customer in self.routes[route_idx].customers_list:
            demand += self.nodes_df.loc[customer, "demand"].item()
        self.routes[route_idx].demand = demand

    def update_attributes(self):
        self.update_unassigned_list()
        self.n_planned_customers = self.n_served_customers() 


