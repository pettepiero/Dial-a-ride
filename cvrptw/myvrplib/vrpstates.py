from cvrptw.myvrplib.data_module import (
    # data,
    generate_dynamic_df,
    dynamic_df_from_dict,
    cost_matrix_from_coords,
    create_depots_dict,
    dynamic_extended_df,
    generate_twc_matrix,
    create_cust_stops_mapping,
    create_nodes_cust_mapping,
    create_cust_nodes_mapping,
)
from cvrptw.output.analyze_solution import verify_time_windows
from cvrptw.myvrplib.myvrplib import END_OF_DAY, UNASSIGNED_PENALTY, LOGGING_LEVEL, LATE_PENALTY, EARLY_PENALTY
from cvrptw.myvrplib.route import Route
import numpy as np
import pandas as pd
import logging
from typing import Union
from cvrptw.maps.maps import setup, get_dict_of_stops
import networkx as nx

logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)

DEPOT_ID = 3404


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
    twc_format_nodes_df: pd.DataFrame
        DataFrame containing the dataset in the format used 
        for time window compatibility matrix computation.
    cust_to_nodes: dict
        Dictionary mapping customer IDs to node IDs.
    unassigned: list
        List of unassigned customers' IDs (NOT node IDs). 
        To get node IDs use mapping cust_to_nodes.
    cust_df: pd.DataFrame
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
#        n_vehicles: int,
#        vehicle_capacity: int,
        dataset: Union[dict, pd.DataFrame, str],
        list_of_depot_stops: list = [DEPOT_ID],
        routes: list[Route] = None,
        routes_cost: list = None,
        given_unassigned: list = None,
        distances: np.ndarray = None,
        map_file: str = "./data/DataSetActvAut(Fermate).csv",
        current_time: int = 0,
        seed: int = 0,
        show_map: bool = False,
    ):
        # Set up map information
        graph, stops_df, segments_df = setup(
                list_of_depots_stops=list_of_depot_stops, 
                stops_data=map_file,
                show_map=show_map
        )
        self.predecessors, shortest_paths = nx.floyd_warshall_predecessor_and_distance(
            graph, weight="weight"
        )

        self.stop_id_to_node = get_dict_of_stops(stops_df, segments_df)
        self.nodes_to_stop = {v: k for k, v in self.stop_id_to_node.items()}

        stop_nodes = [node_id for node_id in stops_df["node_id"]]

        # # Initialize distances matrix
        # Convert to a fast lookup dictionary
        self.distances = {
            (src, dst): shortest_paths[src][dst]
            for src in stop_nodes
            for dst in stop_nodes
        }

        # Set up requests information
        if isinstance(dataset, dict):
            print("from dict")
            self.dataset = dataset
            self.seed = seed
            self.cust_df = dynamic_df_from_dict(dataset, seed=seed)
            self.n_vehicles = dataset["vehicles"]
            self.vehicle_capacity = dataset["capacity"] 
        elif isinstance(dataset, pd.DataFrame):
            raise ValueError(f"ERROR: not implemented yet -> init from dataframe")
            print("From dataframe")
            self.cust_df = dataset
            self.seed = seed
            self.depots = dataset.loc[dataset["service_time"] == 0, "cust_id"].tolist()
        else:
            raise ValueError(f"Dataset must be a dictionary or a DataFrame. Passed: {type(dataset)}")

        self.routes = routes if routes is not None else []
        # Update self.cust_df with the routes
        self.cust_df["route"] = None
        self.nodes_to_cust = create_nodes_cust_mapping(self.cust_df)

        # TODO: check if this is actually doing anything
        for idx, route in enumerate(self.routes):
            for node in route.nodes_list:
                customer_id = self.nodes_to_cust[node]
                self.cust_df.loc[self.cust_df["cust_id"] == customer_id, "route"] = int(idx)

        # Change data format for twc and distances computation
        self.dataset = self.cust_df
        # self.twc_format_nodes_df = dynamic_extended_df(self.cust_df)
        self.twc_format_nodes_df = self.cust_df.loc[self.cust_df["demand"] != 0] # simply exclude depots?
        self.cust_to_stops = create_cust_stops_mapping(
            cust_df=self.cust_df, list_of_depot_stops=list_of_depot_stops)
        self.cust_to_nodes = create_cust_nodes_mapping(
            cust_to_stops=self.cust_to_stops, stop_id_to_node=self.stop_id_to_node
        )

        # if distances is not None:
        #     self.distances = distances
        # else:
        #     self.distances = cost_matrix_from_coords(
        #         coords=self.twc_format_nodes_df[["x", "y"]].values,
        #         cordeau=False
        #     )

        if routes_cost is not None:
            self.routes_cost = routes_cost
            logger.debug(f"Passed len(routes_cost): {len(self.routes_cost)}")
            logger.debug(f"passed routes_cost: {self.routes_cost}")
        else:
            self.routes_cost = np.array([self.route_cost_calculator(idx) for idx in range(len(self.routes))])
            logger.debug(f"Calculated len(routes_cost): {len(self.routes_cost)}")
        print(f"cost of routes: {self.routes_cost}")

        # self.routes_cost = routes_cost if routes_cost is not None else [self.route_cost_calculator(idx) for idx in range(len(self.routes))]
        # logger.debug(f"len(routes_cost): {len(self.routes_cost)}")
        # logger.debug(f"len(self.routes) = {len(self.routes)}")

        print(f"DEBUG: self.cust_df = \n{self.cust_df}")

        assert len(self.routes) == len(self.routes_cost), "Routes and routes_cost must have the same length."
        self.depots = create_depots_dict(list_of_depot_stops=list_of_depot_stops, cust_df=self.cust_df, cust_to_nodes=self.cust_to_nodes,num_vehicles=n_vehicles)

        if given_unassigned is not None:
            self.unassigned = given_unassigned
        else:
            unassigned = self.cust_df.loc[self.cust_df["demand"] != 0, ["cust_id", "route"]]
            unassigned = unassigned.loc[pd.isna(unassigned["route"]), "cust_id"].tolist()
            # filter out depots
            unassigned = [customer for customer in unassigned if customer not in self.depots["depots_indices"]]      

            self.unassigned = unassigned

        # Initialize time window compatibility matrix
        self.twc = generate_twc_matrix(
            self.cust_df,
            # self.twc_format_nodes_df[["start_time", "end_time"]].values.tolist(),
            self.distances,
            stop_to_nodes=self.stop_id_to_node,
            cordeau=False, #node indices start from 0 (cust indices start from 1 because of cordeau)
        )

        # del self.twc_format_nodes_df

        self.current_time = current_time

        self.qmax = self.get_qmax()
        self.dmax = self.get_dmax()
        # normalize dict
        self.norm_tw = {
            couple: self.twc[couple] / self.dmax 
            for couple in self.twc.keys()
        }
        # self.norm_tw = (
        #     self.twc / self.dmax
        # )  # Note: maybe use only norm_tw in the future?

        #self.n_vehicles = n_vehicles
        self.n_customers = self.cust_df.loc[
            self.cust_df["demand"] != 0
        ].shape[0]
        #self.vehicle_capacity = vehicle_capacity

    def __str__(self):
        return  f"Planned routes: {[route.nodes_list for route in self.routes]}, \
                \nUnassigned customer IDs:  {self.unassigned}"

    def copy(self):
        return CvrptwState(
            self.n_vehicles,
            self.vehicle_capacity,
            [route.copy() for route in self.routes],  # Deep copy each Route
            self.routes_cost.copy(),
            self.dataset.copy(),
            self.unassigned.copy(),
            self.distances.copy(),
            self.current_time,
            seed=self.seed
        )

    def route_cost_calculator(self, route_id: int):
        """
        Compute the cost of a route.
        """
        route = self.routes[route_id].nodes_list
        cost = 0
        for idx, node in enumerate(route[:-1]):
            next_node = route[idx + 1]
            cost += self.distances[(node, next_node)]
        return round(cost, 2)

        # picked_up_customers = []
        # for idx, customer in enumerate(route[:-1]):
        #     if customer not in picked_up_customers:
        #         start_node = self.cust_to_nodes[customer][0]
        #         picked_up_customers.append(customer)
        #     else:
        #         start_node = self.cust_to_nodes[customer][1]
        #     next_customer = route[idx + 1]
        #     if next_customer not in picked_up_customers:
        #         next_node = self.cust_to_nodes[next_customer][0]
        #         # add next_customer to picked_up_customers only
        #         # when it is considered the current customer, not now
        #     else:
        #         next_node = self.cust_to_nodes[next_customer][1]

        #     cost += self.distances[start_node][next_node]

        # return round(cost, 2)

    def objective(self):
        """
        Computes the total route costs.
        """
        assert len(self.routes) == len(self.routes_cost), "Routes and routes_cost must have the same length."
        unassigned_penalty = UNASSIGNED_PENALTY * len(self.unassigned)
        late_penalty = LATE_PENALTY * sum([route.sum_late for route in self.routes])
        early_penalty = EARLY_PENALTY * sum([route.sum_early for route in self.routes])

        return sum([self.routes_cost[idx] for idx in range(len(self.routes))])  \
            + unassigned_penalty + late_penalty + early_penalty

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
            customer <= self.cust_df.shape[0]
        ), f"Customer ID must be leq than {self.cust_df.shape[0]}, got {customer}."

        start_node, end_node = self.cust_to_nodes[customer]
        found_start = False
        found_end = False
        for idx, route in enumerate(self.routes):
            if start_node in route.nodes_list:
                found_start = True
                if end_node in route.nodes_list:
                    found_end = True
                    return route, idx
                else:
                    raise ValueError(f"Customer {customer} end node not found in \
                                     route {idx} which contains start_node.")

        if not found_start or not found_end:
            # raise ValueError(f"Customer {customer} not found in any route.")
            raise ValueError(f"Customer {customer} start and end nodes not both found in any route.")

    def find_index_in_route(self, node, route: Route):
        """
        Return index of the node in the route.
            Arguments:
                node: int
                    The node to find.
                route: Route
                    The route where to find the node.
        """
        assert route is not None, "Route must be provided."
        if node in route.nodes_list:
            return route.nodes_list.index(node)

        raise ValueError(f"Given route does not contain customer {node}.")

    def update_times_attributes_routes(self, route_index: int):
        """
        Update the start, end and planned times for each customer in the routes.
        """        
        self.update_est_lst(route_index)
        # TODO udpate planned windows
        self.calculate_planned_times(route_index)
        self.routes[route_index].compute_early_sum(self.twc_format_nodes_df)
        self.routes[route_index].compute_late_sum(self.twc_format_nodes_df)

    def get_dmax(self):
        """
        Get the maximum distance between any two customers.
        """
        # return np.max(self.dataset["edge_weight"])
        return max(self.distances.values())

    def get_qmax(self):
        """
        Get the maximum demand of any customer.
        """
        return self.cust_df["demand"].max()

    @property
    def n_planned_customers(self):
        """
        Return the number of served customers.
        """
        return len(self.planned_customers())

    def planned_customers(self):
        """
        Return the list of served customers.
        """
        planned_customers = set()
        for route in self.routes:
            for node in route.nodes_list[1:-1]:
                if self.twc_format_nodes_df.loc[node, "type"] == "pickup":
                    cust = self.nodes_to_cust[node]
                    end_node = self.cust_to_nodes[cust][1]
                    if end_node in route.nodes_list:
                        planned_customers.add(cust)
                elif self.twc_format_nodes_df.loc[node, "type"] == "delivery":
                    cust = self.nodes_to_cust[node]
                    start_node = self.cust_to_nodes[cust][0]
                    if start_node in route.nodes_list:
                        planned_customers.add(cust)
                elif self.twc_format_nodes_df.loc[node, "type"] == "depot":
                    AssertionError ("Depot should not be in the route")

        return list(planned_customers)

    def served_nodes(self):
        """
        Return the list of served nodes.
        """
        served_nodes = set()
        for route in self.routes:
            for node in route.nodes_list[1:-1]:
                served_nodes.add(node)

        return list(served_nodes)

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
        route = self.routes[route_index].nodes_list
        # df = self.cust_df
        df = dynamic_extended_df(self.cust_df)

        # If first element is a depot, then the earliest start time is 0
        if route[0] in self.depots["depots_indices"]:
            est.append(0)
        else:
            print(f"ERROR: first node {route[0]} in route {route_index} is not a depot, which are: {self.depots['depots_indices']}")
            print(f"Passed route: {route}") 
            # TODO: This will have to be changed for dynamic case
            raise AssertionError("First node in route is not a depot")

        # Implementation of formula 3b of Wang et al. (2024)
        for i in range(1, len(route)-1):
            current_node = route[i]
            # current_cust = self.nodes_to_cust[current_node]
            prev_node = route[i-1]
            # prev_cust = self.nodes_to_cust[prev_node]
            time = float(
                round(
                    max(
                        est[i - 1]
                        + df.loc[prev_node, "service_time"].item()
                        + self.distances[current_node][prev_node],
                        df.loc[current_node, "start_time"].item(),
                    ),
                    2,
                )
            )
            est.append(time)

        if len(est) != len(route):
            AssertionError("Error in calculating earliest start times")

        lst = [None] * len(route)
        lst[-1] = END_OF_DAY
        for i in reversed(range(len(route) - 1)):
            next_node = route[i + 1]
            # next_cust = self.nodes_to_cust[next_node]
            current_node = route[i]
            # current_cust = self.nodes_to_cust[current_node]
            time = round(
                min(
                    lst[i + 1]
                    - df.loc[current_node, "service_time"].item()
                    - self.distances[current_node][next_node],
                    df.loc[current_node, "end_time"].item(),
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
        df = dynamic_extended_df(self.cust_df)
        tw = []
        first_node = route.nodes_list[1]

        tw.append(
            [
                0,
                float(
                    round(
                        max(
                            0,
                            df.loc[first_node, "start_time"].item()
                            - self.distances[0][first_node]
                        ),
                        2,
                    )
                ),
            ]
        )

        last_departure = tw[0][1]
        last_node = route.nodes_list[0]
        for customer in route.nodes_list[1:]:
            # Planned arrival time at customer idx
            # Planned departure time is the planned arrival time + service time
            arr = last_departure + self.distances[last_node][customer]
            dep = arr + df.loc[customer, "service_time"].item()
            tw.append([float(round(arr, 2)), float(round(dep, 2))])
            last_departure = dep
            last_node = customer
        self.routes[route_index].planned_windows = tw

    def update_unassigned_list(self):
        """
        Update the list of unassigned customers.
        """
        # check that both start and end node are assigned
        for route in self.routes:
            for node in route.nodes_list:
                if self.twc_format_nodes_df.loc[node, "type"] == "pickup":
                    cust = self.nodes_to_cust[node]
                    end_node = self.cust_to_nodes[cust][1]
                    if end_node not in route.nodes_list:
                        if cust not in self.unassigned:
                            self.unassigned.append(cust)
                    else:
                        if cust in self.unassigned:
                            self.unassigned.remove(cust)
                elif self.twc_format_nodes_df.loc[node, "type"] == "delivery":
                    cust = self.nodes_to_cust[node]
                    start_node = self.cust_to_nodes[cust][0]
                    if start_node not in route.nodes_list:
                        if cust not in self.unassigned:
                            self.unassigned.append(cust)
                    else:
                        if cust in self.unassigned:
                            self.unassigned.remove(cust)

        # self.unassigned = self.cust_df.loc[pd.isna(self.cust_df["route"]), "id"].tolist()
        # filter out depots
        depots_ids = [self.nodes_to_cust[cust_id] for cust_id in self.depots["depots_indices"]]
        self.unassigned = [customer for customer in self.unassigned if customer not in depots_ids]

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
        for node in self.routes[route_idx].nodes_list:
            demand += self.twc_format_nodes_df.loc[node, "demand"].item()
        self.routes[route_idx].demand = demand

    def insert_node_in_route_at_idx(self, node: int, route_idx: int, node_idx: int) -> None:
        """
        Insert a node in a given route. This means updating the route, the customer DataFrame, 
        the twc_format_nodes_df, the route cost, the route demand, the planned times and
        the unassigned list if the delivery node is already inserted.
        Parameters:
            node: int
                The node to be inserted.
            route_idx: int
                The index of the route where the node will be inserted.
            node_idx: int
                The index where the node will be inserted.
        Returns:
            None
        """
        assert node >= 0, f"Node must be non-negative, got {node}."
        assert 0 <= route_idx <= len(self.routes), f"Route index must be between 0 and {len(self.routes)}, got {route_idx}."
        customer = self.nodes_to_cust[node] # corresponding customer ID
        assert customer in self.unassigned, f"Customer {customer} (node {node}) is not in unassigned list: \n{self.unassigned}."
        assert 0 < node_idx < len(self.routes[route_idx].nodes_list), f"Node index must be between 0 and {len(self.routes[route_idx].nodes_list)}, got {node_idx}."

        self.routes[route_idx].insert(position=node_idx, node=node)
        self.cust_df.loc[customer, "route"] = route_idx
        self.twc_format_nodes_df.loc[node, "route"] = route_idx
        self.update_times_attributes_routes(route_idx)
        self.routes_cost[route_idx] = self.route_cost_calculator(route_idx)
        self.compute_route_demand(route_idx)
        # if this is the delivery node, remove the customer from the unassigned list
        if self.twc_format_nodes_df.loc[node, "type"] == "delivery":
            self.unassigned.remove(customer)
        else:
            # get delivery node and see if it is in the route
            delivery_node = self.cust_to_nodes[customer][1]
            if delivery_node in self.routes[route_idx].nodes_list:
                idx = self.routes[route_idx].nodes_list.index(delivery_node)
                assert idx > node_idx, "Delivery node must be inserted after the pickup node."
                self.unassigned.remove(customer)
