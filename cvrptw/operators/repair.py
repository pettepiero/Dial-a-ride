from cvrptw.myvrplib.data_module import data
import numpy as np
from cvrptw.myvrplib.myvrplib import (
    END_OF_DAY,
    time_window_check,
    route_time_window_check,
    LOGGING_LEVEL,
)
from cvrptw.myvrplib.vrpstates import CvrptwState
import logging
from cvrptw.myvrplib.route import Route
from copy import deepcopy


logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)

# def greedy_repair(state: CvrptwState, rng: np.random) -> CvrptwState:
#     """
#     Inserts the unassigned customers in the best route. If there are no
#     feasible insertions, then a new route is created. Only checks capacity constraints.
#         Parameters:
#             state: CvrptwState
#                 The current solution state.
#             rng: np.random
#                 The random number generator.
#         Returns:
#             CvrptwState
#                 The repaired solution state.
#     """
#     new_state = state.copy()
#     rng.shuffle(new_state.unassigned)

#     while len(new_state.unassigned) != 0:
#         customer = new_state.unassigned.pop()
#         route_idx, idx = best_insert(customer, new_state)

#         if route_idx is not None:
#             new_state.routes[route_idx].insert(idx, customer)
#             new_state.update_est_lst(route_idx)
#             new_state.calculate_planned_times(route_idx)
#             new_state.routes_cost[route_idx] = new_state.route_cost_calculator(route_idx)
#         # If possible, create a new route
#         elif len(new_state.routes) < state.n_vehicles:
#             vehicle_number = len(new_state.routes)
#             new_state.routes.append(
#                     Route(
#                         [
#                             state.depots["vehicle_to_depot"],
#                             customer,
#                             state.depots["vehicle_to_depot"],
#                         ]
#                     )
#             )
#             # append to cost vector new cost
#             new_state.routes_cost.append(new_state.route_cost_calculator(len(new_state.routes) - 1))
#         # debug
#     logger.debug("At the end of greedy repair:")
#     [
#         logger.debug(f"Route {idx}: {route.planned_windows}")
#         for idx, route in enumerate(new_state.routes)
#     ]
#     new_state.update_unassigned_list()

#     return new_state


def greedy_repair_tw(state: CvrptwState, rng: np.random) -> CvrptwState:
    """
    Inserts the unassigned customers in the best route. If there are no
    feasible insertions, then a new route is created.
    Checks capacity and time window constraints.

    Parameters:
        state: CvrptwState
            The current solution state.
        rng: np.random
            The random number generator.
    Returns:
        CvrptwState
            The repaired solution state.
    """
    new_state = state.copy()

    logger.debug(f"In greedy_repair_tw, before loop unassigned = {sorted(new_state.unassigned)}")
    logger.debug(f"Routes =")
    for idx, route in enumerate(new_state.routes):
        logger.debug(f"Route {idx}: {route.nodes_list}")

    counter = 0
    n_unassigned = len(new_state.unassigned)
    rng.shuffle(new_state.unassigned)

    while counter < n_unassigned:   # At most loop over n_unassigned customers
        counter += 1    
        customer = new_state.unassigned[-1]
        pickup_node, delivery_node = new_state.cust_to_nodes[customer]
        # try placing the start node
        route_idx, pickup_idx = best_insert_tw(pickup_node, new_state)

        if route_idx is not None:
            # remove the end node from solution if present
            if delivery_node in new_state.routes[route_idx].nodes_list:
                new_state.routes[route_idx].remove([delivery_node])
                new_state.update_times_attributes_routes(route_idx)
                new_state.routes_cost[route_idx] = new_state.route_cost_calculator(route_idx)

            # insert the start node
            new_state.insert_node_in_route_at_idx(pickup_node, route_idx, pickup_idx)

            # check if the end node can be inserted in the same route but after pickup
            cost, idx = best_insert_given_route(
                node=delivery_node, 
                route_idx=route_idx,
                start_idx=pickup_idx + 1,
                end_idx=len(new_state.routes[route_idx].nodes_list) - 1,
                state=new_state)

            if cost is not None and cost != np.inf:
                new_state.insert_node_in_route_at_idx(delivery_node, route_idx, idx)
            else:
                new_state.routes[route_idx].remove([pickup_node])
                new_state.update_times_attributes_routes(route_idx)
                new_state.routes_cost[route_idx] = new_state.route_cost_calculator(route_idx)
                new_state.cust_df.loc[customer, "route"] = None
                new_state.compute_route_demand(route_idx)

            # done = False
            # for idx in range(pickup_idx, len(new_state.routes[route_idx].nodes_list) - 1):
            #     if can_insert_tw(delivery_node, route_idx, idx, new_state):
            #         # insert the end node
            #         new_state.insert_node_in_route_at_idx(delivery_node, route_idx, idx)
            #         done = True
            #         break
            # if not done: # remove the start node from new_state
            #     new_state.routes[route_idx].remove([pickup_node])
            #     new_state.update_times_attributes_routes(route_idx)
            #     new_state.routes_cost[route_idx] = new_state.route_cost_calculator(route_idx)
            #     new_state.compute_route_demand(route_idx)
            #     new_state.cust_df.loc[customer, "route"] = None

        # Check if the number of routes is less than the number of vehicles
        elif len(new_state.routes) < new_state.n_vehicles:
            # Initialize a new route and corresponding timings
            depot = new_state.depots["vehicle_to_depot"][len(new_state.routes)]
            depot = state.cust_to_nodes[depot][0]
            new_state.routes.append(
                Route(
                    [depot, pickup_node, delivery_node, depot],
                    vehicle=len(new_state.routes),
                    planned_windows=deepcopy(new_state.routes[-1].planned_windows.append([0, END_OF_DAY]))
                    )
            )
            new_state.cust_df.loc[customer, "route"] = len(new_state.routes) - 1
            # append to cost vector new cost
            new_state.routes_cost = np.append(
                new_state.routes_cost,
                new_state.route_cost_calculator(len(new_state.routes) - 1),
            )
            new_state.update_times_attributes_routes(len(new_state.routes) - 1)
        else:
            if customer not in new_state.unassigned:
                new_state.unassigned.insert(0, customer)

    new_state.update_unassigned_list()
    return new_state


# def best_insert(customer: int, state: CvrptwState) -> tuple:
#     """
#     Finds the best feasible route and insertion idx for the customer.
#     Return (None, None) if no feasible route insertions are found.
#     Only checks capacity constraints.
#         Parameters:
#             customer: int
#                 The customer to be inserted.
#             state: CvrptwState
#                 The current solution state.
#         Returns:
#             tuple
#                 The best route and insertion indices for the customer.
#     """
#     best_cost, best_route_idx, best_idx = None, None, None

#     for route_number, route in enumerate(state.routes):
#         for idx in range(1, len(route)-1):
#             if can_insert(customer, route_number, idx, state):
#                 cost = insert_cost(customer, route.nodes_list, idx, state)

#                 if best_cost is None or cost < best_cost:
#                     best_cost, best_route_idx, best_idx = cost, route_number, idx

#     return best_route_idx, best_idx


def best_insert_given_route(
        node: int, 
        route_idx: int, 
        start_idx: int, 
        end_idx: int, 
        state: CvrptwState
        ) -> tuple:
    """
    Finds the best feasible insertion idx for the given node in the given route.
    Starts the search from start_idx and ends at end_idx. Return None if no feasible 
    route insertions are found, otherwise return the best insertion idx.

    Parameters:
        node: int
            The node to be inserted.
        route_idx: int
            The route number.
        start_idx: int
            The start index.
        end_idx: int
            The end index.
        state: CvrptwState
            The current solution state.
    """
    route = state.routes[route_idx]
    best_cost = np.inf
    best_idx = None
    for idx in range(start_idx, end_idx):
        if can_insert_tw(node, route_idx, idx, state):
            cost = insert_cost(node, route.nodes_list, idx, state)

            if cost < best_cost:
                best_cost, best_idx = cost, idx

    return best_cost, best_idx

def best_insert_tw(node: int, state: CvrptwState) -> tuple:
    """
    Finds the best feasible route and insertion idx for the given node.
    Return (None, None) if no feasible route insertions are found.
    Checks both capacity and time window constraints.

    Parameters:
        node: int
            The node to be inserted.
        state: CvrptwState
            The current solution state.
    Returns:
        tuple
            The best route and insertion indices for the node.
    """
    best_cost, best_route_idx, best_idx = np.inf, None, None

    for route_number, route in enumerate(state.routes):
        cost, idx = best_insert_given_route(
            node=node, 
            route_idx=route_number, 
            start_idx=1, 
            end_idx=len(route)-1, 
            state=state
        )
        if cost is not None and cost < best_cost:
                best_cost, best_route_idx, best_idx = cost, route_number, idx

    return best_route_idx, best_idx


# NOTE: I think performance can be improved by changing this function
# maybe insert total demand in route
def can_insert_tw(
    node: int, route_number: int, idx: int, state: CvrptwState
) -> bool:
    """
    Checks if inserting node in route 'route_number' at position 'idx'
    does not exceed vehicle capacity and time window constraints.
        
    Parameters:
        node: int
            The node to be inserted.
        route_number: int
            The route number.
        idx: int
            The insertion index.
        state: CvrptwState
            The current solution state.
    Returns:
        bool
            True if the insertion is feasible, False otherwise.
    """
    df = state.twc_format_nodes_df
    route = state.routes[route_number]

    # Capacity check
    if route.demand is not None:
        total = route.demand + df.loc[node, "demand"]
    else:
        sub_df = df[df["node_id"].isin(route.nodes_list)]["demand"] # demand of all customers in route
        total = sub_df.sum() + df.loc[node, "demand"].item()
    if total > state.vehicle_capacity:
        return False

    previous_customer = route.nodes_list[idx - 1]

    # Time window check
    if time_window_check(
        prev_node_time=route.planned_windows[idx - 1][0],
        prev_service_time=df.loc[previous_customer, "service_time"].item(),
        edge_time=state.distances[previous_customer][node],
        candidate_end_time=df.loc[node, "end_time"].item()):
        return route_time_window_check(state, route, idx)
    return False

def insert_cost(node: int, route: list, idx: int, state: CvrptwState) -> float:
    """
    Computes the insertion cost for inserting node in route at idx.
        
    Parameters:
        node: int
            The node to be inserted.
        route: list
            The route where the node is to be inserted.
        idx: int
            The insertion index.
        state: CvrptwState
            The current solution state.
    Returns:
        float
            The insertion cost.
    """
    dist = state.distances
    pred = 0 if idx == 0 else route[idx - 1]
    succ = 0 if idx == len(route) else route[idx]

    # Increase in cost of adding node, minus cost of removing old edge
    return dist[pred][node] + dist[node][succ] - dist[pred][succ]
