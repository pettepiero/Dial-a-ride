import logging
import numpy as np
from cvrptw.myvrplib.data_module import data
from cvrptw.myvrplib.myvrplib import END_OF_DAY, LOGGING_LEVEL
from cvrptw.myvrplib.vrpstates import CvrptwState
from cvrptw.myvrplib.route import Route
from cvrptw.operators.repair import insert_cost
from copy import deepcopy


logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)

def wang_greedy_repair(state: CvrptwState, rng: np.random) -> CvrptwState:
    """
    Inserts the unassigned customers in the best route. If there are no
    feasible insertions, then a new route is created. Uses the Wang et al (2024)
    insertion heuristics with time window compatibility checks.
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

    counter = 0
    n_unassigned = len(new_state.unassigned)
    rng.shuffle(new_state.unassigned)

    while counter < n_unassigned:
        counter += 1
        customer = new_state.unassigned[-1]
        pickup_node, delivery_node = state.cust_to_nodes[customer]
        logger.debug(f"Candidate customer: {customer}")
        route_idx, pickup_idx = wang_best_insert(customer, new_state)

        if route_idx is not None:
            if delivery_node in new_state.routes[route_idx].nodes_list:
                new_state.routes[route_idx].remove([delivery_node])
                new_state.update_times_attributes_routes(route_idx)
                new_state.routes_cost[route_idx] = new_state.route_cost_calculator(
                    route_idx
                )

            # insert the start node
            new_state.insert_node_in_route_at_idx(pickup_node, route_idx, pickup_idx)

            # check if the end node can be inserted in the same route but after pickup
            cost, idx = wang_best_insert_given_route(
                node=delivery_node,
                route_idx=route_idx,
                start_idx=pickup_idx + 1,
                end_idx=len(new_state.routes[route_idx].nodes_list) - 1,
                state=new_state,
            )

            if cost is not None and cost != np.inf:
                new_state.insert_node_in_route_at_idx(delivery_node, route_idx, idx)
            else:
                new_state.routes[route_idx].remove([pickup_node])
                new_state.update_times_attributes_routes(route_idx)
                new_state.routes_cost[route_idx] = new_state.route_cost_calculator(
                    route_idx
                )
                new_state.cust_df.loc[customer, "route"] = None
                new_state.compute_route_demand(route_idx)

        # Check if the number of routes is less than the number of vehicles
        elif len(new_state.routes) < state.n_vehicles:
            # Initialize a new route and corresponding timings
            depot = state.depots["vehicle_to_depot"][len(new_state.routes)]
            depot = state.cust_to_nodes[depot][0]
            new_state.routes.append(
                Route(
                    [
                        depot, pickup_node, delivery_node, depot],
                    vehicle=len(new_state.routes),
                    planned_windows=deepcopy(
                        state.routes[-1].planned_windows.append([0, END_OF_DAY])
                    ),
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
            logger.debug(f"Could not satisfy customer: {customer}")
            if customer not in new_state.unassigned:
                new_state.unassigned.insert(0, customer)

    new_state.update_unassigned_list()
    return new_state


def wang_best_insert_given_route(
        node: int,
        route_idx: int,
        start_idx: int,
        end_idx: int,
        state: CvrptwState,
) -> tuple:
    """
    """
    route = state.routes[route_idx]
    best_cost = np.inf
    best_idx = None
    for idx in range(start_idx, end_idx + 1):
        if wang_can_insert(
            node=node, 
            route=route, 
            mu=idx, 
            distances=state.distances, 
            tsart=state.twc_format_nodes_df.loc[node, "start_time"],
            tend=state.twc_format_nodes_df.loc[node, "end_time"]
            ):
            cost = insert_cost(node, route.nodes_list, idx, state)
            if cost < best_cost:
                best_cost = cost
                best_idx = idx
    
    return best_cost, best_idx

def wang_best_insert(node: int, state: CvrptwState) -> tuple:
    """
    Finds the best feasible route and insertion indices (pickup and delivery) for the node.
    Return (None, None, None) if no feasible route insertions are found.
    Only checks capacity constraints. Uses the Wang et al (2024) 
    insertion heuristics.
        Parameters:
            node: int
                The node to be inserted.
            state: CvrptwState
                The current solution state.
        Returns:
            tuple
                The best route and insertion indices for the node (route_idx, pickup_idx,\
                    delivery_idx).
    """
    best_cost, best_route_idx, best_idx = np.inf, None, None

    for route_number, route in enumerate(state.routes):
        cost, idx = wang_best_insert_given_route(
            node=node, 
            route_idx=route_number, 
            start_idx=1,
            end_idx=len(route.nodes_list) - 1,
            state=state
        )
        if cost is not None and cost < best_cost:
            best_cost, best_route_idx, best_idx = cost, route_number, idx

    return best_route_idx, best_idx

def wang_can_insert(node: int, route: Route, mu: int, distances: np.ndarray, tsart: int, tend: int) -> bool:
    """
    Check if the node can be inserted at the given position in the route. Based on formula (15)
    of Wang et al (2024).
        Parameters:
            node: int
                The node to be inserted. (c in the paper)
            route: Route
                The route where the node is to be inserted.
            mu: int 
                The position where the node is to be inserted (i_mu, i_mu+1).
            distances: np.ndarray
                The distance matrix.
            tsart: int
                The start time of the node to be inserted.
            tend: int
                The end time of the node to be inserted.
        Returns:
            bool
                True if the node can be inserted, False otherwise.
    """
    # NOTE: should we insert the service time too?
    if mu < 0 or mu >= len(route) - 2:
        return False
    i_mu = route.nodes_list[mu]
    i_mu_plus_1 = route.nodes_list[mu + 1]
    est_mu = route.start_times[mu][0]
    tic = distances[i_mu][node]
    a_c = tsart
    lst_mu_plus_1 = route.start_times[mu+1][1]
    tci = distances[node][i_mu_plus_1]
    b_c = tend
    # NOTE: service time?
    if max(est_mu + tic, a_c) <= min(lst_mu_plus_1 - tci, b_c):
        return True
    else:
        return False
