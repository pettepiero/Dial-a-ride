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
    rng.shuffle(new_state.unassigned)

    counter = 0
    n_unassigned = len(new_state.unassigned)
    while counter < n_unassigned:
        counter += 1
        customer = new_state.unassigned.pop()
        start_node, end_node = state.cust_to_nodes[customer]
        logger.debug(f"Candidate customer: {customer}")
        route_idx, idx = wang_best_insert(customer, new_state)

        if route_idx is not None:
            new_state.routes[route_idx].insert(idx, customer)
            new_state.cust_df.loc[customer, "route"] = route_idx
            new_state.update_times_attributes_routes(route_index=route_idx)
            new_state.routes_cost[route_idx] = new_state.route_cost_calculator(route_idx)
            new_state.compute_route_demand(route_idx)

        elif len(new_state.routes) < state.n_vehicles:
            depot = state.depots["vehicle_to_depot"][len(new_state.routes)]
            new_state.routes.append(
                Route(
                    [
                        depot,
                        customer,
                        depot,
                    ],
                    vehicle=len(new_state.routes),
                    planned_windows=deepcopy(
                        state.routes[-1].planned_windows.append([0, END_OF_DAY])
                    ),
                )
            )
            new_state.cust_df.loc[customer, "route"] = len(new_state.routes) - 1
            new_state.routes_cost = np.append(
                new_state.routes_cost,
                new_state.route_cost_calculator(len(new_state.routes) - 1),
            )
            new_state.update_times_attributes_routes(len(new_state.routes) - 1)
        else:
            logger.debug(f"Could not satisfy customer: {customer}")
            new_state.unassigned.insert(0, customer)
    return new_state


def wang_best_insert(customer: int, state: CvrptwState) -> tuple:
    """
    Finds the best feasible route and insertion indices (pickup and delivery) for the customer.
    Return (None, None, None) if no feasible route insertions are found.
    Only checks capacity constraints. Uses the Wang et al (2024) 
    insertion heuristics.
        Parameters:
            customer: int
                The customer to be inserted.
            state: CvrptwState
                The current solution state.
        Returns:
            tuple
                The best route and insertion indices for the customer (route_idx, pickup_idx,\
                    delivery_idx).
    """

    print(f"DEBUG: Wang best insert for customer {customer}")
    print(f"DEBUG: state.cust_df = \n{state.cust_df}")

    best_cost, best_route_idx, best_idx_pickup, best_idx_delivery = None, None, None, None, None
    for route_idx, route in enumerate(state.routes):
        # start by inserting pickup node
        for idx in range(1, len(route)):
            start, end = state.cust_df.loc[route.nodes_list[idx], ["start_time", "end_time"]]
            # if can_insert(customer, route_number, idx, state):
            if wang_can_insert(customer, route, idx, state.distances, start, end):
                cost = insert_cost(customer, route.nodes_list, idx, state)

                if best_cost is None or cost < best_cost:
                    best_cost, best_route_idx, best_idx = cost, route_idx, idx

    return best_route_idx, best_idx

def wang_can_insert(customer: int, route: Route, mu: int, distances: np.ndarray, tsart: int, tend: int) -> bool:
    """
    Check if the customer can be inserted at the given position in the route. Based on formula (15)
    of Wang et al (2024).
        Parameters:
            customer: int
                The customer to be inserted. (c in the paper)
            route: Route
                The route where the customer is to be inserted.
            mu: int 
                The position where the customer is to be inserted (i_mu, i_mu+1).
            distances: np.ndarray
                The distance matrix.
            tsart: int
                The start time of the customer to be inserted.
            tend: int
                The end time of the customer to be inserted.
        Returns:
            bool
                True if the customer can be inserted, False otherwise.
    """
    # NOTE: should we insert the service time too?
    if mu < 0 or mu >= len(route) - 2:
        return False
    i_mu = route.nodes_list[mu]
    i_mu_plus_1 = route.nodes_list[mu + 1]
    est_mu = route.start_times[mu][0]
    tic = distances[i_mu][customer]
    a_c = tsart
    lst_mu_plus_1 = route.start_times[mu+1][1]
    tci = distances[customer][i_mu_plus_1]
    b_c = tend
    # NOTE: service time?
    # DEBUG
    # print(f"Customer = {customer}")
    # print(f"est_mu = {est_mu}, tic = {tic}, a_c = {a_c}, lst_mu_plus_1 = {lst_mu_plus_1}, tci = {tci}, b_c = {b_c}")
    if max(est_mu + tic, a_c) <= min(lst_mu_plus_1 - tci, b_c):
        return True
    else:
        return False
