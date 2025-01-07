import logging
import numpy as np
from data_module import data
from myvrplib import END_OF_DAY
from vrpstates import CvrptwState
from route import Route
from operators.repair import insert_cost


logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

def wang_greedy_repair(state, rng):
    """
    Inserts the unassigned customers in the best route. If there are no
    feasible insertions, then a new route is created. Uses the Wang et al (2024)
    insertion heuristics with time window compatibility checks.
    """
    rng.shuffle(state.unassigned)

    counter = 0
    n_unassigned = len(state.unassigned)
    while counter < n_unassigned:
        counter += 1
        customer = state.unassigned.pop()
        route, idx = wang_best_insert(customer, state)

        if route is not None:
            route.insert(idx, customer)
            state.update_times()
            # logger.debug(f"WGR: Customer {customer} inserted.")
        else:
            state.unassigned.insert(0, customer)
            # logger.debug(f"WGR: Customer {customer} not inserted.")
        #     if len(state.routes) < data["vehicles"]:
        #         vehicle_number = len(state.routes)
        #         state.routes.append(
        #             Route(
        #                 [
        #                     data["vehicle_to_depot"][vehicle_number],
        #                     customer,
        #                     data["vehicle_to_depot"][vehicle_number],
        #                 ]
        #             )
        #         )
        #         state.update_times()  # NOTE: maybe not needed
    logger.debug(f"Finished Wang greedy repair.")
    return state


def wang_best_insert(customer, state: CvrptwState):
    """
    Finds the best feasible route and insertion idx for the customer.
    Return (None, None) if no feasible route insertions are found.
    Only checks capacity constraints.
    """
    best_cost, best_route, best_idx = None, None, None
    for route in state.routes:
        for idx in range(1, len(route)):
            # if can_insert(customer, route_number, idx, state):
            if wang_can_insert(customer, route, idx):
                cost = insert_cost(customer, route.customers_list, idx)

                if best_cost is None or cost < best_cost:
                    best_cost, best_route, best_idx = cost, route, idx

    return best_route, best_idx

def wang_can_insert(customer, route: Route, mu: int) -> bool:
    """
    Check if the customer can be inserted at the given position in the route. Based on formula (15)
    of Wang et al (2024).
    Parameters:
        customer: int, the customer to be inserted. (c in the paper)
        route: Route, the route where the customer is to be inserted.
        mu: int, the position where the customer is to be inserted (i_mu, i_mu+1).
    """
    # NOTE: should we insert the service time too?
    if mu < 0 or mu >= len(route) - 2:
        return False
    i_mu = route.customers_list[mu]
    i_mu_plus_1 = route.customers_list[mu + 1]
    est_mu = route.earliest_start_times[mu]
    tic = data["edge_weight"][i_mu][customer]
    a_c = data["time_window"][customer][0]
    logger.debug(f"mu = {mu}, i_mu = {i_mu}, i_mu_plus_1 = {i_mu_plus_1}, est_mu = {est_mu}, tic = {tic}, a_c = {a_c}")
    logger.debug(f"len(route.latest_start_times) = {len(route.latest_start_times)}, len(route) = {len(route)}\n")
    lst_mu_plus_1 = route.latest_start_times[mu + 1]
    tci = data["edge_weight"][customer][i_mu_plus_1]
    b_c = data["time_window"][customer][1]
    # NOTE: service time?
    #DEBUG
    # print(f"Customer = {customer}")
    # print(f"est_mu = {est_mu}, tic = {tic}, a_c = {a_c}, lst_mu_plus_1 = {lst_mu_plus_1}, tci = {tci}, b_c = {b_c}")
    if max(est_mu + tic, a_c) <= min(lst_mu_plus_1 - tci, b_c):
        return True
    else:
        return False
