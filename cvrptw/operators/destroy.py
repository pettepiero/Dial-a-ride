from data_module import data
import numpy as np
from myvrplib import END_OF_DAY
from vrpstates import CvrptwState
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)
degree_of_destruction = 0.05
customers_to_remove = int((data["dimension"] - 1) * degree_of_destruction)


def random_removal(state: CvrptwState, rng) -> CvrptwState:
    """
    Removes a number of randomly selected customers from the passed-in solution.
    """
    destroyed: CvrptwState = state.copy()

    for customer in rng.choice(
        range(data["dimension"]), customers_to_remove, replace=False
    ):
        destroyed.unassigned.append(customer)
        route = destroyed.find_route(customer)
        if route is not None:
            route.remove(customer)

    # NOTE: now evaluate the time of the modified routes and return them
    destroyed.update_times()

    return remove_empty_routes(destroyed)


def remove_empty_routes(state: CvrptwState) -> CvrptwState:
    """
    Remove empty routes and timings after applying the destroy operator.
    """
    routes_idx_to_remove = [
        idx for idx, route in enumerate(state.routes) if len(route) == 2
    ]
    state.routes = [
        route
        for idx, route in enumerate(state.routes)
        if idx not in routes_idx_to_remove
    ]
    state.times = [
        timing
        for idx, timing in enumerate(state.times)
        if idx not in routes_idx_to_remove
    ]
    state.times = [
        timing for idx, timing in enumerate(state.times) if len(state.routes[idx]) != 2
    ]
    return state


def random_route_removal(state: CvrptwState, rng) -> CvrptwState:
    """
    Based on [1]. This operator randomly selects routes
    from a given solution, extracting customers associated with these routes
    and transferring them to the request list L. The operator repeats the
    selection until the number of removal customers reaches or exceeds .
    """

    destroyed: CvrptwState = state.copy()

    for route in rng.choice(destroyed.routes, customers_to_remove, replace=True):
        if len(route.customers_list[1:-1]) != 0:
            customer = rng.choice(route.customers_list[1:-1], 1, replace=False)
            destroyed.unassigned.append(customer.item())
            route.remove(customer)
            destroyed.update_times()

    return remove_empty_routes(destroyed)


def relatedness_function(state: CvrptwState, i: int, j: int) -> float:
    """
    Calculates how related two requests are. The lower the value, the more related.
    Based on [1], formula (4), which is itself based on the work of Shaw (1997) and
    Ropke and Pisinger (2006).
    """
    a1 = 0.4
    a2 = 0.8
    a3 = 0.3
    i_route = state.find_route(i)
    j_route = state.find_route(j)

    i_index_in_route = state.find_index_in_route(i, i_route)
    j_index_in_route = state.find_index_in_route(j, j_route)
    e_i = i_route.earliest_start_times[i_index_in_route]
    e_j = j_route.earliest_start_times[j_index_in_route]
    q_i = data["demand"][i]
    q_j = data["demand"][j]
    value = (
        a1 * (data["edge_weight"][i][j] / state.dmax)
        + a2 * (abs(e_i - e_j) / END_OF_DAY)
        + a3 * (abs(q_i - q_j) / state.qmax)
    )

    return value


def shaw_removal(state: CvrptwState, rng) -> CvrptwState:
    """
    Based on [1], formula (4).
    """

    destroyed: CvrptwState = state.copy()

    min_value = np.inf
    j_star = None
    route_star_idx = None

    route_i_idx = rng.choice(
        range(len(destroyed.routes)), 1
    ).item()  # Randomly select first customer to remove from the solution
    # route_i = rng.choice(destroyed.routes, 1).item()  # Randomly select first customer to remove from the solution
    route_i = destroyed.routes[route_i_idx]
    if len(route_i.customers_list[1:-1]) == 0:
        print(f"DEBUG: route_i.customers_list: {route_i.customers_list}")
    first_customer = rng.choice(route_i.customers_list[1:-1], 1, replace=False).item()
    i_selection = [first_customer]

    while len(i_selection) < customers_to_remove:
        i = rng.choice(i_selection, 1, replace=False).item()

        for route_idx, route in enumerate(destroyed.routes):
            for j in route.customers_list[1:-1]:
                if j in i_selection:
                    continue
                if i == j and route_i == route:
                    continue
                value = relatedness_function(state, i, j)
                if value < min_value:
                    min_value = value
                    j_star = j
                    route_star_idx = route_idx

        if j_star is None:
            continue
        # DEBUG
        # print(f"Selected j_star: {j_star}")
        # print(f"Selected route_star_idx: {route_star_idx}")
        # print(f"Selected route: {destroyed.routes[route_star_idx].customers_list}")
        # print(f"i_selection {i_selection}")
        destroyed.routes[route_star_idx].remove(j_star)
        i_selection.append(j_star)
        destroyed.unassigned.append(j_star)
        j_star = None
        min_value = np.inf
        route_star_idx = None

    route_i.remove(first_customer)
    # destroyed.unassigned.append(first_customer)
    destroyed.update_times()

    return remove_empty_routes(destroyed)


def cost_reducing_removal(state, rng):
    """
    Cost reducing removal operator based on 'An adaptive large neighborhood
    search for the multi-depot dynamic vehicle routing problem with time windows'.
    """
    destroyed = state.copy()

    for first_route_index in rng.choice(len(state.routes), 1):
        customers = state.routes[first_route_index].customers_list
        if (
            len(customers) <= 2
        ):  # route has only depot and customer :TODO: what to do in this case?
            break
        change_route = False
        v = rng.choice(customers[1:-1])
        finished_v = False
        i1 = customers[customers.index(v) - 1]  # previous customer
        j1 = customers[customers.index(v) + 1]  # next customer

        # DEBUG
        # logger.debug(f"\n\nv: {v}, i1: {i1}, j1: {j1}, customers: {customers}")
        di1v = data["edge_weight"][i1][v]
        dvj1 = data["edge_weight"][v][j1]
        di1j1 = data["edge_weight"][i1][j1]

        for second_route_index in list(range(len(state.routes))):
            customers2 = state.routes[second_route_index].customers_list
            for i2 in customers2[:-1]:  # first customer of insertion arc
                j2 = customers2[
                    customers2.index(i2) + 1
                ]  # second customer of insertion arc
                if state.twc[v][i2] != -np.inf and state.twc[v][j2] != -np.inf:
                    di2j2 = data["edge_weight"][i2][j2]
                    di2v = data["edge_weight"][i2][v]
                    dvj2 = data["edge_weight"][v][j2]
                    if di1v + dvj1 + di2j2 > di1j1 + di2v + dvj2:
                        logger.debug(
                            f"\nBEFORE:\nfirst_route_index: {first_route_index}, second_route_index: {second_route_index}"
                        )
                        logger.debug(
                            f"len(first_route): {len(state.routes[first_route_index])}, len(second_route): {len(state.routes[second_route_index])}"
                        )
                        logger.debug(f"Customers: {i1} -> {v} -> {j1} | {i2} -> {j2}")
                        logger.debug(
                            f"first_route customers: {state.routes[first_route_index].customers_list}"
                        )
                        logger.debug(f"customers = {customers}")
                        logger.debug(
                            f"second_route customers: {state.routes[second_route_index].customers_list}"
                        )
                        logger.debug(f"customers2 = {customers2}")

                        state.routes[first_route_index].remove(v)
                        state.routes[second_route_index].insert(
                            customers2.index(j2), v.item()
                        )
                        # customers2.insert(customers2.index(i2), v)
                        logger.debug(
                            f"\nAFTER:\nfirst_route customers: {state.routes[first_route_index].customers_list}"
                        )
                        logger.debug(f"customers = {customers}")
                        logger.debug(
                            f"second_route customers: {state.routes[second_route_index].customers_list}"
                        )
                        logger.debug(f"customers2 = {customers2}")

                        logger.debug(
                            f"len(first_route): {len(state.routes[first_route_index])}, len(second_route): {len(state.routes[second_route_index])}\n"
                        )
                        destroyed.update_times()
                        finished_v = True
                        change_route = True
                        break
            if finished_v:
                break
        if change_route:
            break

    return remove_empty_routes(destroyed)
