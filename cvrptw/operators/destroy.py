from data_module import data
import numpy as np
from myvrplib import END_OF_DAY
from vrpstates import CvrptwState
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
degree_of_destruction = 0.05
customers_to_remove = int((data["dimension"] - 1) * degree_of_destruction)


def random_removal(state: CvrptwState, rng) -> CvrptwState:
    """
    Removes a number of randomly selected customers from the passed-in solution.
    Ignores first customer following cordeau dataset notation.
    """
    destroyed: CvrptwState = state.copy()

    # list of customers in solution
    solution_customers = state.served_customers()

    for customer in rng.choice(
        solution_customers, customers_to_remove, replace=False
    ):
        destroyed.unassigned.append(customer.item())
        route = destroyed.find_route(customer.item())
        if route is not None:
            route.remove(customer.item())
        else:
            logger.debug(
                f"Error: customer {customer.item()} not found in any route but picked from served customers."
            )

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
    #debug
    print(f"len(destroyed.routes): {len(destroyed.routes)}")
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
    #DEBUG
    print(f"Starting Shaw removal with routes: {[route.customers_list for route in destroyed.routes]}")

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
    print(f"i_selection = {i_selection}")
    while len(i_selection) < customers_to_remove:
        i = rng.choice(i_selection, 1, replace=False).item()

        for route_idx, route in enumerate(destroyed.routes):
            for j in route.customers_list[1:-1]:
                if j in i_selection:
                    continue
                if i == j and route_i == route:
                    continue
                value = relatedness_function(state, i, j)   #NOTE: maybe use dynamic programming to store values
                #DEBUG
                print(f"Checking i: {i}, j: {j}, value: {value}")
                if value < min_value:
                    print(f"New min_value: {value}")
                    min_value = value
                    j_star = j
                    route_star_idx = route_idx

        if j_star is None:
            print(f"\nFinished checking all customers and no j_star found")
            print(f"Left routes: {[route.customers_list for route in destroyed.routes]}")
            continue
        # DEBUG
        # print(f"Selected j_star: {j_star}")
        # print(f"Selected route_star_idx: {route_star_idx}")
        # print(f"Selected route: {destroyed.routes[route_star_idx].customers_list}")
        # print(f"i_selection {i_selection}")
        destroyed.routes[route_star_idx].remove(j_star)
        i_selection.append(j_star)
        print("Now i_selection: ", i_selection)
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
        v = rng.choice(customers[1:-1])
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
                        # Remove v from first route and insert into second
                        state.routes[first_route_index].remove(v)
                        state.routes[second_route_index].insert(
                            customers2.index(j2), v.item()
                        )
                        destroyed.update_times()
                        return remove_empty_routes(destroyed)

    return remove_empty_routes(destroyed)


def worst_removal_cost_function(distances: np.ndarray) -> float:
    """
    Cost function for the worst removal operator.
    """

    return

def worst_removal(state: CvrptwState, rng: np.random.Generator) -> CvrptwState:
    """
    Removes customers in decreasing order of service cost.
    """
    destroyed = state.copy()
    max_service_cost = 0

    for route in destroyed.routes:
        for i in route.customers_list[1:-1]:
            j = route.customers_list[route.customers_list.index(i) - 1]
            k = route.customers_list[route.customers_list.index(i) + 1]
            service_cost = (data["edge_weight"][j][i] +
                            data["edge_weight"][i][k] -
                            data["edge_weight"][j][k]
                            )
            if service_cost > max_service_cost:
                max_service_cost = service_cost
                worst_customer = i
                worst_route = route

    # Removes the worst customer
    worst_route.remove(worst_customer)
    destroyed.unassigned.append(worst_customer)
    destroyed.update_times()

    return state

def exchange_reducing_removal(state: CvrptwState, rng: np.random.Generator) -> CvrptwState:
    """
    Variation of the cost-reducing removal based on Wang et al (2024)
    """
    destroyed = state.copy()

    route1 = rng.choice(destroyed.routes, 1).item()

    for idx1 in range(1, len(route1.customers_list)-1):
        v1 = route1.customers_list[idx1]
    # for v1 in route1.customers_list:
    #     idx1 = route1.customers_list.index(v1)
        i1 = route1.customers_list[idx1 - 1]    #previous node
        j1 = route1.customers_list[idx1 + 1]    #next node

        for route2 in destroyed.routes:
            for v2 in route2.customers_list[1:-1]:
                if route2 == route1 and v2 == v1:
                    continue

                idx2 = route2.customers_list.index(v2)
                i2 = route2.customers_list[idx2 - 1]  # previous node
                j2 = route2.customers_list[idx2 + 1]  # next node
                # Check Time Window Compatibility
                if destroyed.twc[v1][i2] != -np.inf and destroyed.twc[v2][i1] != np.inf:
                    di1v1 = data["edge_weight"][i1][v1]
                    dv1j1 = data["edge_weight"][v1][j1]
                    di2v2 = data["edge_weight"][i2][v2]
                    dv2j2 = data["edge_weight"][v2][j2]

                    di1v2 = data["edge_weight"][i1][v2]
                    dv2j1 = data["edge_weight"][v2][j1]
                    di2v1 = data["edge_weight"][i2][v1]
                    dv1j2 = data["edge_weight"][v1][j2]

                    if di1v1 + dv1j1 + di2v2 + dv2j2 > di1v2 + dv2j1 + di2v1 + dv1j2:
                        # swap v1 and v2
                        route1.customers_list[idx1] = v2
                        route2.customers_list[idx2] = v1
                        destroyed.update_times()
                        return remove_empty_routes(destroyed)
    
    return remove_empty_routes(destroyed)
