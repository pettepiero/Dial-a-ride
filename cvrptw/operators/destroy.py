from cvrptw.myvrplib.data_module import data
import numpy as np
from cvrptw.myvrplib.myvrplib import END_OF_DAY, LOGGING_LEVEL
from cvrptw.myvrplib.vrpstates import CvrptwState
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOGGING_LEVEL)
degree_of_destruction = 0.05
customers_to_remove = int((data["dimension"] - 1) * degree_of_destruction)


def random_removal(state: CvrptwState, rng: np.random) -> CvrptwState:
    """
    Removes customers_to_remove randomly selected customers from the passed-in solution.
    Ignores first and last customers in the routes following cordeau dataset notation.
        Parameters:
            state: CvrptwState
                The solution from which to remove customers.
            rng: np.random
                Random number generator.
        Returns:
            CvrptwState
                The solution after applying the destroy operator.
    """
    destroyed: CvrptwState = state.copy()

    # list of customers in solution
    solution_customers = state.served_customers()

    for customer in rng.choice(
        solution_customers, customers_to_remove, replace=False
    ):
        route, idx = destroyed.find_route(customer.item())
        if route is not None:
            destroyed.routes[idx].remove(customer.item())
            destroyed.unassigned.append(customer.item())
            destroyed.update_times_attributes_routes()
            # route.remove(customer.item())
            # route.calculate_planned_times()
            logger.debug(f"Customer {customer.item()} removed from route {idx}.")
        else:
            logger.debug(
                f"Error: customer {customer.item()} not found in any route but picked from served customers."
            )

    return remove_empty_routes(destroyed)


def remove_empty_routes(state: CvrptwState) -> CvrptwState:
    """
    Remove empty routes and timings after applying the destroy operator.
    Cordeau dataset notation is followed, so empty routes ar those with two elements.
        Parameters:
            state: CvrptwState
                The solution from which to remove empty routes.
        Returns:
            CvrptwState
                The solution after removing empty routes.
    """
    routes_idx_to_remove = [
        idx for idx, route in enumerate(state.routes) if len(route) == 2
    ]
    state.routes = [
        route
        for idx, route in enumerate(state.routes)
        if idx not in routes_idx_to_remove
    ]
    return state


def random_route_removal(state: CvrptwState, rng: np.random) -> CvrptwState:
    """
    Based on (Wang et. al, 2024). This operator randomly selects customers_to_remove 
    routes from a given solution and then removes a random customer from each route.
        Parameters:
            state: CvrptwState
                The solution from which to remove customers.
            rng: np.random
                Random number generator.
        Returns:
            CvrptwState
                The solution after applying the destroy operator.
    """    
    destroyed: CvrptwState = state.copy()
    for route_idx in rng.choice(range(len(destroyed.routes)), min(customers_to_remove, state.n_served_customers()), replace=True):
        route = destroyed.routes[route_idx]
        # for route in rng.choice(destroyed.routes, customers_to_remove, replace=True):
        if len(route.customers_list[1:-1]) != 0:
            customer = rng.choice(route.customers_list[1:-1], 1, replace=False)
            destroyed.unassigned.append(customer.item())
            destroyed.routes[route_idx].remove(customer)
            destroyed.routes[route_idx].calculate_planned_times()

    destroyed.update_times_attributes_routes()

    return remove_empty_routes(destroyed)


def relatedness_function(state: CvrptwState, i: int, j: int) -> float:
    """
    Calculates how related two requests are. The lower the value, the more related.
    Based on (Wang et. al, 2024), formula (4), which is itself based on the work of 
    Shaw (1997) and Ropke and Pisinger (2006).
        Parameters:
            state: CvrptwState
                The solution from which to remove customers.
            i: int
                The first customer.
            j: int
                The second customer.
        Returns:    
            float
                The relatedness value between the two customers.
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
    Based on (Wang et. al, 2024), formula (4), which is itself based on the work of
    Shaw (1997) and Ropke and Pisinger (2006). This operator removes customers_to_remove
    customers from the solution by selecting the most related customers.
        Parameters:
            state: CvrptwState
                The solution from which to remove customers.
            rng: np.random
                Random number generator.
        Returns:
            CvrptwState
                The solution after applying the destroy operator.
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
                value = relatedness_function(state, i, j)   #NOTE: maybe use dynamic programming to store values
                if value < min_value:
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
        destroyed.routes[route_star_idx].calculate_planned_times()
        i_selection.append(j_star)
        print("Now i_selection: ", i_selection)
        destroyed.unassigned.append(j_star)
        j_star = None
        min_value = np.inf
        route_star_idx = None

    route_i.remove(first_customer)
    route_i.calculate_planned_times()
    # destroyed.unassigned.append(first_customer)
    destroyed.update_times()

    return remove_empty_routes(destroyed)


def cost_reducing_removal(state: CvrptwState, rng: np.random) -> CvrptwState:
    """
    Cost reducing removal operator based on algorithm 2 of (Wang et al, 2024). 
    Identifies customers that can be inserted into a solution route at a lower cost.
    A limit on iterations is proposed to terminate the search for potential 
    customers using this operator.
        Parameters:
            state: CvrptwState
                The solution from which to remove customers.
            rng: np.random
                Random number generator.
        Returns:
            CvrptwState
                The solution after applying the destroy operator.
    """

    # TODO: Implement the limit on the iterations for this operator
    destroyed = state.copy()
    
    iterations = 10
    for _ in range(iterations):
        for first_route_index in rng.choice(len(state.routes), 1):
            customers = state.routes[first_route_index].customers_list
            if (len(customers) <= 2):   # rout has only depots
                break
            for v in customers[1:-1]:
                i1 = customers[customers.index(v) - 1]  # previous customer
                j1 = customers[customers.index(v) + 1]  # next customer
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
                                destroyed.routes[first_route_index].remove(v)
                                destroyed.routes[first_route_index].calculate_planned_times()
                                # state.routes[second_route_index].insert(
                                #     customers2.index(j2), v.item()
                                # )
                                destroyed.unassigned.append(v)
                                destroyed.update_times_attributes_routes()
                            
                                return remove_empty_routes(destroyed)
    logger.debug(f"No customers found to remove in {iterations} iterations of cost_reducing_removals.")
        
    destroyed.update_times_attributes_routes()
    return remove_empty_routes(destroyed)

def worst_removal(state: CvrptwState, rng: np.random.Generator) -> CvrptwState:
    """
    Removes customers in decreasing order of service cost.
        Parameters:
            state: CvrptwState
                The solution from which to remove customers.
            rng: np.random.Generator
                Random number generator.
        Returns:
            CvrptwState
                The solution after applying the destroy operator.
    """
    destroyed = state.copy()
    max_service_cost = 0

    for route_idx, route in enumerate(destroyed.routes):
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
                worst_route = route_idx
    # Removes the worst customer
    destroyed.routes[worst_route].remove(worst_customer)
    destroyed.update_times_attributes_routes()
    destroyed.unassigned.append(worst_customer)

    return destroyed

def exchange_reducing_removal(state: CvrptwState, rng: np.random.Generator) -> CvrptwState:
    """
    Variation of the cost-reducing removal based on Wang et. al (2024). Selects 
    customers in pairs, allowing one customer to be replaced by another simultaneously.
        Parameters:
            state: CvrptwState
                The solution from which to remove customers.
            rng: np.random.Generator
                Random number generator.
        Returns:
            CvrptwState
                The solution after applying the destroy operator.
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
                        destroyed.update_times_attributes_routes()

                        return remove_empty_routes(destroyed)

    destroyed.update_times_attributes_routes()

    return remove_empty_routes(destroyed)
