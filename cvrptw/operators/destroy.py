from cvrptw.myvrplib.data_module import data
import numpy as np
from cvrptw.myvrplib.myvrplib import END_OF_DAY, LOGGING_LEVEL
from cvrptw.myvrplib.vrpstates import CvrptwState
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOGGING_LEVEL)
degree_of_destruction = 0.05


def random_removal(state: CvrptwState, rng: np.random) -> CvrptwState:
    """
    Removes customers_to_remove randomly selected customers from the passed-in solution.
    Ignores first and last customers in the routes because they are depots.
        
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
    customers_to_remove = int(destroyed.n_customers * degree_of_destruction)

    # list of customers in solution
    solution_customers = state.planned_customers()
    assert len(solution_customers) > 0, "No customers in solution."
    for customer in rng.choice(
        solution_customers,
        min(customers_to_remove, state.n_planned_customers),
        replace=False,
    ):
        customer = customer.item()
        pickup_node, delivery_node = state.cust_to_nodes[customer]
        assert (
            pickup_node not in destroyed.depots["depots_indices"]
        ), "Depot selected for removal."
        assert (
            delivery_node not in destroyed.depots["depots_indices"]
        ), "Depot selected for removal."

        route, idx = destroyed.find_route(customer)
        if route is not None:
            destroyed.routes[idx].remove([pickup_node, delivery_node])
            # Update df
            destroyed.cust_df.loc[customer, "route"] = None
            destroyed.cust_df.loc[customer, "done"] = False
            destroyed.unassigned.append(customer)
            destroyed.update_times_attributes_routes(idx)
            if len(destroyed.routes[idx]) != 2:
                destroyed.routes_cost[idx] = destroyed.route_cost_calculator(idx)
            logger.debug(f"Customer {customer} removed from route {idx}.")
        else:
            logger.debug(
                f"Error: customer {customer} not found in any route but picked from served customers."
            )

    return remove_empty_routes(destroyed)


def remove_empty_routes(state: CvrptwState) -> CvrptwState:
    """
    Remove empty routes and corresponding cost after applying the destroy operator.
    Cordeau dataset notation is followed, so empty routes ar those with two elements.
        
    Parameters:
        state: CvrptwState
            The solution from which to remove empty routes.
    Returns:
        CvrptwState
            The solution after removing empty routes.
    """
    for idx, route in enumerate(state.routes):
        if len(route) == 2:
            logger.debug(f"Route {idx} is empty and will be removed.")
            del state.routes[idx]
            state.routes_cost = np.delete(state.routes_cost, idx)

    # routes_idx_to_remove = [
    #     idx for idx, route in enumerate(state.routes) if len(route) == 2
    # ]
    # state.routes = [
    #     route
    #     for idx, route in enumerate(state.routes)
    #     if idx not in routes_idx_to_remove
    # ]

    # state.routes_cost = [
    #     state.route_cost_calculator(idx) for idx in range(len(state.routes))
    # ]
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
    customers_to_remove = int(destroyed.n_customers * degree_of_destruction)
    for route_idx in rng.choice(
        range(len(destroyed.routes)),
        min(customers_to_remove, state.n_planned_customers),
        replace=True,
    ):
        route = destroyed.routes[route_idx]
        # for route in rng.choice(destroyed.routes, customers_to_remove, replace=True):
        if len(route.nodes_list[1:-1]) != 0:
            sampled_node = rng.choice(route.nodes_list[1:-1], 1, replace=False)
            cust = destroyed.nodes_to_cust[sampled_node.item()]
            pickup_node, delivery_node = destroyed.cust_to_nodes[cust]
            destroyed.routes[route_idx].remove([pickup_node, delivery_node])
            # Update df
            destroyed.cust_df.loc[cust, "route"] = None
            destroyed.cust_df.loc[cust, "done"] = False
            destroyed.unassigned.append(cust)
            if len(destroyed.routes[route_idx]) != 2:
                destroyed.update_times_attributes_routes(route_idx)
                destroyed.routes_cost[route_idx] = destroyed.route_cost_calculator(
                    route_idx
                )
    return remove_empty_routes(destroyed)


def relatedness_function(state: CvrptwState, cust_i: int, cust_j: int) -> float:
    """
    Calculates how related two requests are. The lower the value, the more related.
    Based on (Wang et. al, 2024), formula (4), which is itself based on the work of
    Shaw (1997) and Ropke and Pisinger (2006). The formula is modified for the pickup
    and delivery problem, and the relatedness value is calculated as the sum of the
    relatedness between pickup nodes and the relatedness between delivery nodes.
    The weights a1, a2 and a3 are given by Wang et. al, 2024.

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

    i_route, _ = state.find_route(cust_i)
    j_route, _ = state.find_route(cust_j)

    # Pikcup
    i = state.cust_to_nodes[cust_i][0]
    j = state.cust_to_nodes[cust_j][0]
    i_index_in_route = state.find_index_in_route(i, i_route)
    j_index_in_route = state.find_index_in_route(j, j_route)
    e_i = i_route.start_times[i_index_in_route][0]
    e_j = j_route.start_times[j_index_in_route][0]
    q_i = state.cust_df.loc[cust_i, "demand"].item()
    q_j = state.cust_df.loc[cust_j, "demand"].item()
    value_pickup = (
        a1 * (state.distances[i][j] / state.dmax)
        + a2 * (abs(e_i - e_j) / END_OF_DAY)
        + a3 * (abs(q_i - q_j) / state.qmax)
    )
    # Delivery
    i = state.cust_to_nodes[cust_i][1]
    j = state.cust_to_nodes[cust_j][1]
    i_index_in_route = state.find_index_in_route(i, i_route)
    j_index_in_route = state.find_index_in_route(j, j_route)
    e_i = i_route.start_times[i_index_in_route][0]
    e_j = j_route.start_times[j_index_in_route][0]
    q_i = state.cust_df.loc[cust_i, "demand"].item()
    q_j = state.cust_df.loc[cust_j, "demand"].item()
    value_delivery = (
        a1 * (state.distances[i][j] / state.dmax)
        + a2 * (abs(e_i - e_j) / END_OF_DAY)
        + a3 * (abs(q_i - q_j) / state.qmax)
    )

    return value_delivery + value_pickup


def shaw_removal(state: CvrptwState, rng) -> CvrptwState:
    """
    Based on (Wang et. al, 2024), formula (4), which is itself based on the work of
    Shaw (1997) and Ropke and Pisinger (2006). This operator removes a single
    customers from the solution by selecting the most related ones, according to the formula.
    
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

    # Search phase
    # Randomly select seed customer
    # route_i_idx = rng.choice(range(len(destroyed.routes)), 1).item()
    # route_i = destroyed.routes[route_i_idx]
    # Set of both planned and not planned yet customers
    first_customer = rng.choice(destroyed.planned_customers(), 1, replace=False).item()

    # i_selection = [first_customer]

    for j in destroyed.planned_customers():
        # if j in i_selection:
        #     continue
        value = relatedness_function(
            state, first_customer, j
        )
        if value < min_value:
            min_value = value
            j_star = j
            route_star_idx = destroyed.find_route(j)[1]

    if j_star is None:
        print(f"\nFinished checking all customers and no j_star found")
        print(f"Left routes: {[route.nodes_list for route in destroyed.routes]}")
        return destroyed

    start_j, end_j = destroyed.cust_to_nodes[j_star]

    # Modify phase
    destroyed.routes[route_star_idx].remove([start_j, end_j])
    # Update df
    destroyed.cust_df.loc[j_star, "route"] = None
    destroyed.cust_df.loc[j_star, "done"] = False
    destroyed.unassigned.append(j_star)
    if len(destroyed.routes[route_star_idx]) != 2:
        destroyed.update_times_attributes_routes(route_star_idx)
        destroyed.routes_cost[route_star_idx] = destroyed.route_cost_calculator(
            route_star_idx
        )
    # i_selection.append(j_star)

    # Update df
    if first_customer in destroyed.planned_customers():
        start_i, end_i = destroyed.cust_to_nodes[first_customer]
        route_i, route_i_idx = destroyed.find_route(first_customer)
        destroyed.routes[route_i_idx].remove([start_i, end_i])
        destroyed.unassigned.append(first_customer)
        destroyed.cust_df.loc[first_customer, "route"] = None
        destroyed.cust_df.loc[first_customer, "done"] = False
        if len(route_i) != 2:
            destroyed.update_times_attributes_routes(route_i_idx)
            destroyed.routes_cost[route_i_idx] = destroyed.route_cost_calculator(
                route_i_idx
            )

    return remove_empty_routes(destroyed)


def cost_reducing_removal(state: CvrptwState, rng: np.random) -> CvrptwState:
    """
    Cost reducing removal operator based on (Wang et al, 2024). Identifies
    customers that can be inserted into a solution route at a lower cost. This
    includes testing both nodes for the selected customer. The first better solution 
    is selected, which means that there might be better solutions that are not found.
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

    logger.debug(f"In cost reducing removal")
    destroyed = state.copy()
    initial_cost = destroyed.objective()

    iterations = 10
    for i in range(iterations):
        for first_route_index in rng.choice(len(state.routes), 1):
            nodes = state.routes[first_route_index].nodes_list
            if (
                len(nodes) <= 2
            ):  # route has only depot and customer :TODO: what to do in this case?
                break
            for v in nodes[1:-1]:
                i1 = nodes[nodes.index(v) - 1]  # previous customer
                j1 = nodes[nodes.index(v) + 1]  # next customer

                # DEBUG
                # logger.debug(f"\n\nv: {v}, i1: {i1}, j1: {j1}, nodes: {nodes}")
                di1v = state.distances[i1][v]
                dvj1 = state.distances[v][j1]
                di1j1 = state.distances[i1][j1]

                for second_route_index in list(range(len(state.routes))):
                    nodes2 = state.routes[second_route_index].nodes_list
                    for i2 in nodes2[:-2]:  # first customer of insertion arc
                        i2_idx = nodes2.index(i2)
                        j2 = nodes2[i2_idx + 1]  # second customer of insertion arc
                        if state.twc[v][i2] != -np.inf and state.twc[v][j2] != -np.inf:
                            di2j2 = state.distances[i2][j2]
                            di2v = state.distances[i2][v]
                            dvj2 = state.distances[v][j2]
                            logger.debug(
                                f"Checking if {di1v} + {dvj1} + {di2j2} > {di1j1} + {di2v} + {dvj2}"
                            )
                            if di1v + dvj1 + di2j2 > di1j1 + di2v + dvj2: # Check if cost is lower
                                # Now we have to check is partner node is valid or has to be inserted
                                # in a new position and if the obtained cost is lower

                                cust = state.nodes_to_cust[v]
                                if state.twc_format_nodes_df.loc[v, "type"] == "pickup":
                                    partner_node = state.cust_to_nodes[cust][1]
                                    goto = "after"
                                elif state.twc_format_nodes_df.loc[v, "type"] == "delivery":
                                    partner_node = state.cust_to_nodes[cust][0]
                                    goto = "before"
                                else:
                                    raise ValueError("Customer type is wrong.")

                                # Check if partner node is in the same route
                                if partner_node in nodes:
                                    partner_node_idx = nodes.index(partner_node)
                                    if goto == "after":
                                        if partner_node_idx > i2_idx+1:
                                            # Then partner delivery is correctly placed
                                            # after pickup node and a better solution is found
                                            
                                            removal_procedure(destroyed, cust)
                                            return remove_empty_routes(destroyed)
                                        
                                        else:
                                            # the position of partner node is not valid.
                                            # We need to find a valid position after the pickup node
                                            start_idx = i2_idx + 1
                                            end_idx = len(nodes2) - 1
                                            valid_position = None
                                            for idx in range(start_idx, end_idx):
                                                valid_position = find_valid_position(
                                                    state=state, start_idx=start_idx, end_idx=end_idx, route_idx=second_route_index, node_id=partner_node
                                                )
                                                if valid_position is None:
                                                    break
                                                else:
                                                    removal_procedure(destroyed, cust)
                                                    return remove_empty_routes(destroyed)
                                            # No valid position was possible, abort search for
                                            # customer v in this route and continue with the next route
                                            break
                                    elif goto == "before":
                                        if partner_node_idx < i2_idx:
                                            removal_procedure(destroyed, cust)
                                            return remove_empty_routes(destroyed)
                                        else:
                                            # the position of partner node is not valid.
                                            # We need to find a valid position before the pickup node
                                            start_idx = 1
                                            end_idx = i2_idx
                                            valid_position = None
                                            for idx in range(start_idx, end_idx):
                                                valid_position = find_valid_position(
                                                    state=state, start_idx=start_idx, end_idx=end_idx, route_idx=second_route_index, node_id=partner_node
                                                )
                                                if valid_position is None:
                                                    break
                                                else:
                                                    removal_procedure(destroyed, cust)
                                                    return remove_empty_routes(destroyed)
                                            # No valid position was possible, abort search for
                                            # customer v in this route and continue with the next route
                                            break

                            else:
                                logger.debug(f"No customer found to remove.")
                        # else:
                        # print(f"Time window check failed.")

    destroyed.update_unassigned_list()
    print(f"Finished")
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
        for i in route.nodes_list[1:-1]:
            j = route.nodes_list[route.nodes_list.index(i) - 1]
            k = route.nodes_list[route.nodes_list.index(i) + 1]
            service_cost = (
                state.distances[j][i] + state.distances[i][k] - state.distances[j][k]
            )
            if service_cost > max_service_cost:
                max_service_cost = service_cost
                worst_node = i
                worst_route = route_idx

    # Removes the customer associated with worst node
    worst_customer = state.nodes_to_cust[worst_node]
    removal_procedure(destroyed, worst_customer)
    return destroyed


def exchange_reducing_removal(
    state: CvrptwState, rng: np.random.Generator
) -> CvrptwState:
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

    iterations = 50
    twc_checks = 0
    for _ in range(iterations):
        for first_route_index in rng.choice(len(destroyed.routes), 1):
            route1 = destroyed.routes[first_route_index]
            customers = route1.nodes_list
            if len(customers) <= 2:
                break
            for v1 in customers[1:-1]:
                idx1 = customers.index(v1)
                i1 = customers[idx1 - 1]  # previous customer
                j1 = customers[idx1 + 1]  # next customer
                di1v1 = destroyed.distances[i1][v1]
                dv1j1 = destroyed.distances[v1][j1]
                di1j1 = destroyed.distances[i1][j1]

            for second_route_index in list(range(len(destroyed.routes))):
                route2 = destroyed.routes[second_route_index]
                customers2 = route2.nodes_list
                for v2 in customers2[1:-1]:
                    idx2 = customers2.index(v2)
                    if second_route_index == first_route_index and idx2 == idx1:
                        continue
                    i2 = customers2[idx2 - 1]
                    j2 = customers2[idx2 + 1]

                    # Check Time Window Compatibility
                    twc_checks += 1
                    if twc_checks > iterations:
                        break

                    if (
                        destroyed.twc[i2][v1] != -np.inf
                        and destroyed.twc[v1][j2] != np.inf
                    ):
                        if (
                            destroyed.twc[i1][v2] != -np.inf
                            and destroyed.twc[v2][j1] != np.inf
                        ):
                            di1v1 = destroyed.distances[i1][v1]
                            dv1j1 = destroyed.distances[v1][j1]
                            di2v2 = destroyed.distances[i2][v2]
                            dv2j2 = destroyed.distances[v2][j2]

                            di1v2 = destroyed.distances[i1][v2]
                            dv2j1 = destroyed.distances[v2][j1]
                            di2v1 = destroyed.distances[i2][v1]
                            dv1j2 = destroyed.distances[v1][j2]

                            if (
                                di1v1 + dv1j1 + di2v2 + dv2j2
                                > di1v2 + dv2j1 + di2v1 + dv1j2
                            ):
                                # swap v1 and v2
                                route1.nodes_list[idx1] = v2
                                route2.nodes_list[idx2] = v1
                                destroyed.update_times_attributes_routes(
                                    first_route_index
                                )
                                destroyed.update_times_attributes_routes(
                                    second_route_index
                                )
                                destroyed.update_unassigned_list()

                                return remove_empty_routes(destroyed)

    return remove_empty_routes(destroyed)


def find_valid_position(
    state: CvrptwState, start_idx: int, end_idx: int, route_idx: int, node_id: int
) -> int:
    """
    Find a valid position to insert a node in a route. The node is inserted after the
    start_idx and before the end_idx. The route is given by route_idx and the node by node_id.

    Parameters:
        state: CvrptwState
            The solution from which to remove customers.
        start_idx: int
            The index of the node after which the node is to be inserted.
        end_idx: int
            The index of the node before which the node is to be inserted.
        route_idx: int
            The index of the route in which the node is to be inserted.
        node_id: int
            The id of the node to be inserted.
    Returns:
        int
            The index of the node after which the node is to be inserted.
    """
    nodes = state.routes[route_idx].nodes_list
    for i in range(start_idx, end_idx):
        if (
            state.twc[nodes[i]][node_id] != -np.inf
            and state.twc[node_id][nodes[i + 1]] != -np.inf
        ):
            return i
    return None


def removal_procedure(state: CvrptwState, cust_id: int):
    """
    Performs the removal procedure as descibed by the documentation.

    Parameters:
        state: CvrptwState
            The solution from which to remove customers.
        cust_id: int
            The id of the customer to be removed.

    Returns:
        None
    """
    pickup_node, delivery_node = state.cust_to_nodes[cust_id]
    route, route_idx = state.find_route(cust_id)

    state.routes[route_idx].remove([pickup_node, delivery_node])
    state.unassigned.append(cust_id)
    state.cust_df.loc[cust_id, "route"] = None
    state.cust_df.loc[cust_id, "done"] = False
    if len(state.routes[route_idx]) != 2:
        state.update_times_attributes_routes(route_idx)
        state.routes_cost[route_idx] = state.route_cost_calculator(route_idx)
