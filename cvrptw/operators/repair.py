from data_module import data
from copy import deepcopy
import numpy as np
from myvrplib import END_OF_DAY
from vrpstates import CvrptwState
import logging
from route import Route


def greedy_repair(state: CvrptwState, rng: np.random) -> CvrptwState:
    """
    Inserts the unassigned customers in the best route. If there are no
    feasible insertions, then a new route is created. Only checks capacity constraints.
        Parameters:
            state: CvrptwState
                The current solution state.
            rng: np.random
                The random number generator.
        Returns:
            CvrptwState
                The repaired solution state.
    """
    new_state = deepcopy(state)
    rng.shuffle(new_state.unassigned)

    while len(new_state.unassigned) != 0:
        customer = new_state.unassigned.pop()
        route, idx = best_insert(customer, new_state)

        if route is not None:
            route.insert(idx, customer)
            new_state.update_times()
        else:
            if len(new_state.routes) < data["vehicles"]:
                vehicle_number = len(new_state.routes)
                new_state.routes.append(
                    Route(
                        [
                            data["vehicle_to_depot"][vehicle_number],
                            customer,
                            data["vehicle_to_depot"][vehicle_number],
                        ]
                    )
                )
                new_state.update_times()  # NOTE: maybe not needed
    return new_state


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
    new_state = deepcopy(state)
    rng.shuffle(new_state.unassigned)

    while len(new_state.unassigned) != 0:
        customer = new_state.unassigned.pop()
        route, idx = best_insert_tw(customer, new_state)

        if route is not None:
            route.insert(idx, customer.item())
            new_state.update_times()
        else:
            # Initialize a new route and corresponding timings
            # Check if the number of routes is less than the number of vehicles
            if len(new_state.routes) < data["vehicles"]:
                new_state.routes.append([customer.item()])
                new_state.times.append([0])
                new_state.update_times()  # NOTE: maybe not needed
            # else:
            # print(f"Customer {customer} could not be inserted in any route. Maximum number of routes/vehicles reached.")
    return new_state


def best_insert(customer: int, state: CvrptwState) -> tuple:
    """
    Finds the best feasible route and insertion idx for the customer.
    Return (None, None) if no feasible route insertions are found.
    Only checks capacity constraints.
        Parameters:
            customer: int
                The customer to be inserted.
            state: CvrptwState
                The current solution state.
        Returns:
            tuple
                The best route (Route) and insertion idx (int).
    """
    best_cost, best_route, best_idx = None, None, None

    for route_number, route in enumerate(state.routes):
        for idx in range(1, len(route)):
            if can_insert(customer, route_number, idx, state):
                cost = insert_cost(customer, route.customers_list, idx)

                if best_cost is None or cost < best_cost:
                    best_cost, best_route, best_idx = cost, route, idx

    return best_route, best_idx


def best_insert_tw(customer: int, state: CvrptwState) -> tuple:
    """
    Finds the best feasible route and insertion idx for the customer.
    Return (None, None) if no feasible route insertions are found.
    Checks both capacity and time window constraints.
    Only checks capacity constraints.
        Parameters:
            customer: int
                The customer to be inserted.
            state: CvrptwState
                The current solution state.
        Returns:
            tuple
                The best route (Route) and insertion idx (int).
    """
    best_cost, best_route, best_idx = None, None, None

    for route_number, route in enumerate(state.routes):
        # for idx in range(1, len(route) + 1):
        for idx in range(1, len(route)):

            # DEBUG
            # print(f"In best_insert: route_number = {route_number}, idx = {idx}")
            if can_insert_tw(customer, route_number, idx, state):
                cost = insert_cost(customer, route.customers_list, idx)

                if best_cost is None or cost < best_cost:
                    best_cost, best_route, best_idx = cost, route, idx

    return best_route, best_idx


# NOTE: I think performance can be improved by changing this function
def can_insert(customer: int, route_number: int, idx: int, state: CvrptwState) -> bool:
    """
    Checks if inserting customer in route 'route_number' at position 'idx' does not 
    exceed vehicle capacity.
        Parameters:
            customer: int
                The customer to be inserted.
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
    route: Route = state.routes[route_number]

    # Capacity check
    total = data["demand"][route.customers_list].sum() + data["demand"][customer]
    if total > data["capacity"]:
        return False
    return True


# NOTE: I think performance can be improved by changing this function
def can_insert_tw(
    customer: int, route_number: int, idx: int, state: CvrptwState
) -> bool:
    """
    Checks if inserting customer in route 'route_number' at position 'idx'
    does not exceed vehicle capacity and time window constraints.
        Parameters:
            customer: int
                The customer to be inserted.
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

    route = state.routes[route_number]
    # Capacity check
    total = data["demand"][route].sum() + data["demand"][customer]
    if total > data["capacity"]:
        return False
    # Time window check
    if time_window_check(state.times[route_number][idx - 1], route[idx - 1], customer):
        return route_time_window_check(route, state.times[route_number])


def route_time_window_check(route: Route, times: list) -> bool:
    """
    Check if the route satisfies time-window constraints. Ignores the depots as
    they are considered available 24h. Depots are first and last elements
    according to Cordeau notation.
        Parameters:
            route: Route
                The route to be checked.
            times: list
                The arrival times of the customers in the route.
        Returns:
            bool
                True if the route satisfies time-window constraints, False otherwise.
    """
    route = route[1:-1]  # Ignore the depot
    for idx, customer in enumerate(route):
        if times[idx] > data["time_window"][customer][1]:
            return False

    return True


# NOTE: this is a terrible check.
# It will accept any customer whose time window is after the calculated arrival time,
# even if the vehicle is early.
# Is the vehicle allowed to be early?
def time_window_check(prev_customer_time: float, prev_customer: int, candidate_customer: int):
    """
    Check if the candidate customer satisfies time-window constraints. Ignores the depots as
    they are considered available 24h.
        Parameters:
            prev_customer_time: float
                The arrival time of the previous customer.
            prev_customer: int
                The previous customer.
            candidate_customer: int
                The candidate customer.
        Returns:
            bool
                True if the candidate customer satisfies time-window constraints, False otherwise.
    """
    return (
        prev_customer_time
        + data["service_time"][prev_customer]
        + data["edge_weight"][prev_customer][candidate_customer]
        <= data["time_window"][candidate_customer][1]
    )


def insert_cost(customer: int, route: list, idx: int) -> float:
    """
    Computes the insertion cost for inserting customer in route at idx.
        Parameters:
            customer: int
                The customer to be inserted.
            route: list
                The route where the customer is to be inserted.
            idx: int
                The insertion index.
        Returns:
            float
                The insertion cost.
    """
    dist = data["edge_weight"]
    pred = 0 if idx == 0 else route[idx - 1]
    succ = 0 if idx == len(route) else route[idx]

    # Increase in cost of adding customer, minus cost of removing old edge
    return dist[pred][customer] + dist[customer][succ] - dist[pred][succ]
