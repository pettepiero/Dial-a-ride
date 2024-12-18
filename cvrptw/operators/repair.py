from data_module import data
import numpy as np
from myvrplib import END_OF_DAY
from vrpstates import CvrptwState
import logging
from route import Route


def greedy_repair(state, rng):
    """
    Inserts the unassigned customers in the best route. If there are no
    feasible insertions, then a new route is created. Only checks capacity constraints.
    """
    rng.shuffle(state.unassigned)

    while len(state.unassigned) != 0:
        customer = state.unassigned.pop()
        route, idx = best_insert(customer, state)

        if route is not None:
            route.insert(idx, customer)
            state.update_times()
        else:
            if len(state.routes) < data["vehicles"]:
                vehicle_number = len(state.routes)
                state.routes.append(
                    Route(
                        [
                            data["vehicle_to_depot"][vehicle_number],
                            customer,
                            data["vehicle_to_depot"][vehicle_number],
                        ]
                    )
                )
                state.update_times()  # NOTE: maybe not needed
    return state


def greey_repair_wang(state, rng):
    """
    Inserts the unassigned customers in the best route. If there are no
    feasible insertions, then a new route is created. Uses the Wang et al (2024)
    insertion heuristics with time window compatibility checks.
    """
    rng.shuffle(state.unassigned)

    while len(state.unassigned) != 0:
        customer = state.unassigned.pop()
        route, idx = best_insert_wang(customer, state)

        if route is not None:
            route.insert(idx, customer)
            state.update_times()
        else:
            if len(state.routes) < data["vehicles"]:
                vehicle_number = len(state.routes)
                state.routes.append(
                    Route(
                        [
                            data["vehicle_to_depot"][vehicle_number],
                            customer,
                            data["vehicle_to_depot"][vehicle_number],
                        ]
                    )
                )
                state.update_times()  # NOTE: maybe not needed
    return state


def greedy_repair_tw(state, rng):
    """
    Inserts the unassigned customers in the best route. If there are no
    feasible insertions, then a new route is created. Check capacity and time window constraints.
    """
    rng.shuffle(state.unassigned)

    while len(state.unassigned) != 0:
        customer = state.unassigned.pop()
        route, idx = best_insert_tw(customer, state)

        if route is not None:
            route.insert(idx, customer.item())
            state.update_times()
        else:
            # Initialize a new route and corresponding timings
            # Check if the number of routes is less than the number of vehicles
            if len(state.routes) < data["vehicles"]:
                state.routes.append([customer.item()])
                state.times.append([0])
                state.update_times()  # NOTE: maybe not needed
            # else:
            # print(f"Customer {customer} could not be inserted in any route. Maximum number of routes/vehicles reached.")

    return state


def best_insert_wang(customer, state: CvrptwState):
    """
    Finds the best feasible route and insertion idx for the customer.
    Return (None, None) if no feasible route insertions are found.
    Only checks capacity constraints.
    """
    best_cost, best_route, best_idx = None, None, None
    for route in enumerate(state.routes):
        for idx in range(1, len(route)):
            # if can_insert(customer, route_number, idx, state):
            if state.can_be_inserted_wang(customer, route, idx):
                cost = insert_cost(customer.item(), route.customers_list, idx)

                if best_cost is None or cost < best_cost:
                    best_cost, best_route, best_idx = cost, route, idx

    return best_route, best_idx


def best_insert(customer, state: CvrptwState):
    """
    Finds the best feasible route and insertion idx for the customer.
    Return (None, None) if no feasible route insertions are found.
    Only checks capacity constraints.
    """
    best_cost, best_route, best_idx = None, None, None

    for route_number, route in enumerate(state.routes):
        for idx in range(1, len(route)):
            if can_insert(customer, route_number, idx, state):
                cost = insert_cost(customer, route.customers_list, idx)

                if best_cost is None or cost < best_cost:
                    best_cost, best_route, best_idx = cost, route, idx

    return best_route, best_idx


def best_insert_tw(customer, state):
    """
    Finds the best feasible route and insertion idx for the customer.
    Return (None, None) if no feasible route insertions are found.
    Checks both capacity and time window constraints.
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
def can_insert(customer, route_number, idx, state):
    """
    Checks if inserting customer in route 'route_number' at position 'idx' does not exceed vehicle capacity.
    """
    route: Route = state.routes[route_number]

    # Capacity check
    total = data["demand"][route.customers_list].sum() + data["demand"][customer]
    if total > data["capacity"]:
        return False
    return True


# NOTE: I think performance can be improved by changing this function
def can_insert_tw(customer, route_number, idx, state):
    """
    Checks if inserting customer in route 'route_number' at position 'idx' does not exceed vehicle capacity and time window constraints.
    """

    route = state.routes[route_number]
    # Capacity check
    total = data["demand"][route].sum() + data["demand"][customer]
    if total > data["capacity"]:
        return False
    # Time window check
    if time_window_check(state.times[route_number][idx - 1], route[idx - 1], customer):
        return route_time_window_check(route, state.times[route_number])


def route_time_window_check(route, times):
    """
    Check if the route satisfies time-window constraints. Ignores the depots as
    they are considered available 24h.
    """
    route = route[1:]  # Ignore the depot
    for idx, customer in enumerate(route):
        if times[idx] > data["time_window"][customer][1]:
            return False

    return True


# NOTE: this is a terrible check.
# It will accept any customer whose time window is after the calculated arrival time,
# even if the vehicle is early.
# Is the vehicle allowd to be early?
def time_window_check(prev_customer_time, prev_customer, candidate_customer):
    """
    Check if the candidate customer satisfies time-window constraints. Ignores the depots as
    they are considered available 24h.
    """
    return (
        prev_customer_time
        + data["service_time"][prev_customer]
        + data["edge_weight"][prev_customer][candidate_customer]
        <= data["time_window"][candidate_customer][1]
    )


def insert_cost(customer, route, idx):
    """
    Computes the insertion cost for inserting customer in route at idx.
    """
    dist = data["edge_weight"]
    pred = 0 if idx == 0 else route[idx - 1]
    succ = 0 if idx == len(route) else route[idx]

    # Increase in cost of adding customer, minus cost of removing old edge
    return dist[pred][customer] + dist[customer][succ] - dist[pred][succ]
