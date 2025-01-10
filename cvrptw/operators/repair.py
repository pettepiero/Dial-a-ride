from data_module import data
from copy import deepcopy
import numpy as np
from myvrplib import END_OF_DAY, time_window_check, route_time_window_check
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
        # If possible, create a new route
        elif len(new_state.routes) < data["vehicles"]:
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
            new_state.times.append([0])
            # evaluate times of new route
            new_state.update_times_attributes_routes()
    
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
            route.insert(idx, customer)
            new_state.update_times()
        else:
            # Initialize a new route and corresponding timings
            # Check if the number of routes is less than the number of vehicles
            depot = data["vehicle_to_depot"][len(new_state.routes)]
            new_state.routes.append(
                Route(
                    [
                        depot(new_state.routes[-1], customer),
                        customer,
                        depot(new_state.routes[-1], customer),
                    ],
                    vehicle=len(new_state.routes),
                )
            )
            # new_state.routes.append([customer])
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
        for idx in range(1, len(route)-1):
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
        for idx in range(1, len(route)-1):

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
    total = data["demand"][route.customers_list].sum() + data["demand"][customer]
    if total > data["capacity"]:
        return False
    # Time window check
    if time_window_check(state.times[route_number][idx - 1], route.customers_list[idx - 1], customer):
        return route_time_window_check(route, state.times[route_number])
    return False

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
