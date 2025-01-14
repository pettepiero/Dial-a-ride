from data_module import data
import numpy as np
from myvrplib import END_OF_DAY, time_window_check, route_time_window_check, LOGGING_LEVEL
from vrpstates import CvrptwState
import logging
from route import Route
from copy import deepcopy


logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)

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
    # debug
    logger.debug("At the beginning of greedy repair:")
    [
        logger.debug(f"Route {idx}: {route.planned_windows}")
        for idx, route in enumerate(state.routes)
    ]

    new_state = state.copy()
    rng.shuffle(new_state.unassigned)

    while len(new_state.unassigned) != 0:
        customer = new_state.unassigned.pop()
        route_idx, idx = best_insert(customer, new_state)

        if route_idx is not None:
            new_state.routes[route_idx].insert(idx, customer)
            new_state.routes[route_idx].calculate_planned_times()
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
        # debug
    logger.debug("At the end of greedy repair:")
    [
        logger.debug(f"Route {idx}: {route.planned_windows}")
        for idx, route in enumerate(new_state.routes)
    ]

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
    new_state = state.copy()
    rng.shuffle(new_state.unassigned)

    counter = 0
    n_unassigned = len(new_state.unassigned)
    while counter < n_unassigned:
        counter += 1
        customer = new_state.unassigned.pop()
        route_idx, idx = best_insert_tw(customer, new_state)

        if route_idx is not None:
            new_state.routes[route_idx].insert(idx, customer)
            new_state.update_times_attributes_routes()
            
            # Check if the number of routes is less than the number of vehicles
        elif len(new_state.routes) < data["vehicles"]:
            # Initialize a new route and corresponding timings
            depot = data["vehicle_to_depot"][len(new_state.routes)]
            new_state.routes.append(
                Route(
                    [depot, customer, depot],
                        # depot(new_state.routes[-1], customer),
                        # customer,
                        # depot(new_state.routes[-1], customer),
                    # ],
                    vehicle=len(new_state.routes),
                    planned_windows=deepcopy(state.routes[-1].planned_windows.append([0, END_OF_DAY]))
                    )
            )
            new_state.update_times_attributes_routes()
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
                The best route and insertion indices for the customer.
    """
    best_cost, best_route_idx, best_idx = None, None, None

    for route_number, route in enumerate(state.routes):
        for idx in range(1, len(route)-1):
            if can_insert(customer, route_number, idx, state):
                cost = insert_cost(customer, route.customers_list, idx)

                if best_cost is None or cost < best_cost:
                    best_cost, best_route_idx, best_idx = cost, route_number, idx

    return best_route_idx, best_idx


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
                The best route and insertion indices for the customer.
    """
    best_cost, best_route_idx, best_idx = None, None, None

    for route_number, route in enumerate(state.routes):
        for idx in range(1, len(route) - 1):
            if can_insert_tw(customer, route_number, idx, state):
                cost = insert_cost(customer, route.customers_list, idx)

                if best_cost is None or cost < best_cost:
                    best_cost, best_route_idx, best_idx = cost, route_number, idx

    return best_route_idx, best_idx


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
    
    #debug
    # print(f"route_number = {route_number}, idx = {idx}")
    # print(f"len(route) = {len(route)}, len(route.planned_windows) = {len(route.planned_windows)}")
    # print(f"len(route.customers_list) = {len(route.customers_list)}")
    # print(f"route.planned_windows[idx - 1]= {route.planned_windows[idx - 1]}")


    # Time window check
    if time_window_check(route.planned_windows[idx - 1][0], route.customers_list[idx - 1], customer):
        return route_time_window_check(route, idx)
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
