import numpy as np
from lib.myvrplib.myvrplib import (
    END_OF_DAY,
    time_window_check,
    route_time_window_check,
    LOGGING_LEVEL,
)
from lib.myvrplib.CVRPTWState import CVRPTWState
from lib.myvrplib.CVRPState import CVRPState
import logging
from lib.myvrplib.route import Route
from copy import deepcopy


logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)


def lowest_delta_position(state: CVRPState, customer: int, route_idx: int) -> tuple[float, int]:
    route = state.routes[route_idx]
    base_cost = state.route_cost_calculator(route_idx)

    best_delta = float("inf")
    best_pos = 0
    for pos in range(len(route) + 1):
        route.insert(pos, customer)
        delta = state.route_cost_calculator(route_idx) - base_cost
        route.remove(customer)
        if delta < best_delta:
            best_delta, best_pos = delta, pos
    return best_delta, best_pos

def regret3_insertion(state: CVRPState, rng: np.random.Generator) -> CVRPState:
    """
    Regret-k insertion with k=3, based on [WaSH24]
    """
    new_state = state.copy()
    while len(new_state.unassigned) > 0: #while there are customers to serve
        R = len(new_state.routes) # number of routes
        U = len(new_state.unassigned) # number of unassigned customers

        regrets          = np.full(U, -np.inf, dtype=float) # for each customer
        best_first_delta = np.full(U, np.inf, dtype=float)
        best_first_route = np.full(U, -1, dtype=int)
        best_first_pos   = np.full(U, -1, dtype=int)

        for ui, cust in enumerate(new_state.unassigned): # loop over unassigned
            deltas = np.full(R, np.inf, dtype=float) # for customer cust, prepare deltas  and pos list
            positions  = np.full(R, -1, dtype=int)

            for r in range(R): # loop over routes and store best position and delta
                delta, pos = lowest_delta_position(new_state, cust, r)
                deltas[r] = delta
                positions[r] = pos
                
            valid = np.isfinite(deltas) & (positions >= 0) #boolean mask
            if not np.any(valid):
                continue # this means no feasible solutions for customer cust

            valid_routes = np.nonzero(valid)[0] #feasible routes indices
            valid_deltas = deltas[valid] 
            valid_positions = positions[valid] 

            k = min(3, valid_deltas.size)
            sel = np.argpartition(valid_deltas, k-1)[:k]
            ordk = np.argsort(valid_deltas[sel])

            top_delta = valid_deltas[sel][ordk]
            top_route= valid_routes[sel][ordk]
            top_pos = valid_positions[sel][ordk]


            best_first_delta[ui] = float(top_delta[0])
            best_first_route[ui] = int(top_route[0])
            best_first_pos[ui] = int(top_pos[0])
            regrets[ui] = float(np.sum(top_delta[1:] - top_delta[:1]))

        if not np.isfinite(regrets).any():
            print(f"DEBUG: regrets")
            raise RuntimeError("no feasible insertion for any remaining customer")

        max_regret = np.max(regrets)
        cand = np.where(regrets == max_regret)[0]

        # Tie-breaker : random choice of cand if there is more than one
        pick_idx = int(cand[rng.integers(cand.size)])
        customer = new_state.unassigned[pick_idx]
        r_ins    = int(best_first_route[pick_idx])
        p_ins    = int(best_first_pos[pick_idx])
       
        if r_ins < 0 or p_ins < 0 or not np.isfinite(best_first_delta[pick_idx]):
            raise RuntimeError(f"regret3_insertion: chosen customer {customer} has no feasible insertion.")

        new_state.routes[r_ins].insert(p_ins, customer)
        new_state.unassigned.pop(pick_idx)
    new_state.update_attributes()

    return new_state




    

#def greedy_repair(state: CVRPState, rng: np.random, tw: bool = True) -> CVRPState:
#    if tw:
#        return greedy_repair_tw(state=state, rng=rng)
#    else:
#        return greedy_repair_no_tw(state=state, rng=rng)


def greedy_repair_no_tw(state: CVRPState, rng: np.random) -> CVRPState:
    """
    Inserts the unassigned customers in the best route. If there are no
    feasible insertions, then a new route is created. Only checks capacity constraints.
        Parameters:
            state: CVRPState
                The current solution state.
            rng: np.random
                The random number generator.
        Returns:
            CVRPState
                The repaired solution state.
    """
    new_state = state.copy()
    rng.shuffle(new_state.unassigned)

    while len(new_state.unassigned) != 0:
        customer = new_state.unassigned.pop()
        route_idx, idx = best_insert(customer, new_state)

        if route_idx is not None:
            new_state.routes[route_idx].insert(idx, customer)
            if isinstance(state, CVRPTWState):
                new_state.update_est_lst(route_idx)
                new_state.calculate_planned_times(route_idx)
            new_state.routes_cost[route_idx] = new_state.route_cost_calculator(route_idx)
        # If possible, create a new route
        elif len(new_state.routes) < state.n_vehicles:
            vehicle_number = len(new_state.routes)
            new_state.routes.append(
                    Route(
                        [
                            state.depots["vehicle_to_depot"],
                            customer,
                            state.depots["vehicle_to_depot"],
                        ]
                    )
            )
            # append to cost vector new cost
            new_state.routes_cost.append(new_state.route_cost_calculator(len(new_state.routes) - 1))
        # debug
    logger.debug("At the end of greedy repair:")
    [
        logger.debug(f"Route {idx}: {route.planned_windows}")
        for idx, route in enumerate(new_state.routes)
    ]
    #new_state.update_unassigned_list()
    new_state.update_attributes()

    return new_state


def greedy_repair_tw(state: CVRPTWState, rng: np.random) -> CVRPTWState:
    """
    Inserts the unassigned customers in the best route. If there are no
    feasible insertions, then a new route is created.
    Checks capacity and time window constraints.

    Parameters
    ----------
    state: CVRPTWState
        The current solution state.
    rng: np.random
        The random number generator.

    Returns
    -------
    CVRPTWState
        The repaired solution state.
    """
    new_state = state.copy()

    logger.debug(f"In greedy_repair_tw, before loop unassigned = {sorted(new_state.unassigned)}")
    logger.debug(f"Routes =")
    for idx, route in enumerate(new_state.routes):
        logger.debug(f"Route {idx}: {route.customers_list}")
    
    counter = 0
    n_unassigned = len(new_state.unassigned)
    rng.shuffle(new_state.unassigned)
    
    while counter < n_unassigned:
        counter += 1
        customer = new_state.unassigned.pop()
        route_idx, idx = best_insert_tw(customer, new_state)

        if route_idx is not None:
            new_state.routes[route_idx].insert(idx, customer)
            new_state.nodes_df.loc[customer, "route"] = route_idx
            new_state.update_times_attributes_routes(route_idx)
            new_state.routes_cost[route_idx] = new_state.route_cost_calculator(route_idx)
            new_state.compute_route_demand(route_idx)

            # Check if the number of routes is less than the number of vehicles
        elif len(new_state.routes) < state.n_vehicles:
            # Initialize a new route and corresponding timings
            depot = state.depots["vehicle_to_depot"][len(new_state.routes)]
            pw = deepcopy(state.routes[-1].planned_windows)
            pw.append([0, END_OF_DAY])
            new_state.routes.append(
                Route(
                    [depot, customer, depot],
                        # depot(new_state.routes[-1], customer),
                        # customer,
                        # depot(new_state.routes[-1], customer),
                    # ],
                    vehicle=len(new_state.routes),
                    planned_windows=pw
                    )
            )
            new_state.nodes_df.loc[customer, "route"] = len(new_state.routes) - 1
            # append to cost vector new cost
            new_state.routes_cost = np.append(
                new_state.routes_cost,
                new_state.route_cost_calculator(len(new_state.routes) - 1),
            )
            new_state.update_times_attributes_routes(len(new_state.routes) - 1)

        else:
            new_state.unassigned.insert(0, customer)

    #new_state.update_unassigned_list()
    new_state.update_attributes()
    return new_state


def best_insert(customer: int, state: CVRPState) -> tuple:
    """
    Finds the best feasible route and insertion idx for the customer.
    Return (None, None) if no feasible route insertions are found.
    Only checks capacity constraints.
        Parameters:
            customer: int
                The customer to be inserted.
            state: CVRPState
                The current solution state.
        Returns:
            tuple
                The best route and insertion indices for the customer.
    """
    best_cost, best_route_idx, best_idx = None, None, None

    for route_number, route in enumerate(state.routes):
        for idx in range(1, len(route)-1):
            if can_insert(customer, route_number, idx, state):
                cost = insert_cost(customer, route.customers_list, idx, state)

                if best_cost is None or cost < best_cost:
                    best_cost, best_route_idx, best_idx = cost, route_number, idx

    return best_route_idx, best_idx


def best_insert_tw(customer: int, state: CVRPTWState) -> tuple:
    """
    Finds the best feasible route and insertion idx for the customer.
    Return (None, None) if no feasible route insertions are found.
    Checks both capacity and time window constraints.
    Only checks capacity constraints.

    Parameters
    ----------
    customer: int
        The customer to be inserted.
    state: CVRPTWState
        The current solution state.

    Returns
    -------
    tuple
        The best route and insertion indices for the customer.
    """
    best_cost, best_route_idx, best_idx = None, None, None

    for route_number, route in enumerate(state.routes):
        for idx in range(1, len(route) - 1):
            if can_insert_tw(customer, route_number, idx, state):
                cost = insert_cost(customer, route.customers_list, idx, state)

                if best_cost is None or cost < best_cost:
                    best_cost, best_route_idx, best_idx = cost, route_number, idx

    return best_route_idx, best_idx


def can_insert(customer: int, route_number: int, idx: int, state: CVRPState) -> bool:
    """
    Checks if inserting customer in route 'route_number' at position 'idx'
    does not exceed vehicle capacity and time window constraints.

    Parameters
    ----------
    customer: int
        The customer to be inserted.
    route_number: int
        The route number.
    idx: int
        The insertion index.
    state: CVRPState
        The current solution state.

    Returns
    -------
    bool
        True if the insertion is feasible, False otherwise.
    """
    df = state.nodes_df
    route = state.routes[route_number]

    # Capacity check
    if route.demand is not None:
        total = route.demand + df.loc[customer, "demand"]
    else:
        sub_df = df[df["id"].isin(route.customers_list)]["demand"]
        total = sub_df.sum() + df.loc[customer, "demand"].item()
    if total > state.vehicle_capacity:
        return False

    return True 



# NOTE: I think performance can be improved by changing this function
# maybe insert total demand in route
def can_insert_tw(
    customer: int, route_number: int, idx: int, state: CVRPTWState
) -> bool:
    """
    Checks if inserting customer in route 'route_number' at position 'idx'
    does not exceed vehicle capacity and time window constraints.

    Parameters
    ----------
    customer: int
        The customer to be inserted.
    route_number: int
        The route number.
    idx: int
        The insertion index.
    state: CVRPTWState
        The current solution state.

    Returns
    -------
    bool
        True if the insertion is feasible, False otherwise.
    """
    df = state.nodes_df
    route = state.routes[route_number]

    # Capacity check
    if route.demand is not None:
        total = route.demand + df.loc[customer, "demand"]
    else:
        sub_df = df[df["id"].isin(route.customers_list)]["demand"]
        total = sub_df.sum() + df.loc[customer, "demand"].item()
    if total > state.vehicle_capacity:
        return False

    previous_customer = route.customers_list[idx - 1]

    # Time window check
    if time_window_check(
        prev_customer_time=route.planned_windows[idx - 1][0],
        prev_service_time=df.loc[previous_customer, "service_time"].item(),
        edge_time=state.distances[previous_customer][customer],
        candidate_end_time=df.loc[customer, "end_time"].item()):
        return route_time_window_check(state, route, idx)
    return False

def insert_cost(customer: int, route: list, idx: int, state: CVRPState) -> float:
    """
    Computes the insertion cost for inserting customer in route at idx.

    Parameters
    ----------
    customer: int
        The customer to be inserted.
    route: list
        The route where the customer is to be inserted.
    idx: int
        The insertion index.
    state: CVRPState
        The current solution state.

    Returns
    -------
    float
        The insertion cost.
    """
    dist = state.distances
    pred = 0 if idx == 0 else route[idx - 1]
    succ = 0 if idx == len(route) else route[idx]

    # Increase in cost of adding customer, minus cost of removing old edge
    return dist[pred][customer] + dist[customer][succ] - dist[pred][succ]
