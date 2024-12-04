import vrplib

import copy
from types import SimpleNamespace

import numpy as np
import numpy.random as rnd
from alns.select import *
from alns.accept import *


def greedy_repair(data, state, rng):
    """
    Inserts the unassigned customers in the best route. If there are no
    feasible insertions, then a new route is created.
    """
    rng.shuffle(state.unassigned)

    while len(state.unassigned) != 0:
        customer = state.unassigned.pop()
        route, idx = best_insert(data, customer, state)

        if route is not None:
            route.insert(idx, customer.item())
            state.update_times()
        else:
            # Initialize a new route and corresponding timings
            state.routes.append([customer.item()])
            state.times.append([0])
            state.update_times()  # NOTE: maybe not needed

    return state


def best_insert(data, customer, state):
    """
    Finds the best feasible route and insertion idx for the customer.
    Return (None, None) if no feasible route insertions are found.
    """
    best_cost, best_route, best_idx = None, None, None

    for route_number, route in enumerate(state.routes):
        for idx in range(len(route) + 1):
            # DEBUG
            # print(f"In best_insert: route_number = {route_number}, idx = {idx}")
            if can_insert(data, customer, route_number, idx, state):
                cost = insert_cost(data, customer, route, idx)

                if best_cost is None or cost < best_cost:
                    best_cost, best_route, best_idx = cost, route, idx

    return best_route, best_idx


# NOTE: I think performance can be improved by changing this function
def can_insert(data, customer, route_number, idx, state):
    """
    Checks if inserting customer in route 'route_number' at position 'idx' does not exceed vehicle capacity and time window constraints.
    """

    route = state.routes[route_number]
    # Capacity check
    total = data["demand"][route].sum() + data["demand"][customer]
    if total > data["capacity"]:
        return False
    # Time window check
    return time_window_check(
        data, state.times[route_number][idx - 1], route[idx - 1], customer
    )


# NOTE: this is a terrible check.
# It will accept any customer whose time window is after the calculated arrival time,
# even if the vehicle is early.
# Is the vehicle allowd to be early?
def time_window_check(data, prev_customer_time, prev_customer, candidate_customer):
    """
    Check if the candidate customer satisfies time-window constraints.
    """

    return (
        prev_customer_time
        + data["service_time"][prev_customer]
        + data["edge_weight"][prev_customer][candidate_customer]
        <= data["time_window"][candidate_customer][1]
    )

def insert_cost(data, customer, route, idx):
    """
    Computes the insertion cost for inserting customer in route at idx.
    """
    dist = data["edge_weight"]
    pred = 0 if idx == 0 else route[idx - 1]
    succ = 0 if idx == len(route) else route[idx]

    # Increase in cost of adding customer, minus cost of removing old edge
    return dist[pred][customer] + dist[customer][succ] - dist[pred][succ]
