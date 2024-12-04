import vrplib

import copy
from types import SimpleNamespace

import numpy as np
import numpy.random as rnd
from alns.select import *


def random_removal(state, data, customers_to_remove, rng):
    """
    Removes a number of randomly selected customers from the passed-in solution.
    """
    destroyed = state.copy()

    for customer in rng.choice(
        range(1, data["dimension"]), customers_to_remove, replace=False
    ):
        destroyed.unassigned.append(customer)
        route = destroyed.find_route(customer)
        route.remove(customer)

    destroyed.update_times()

    return remove_empty_routes(destroyed)


def remove_empty_routes(state):
    """
    Remove empty routes and timings after applying the destroy operator.
    """
    state.routes = [route for route in state.routes if len(route) != 0]
    state.times = [
        timing for idx, timing in enumerate(state.times) if len(state.routes[idx]) != 0
    ]
    return state
