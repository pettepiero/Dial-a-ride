import copy
from types import SimpleNamespace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd

from alns import ALNS
from alns.accept import RecordToRecordTravel
from alns.accept import HillClimbing
from alns.select import RouletteWheel
from alns.stop import MaxIterations

from repair import time_window_check

def get_customer_info(data, idx):
    """
    Get the customer information for the passed-in index.
    """
    return {
        "index": idx,
        "coords": data["node_coord"][idx],
        "demand": data["demand"][idx],
        "ready_time": data["time_window"][idx, 0],
        "due_time": data["time_window"][idx, 1],
        "service_time": data["service_time"][idx],
    }


# NOTE: maybe add time influence on cost of solution ?
def route_cost(data, route):
    """
    Compute the cost of a route.
    """
    distances = data["edge_weight"]
    tour = [0] + route + [0]

    return sum(distances[tour[idx]][tour[idx + 1]] for idx in range(len(tour) - 1))


class CvrptwState:
    """
    Solution state for CVRPTW. It has three data members: routes, times and unassigned.
    Routes is a list of list of integers, where each inner list corresponds to
    a single route denoting the sequence of customers to be visited. A route
    does not contain the start and end depot. Times is a list of list, containing
    the planned arrival times for each customer in the routes. The outer list
    corresponds to the routes, and the inner list corresponds to the customers of
    the route. Unassigned is a list of integers, each integer representing an
    unassigned customer.
    """

    def __init__(self, routes, times, unassigned=None):
        self.routes = routes
        self.times = times  # planned arrival times for each customer
        self.unassigned = unassigned if unassigned is not None else []

    def copy(self):
        return CvrptwState(
            copy.deepcopy(self.routes), self.times.copy(), self.unassigned.copy()
        )

    def objective(self, data):
        """
        Computes the total route costs.
        """
        return sum(route_cost(data, route) for route in self.routes)

    @property
    def cost(self, data):
        """
        Alias for objective method. Used for plotting.
        """
        return self.objective(data)

    def find_route(self, customer):
        """
        Return the route that contains the passed-in customer.
        """
        for route in self.routes:
            if customer in route:
                return route

        raise ValueError(f"Solution does not contain customer {customer}.")

    def update_times(self):
        """
        Update the arrival times of each customer in the routes.
        """
        for idx, route in enumerate(self.routes):
            self.times[idx] = self.evaluate_times_of_route(route)

    def evaluate_times_of_route(self, data, route):
        """
        Update the arrival times of each customer in a given route.
        """
        timings = [0]
        current_position = 0
        for customer in route:
            movement_time = data["edge_weight"][current_position][customer]
            # add the service time of the last customer
            timings.append(timings[-1] + data["service_time"][current_position])
            # add the movement time to reach next customer
            timings[-1] = float(timings[-1] + movement_time)

        return timings

    def print_state_dimensions(self):
        """
        Print the dimensions of the state.
        """
        print(f"Number of routes: {len(self.routes)}")
        print(f"Length of routes: {[len(route) for route in self.routes]}")
        print(f"Dimensions of times: {[len(route) for route in self.times]}")
        print(f"Number of unassigned customers: {len(self.unassigned)}")
        print(
            f"Number of customers in routes: {sum(len(route) for route in self.routes)}"
        )


def neighbors(data, customer):
    """
    Return the nearest neighbors of the customer, excluding the depot.
    """
    locations = np.argsort(data["edge_weight"][customer])
    return locations[locations != 0]


def nearest_neighbor_tw(data):
    """
    Build a solution by iteratively constructing routes, where the nearest
    time-window compatible customer is added until the route has met the
    vehicle capacity limit.
    """
    routes = []
    full_schedule = []
    unvisited = set(range(1, data["dimension"]))

    while unvisited:
        route = [0]  # Start at the depot
        route_schedule = [0]
        route_demands = 0

        while unvisited:
            # Add the nearest unvisited customer to the route till max capacity
            current = route[-1]
            nearest = [nb for nb in neighbors(data, current) if nb in unvisited][0]
            nearest = int(nearest)

            if route_demands + data["demand"][nearest] > data["capacity"]:
                break

            if not time_window_check(data, route_schedule[-1], current, nearest):
                break

            route.append(nearest)

            route_schedule.append(
                data["edge_weight"][current][nearest].item()
                + data["service_time"][current].item()
            )

            unvisited.remove(nearest)
            route_demands += data["demand"][nearest]

        customers = route[1:]  # Remove the depot
        routes.append(customers)
        full_schedule.append(route_schedule)

    return CvrptwState(routes, full_schedule)
