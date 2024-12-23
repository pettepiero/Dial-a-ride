from typing import List
import numpy as np
from data_module import data, calculate_depots
from vrpstates import CvrptwState
from route import Route
from myvrplib import time_window_check, route_time_window_check


def neighbors(customer, depots: list = []):
    """
    Return the nearest neighbors of the customer, excluding the depot.
    """
    locations = np.argsort(data["edge_weight"][customer])
    return [loc for loc in locations if loc not in depots]


def nearest_neighbor_tw(cordeau:bool = True) -> CvrptwState:
    """
    Build a solution by iteratively constructing routes, where the nearest
    time-window compatible customer is added until the route has met the
    vehicle capacity limit.
    """
    routes: list[Route] = []
    full_schedule = []
    start_idx = 1 if cordeau else 0
    unvisited = set(range(start_idx, data["dimension"]))
    vehicle = 0

    # while unvisited and vehicle < data["vehicles"]:
    while unvisited:    
        initial_depot = data["vehicle_to_depot"][vehicle]
        route = [initial_depot]  # Start at the depot
        route_schedule = [0]
        route_demands = 0
        while unvisited:
            # Add the nearest unvisited customer to the route till max capacity
            current = route[-1]
            nearest = [
                nb
                for nb in neighbors(current, depots=data["depots"])
                if nb in unvisited
            ][0]
            nearest = int(nearest)
            if route_demands + data["demand"][nearest] > data["capacity"]:
                break
            if not time_window_check(route_schedule[-1], current, nearest):
                break
            if not route_time_window_check(route, route_schedule):
                break

            route.append(nearest)
            route_schedule.append(
                data["edge_weight"][current][nearest].item()
                + data["service_time"][current].item()
            )

            unvisited.remove(nearest)
            route_demands += data["demand"][nearest]

        route.append(route[0])  # Return to the depot
        route = Route(route, vehicle)
        routes.append(route)
        full_schedule.append(route_schedule)
        # Consider new vehicle
        print(f"vehicle: {vehicle}")
        vehicle += 1
        if vehicle == data["vehicles"]:
            vehicle = 0



    if unvisited:
        print(f"Unvisited customers after nearest neighbor solution: {unvisited}")
    if vehicle < data["vehicles"]:
        print(f"Vehicles left: {data['vehicles'] - vehicle}")

    return CvrptwState(routes, full_schedule, unassigned=list(unvisited))
