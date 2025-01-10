from typing import List
import numpy as np
from data_module import data, calculate_depots
from vrpstates import CvrptwState
from route import Route
from myvrplib import time_window_check, route_time_window_check


def neighbours(customer: int, depots: list = []) -> list:
    """
    Return the customers in order of increasing distance from the given
    customer, excluding the depots.
        Parameters:
            customer: int
                The customer whose neighbors are to be found.
            depots: list
                The list of depot locations.
        Returns:
            list
                The list of customers in order of increasing distance from the
                given customer, excluding the depots.
    """
    locations = np.argsort(data["edge_weight"][customer])
    return [loc for loc in locations if loc not in depots and loc != customer]

def time_neighbours(customer: int, depots: list = []) -> list:
    """
    Return the customers in order of increasing start time from the given
    customer. Considers the customers that can be reached in time from the
    given customer. Therefore, this can exclude customers that request a 
    service later in the day but that are not reachable in time.
        Parameters:
            customer: int
                The customer whose neighbors are to be found.
            depots: list
                The list of depot locations.
        Returns:
            list
                The list of customers in order of increasing time from the
                given customer, excluding the depots.
    """
    # assert customer not in depots, "Customer cannot be a depot"
    locations = [(loc, data["time_window"][loc][0]) for loc in range(len(data["time_window"]))]
    #order by soonest start time after current customer
    current_start_time = locations[customer][1]
    # Filter customers from the past
    locations = [loc for loc in locations if loc[1] > current_start_time]
    # Filter reachable customers
    locations = [loc for loc in locations if data["edge_weight"][customer][loc[0]] + current_start_time <= data["time_window"][loc[0]][1]]

    locations = sorted(locations, key=lambda x: x[1])
    locations = [loc for loc in locations if loc[0] != customer]

    return [loc[0] for loc in locations]


def nearest_neighbor_tw(cordeau:bool = True) -> CvrptwState:
    """
    Build a solution by iteratively constructing routes, where the nearest
    time-window compatible customer is added until the route has met the
    vehicle capacity limit.
        Parameters:
            cordeau: bool
                If True, the Cordeau dataset notation is used, else the
                Solomon dataset notation is used.
        Returns:
            CvrptwState
                The initial solution to the CVRPTW problem.
    """
    routes: list[Route] = []
    # full_schedule = []
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
            # Add the nearest compatible unvisited customer to the route till max capacity
            current = route[-1]
            reachable = time_neighbours(current, depots=data["depots"])
            if len(reachable) == 0:
                break
            nearest = [nb for nb in reachable if nb in unvisited]
            if len(nearest) == 0:
                break
            nearest = nearest[0]
            nearest = int(nearest)
            if route_demands + data["demand"][nearest] > data["capacity"]:
                break
            if not time_window_check(route_schedule[-1], current, nearest):
                break
            # if not route_time_window_check(route, route_schedule):
            #     break

            route.append(nearest)
            route_schedule.append(
                data["edge_weight"][current][nearest].item()
                + data["service_time"][current].item()
            )

            unvisited.remove(nearest)
            route_demands += data["demand"][nearest]

        route.append(route[0])  # Return to the depot
        route = Route(route, vehicle)
        route.calculate_planned_times()
        routes.append(route)
        # full_schedule.append(route_schedule)
        # Consider new vehicle
        vehicle += 1
        if vehicle == data["vehicles"]:
            vehicle = 0

    if unvisited:
        print(f"Unvisited customers after nearest neighbor solution: {unvisited}")
    if vehicle < data["vehicles"]:
        print(f"Vehicles left: {data['vehicles'] - vehicle}")

    solution = CvrptwState(routes, unassigned=list(unvisited))
    solution.update_times_attributes_routes()

    return solution
