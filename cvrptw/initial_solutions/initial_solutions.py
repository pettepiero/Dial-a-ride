from typing import List
import numpy as np
import pandas as pd
from cvrptw.myvrplib.data_module import (
    data,
    d_data,
    calculate_depots,
    get_initial_data,
    get_ids_of_time_slot,
)
from cvrptw.myvrplib.vrpstates import CvrptwState
from cvrptw.myvrplib.route import Route
from cvrptw.myvrplib.myvrplib import time_window_check, route_time_window_check

def neighbours(state: CvrptwState, customer: int) -> list:
    """
    Return the customers IDs in order of increasing distance from the given
    customer, excluding the depots.
        Parameters:
            state: Cvrptw
                The current state of the CVRPTW problem.s
            customer: int
                The customer whose neighbors are to be found.
        Returns:
            list
                The list of customers in order of increasing distance from the
                given customer, excluding the depots.
    """
    locations = np.argsort(state.distances[customer]).tolist()

    return [loc for loc in locations if loc not in state.depots["depots_indices"] and loc != customer]

def time_neighbours(state: CvrptwState, customer: int) -> list:
    """
    Return the customers in order of increasing start time after the given
    customer. Considers the customers that can be reached in time from the
    given customer. Therefore, this can exclude customers that request a 
    service later in the day but that are not reachable in time.
        Parameters:
            state: Cvrptw
                The current state of the CVRPTW problem.
            customer: int
                The customer whose neighbors are to be found.

        Returns:
            list
                The list of customers in order of increasing time from the
                given customer, excluding the depots.
    """
    current_start_time = state.nodes_df.loc[state.nodes_df['id'] == customer, "start_time"]
    current_start_time = current_start_time.item()

    locations = state.nodes_df[state.nodes_df['start_time'] > current_start_time][["id", "start_time"]].values.tolist()
    locations = [[loc, time] for loc, time in locations if state.distances[customer][loc] + current_start_time <= state.nodes_df.loc[state.nodes_df['id'] == loc, "end_time"].item()]
    locations = sorted(locations, key=lambda x: x[1])
    locations = [loc for loc in locations if loc != customer]

    return [loc[0] for loc in locations]

    # # assert customer not in depots, "Customer cannot be a depot"
    # locations = [(loc, state.nodes_df.loc[state.nodes_df['id'] == loc, "start_time"]) for loc in range(state.n_customers)]
    # locations = [loc for loc in locations if loc[0] not in state.depots["depots_indices"]]
    # #order by soonest start time after current customer
    # current_start_time = locations[customer][1]
    # # Filter customers from the past
    # locations = [loc for loc in locations if loc[1] > current_start_time]
    # # Filter reachable customers
    # locations = [loc for loc in locations if data["edge_weight"][customer][loc[0]] + current_start_time <= data["time_window"][loc[0]][1]]

    # locations = sorted(locations, key=lambda x: x[1])
    # locations = [loc for loc in locations if loc[0] != customer]

    # return [loc[0] for loc in locations]


def nearest_neighbor_tw(state: CvrptwState, cordeau:bool = True, initial_time_slot: bool = True) -> CvrptwState:
    """
    Build a solution by iteratively constructing routes, where the nearest
    time-window compatible customer is added until the route has met the
    vehicle capacity limit.
        Parameters:
            state: CvrptwState
                The current state of the CVRPTW problem.
            cordeau: bool
                If True, the Cordeau dataset notation is used, else the
                Solomon dataset notation is used.
            intial_time_slot: bool
                If True, only data related to customers that are called
                in at the initial time step are considered.
        Returns:
            CvrptwState
                The initial solution to the CVRPTW problem.
    """

    routes: list[Route] = []

    start_idx = 1 if cordeau else 0
    if initial_time_slot:
        valid_customers = get_ids_of_time_slot(state.nodes_df, 0)
        unvisited = valid_customers
    else:
        unvisited = set(range(start_idx, len(state.nodes_df["demand"])))

    vehicle = 0

    while vehicle < state.n_vehicles:
        initial_depot = state.depots["vehicle_to_depot"][vehicle]
        route = [initial_depot]
        route_schedule = [0]
        route_demands = 0
        while unvisited:
            # Add the nearest compatible unvisited customer to the route till max capacity
            current = route[-1]
            reachable = time_neighbours(
                state,
                current
            )  # Customers reachable in time
            if len(reachable) == 0:
                break
            nearest = [
                nb for nb in reachable if nb in unvisited
            ]  # Keep only unvisited customers
            if len(nearest) == 0:
                break
            nearest = int(nearest[0])  # Nearest unvisited reachable customer
            # Check vehicle capacity and time window constraints
            if route_demands + state.nodes_df[state.nodes_df["id"] == nearest]["demand"].item() > state.vehicle_capacity:
                break
            if not time_window_check(route_schedule[-1], current, nearest):
                break

            route.append(nearest)
            route_schedule.append(
                state.distances[current][nearest].item()
                + state.nodes_df[state.nodes_df['id'] == current]["service_time"].item()
            )

            unvisited.remove(nearest)
            route_demands += state.nodes_df[state.nodes_df["id"] == nearest][
                "demand"
            ].item()

        route.append(route[0])  # Return to the depot
        route = Route(route, vehicle)
        # route.calculate_planned_times()
        routes.append(route)
        # full_schedule.append(route_schedule)
        # Consider new vehicle
        vehicle += 1
        # if vehicle == full_data["vehicles"]:
        #     vehicle = 0

    # Assign routes to customers in nodes_df
    for route_num, route in enumerate(routes):
        for customer in route.customers_list:
            state.nodes_df.loc[state.nodes_df["id"] == customer, "route"] = route_num        

    if unvisited:
        print(f"#Unvisited customers after nearest neighbor solution: {len(unvisited)}")
    if vehicle < state.n_vehicles:
        print(f"Vehicles left: {state.n_vehicles - vehicle}")

    solution = CvrptwState(routes, nodes_df=state.nodes_df, given_unassigned=list(unvisited))
    for route_idx in range(len(solution.routes)):
        solution.update_times_attributes_routes(route_idx)
        for customer in solution.routes[route_idx].customers_list:
            solution.nodes_df.loc[solution.nodes_df["id"] == customer, "route"] = route_idx


    return solution
