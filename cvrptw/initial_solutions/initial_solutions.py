from typing import List
import numpy as np
import pandas as pd
from cvrptw.myvrplib.data_module import get_ids_of_time_slot, dynamic_extended_df
from cvrptw.myvrplib.vrpstates import CvrptwState
from cvrptw.myvrplib.route import Route
from cvrptw.myvrplib.myvrplib import time_window_check

# TODO: modify this to make it work for pickup and delivery
# def neighbours(state: CvrptwState, customer: int) -> list:
#     """
#     Return the customers IDs in order of increasing distance from the given
#     customer, excluding the depots.
#         Parameters:
#             state: Cvrptw
#                 The current state of the CVRPTW problem.s
#             customer: int
#                 The customer whose neighbors are to be found.
#         Returns:
#             list
#                 The list of customers in order of increasing distance from the
#                 given customer, excluding the depots.
#     """
#     locations = np.argsort(state.distances[customer]).tolist()

#     return [loc for loc in locations if loc not in state.depots["depots_indices"] and loc != customer]

def pick_up_time_neighbours(state: CvrptwState, customer: int, df: pd.DataFrame) -> list:
    """
    Return the customers in order of increasing start time after the given
    customer. Considers the customers that can be reached in time from the
    given customer, based only on the pickup time window. Therefore, this 
    can exclude customers that request a service later in the day but that 
    are not reachable in time.
        Parameters:
            state: Cvrptw
                The current state of the CVRPTW problem.
            customer: int
                The customer whose neighbors are to be found.
            df: pd.DataFrame
                The dataframe containing the nodes. Format is the same as the
                one returned by the dynamic_extended_df function.

        Returns:
            list
                The list of customers in order of increasing time from the
                given customer, excluding the depots.
    """
    cust_pickup_node = state.cust_to_nodes[customer][0]

    current_start_time = df.loc[
        df["node_id"] == cust_pickup_node, "start_time"
    ].values[0]
    nodes = df[df['start_time'] > current_start_time][["node_id", "start_time"]].values.tolist()
    nodes = [
        [node, time]
        for node, time in nodes
        if state.distances[cust_pickup_node][node] + current_start_time
        <= df.loc[df["node_id"] == node, "end_time"].values[0]
    ]   # Pickup nodes that are reachable in time
    nodes = sorted(nodes, key=lambda x: x[1]) # sort and drop start times 
    nodes = [node for node in nodes if node != cust_pickup_node] # drop the customer itself

    # locations = state.cust_df[state.cust_df['pstart_time'] > current_start_time][["id", "pstart_time"]].values.tolist()
    # locations = [[loc, time] for loc, time in locations if state.distances[customer][loc] + current_start_time <= state.cust_df.loc[loc, "pend_time"].item()]
    # locations = sorted(locations, key=lambda x: x[1])
    # locations = [loc for loc in locations if loc != customer]

    return [node[0] for node in nodes]

def close_route(route: list, route_schedule: list, state: CvrptwState, df: pd.DataFrame):
    current = route[-1]
    current_time = route_schedule[-1]
    route.append(route[0])
    dist = state.distances[current][route[0]].item()
    current_service_time = df.loc[current, "service_time"].item()
    # add to schedule sum of current time, service time and travel time
    planned_time = current_time + current_service_time + dist
    route_schedule.append(planned_time)
    return route, route_schedule

def nearest_neighbor(
    state: CvrptwState, cordeau: bool = True, initial_time_slot: bool = True
) -> CvrptwState:

    routes: list[Route] = []
    # filter only the requests that are visible at the initial time slot
    state.cust_df = state.cust_df[state.cust_df["call_in_time_slot"] == 0]
    df = dynamic_extended_df(state.cust_df)
    print(f"DEBUG: dynamic df:\n{df}")
    if initial_time_slot:
        visible_customers = get_ids_of_time_slot(state.cust_df, 0)
    else:
        AssertionError("Not implemented yet")

    # Extend to include both pick up and delivery nodes
    visible_nodes = [
        node for cust in visible_customers for node in state.cust_to_nodes[cust]
    ]
    unvisited_nodes = visible_nodes.copy()

    # These will be used intersecting unvisited_nodes
    pick_up_nodes = [state.cust_to_nodes[cust][0] for cust in visible_customers] 
    delivery_nodes = [state.cust_to_nodes[cust][1] for cust in visible_customers]

    remaining_pick_up_nodes = pick_up_nodes.copy()

    # If there are available vehicles, create a new route
    while len(routes) < state.n_vehicles and remaining_pick_up_nodes:
        # Initialize route
        initial_depot_id = state.depots["vehicle_to_depot"][len(routes)] # depot for the current vehicle
        depot_node = state.cust_to_nodes[initial_depot_id][0] # depot node
        route = [depot_node]
        route_schedule = []
        vehicle_load = 0
        n = 3 # how many new unvisited customers to consider at each iteration
        # this is a parameter that can be tuned

        # add nearest from list of pickup nodes (only for first customer)
        candidates = [
            (node, float(state.distances.get((depot_node, node), float('inf'))))
            for node in remaining_pick_up_nodes
        ]
        candidates = sorted(candidates, key=lambda x: x[1])  # sort by distance
        candidates = [(node, distance) for node, distance in candidates if node != depot_node]
        print(f"DEBUG: candidates = {candidates}")
        nearest_node, nearest_dist = candidates[0]
        nearest_stop = state.nodes_to_stop[nearest_node]

        if nearest_stop is not None:
            route.append(nearest_stop)
            print(f"DEBUG: route = {route}")
            requests_of_nn = df.loc[
                nearest_stop
            ]  # all the requests of the nearest neighbor
            if isinstance(requests_of_nn, pd.DataFrame): # more than one request for node 'nearest_stop' is present, consider only the first one
                # only consider the first one
                start_time, load, node_type = requests_of_nn.iloc[0][["start_time", "demand", "type"]].values.tolist()
            elif isinstance(requests_of_nn, pd.Series): # only one request for node 'nearest_stop' is present
                start_time, load, node_type = df.loc[
                    nearest_stop, ["start_time", "demand", "type"]
                ].values.tolist()
            # add to schedule 'departure from depot' time and 'arrival at customer' time
            route_schedule.append(start_time - nearest_dist) #departure from depot
            route_schedule.append(start_time) # arrival at customer
            vehicle_load += load
            assert vehicle_load <= state.vehicle_capacity, f"Vehicle load is \
                    bigger than allowed: {vehicle_load}"
            print(f"DEBUG: remaining_pick_up_nodes = {remaining_pick_up_nodes}")    
            print(f"DEBUG: nearest_stop = {nearest_stop} -> node = {state.stop_id_to_node[nearest_stop]}")
            remaining_pick_up_nodes.remove(nearest_stop)
            if node_type == "pickup":
                unvisited_nodes.remove(nearest_stop)
        else:
            AssertionError("Could not begin a new route")

        # Begin the loop of the heuristic
        while remaining_pick_up_nodes and n > 0:
            current = route[-1]
            # Update candidate
            candidates = [
                (node, float(distance))
                for node, distance in zip(
                    remaining_pick_up_nodes, state.distances[current][remaining_pick_up_nodes]
                )
            ]

            # Add delivery nodes of planned pickups, with dist from current node to candidates
            for planned_node in route:
                if df.loc[planned_node, "type"] == "pick_up":
                    customer_id = df.loc[planned_node, "cust_id"].item()
                    delivery_node = state.cust_to_nodes[customer_id][1]
                    distance = state.distances[current][delivery_node]
                    candidate_tuple = (delivery_node, distance)
                    if candidate_tuple not in candidates:
                        candidates.append(candidate_tuple)

            candidates = sorted(candidates, key=lambda x: x[1])  # sort by distance 
            # Add nearest node of candidates to route
            nearest_node, nearest_dist = candidates[0]
            if nearest_node is not None:
                route.append(nearest_node)
                start_time, demand, node_type = df.loc[nearest_node, ["start_time", "demand", "type"]].values.tolist()
                current_time = route_schedule[-1]
                current_service_time = df.loc[current, "service_time"].item()
                # add to schedule sum of current time, service time and travel time
                planned_time = current_time + current_service_time + nearest_dist
                route_schedule.append(planned_time)
                # update vehicle load
                # TODO: add proper check for vehicle capacity
                if node_type == "pickup":
                    vehicle_load += demand
                    assert vehicle_load <= state.vehicle_capacity, f"Vehicle load is \
                          bigger than allowed: {vehicle_load}"
                    unvisited_nodes.remove(nearest_node)
                elif node_type == "delivery":
                    vehicle_load -= df.loc[nearest_node, "demand"].item()
                    assert vehicle_load >= 0, f"Vehicle load is negative: {vehicle_load}"
                remaining_pick_up_nodes.remove(nearest_node)
            else:
                AssertionError("Could not begin a new route")
            # decrease n
            n -= 1
            if not remaining_pick_up_nodes:
                break

        if not unvisited_nodes:
            assert remaining_pick_up_nodes == [], f"Unvisited pick up nodes are not empty while \
                                                    route is being closed. nodes: {remaining_pick_up_nodes}"
            assert vehicle_load == 0, f"Vehicle load is not zero while route is being closed. load: {vehicle_load}"
            # Close the route
            route, route_schedule = close_route(route, route_schedule, state, df)
            routes.append(Route(route, len(routes)))
        elif unvisited_nodes:
            # serve remaining delivery nodes in order of distance
            current = route[-1]
            current_time = route_schedule[-1]
            planned_customers_set = set([df.loc[node, "cust_id"] for node in route])
            planned_delivery_nodes = [node for node in route if df.loc[node, "type"] == "delivery"]
            total_delivery_nodes = [state.cust_to_nodes[cust][1] for cust in planned_customers_set]
            remaining_delivery_nodes = [node for node in total_delivery_nodes if node not in planned_delivery_nodes]

            while remaining_delivery_nodes:
                current = route[-1]
                # Update candidate
                candidates = [
                    (node, float(distance))
                    for node, distance in zip(
                        remaining_delivery_nodes, state.distances[current][remaining_delivery_nodes]
                    )
                ]
                candidates = sorted(candidates, key=lambda x: x[1])  # sort by distance
                # Add nearest node of candidates to route
                nearest_node, nearest_dist = candidates[0]
                if nearest_node is not None:
                    route.append(nearest_node)
                    start_time, demand = df.loc[nearest_node, ["start_time", "demand"]].values.tolist()
                    current_time = route_schedule[-1]
                    current_service_time = df.loc[current, "service_time"].item()
                    # add to schedule sum of current time, service time and travel time
                    planned_time = current_time + current_service_time + nearest_dist
                    route_schedule.append(planned_time)
                    # update vehicle load
                    vehicle_load -= demand
                    assert vehicle_load >= 0, f"Vehicle load is negative: {vehicle_load}"
                    remaining_delivery_nodes.remove(nearest_node)
                else:
                    AssertionError("Could not serve all remaining delivery nodes")
            # Close the route
            if route[0] != route[-1]:
                route, route_schedule = close_route(route, route_schedule, state, df)
            routes.append(Route(route, len(routes)))

    # assign route to customers in cust_df
    for route_idx, route in enumerate(routes):
        for customer in route.nodes_list:
            state.cust_df.loc[state.cust_df["id"] == customer, "route"] = route_idx

    print(f"\n\n********************************************************")

    # Create the solution object of type CvrptwState
    solution = CvrptwState(
        routes=routes,
        n_vehicles=state.n_vehicles,
        vehicle_capacity=state.vehicle_capacity,
        dataset=state.cust_df,
    )
    # Update the time and cost attributes of the solution
    for route_idx in range(len(solution.routes)):
        solution.update_times_attributes_routes(route_idx)
        print(f"DEBUG: route {route_idx}:\n cost = {solution.routes_cost[route_idx]}")
        for customer in solution.routes[route_idx].nodes_list:
            solution.cust_df.loc[solution.cust_df["id"] == customer, "route"] = route_idx

    print(f"DEBUG: solution.objective() :\n{solution.objective()}")
    return solution


def nearest_neighbor_tw(state: CvrptwState, cordeau:bool = True, initial_time_slot: bool = True) -> CvrptwState:
    """
    Build a solution by iteratively constructing routes, where the nearest
    time-window compatible customer is added until the route has met the
    vehicle capacity limit. First, the compatibility is checked only on the
    pickup time window. If the customer is compatible, the check is extended
    to the delivery time window.
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
    df = dynamic_extended_df(state.cust_df)
    print(f"dynamic df:\n{df}")

    start_idx = 1 if cordeau else 0
    if initial_time_slot:
        valid_customers = get_ids_of_time_slot(state.cust_df, 0)
        unvisited = valid_customers
    else:
        unvisited = set(range(start_idx, len(state.cust_df["demand"])))

    vehicle = 0

    while vehicle < state.n_vehicles:
        initial_depot = state.depots["vehicle_to_depot"][vehicle]
        route = [initial_depot]
        route_schedule = [0]
        route_demands = 0
        while unvisited:
            # Add the nearest compatible unvisited customer to the route till max capacity
            current = route[-1]
            current_pickup_node = state.cust_to_nodes[current][0]
            reachable = pick_up_time_neighbours(
                state,
                current,
                df
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
            nearest_demand, nearest_end_time = df.loc[
                nearest, ["demand", "end_time"]
            ].values
            edge_time = state.distances[current_pickup_node][nearest].item()

            if route_demands + nearest_demand > state.vehicle_capacity:
                break
            current_service_time = df.loc[current, "service_time"].item()
            if not time_window_check(route_schedule[-1], current_service_time, edge_time, nearest_end_time):
                break

            route.append(nearest)
            route_schedule.append(
                state.distances[current_pickup_node][nearest].item()
                + current_service_time
            )

            unvisited.remove(nearest)
            route_demands += state.cust_df.loc[nearest, "demand"].item()

        route.append(route[0])  # Return to the depot
        route = Route(route, vehicle)
        routes.append(route)
        vehicle += 1

    # Assign routes to customers in cust_df
    for route_num, route in enumerate(routes):
        for customer in route.nodes_list:
            state.cust_df.loc[customer, "route"] = route_num

    # Create the solution object of type CvrptwState
    solution = CvrptwState(
        routes=routes,
        n_vehicles=state.n_vehicles,
        vehicle_capacity=state.vehicle_capacity,
        given_unassigned=list(unvisited),
    )
    # Update the time and cost attributes of the solution
    for route_idx in range(len(solution.routes)):
        solution.update_times_attributes_routes(route_idx)
        for customer in solution.routes[route_idx].nodes_list:
            solution.cust_df.loc[customer, "route"] = route_idx
    return solution
