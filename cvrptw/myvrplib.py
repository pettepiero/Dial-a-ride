import copy
import random
from types import SimpleNamespace

import vrplib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd

from alns import ALNS
from alns.accept import RecordToRecordTravel
from alns.select import RouletteWheel
from alns.stop import MaxIterations


def cost_matrix_from_coords(coords: list) -> list:
    """
    Create a cost matrix from a list of coordinates. Uses the Euclidean distance as cost.
    """
    n = len(coords)
    cost_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cost_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])
    return cost_matrix


def read_cordeau_data(file: str, print_data: bool = False) -> dict:
    """
    Read the Cordeau et al. (2001) benchmark data.
    """
    filename = file.split("/")[-1]
    data_file = open(file, "r")
    if print_data:
        print("Reading data from file: ", filename)
    data = data_file.readlines()
    data_file.close()

    # Read problem type, number of vehicles, number of customers and number of days/depots/vehicle types
    # according to: https://www.bernabe.dorronsoro.es/vrp/

    type_dict = {
        0: "VRP",
        1: "PVRP",
        2: "MDVRP",
        3: "SDVRP",
        4: "VRPTW",
        5: "PVRPTW",
        6: "MDVRPTW",
        7: "SDVRPTW",
    }

    key = data[0].split()[0]
    problem_type = type_dict.get(key)  # Problem type
    problem_type = type_dict.get(6)
    m = int(data[0].split()[1])  # number of vehicles
    n = int(data[0].split()[2])  # number of customers
    t = int(data[0].split()[3])  # number of days/depots/vehicle types

    # Save depots max duration and max load in array
    depots_info = []
    for i in range(t):
        line = data[i + 1]
        depots_info.append(line.split())

    # Save customers in array
    customers = []
    for i in range(n):
        line = data[t + 1 + i]
        customers.append(line.split())

    customers = np.array(customers, dtype=np.float64)

    # Save depots in array
    depots = []
    for i in range(t):
        line = data[t + 1 + n + i]
        depots.append(line.split())

    depots = np.array(depots, dtype=np.float64)

    # Save in dict structure
    data_dict = {}
    data_dict["name"] = filename
    data_dict["vehicles"] = m
    data_dict["capacity"] = int(depots_info[0][1])
    data_dict["dimension"] = n  # Number of customers only, not depots
    data_dict["n_depots"] = t
    data_dict["depot_to_vehicles"] = {}  # {depot: [vehicles]}
    data_dict["vehicle_to_depot"] = {}  # {vehicle: depot}
    data_dict["depots"] = [i for i in range(n, n + t)]
    data_dict["node_coord"] = np.array([rows[1:3] for rows in customers])
    data_dict["node_coord"] = np.append(
        data_dict["node_coord"], np.array([rows[1:3] for rows in depots]), axis=0
    )
    data_dict["demand"] = [int(row[4]) for row in customers]
    data_dict["demand"] = np.array(data_dict["demand"] + [0 for row in depots])
    begin_times = [row[11] for row in customers]
    end_times = [row[12] for row in customers]
    data_dict["time_window"] = [[a, b] for a, b in zip(begin_times, end_times)]

    data_dict["service_time"] = [row[3] for row in customers]
    data_dict["service_time"] += [row[3] for row in depots]
    data_dict["edge_weight"] = cost_matrix_from_coords(data_dict["node_coord"])

    if print_data:
        print("Problem type: ", problem_type)
        print("Number of vehicles: ", m)
        print("Number of customers: ", n)
        print("Number of days/depots/vehicle types: ", t)

    return data_dict


def plot_data(data, name="VRPTW Data"):
    """
    Plot the routes of the passed-in solution.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    n = data["dimension"]

    ax.plot(
        data["node_coord"][: n - 1, 0],
        data["node_coord"][: n - 1, 1],
        "o",
        label="Customers",
    )
    ax.plot(
        data["node_coord"][n:, 0],
        data["node_coord"][n:, 1],
        "X",
        label="Depot",
    )
    ax.set_title(f"{name}")
    ax.set_xlabel("X-coordinate")
    ax.set_ylabel("Y-coordinate")
    ax.legend(frameon=False, ncol=3)


def plot_solution(
    data, solution, name="CVRP solution", idx_annotations=False, figsize=(12, 10), save=False
):
    """
    Plot the routes of the passed-in solution.
    """
    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap("Set2", data["vehicles"])
    cmap

    for idx, route in enumerate(solution.routes):
        ax.plot(
            [data["node_coord"][loc][0] for loc in route.customers_list],
            [data["node_coord"][loc][1] for loc in route.customers_list],
            color=cmap(idx),
            marker=".",
            label=f"Vehicle {idx}",
        )

    for i in range(data["dimension"]):
        customer = data["node_coord"][i]
        ax.plot(customer[0], customer[1], "o", c="tab:blue")
        if idx_annotations:
            ax.annotate(i, (customer[0], customer[1]))

    # for idx, customer in enumerate(data["node_coord"][:data["dimension"]]):
    #     ax.plot(customer[0], customer[1], "o", c="tab:blue")
    #     ax.annotate(idx, (customer[0], customer[1]))

    # Plot the depot
    kwargs = dict(zorder=3, marker="X")

    for i in range(data["dimension"], data["dimension"] + data["n_depots"]):
        depot = data["node_coord"][i]
        ax.plot(depot[0], depot[1], c="tab:red", **kwargs, label=f"Depot {i}")
        if idx_annotations:
            ax.annotate(i, (depot[0], depot[1]))

    # for idx, depot in enumerate(data["depots"]):
    #     ax.scatter(*data["node_coord"][depot], label=f"Depot {depot}", c=cmap(idx), **kwargs)
    #     ax.annotate(idx, (data["node_coord"][depot][0], data["node_coord"][depot][1]))

    ax.scatter(*data["node_coord"][0], c="tab:red", label="Depot 0", **kwargs)

    ax.set_title(f"{name}\n Total distance: {solution.cost}")
    ax.set_xlabel("X-coordinate")
    ax.set_ylabel("Y-coordinate")
    ax.legend(frameon=False, ncol=3)

    if save:
        plt.savefig(f"./plots/{name}.png")
        plt.close()


def calculate_depots(data):
    """
    Calculate the depot index for the vehicles. If the number of vehicles is equal to the number of depots,
    then vehicle i is mapped to depot i. If the number of vehicles is greater than the number of depots, then
    round robin assignment is used. If the number of vehicles is less than the number of depots, then random
    assignment is used, but load balancing between depots is guaranteed. The mapping is stored in the data
    dictionaries "depot_to_vehicles" and "vehicle_to_depot".
    """
    n_customers = data["dimension"]
    n_vehicles = data["vehicles"]
    n_depots = data["n_depots"]
    depots = data["depots"]
    # print(f"Number of customers: {n_customers}")
    # print(f"Before, depot to vehicles: {data['depot_to_vehicles']}")
    # Initialization of the dictionaries
    for depot in depots:
        data["depot_to_vehicles"][depot] = []
    for vehicle in range(n_vehicles):
        data["vehicle_to_depot"][vehicle] = None
    # vehicle i -> depot i
    if n_vehicles == n_depots:
        for depot in depots:
            data["depot_to_vehicles"][depot].append(depot)
            data["vehicle_to_depot"][depot] = depot

    elif n_vehicles > n_depots:
        # Round robin assignment
        for vehicle in range(n_vehicles):
            depot = vehicle % n_depots
            # print(f"Vehicle {vehicle} assigned to depot {depot}.")
            # print(f"Depot to vehicles: {data['depot_to_vehicles']}")
            # print(f"After, depot to vehicles: {data['depot_to_vehicles']}")
            # print(f"n_customer + depot: {n_customers + depot}")
            data["depot_to_vehicles"][n_customers + depot].append(vehicle)
            data["vehicle_to_depot"][vehicle] = n_customers + depot
    else:
        # Random assignment
        depots = rnd.choice(depots, size=n_vehicles, replace=False)
        for vehicle in range(n_vehicles):
            depot = depots[vehicle]
            data["depot_to_vehicles"][depot].append(vehicle)
            data["vehicle_to_depot"][vehicle] = int(depot)


def verify_time_windows(data, state):
    """
    Counts the number of customers that are served late or early in the solution.
    """
    late, early, ontime = 0, 0, 0
    planned_customers = [customer for route in state.routes for customer in route.customers_list]
    planned_customers = [
        customer for customer in planned_customers if customer not in data["depots"]
    ]
    left_out_customers = [
        customer
        for customer in range(data["dimension"])
        if customer not in planned_customers
    ]
    for customer in planned_customers:
        route = state.find_route(customer)
        idx = state.find_index_in_route(customer, route)
        arrival_time = state.times[state.routes.index(route)][idx]
        due_time = data["time_window"][customer][1]
        ready_time = data["time_window"][customer][0]
        if arrival_time > due_time:
            late += 1
        elif arrival_time < ready_time:
            early += 1
        elif arrival_time >= ready_time and arrival_time <= due_time:
            ontime += 1
    return late, early, ontime, len(left_out_customers)


def close_route(route):
    """
    Close the routes in the solution.
    """
    return route + [route[0]]
