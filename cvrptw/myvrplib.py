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
from data_module import data, END_OF_DAY

UNASSIGNED_PENALTY = 10

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
    data, solution, name="CVRP solution", idx_annotations=False, figsize=(12, 10), save=False, cordeau: bool = True
):
    """
    Plot the routes of the passed-in solution. If cordeau is True, the first customer is ignored.
    """
    start_idx = 1 if cordeau else 0
    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap("Set2", len(solution.routes))
    cmap

    for idx, route in enumerate(solution.routes):
        ax.plot(
            [data["node_coord"][loc][0] for loc in route.customers_list],
            [data["node_coord"][loc][1] for loc in route.customers_list],
            color=cmap(idx),
            marker=".",
            label=f"Vehicle {route.vehicle}",
        )

    for i in range(start_idx, data["dimension"]):
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

    # ax.scatter(*data["node_coord"][0], c="tab:red", label="Depot 0", **kwargs)

    ax.set_title(f"{name}\n Total distance: {solution.cost}")
    ax.set_xlabel("X-coordinate")
    ax.set_ylabel("Y-coordinate")
    ax.legend(frameon=False, ncol=3)

    if save:
        plt.savefig(f"./plots/{name}.png")
        plt.close()





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


# NOTE: this is a terrible check.
# It will accept any customer whose time window is after the calculated arrival time,
# even if the vehicle is early.
# Is the vehicle allowd to be early?
def time_window_check(prev_customer_time, prev_customer, candidate_customer):
    """
    Check if the candidate customer satisfies time-window constraints.
    """

    return (
        prev_customer_time
        + data["service_time"][prev_customer]
        + data["edge_weight"][prev_customer][candidate_customer]
        <= data["time_window"][candidate_customer][1]
    )


def route_time_window_check(route, times):
    """
    Check if the route satisfies time-window constraints.
    """
    for idx, customer in enumerate(route):
        if times[idx] > data["time_window"][customer][1]:
            return False

    return True
