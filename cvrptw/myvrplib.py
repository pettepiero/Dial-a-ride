import copy
import random
import logging
from types import SimpleNamespace
import vrplib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
from alns import ALNS
from alns.accept import RecordToRecordTravel
from alns.select import RouletteWheel, RandomSelect
from alns.stop import MaxIterations
from cvrptw.data_module import data, END_OF_DAY

UNASSIGNED_PENALTY = 20
LOGGING_LEVEL = logging.DEBUG


def plot_data(data: dict, idx_annotations=False, name: str = "VRPTW Data", cordeau: bool = True):
    """
    Plot the routes of the passed-in solution.
        Parameters:
            data: dict
                The data to be plotted.
            idx_annotations: bool
                If True, the customer indices are plotted.
            name: str
                The name of the plot.
            cordeau: bool
                If True, the first customer is ignored.
        Returns:
            None
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    n = data["dimension"]
    start_idx = 1 if cordeau else 0

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
    if idx_annotations:
        for i in range(start_idx, n):
            customer = data["node_coord"][i]
            ax.annotate(i, (customer[0], customer[1]))

    ax.set_title(f"{name}")
    ax.set_xlabel("X-coordinate")
    ax.set_ylabel("Y-coordinate")
    ax.legend(frameon=False, ncol=3)


def plot_solution(
    data: dict, solution, name="CVRP solution", idx_annotations=False, figsize=(12, 10), save=False, cordeau: bool = True
):
    """
    Plot the routes of the passed-in solution. If cordeau is True, the first customer is ignored.
        Parameters:
            data: dict
                The data to be plotted.
            solution: CvrptwState
                The solution to be plotted.
            name: str
                The name of the plot.
            idx_annotations: bool
                If True, the customer indices are plotted.
            figsize: tuple
                The size of the plot.
            save: bool
                If True, the plot is saved in "./plots".
            cordeau: bool
                If True, the first customer is ignored.
    
    """
    start_idx = 1 if cordeau else 0
    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap("Set2", len(solution.routes))
    cmap
    # Plot the routes
    for idx, route in enumerate(solution.routes):
        ax.plot(
            [data["node_coord"][loc][0] for loc in route.customers_list],
            [data["node_coord"][loc][1] for loc in route.customers_list],
            color=cmap(idx),
            marker=".",
            label=f"Vehicle {route.vehicle}",
        )
    
    # Plot the customers
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

    ax.set_title(f"{name}\n Total distance: {solution.cost}\n Total unassigned: {len(solution.unassigned)}")
    ax.set_xlabel("X-coordinate")
    ax.set_ylabel("Y-coordinate")
    ax.legend(frameon=False, ncol=3)

    if save:
        plt.savefig(f"./plots/{name}.png")
        plt.close()

def solution_times_statistics(data: dict, state) -> dict:
    """
    Counts the number of customers that are served late or early in the solution.
        Parameters:
            data: dict
                The data to be used.
            state: CvrptwState
                The solution to be verified.
        Returns:
            dict
                A dictionary containing the number of customers served late, early, on-time, left-out customers, and the sum of late and early minutes.
    """
    #debug
    print("Inside solution_times_statistics")
    late, early, ontime = 0, 0, 0
    # Get customers that are planned or absent in the solution
    planned_customers = [customer for route in state.routes for customer in route.customers_list[1:-1]]
    left_out_customers = [
        customer
        for customer in range(data["dimension"])
        if customer not in planned_customers
    ]
    late_minutes_sum = 0
    early_minutes_sum = 0
    # Check time windows for planned customers
    for customer in planned_customers:
        route = state.find_route(customer)
        idx = state.find_index_in_route(customer, route)
        # Use planned arrival times
        arrival_time = route.planned_windows[idx][0]
        due_time = data["time_window"][customer][1]
        ready_time = data["time_window"][customer][0]
        if arrival_time > due_time:
            late += 1
            late_minutes_sum += arrival_time - due_time
        elif arrival_time < ready_time:
            early += 1
            early_minutes_sum += ready_time - arrival_time
            #debug
            print(f"Customer {customer} planned {arrival_time} ready {ready_time}")
        elif arrival_time >= ready_time and arrival_time <= due_time:
            ontime += 1
    dict = {
        "late": late,
        "early": early,
        "ontime": ontime,
        "left_out_customers": len(left_out_customers),
        "late_minutes_sum": round(late_minutes_sum, 2),
        "early_minutes_sum": float(round(early_minutes_sum, 2)),
    }
    return dict


def close_route(route: list) -> list:
    """
    Append to end of route the depot.
        Parameters:
            route: list
                The route to be closed.
        Returns:
            list
                The closed route.
    """
    return route + [route[0]]


def route_time_window_check(route, start_index: int = 1) -> bool:
    """
    Check if the route satisfies time-window constraints. Ignores the depots as
    they are considered available 24h. Depots are first and last elements
    according to Cordeau notation.
        Parameters:
            route: Route
                The route to be checked.
            start_index: int
                The index to start checking from.
        Returns:
            bool
                True if the route satisfies time-window constraints, False otherwise.
    """
    # check if planned arrival time is later than the due time
    for idx, customer in enumerate(route.customers_list[start_index:-1]):
        idx += start_index
        if route.planned_windows[idx][0] > data["time_window"][customer][1]:
            return False
    return True


# NOTE: this is a terrible check.
# It will accept any customer whose time window is after the calculated arrival time,
# even if the vehicle is early.
# Is the vehicle allowed to be early?
# For now, yes. It will stay at the customer until the time window opens.
def time_window_check(
    prev_customer_time: float, prev_customer: int, candidate_customer: int
):
    """
    Check if the candidate customer satisfies time-window constraints. Returns true if the
    candidate customer is not served late. Notice that the vehicle can be early.
        Parameters:
            prev_customer_time: float
                The arrival time of the previous customer.
            prev_customer: int
                The previous customer.
            candidate_customer: int
                The candidate customer.
        Returns:
            bool
                True if the candidate customer satisfies time-window constraints, False otherwise.
    """
    return (
        prev_customer_time
        + data["service_time"][prev_customer]
        + data["edge_weight"][prev_customer][candidate_customer]
        <= data["time_window"][candidate_customer][1]
    )
