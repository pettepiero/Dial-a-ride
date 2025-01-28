import copy
import random
import logging
from types import SimpleNamespace
import vrplib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Union
import numpy.random as rnd
from alns import ALNS
from alns.accept import RecordToRecordTravel
from alns.select import RouletteWheel, RandomSelect
from alns.stop import MaxIterations
from .data_module import data, END_OF_DAY

UNASSIGNED_PENALTY = 50
LOGGING_LEVEL = logging.ERROR


def plot_data(data: Union[dict, pd.DataFrame], idx_annotations=False, name: str = "VRPTW Data", cordeau: bool = True, time_step: int = None):
    """
    Plot the routes of the passed-in solution.
        Parameters:
            data: dict or pd.DataFrame
                The data to be plotted.
            idx_annotations: bool
                If True, the customer indices are plotted.
            name: str
                The name of the plot.
            cordeau: bool
                If True, the first customer is ignored.
            time_step: int
                If not None, only the customers that are active at the given time step are plotted.
        Returns:
            None
    """
    if isinstance(data, dict):
        n = data["dimension"]
        start_idx = 1 if cordeau else 0

        fig, ax = plt.subplots(figsize=(12, 10))
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
    elif isinstance(data, pd.DataFrame):
        fig, ax = plt.subplots(figsize=(12, 10))

        if time_step is not None:
            active_custs = data.loc[data["call_in_time_slot"] <= time_step]
            non_active_custs = data.loc[data["call_in_time_slot"] > time_step]
            ax.plot(
                active_custs.loc[data["demand"] != 0, "x"],
                active_custs.loc[data["demand"] != 0, "y"],
                "o",
                color="tab:blue",
                label="Customers at this time step",
            )
            ax.plot(
                non_active_custs.loc[data["demand"] != 0, "x"],
                non_active_custs.loc[data["demand"] != 0, "y"],
                "o",
                color="tab:gray",
                label="Customers at future time step",
            )
        else:
            ax.plot(
                data.loc[data["demand"] != 0, "x"],
                data.loc[data["demand"] != 0, "y"],
                "o",
                label="Customers",
            )

        ax.plot(
            data.loc[data["demand"] == 0, "x"],
            data.loc[data["demand"] == 0, "y"],
            "X",
            label="Depots",
        )
        if idx_annotations:
            for i in range(data.shape[0]):
                ax.annotate(i, (data["x"].iloc[i], data["y"].iloc[i]))
    else:
        raise ValueError("Data must be a dict or a pandas DataFrame.")
    
    ax.set_title(f"{name}")
    ax.set_xlabel("X-coordinate")
    ax.set_ylabel("Y-coordinate")
    ax.legend(frameon=False, ncol=3)


def plot_solution(
    solution, name="CVRP solution", idx_annotations=False, figsize=(12, 10), save=False, cordeau: bool = True
):
    """
    Plot the routes of the passed-in solution. If cordeau is True, the first customer is ignored.
        Parameters:
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
    df = solution.nodes_df
    start_idx = 1 if cordeau else 0
    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap("Set2", len(solution.routes))
    cmap
    # Plot the routes
    for idx, route in enumerate(solution.routes):
        ax.plot(
            [
                df.loc[df["id"] == loc, "x"].item()
                for loc in route.customers_list
            ],
            [
                df.loc[df["id"] == loc, "y"].item()
                for loc in route.customers_list
            ],
            color=cmap(idx),
            marker=".",
            label=f"Vehicle {route.vehicle}",
        )

    # Plot the customers
    for i in range(start_idx, solution.n_customers +1):
        customer = df.loc[df["id"] == i, ['x', 'y']]
        customer = customer.values[0]
        ax.plot(customer[0], customer[1], "o", c="tab:blue")
        if idx_annotations:
            ax.annotate(i, (customer[0], customer[1]))

    # Plot the depot
    kwargs = dict(zorder=3, marker="X")

    for i, dep in enumerate(solution.depots["depots_indices"]):
        coords = solution.depots["coords"][i]
        ax.plot(coords[0], coords[1], c="tab:red", **kwargs, label=f"Depot {dep}")
        if idx_annotations:
            ax.annotate(dep, (coords[0], coords[1]))

    ax.set_title(f"{name}\n Total distance: {solution.cost}\n Total unassigned: {len(solution.unassigned)}")
    ax.set_xlabel("X-coordinate")
    ax.set_ylabel("Y-coordinate")
    ax.legend(frameon=False, ncol=3)

    if save:
        plt.savefig(f"/home/pettepiero/tirocinio/dial-a-ride/outputs/plots/{name}.png")
        plt.close()

def solution_times_statistics(state) -> dict:
    """
    Counts the number of customers that are served late or early in the solution.
    The considered time step is current_time attribute of state.
        Parameters:
            state: CvrptwState
                The solution to be verified.
        Returns:
            dict
                A dictionary containing the number of customers served late, early,
                 on-time, left-out customers, and the sum of late and early minutes.
    """
    data = state.nodes_df

    late, early, ontime = 0, 0, 0
    # To get customers in the solution, first remove all the depots
    # then get all the customers that were seen by the system
    # until current time step

    available_customers = data.loc[data["demand"] != 0]
    available_customers = available_customers.loc[
        available_customers["call_in_time_slot"] <= state.current_time
    ]
    # remove the already satisfied customers
    available_customers = available_customers.loc[available_customers["done"] == False]

    # Then get the customers that are in the solution, aka planned_customers
    planned_customers = available_customers.loc[
        pd.notnull(available_customers["route"])
    ]
    # left out customers are seen but not planned
    left_out_customers = available_customers.loc[
        pd.isnull(available_customers["route"])
    ]

    late_minutes_sum = 0
    early_minutes_sum = 0

    for _, customer in planned_customers.iterrows():
        id = customer["id"]
        route = customer["route"]
        idx_in_route = state.find_index_in_route(id, state.routes[route])
        print("DEBUG: ", id, route, idx_in_route)
        print(f"DEBUG: {state.routes[route].customers_list}")
        print(f"DEBUG: {state.routes[route].planned_windows}")
        planned_arrival_time = state.routes[route].planned_windows[idx_in_route][0]

        due_time = customer["end_time"]
        ready_time = customer["start_time"]
        if planned_arrival_time > due_time:
            late += 1
            late_minutes_sum += planned_arrival_time - due_time
        elif planned_arrival_time < ready_time:
            early += 1
            early_minutes_sum += ready_time - planned_arrival_time
        elif planned_arrival_time >= ready_time and planned_arrival_time <= due_time:
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
