import logging
import pandas as pd
from .data_module import data, END_OF_DAY

UNASSIGNED_PENALTY = 50
LOGGING_LEVEL = logging.ERROR

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
