from cvrptw.myvrplib.data_module import data
from cvrptw.myvrplib.vrpstates import CvrptwState
from alns import Result
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def verify_time_windows(data: dict, sol: CvrptwState, percentage: bool = False) -> dict:
    """
    Verifies the time windows of the solution and returns the early/late/ontime counts.
    If percentage is True, then the counts are returned as percentages of served customers.
        Parameters:
            data: dict
                The data dictionary.
            sol: CvrptwState
                The solution to be verified.
            percentage: bool
                If True, the counts are returned as percentages.
    """
    late = 0
    early = 0
    ontime = 0
    sum_early = 0
    sum_late = 0

    for route in sol.routes:
        for cust_idx, customer in enumerate(route.customers_list):
            if route.planned_windows[cust_idx][0] < data["time_window"][customer][0]:
                early += 1
                sum_early += data["time_window"][customer][0] - route.planned_windows[cust_idx][0]
            elif route.planned_windows[cust_idx][0] > data["time_window"][customer][1]:
                late += 1
                sum_late += route.planned_windows[cust_idx][1] - data["time_window"][customer][0]
            else:
                ontime += 1
    total = early + late + ontime
    
    if percentage:
        early = round(early / total * 100, 1)
        late = round(late / total * 100, 1)
        ontime = round(ontime / total * 100, 1)

    stats = {
        "total_served": total,
        "early": early,
        "late": late,
        "ontime": ontime,
        "sum_early": sum_early,
        "sum_late": round(sum_late, 2)
    }
    return stats


def plot_n_removals(destruction_counts: np.ndarray, d_operators: dict) -> None:
    """
    Plots the number of removals by destroy operator.
        Parameters:
            destruction_counts: np.ndarray
                The number of removals by destroy operator.
            d_operators: dict
                The destroy operators used.
    """
    cumulative_sums = np.cumsum(destruction_counts, axis=0)
    rows = np.arange(destruction_counts.shape[0])
    fig, ax = plt.subplots(figsize=(10,10))
    for col_idx in range(destruction_counts.shape[1]-1):
        plt.plot(rows, cumulative_sums[:, col_idx], label=f"{d_operators[col_idx]}")

    plt.xlabel("Iteration number")
    plt.ylabel("Number of removals")
    plt.title("Number of removals by destroy operator")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_n_insertions(insertion_counts: np.ndarray, r_operators: dict) -> None:
    """
    Plots the number of insertions by insertion operator.
        Parameters:
            insertion_counts: np.ndarray
                The number of insertions by insertion operator.
            r_operators: dict
                The repair operators used.
    """
    cumulative_sums = np.cumsum(insertion_counts, axis=0)  # Plot each column
    rows = np.arange(insertion_counts.shape[0])
    fig, ax = plt.subplots(figsize=(10, 10))
    for col_idx in range(insertion_counts.shape[1] - 1):
        plt.plot(rows, cumulative_sums[:, col_idx], label=f"{r_operators[col_idx]}")

    # Customize plot
    plt.xlabel("Iteration number")
    plt.ylabel("Number of insertions")
    plt.title("Number of insertions by insertion operator")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_n_destroy_ops_counts(d_operators_log: np.ndarray, d_operators: dict) -> None:
    """
    Plots the number of destroy operator applications.
        Parameters:
            d_operators_log: np.ndarray
                The destroy operators applied.
            d_operators: dict
                The destroy operators used.
    """
    destroy_operators_log_array = np.zeros(shape=(len(d_operators_log), len(d_operators)), dtype=int)
    for i, op in enumerate(d_operators_log):
        destroy_operators_log_array[i, op] +=1
    destroy_operators_log_array = np.cumsum(destroy_operators_log_array, axis=0)
    rows = np.arange(destroy_operators_log_array.shape[0])
    fig, ax = plt.subplots(figsize=(10, 10))
    for col_idx in range(destroy_operators_log_array.shape[1]):
        plt.plot(
            rows, destroy_operators_log_array[:, col_idx], label=f"{d_operators[col_idx]}"
        )

    plt.xlabel("Iteration number")
    plt.ylabel("Number of applications")
    plt.title("Number of destroy operator applications")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_n_repair_ops_counts(r_operators_log: np.ndarray, r_operators: dict) -> None:
    """
    Plots the number of repair operator applications.
        Parameters:
            r_operators_log: np.ndarray
                The repair operators applied.
            r_operators: dict
                The repair operators used.
    """
    repair_operators_log_array = np.zeros(shape=(len(r_operators_log), len(r_operators)), dtype=int)
    for i, op in enumerate(r_operators_log):
        repair_operators_log_array[i, op] += 1
    repair_operators_log_array = np.cumsum(repair_operators_log_array, axis=0)
    rows = np.arange(repair_operators_log_array.shape[0])
    fig, ax = plt.subplots(figsize=(10, 10))
    for col_idx in range(repair_operators_log_array.shape[1]):
        plt.plot(
            rows, repair_operators_log_array[:, col_idx], label=f"{r_operators[col_idx]}"
        )

    plt.xlabel("Iteration number")
    plt.ylabel("Number of applications")
    plt.title("Number of repair operator applications")
    plt.legend()
    plt.grid(True)
    plt.show()


def bar_plot_destroy_ops_count(result: Result) -> None:
    """
    Plots the number of destroy operator applications with a bar plot.
        Parameters:
            result: Result
                The result object.
    """
    results_df = pd.DataFrame(result.statistics.destroy_operator_counts)
    reasons = ["Global best", "Better", "Accepted", "Rejected"]
    x = np.arange(len(reasons))
    width = 0.20
    multiplier = 0

    fig, ax = plt.subplots(layout="constrained")
    fig.tight_layout()
    for attribute, measurement in results_df.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel("Count")
    ax.set_title("Destroy operator counts")
    ax.set_xticks(x + width, reasons)
    ax.legend(loc="right", ncols=1)

    plt.show()


def bar_plot_repair_ops_count(result: Result) -> None:
    """
    Plots the number of repair operator applications with a bar plot.
        Parameters:
            result: Result
                The result object.
    """
    results_df = pd.DataFrame(result.statistics.repair_operator_counts)
    reasons = ["Global best", "Better", "Accepted", "Rejected"]
    x = np.arange(len(reasons))
    width = 0.20
    multiplier = 0

    fig, ax = plt.subplots(layout="constrained")
    fig.tight_layout()
    for attribute, measurement in results_df.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel("Count")
    ax.set_title("Repair operator counts")
    ax.set_xticks(x + width, reasons)
    ax.legend(loc="right", ncols=1)

    plt.show()
