import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Union
import plotly.express as px

def plot_data(
    data: Union[dict, pd.DataFrame],
    idx_annotations=False,
    name: str = "VRPTW Data",
    cordeau: bool = True,
    time_step: int = None,
):
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



    ########################################################################
    # Map related plots

    