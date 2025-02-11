import pandas as pd
from cvrptw.myvrplib.data_module import (
    dynamic_extended_df, 
    cost_matrix_from_coords, 
    generate_twc_matrix,
    create_cust_nodes_mapping,
    read_cordeau_data
    )
from cvrptw.myvrplib.vrpstates import CvrptwState
from cvrptw.myvrplib.route import Route
import numpy as np


if __name__ == "__main__":
    init_data = pd.read_csv("./data/dynamic_df.csv")

    twc_format_nodes_df = dynamic_extended_df(init_data)
    distances = cost_matrix_from_coords(coords=twc_format_nodes_df[["x", "y"]].values, cordeau=False)

    twc_matrix = generate_twc_matrix(
        twc_format_nodes_df[["start_time", "end_time"]].values.tolist(), 
        distances,
        cordeau=False
        )
    cust_to_nodes = create_cust_nodes_mapping(twc_format_nodes_df)
    print(f"twc_format_nodes_df: {twc_format_nodes_df}")
    print(twc_format_nodes_df.dtypes)

    test_route = Route(customers_list=[22, 3, 6, 3, 6, 7, 7, 22])
    print("\n\nCreating state\n")

    state = CvrptwState(n_vehicles=2, vehicle_capacity=195, dataset=init_data, routes=[test_route])
    print(state)
    print(state.served_customers())
