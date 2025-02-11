import pandas as pd
from cvrptw.myvrplib.data_module import (
    dynamic_extended_df, 
    cost_matrix_from_coords, 
    generate_twc_matrix,
    create_cust_nodes_mapping,
    read_cordeau_data
    )
from cvrptw.myvrplib.vrpstates import CvrptwState
import numpy as np


if __name__ == "__main__":
    init_data = pd.read_csv("./data/dynamic_df.csv")

    twc_format_nodes_df = dynamic_extended_df("./data/dynamic_df.csv")
    distances = cost_matrix_from_coords(coords=twc_format_nodes_df[["x", "y"]].values, cordeau=False)

    twc_matrix = generate_twc_matrix(
        twc_format_nodes_df[["start_time", "end_time"]].values.tolist(), 
        distances,
        cordeau=False
        )
    cust_to_nodes = create_cust_nodes_mapping(twc_format_nodes_df)

    state = CvrptwState(n_vehicles=2, vehicle_capacity=195, dataset=init_data)
    print(state.dataset)
    print(state.distances)
