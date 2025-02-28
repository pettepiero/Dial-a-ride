import plotly.express as px
import utm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import networkx as nx
import geopy.distance

def get_stops():
    stops_df = pd.read_csv("./data/DataSetActvAut(Fermate).csv")
    zone = 32
    # convert from utm to lat/lon
    stops_df["centroid_lat"] = stops_df.apply(
        lambda x: utm.to_latlon(x["X"], x["Y"], zone, northern=True)[0], axis=1
    )
    stops_df["centroid_lon"] = stops_df.apply(
        lambda x: utm.to_latlon(x["X"], x["Y"], zone, northern=True)[1], axis=1
    )

    # Filtering stops on Favaro dial a ride area
    with open('./data/fermate_chiamata_actv.txt', 'r', encoding='utf-8') as file:
        content = file.readlines()
    valid_stops = [int(match) for line in content for match in re.findall(r'\((\d+)\)', line)]
    # drop cols
    cols_to_drop = ["CODPRGFERMATA", "NODO", "ARCO", "DENOMINAZIONE", "UBICAZIONE", "COMUNE", "TIPOLOGIA", "ZONATARIFFARIA", "INDFERMATA", "INDPROXFERMATA", "NETWORKVERSION"]
    stops_df = stops_df.drop(columns=cols_to_drop)
    # filter valid stops
    stops_df = stops_df.loc[stops_df["CODFERMATA"].isin(valid_stops)]
    stops_df.to_csv("./data/actv_stops.csv", index=False)

    return stops_df


def add_manual_arcs(list_of_arcs: list):
    # Manually include the arcs that are wrongly tagged
    arcs_to_include = [1299, 1314, 1315, 1316, 1254, 1253, 1175, 1174, 1176, 2304, 2183, 2500, 2502, 332, 232, 1061, 1058, 449, 1046, 234, 1255,
       311, 233, 314, 1062, 1059, 439, 313, 235, 342, 341, 312, 237, 236, 256, 2190, 2187, 
       2189, 2188, 1256, 889, 212, 210, 1169]
    list_of_arcs = np.append(list_of_arcs, arcs_to_include)

    return list_of_arcs

def remove_manual_arcs(list_of_arcs: list):
    # Manually remove the arcs that are wrongly tagged
    arcs_to_remove = [1237, 1238, 1243, 1244, 1161, 1172, 1163, 1162, 1000, 1312, 1311, 
                      1310, 2185, 342, 1295, 1294, 2179, 2184, 730, 2300]
    list_of_arcs = np.setdiff1d(list_of_arcs, arcs_to_remove)

    return list_of_arcs

def get_arcs(list_of_stops: list):
    arcs_df = pd.read_csv("./data/DataSetActvAut(Archi).csv", delimiter=",")
    # filter arcs that only connect the stops in the list
    filtered_arcs_df = arcs_df.loc[arcs_df["FERMDA"].isin(list_of_stops)]
    filtered_arcs_df = filtered_arcs_df.loc[filtered_arcs_df["FERMAA"].isin(list_of_stops)]

########################################################################


    list_of_arcs = filtered_arcs_df["CODPRGARCOFERMATA"].unique()
    # list_of_arcs = arcs_df["CODPRGARCOFERMATA"].unique()


    ##################################################################

    list_of_arcs = add_manual_arcs(list_of_arcs)
    list_of_arcs = remove_manual_arcs(list_of_arcs)

    return list_of_arcs

def get_segments(list_of_arcs: list):
    # Read the CSV
    segments_df = pd.read_csv(
        "./data/DataSetActvAut(Segmenti).csv", delimiter=","
    )
    # Filter the segments DataFrame to only include the arcs in the arcs DataFrame
    segments_df = segments_df.loc[
        segments_df["CODPRGARCOFERMATA"].isin(list_of_arcs)
    ]

    # UTM Zone (You need to specify the correct one for your region)
    zone = 32  # Example for Northern Italy (Veneto)

    # Convert UTM to lat/lon
    segments_df["lat_da"], segments_df["lon_da"] = zip(
        *segments_df.apply(
            lambda x: utm.to_latlon(x["XDA"], x["YDA"], zone, northern=True), axis=1
        )
    )
    segments_df["lat_a"], segments_df["lon_a"] = zip(
        *segments_df.apply(
            lambda x: utm.to_latlon(x["XA"], x["YA"], zone, northern=True), axis=1
        )
    )
    return segments_df

def filter_geographically(segments_df: pd.DataFrame, stops_df: pd.DataFrame):
    # filter favaro area
    # long_min = 12.25766
    # long_max = 12.31817
    # lat_min = 45.490074
    # lat_max = 45.52111
    # epsilon = 0.002
    epsilon = 0.005

    long_min = stops_df["centroid_lon"].min() - epsilon
    long_max = stops_df["centroid_lon"].max() + epsilon
    lat_min = stops_df["centroid_lat"].min() - epsilon
    lat_max = stops_df["centroid_lat"].max() + epsilon
    df_filtered = segments_df[
        (segments_df["lon_da"] >= long_min)
        & (segments_df["lon_da"] <= long_max)  # Start point longitude
        & (segments_df["lat_da"] >= lat_min)
        & (segments_df["lat_da"] <= lat_max)  # Start point latitude
        & (segments_df["lon_a"] >= long_min)
        & (segments_df["lon_a"] <= long_max)  # End point longitude
        & (segments_df["lat_a"] >= lat_min)
        & (segments_df["lat_a"] <= lat_max)  # End point latitude
    ]

    return df_filtered


def plot_map(segments_df: pd.DataFrame, stops_df: pd.DataFrame, nodes: list=None):

    # Create a long-form DataFrame for Plotly (each segment as a line)
    plot_data = []
    for _, row in segments_df.iterrows():
        plot_data.append(
            {"lat": row["lat_da"], "lon": row["lon_da"], "segment": row["CODSEGMENTO"], "CODPRGARCOFERMATA" : row["CODPRGARCOFERMATA"]}
        )
        plot_data.append(
            {
                "lat": row["lat_a"],
                "lon": row["lon_a"],
                "segment": row["CODSEGMENTO"],
                "CODPRGARCOFERMATA": row["CODPRGARCOFERMATA"],
            }
        )

    df_plot = pd.DataFrame(plot_data)

    arcs = df_plot["CODPRGARCOFERMATA"].unique()
    df_plot.to_csv("./data/usable_road_network.csv", index=False)

    # Plot with Plotly Express
    fig = px.line_map(
        # df_plot.loc[df_plot["CODPRGARCOFERMATA"] == 1311],
        df_plot,
        lat="lat",
        lon="lon",
        # mapbox_style="open-street-map",
        zoom=12,
        title="Transport Network",
        hover_data=["segment"],
        line_group="CODPRGARCOFERMATA",
        color="CODPRGARCOFERMATA",
    )

    fig.add_scattermap(
        lat=stops_df["centroid_lat"],
        lon=stops_df["centroid_lon"],
        mode="markers",
        marker=dict(size=8, color="red"),
        hovertext=stops_df["CODFERMATA"],
    )

    # if nodes is not None:
    #     for node in nodes:
    #         fig.add_scattermap(
    #             lat=[node[0]],
    #             lon=[node[1]],
    #             mode="markers",
    #             marker=dict(size=8, color="blue"),
    #             hovertext="node",
    #         )

    fig.show()


def coordinates_in_nodes(coords, nodes):
    return any((coords == node).all() for node in nodes.values())

def euclidian_distance_utm(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def compute_length_of_arc(arc_id: int, segments_df: pd.DataFrame) -> float:
    """
    Compute the length of an arc given its ID and the segments DataFrame
    using euclidean distance because the segments are short lines."""
    arc_segments = segments_df.loc[segments_df["CODPRGARCOFERMATA"] == arc_id]
    length = 0
    for _, row in arc_segments.iterrows():
        length += euclidian_distance_utm(
            row["XDA"], row["YDA"], row["XA"], row["YA"]
        )
    return length


def get_dict_of_stops(stops_df: pd.DataFrame, segments_df: pd.DataFrame) -> dict:
    """
    Gets a dictionary mapping stops to their corresponding node IDs.
    """
    nodes = get_dict_of_nodes(segments_df)  # nodes: {node_id: np.array([x, y])}
    list_of_stops = stops_df["CODFERMATA"].unique()
    stops = {}

    for stop in list_of_stops:
        stop_x, stop_y = stops_df.loc[
            stops_df["CODFERMATA"] == stop, ["centroid_lat", "centroid_lon"]
        ].values[0]
        stop_coords = np.array([stop_x, stop_y])

        # Find the matching node by comparing coordinates
        for node_id, node_coords in nodes.items():
            if np.array_equal(
                node_coords, stop_coords
            ):  # Correct way to compare arrays
                stops[int(stop.item())] = node_id
                break  # Stop after finding the match

    return stops


def get_dict_of_nodes(segments_df: pd.DataFrame) -> dict:
    """
    Gets full dict of nodes where arcs meet.
    keys: "CODPUNTODA" or "CODPUNTOA"
    values: (lat, lon)

    Parameters:
    segments_df: DataFrame with segments data
    """
    list_of_arcs = segments_df["CODPRGARCOFERMATA"].unique()
    nodes = {}
    for arc in list_of_arcs:
        if arc == -1:
            print(
                f"Segments for arc {arc}:\n",
                segments_df.loc[segments_df["CODPRGARCOFERMATA"] == arc],
            )

        # start coordinatess
        lat_da, lon_da, node_id = segments_df.loc[
            segments_df["CODPRGARCOFERMATA"] == arc, ["lat_da", "lon_da", "CODPUNTODA"]
        ].values[0]
        lat_da = lat_da.item()
        lon_da = lon_da.item()
        node_id = int(node_id.item())
        start_coords = np.array([lat_da, lon_da])

        if not coordinates_in_nodes(start_coords, nodes):
            nodes[node_id] = (lat_da, lon_da)

        lat_a, lon_a, node_id = segments_df.loc[
            segments_df["CODPRGARCOFERMATA"] == arc, ["lat_a", "lon_a", "CODPUNTOA"]
        ].values[-1]
        lat_a = lat_a.item()
        lon_a = lon_a.item()
        node_id = int(node_id.item())
        end_coords = np.array([lat_a, lon_a])

        if not coordinates_in_nodes(end_coords, nodes):
            nodes[node_id] = (lat_a, lon_a)

    return nodes


def create_graph(segments_df: pd.DataFrame, stops_df: pd.DataFrame) -> pd.DataFrame:
    # compute real arcs from segments
    # create graph with stops and real arcs

    print(stops_df.head())
    print(segments_df.head())

    nodes = get_dict_of_nodes(segments_df)
    list_of_nodes = list(nodes.keys())

    print(f"list_of_nodes:\n {list_of_nodes}\n")

    graph = nx.DiGraph()
    list_of_stops = stops_df["CODFERMATA"].unique()
    list_of_arcs = get_arcs(list_of_stops)
    for arc in list_of_arcs:
        graph.add_edge(
            segments_df.loc[segments_df["CODPRGARCOFERMATA"] == arc, "CODPUNTODA"].values[0],
            segments_df.loc[segments_df["CODPRGARCOFERMATA"] == arc, "CODPUNTOA"].values[-1],
            weight=compute_length_of_arc(arc, segments_df),
        )

    return graph


def main():
    stops_df = get_stops()
    list_of_stops = stops_df["CODFERMATA"].unique()
    list_of_arcs = get_arcs(list_of_stops)
    segments_df = get_segments(list_of_arcs)
    # manually add segment from stop 702 to stop 720
    segments_df = pd.concat(
        [
            segments_df,
            pd.DataFrame(
                {
                    "CODSEGMENTO": [-1],
                    "CODPRGARCOFERMATA": [-1],
                    "CODPUNTODA": [-1],
                    "CODPUNTOA": [-2],
                    "XDA": [755156.74],
                    "YDA": [5046248.55],
                    "XA": [755230.03],
                    "YA": [5046578.2],
                    "lat_da": [45.52302648],
                    "lon_da": [12.2672934],
                    "lat_a": [45.52596203],
                    "lon_a": [12.26840207],
                }
            ),
        ]
    )

    segments_df = filter_geographically(segments_df, stops_df)

    graph = create_graph(segments_df, stops_df)
    # plot_map(segments_df, stops_df)
    print(graph)
    print(f"Running shortest path algorithm...")
    predecessors, shortest_paths = nx.floyd_warshall_predecessor_and_distance(graph, weight="weight")
    print(f"Done.")

    # Convert to a fast lookup dictionary
    distance_lookup = {
        (src, dst): shortest_paths[src][dst] for src in graph.nodes for dst in graph.nodes
    }

    stops_to_node = get_dict_of_stops(stops_df, segments_df)
    nodes_to_stop = {v: k for k, v in stops_to_node.items()}
    print(stops_to_node)

    node_a = 277
    node_b = 684
    print(f"Distance between nodes {node_a} and {node_b}:")
    print(distance_lookup[(stops_to_node[node_a], stops_to_node[node_b])])
    print("Path between nodes:")
    path = nx.reconstruct_path(
        stops_to_node[node_a], stops_to_node[node_b], predecessors
    )
    path = [int(nodes_to_stop[el]) for el in path]
    print(path)

if __name__ == "__main__":
    main()
