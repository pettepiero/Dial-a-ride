import plotly.express as px
import utm
import pandas as pd
import numpy as np
import re
import networkx as nx

DEPOT_ID = 3404
DEPOT_ID = 1259

def get_stops(
        list_of_depots_stops: list,
        data: str = "./data/DataSetActvAut(Fermate).csv",
        valid_stops_file: str = "./data/fermate_chiamata_actv.txt",
        utm_zone: int = 32,
        output_file: str = "./data/actv_stops.csv",
        all_stops: bool = False
    ) -> pd.DataFrame:
    """
    Get the stops from the CSV file and convert the UTM coordinates to lat/lon. Drops 
    unnecessary columns and filters stops on the Favaro dial a ride area. Saves the 
    filtered stops DataFrame to a CSV file. The list of valid stops is read from the 
    specified text file. Default values are provided for all arguments.

    Parameters:
    list_of_depots_stops: list
        List of depot stops.
    data: str
        Path to the CSV file containing the stops data.
    valid_stops_file: str
        Path to the text file containing the valid stops.
    utm_zone: int
        UTM zone for the region. (32 for Venezia/Mestre area)
    output_file: str
        Path to the output CSV file.
    all_stops: bool
        If True, returns all stops in the dataset.
    Returns:
    stops_df: pd.DataFrame
        DataFrame containing the stops data.
        Attributes:
            - CODFERMATA: Stop ID
            - X: UTM X coordinate
            - Y: UTM Y coordinate
            - centroid_lat: Latitude
            - centroid_lon: Longitude
            - node_id: Node ID for graph
    """
    # Check if the depot stops are in the list of valid stops, otherwise add them
    with open(valid_stops_file, "r", encoding="utf-8") as file:
        content = file.readlines()
    valid_stops = [int(match) for line in content for match in re.findall(r'\((\d+)\)', line)]
    for depot in list_of_depots_stops:
        if depot not in valid_stops:
            valid_stops.append(depot)

    stops_df = pd.read_csv(data, delimiter=",")
    # convert from utm to lat/lon
    stops_df["centroid_lat"] = stops_df.apply(
        lambda x: utm.to_latlon(x["X"], x["Y"], utm_zone, northern=True)[0], axis=1
    )
    stops_df["centroid_lon"] = stops_df.apply(
        lambda x: utm.to_latlon(x["X"], x["Y"], utm_zone, northern=True)[1], axis=1
    )

    if not all_stops:
        # Filtering stops on Favaro dial a ride area
        with open(valid_stops_file, "r", encoding="utf-8") as file:
            content = file.readlines()
        valid_stops = [int(match) for line in content for match in re.findall(r'\((\d+)\)', line)]
    # drop cols that I don't need
    cols_to_drop = ["CODPRGFERMATA", "NODO", "ARCO", "DENOMINAZIONE", "UBICAZIONE", "COMUNE", "TIPOLOGIA", "ZONATARIFFARIA", "INDFERMATA", "INDPROXFERMATA", "NETWORKVERSION"]
    stops_df = stops_df.drop(columns=cols_to_drop)
    if not all_stops:
        # filter valid stops
        stops_df = stops_df.loc[stops_df["CODFERMATA"].isin(valid_stops)]
    stops_df["node_id"] = stops_df.index +1
    stops_df.reset_index(drop=True, inplace=True)
    stops_df.to_csv(output_file, index=False)

    return stops_df


def manual_arcs(
        starting_list: list, 
        arcs_to_add: list = None, 
        arcs_to_remove: list = None
    ) -> list:
    """
    Function to manually add or remove arcs to the list of arcs to include in the network. This 
    is useful for datasets where some arcs are misteriously missing for some reason. The default
    list is hand picked for the dataset used in this project.

    Parameters:
    starting_list: list
        List of arcs to include in the network.
    arcs_to_add: list
        List of arcs to add to the network.
    arcs_to_remove: list
        List of arcs to remove from the network.

    Returns:
    list_of_arcs: list
        List of arcs to include in the network.
    """
    if arcs_to_add is not None:
        result = np.append(starting_list, arcs_to_add)
    else:
        # arcs_to_include = [1299, 1314, 1315, 1316, 1254, 1253,
        # 1175, 1174, 1176, 2304, 2183, 2500, 2502, 332, 232,
        # 1061, 1058, 449, 1046, 234, 1255, 311, 233, 314, 1062,
        # 1059, 439, 313, 235, 342, 341, 312, 237, 236, 256, 2190,
        # 2187, 2189, 2188, 1256, 889, 212, 210, 1169, 1049, 1169,
        # 1682, 1050, 1169, 1053, 2587, 2580, 2579, 2199, 1681,
        # 1682, 2584, 2199, 2202, 2201, 2200, 265, 244, 257,
        # 2190, 256, 1168, 237, 2582, 2583, 2576, 2575, 3015,
        # 1695, 1696, 3014, 317, 269, 943,
        # 267, 258, 2586, 2584, 1052, 1047, 1169]

        arcs_to_include = [
            210,
            212,
            232,
            233,
            234,
            235,
            236,
            237,
            237,
            244,
            256,
            256,
            257,
            258,
            265,
            311,
            312,
            313,
            314,
            332,
            341,
            342,
            439,
            449,
            889,
            1046,
            1047,
            1049,
            1050,
            1052,
            1053,
            1058,
            1059,
            1061,
            1062,
            1168,
            1169,
            1169,
            1169,
            1169,
            1174,
            1175,
            1176,
            1253,
            1254,
            1255,
            1256,
            1299,
            1314,
            1315,
            1316,
            1681,
            1682,
            1682,
            2183,
            2187,
            2188,
            2189,
            2190,
            2199,
            2200,
            2201,
            2202,
            2304,
            2500,
            2502,
            2579,
            2580,
            2584,
            2586,
            2587,
        ]

        result = np.append(starting_list, arcs_to_include)

    if arcs_to_remove is not None:
        result = np.setdiff1d(result, arcs_to_remove)
    else:
        arcs_to_remove = [1237, 1238, 1243, 1244, 1161, 1172, 1163, 
                          1162, 1000, 1312, 1311, 1310, 2185, 342, 
                          1295, 1294, 2179, 2184, 730, 2300, 2186, 
                          2585, 886, 889, 1528, 2498]
        result = np.setdiff1d(result, arcs_to_remove)

    return result

def get_arcs(
        list_of_stops: list, 
        arcs_data: str = "./data/DataSetActvAut(Archi).csv", 
        all_arcs: bool=False) -> list:
    """
    Get the list of arc ids that connect the stops in the given 
    list. The arcs are read from the CSV file and filtered to only
    include the arcs that connect the stops in the list. The 
    default arcs data file is provided. It is possible to return 
    all arcs in the dataset by setting the all_arcs parameter to 
    True.

    Parameters:
    list_of_stops: list
        List of stop IDs to connect.
    arcs_data: str
        Path to the CSV file containing the arcs data.
    all_arcs: bool
        If True, returns all arcs in the dataset.

    Returns:
    list_of_arcs: list
        List of arc IDs that connect the stops in the list.
    """    
    arcs_df = pd.read_csv(arcs_data, delimiter=",")
    if all_arcs:
        return arcs_df["CODPRGARCOFERMATA"].unique()
    # filter arcs that only connect the stops in the list
    filtered_arcs_df = arcs_df.loc[arcs_df["FERMDA"].isin(list_of_stops)]
    list_of_arcs = filtered_arcs_df["CODPRGARCOFERMATA"].unique()
    filtered_arcs_df = filtered_arcs_df.loc[filtered_arcs_df["FERMAA"].isin(list_of_stops)]
    list_of_arcs = np.append(list_of_arcs, filtered_arcs_df["CODPRGARCOFERMATA"].unique())

    return list_of_arcs

def get_segments(list_of_arcs: list):
    """
    Given a list of arcs, returns a dataframe with all the segments that make up the arcs.
    """
    segments_df = pd.read_csv(
        "./data/DataSetActvAut(Segmenti).csv", delimiter=","
    )
    # Filter the segments DataFrame to only include the arcs in list_of_arcs
    segments_df = segments_df.loc[
        segments_df["CODPRGARCOFERMATA"].isin(list_of_arcs)
    ]
    zone = 32  # UTM Zone for Veneto
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

def filter_geographically(
        segments_df: pd.DataFrame,
        stops_df: pd.DataFrame,
        bigger_map: bool = False):
    if not bigger_map:
        epsilon = 0.02
        #filter favaro area
        long_min = 12.2350 - epsilon
        long_max = 12.27066 + epsilon
        lat_min = 45.400074 - epsilon
        lat_max = 45.49451 + epsilon
    else:
        epsilon = 0.010
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

    stops_filtered = stops_df[
        (stops_df["centroid_lon"] >= long_min)
        & (stops_df["centroid_lon"] <= long_max)  # Longitude
        & (stops_df["centroid_lat"] >= lat_min)
        & (stops_df["centroid_lat"] <= lat_max)  # Latitude
    ]
    print(f"DEBUG: len(df_filtered) = {len(df_filtered)}")
    return df_filtered, stops_filtered


def plot_map(
        segments_df: pd.DataFrame, 
        stops_df: pd.DataFrame, 
        nodes: list=None,
        show: bool=True):

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
    if show:
        fig.show()
    return fig


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


def create_graph(
        segments_df: pd.DataFrame, 
        stops_df: pd.DataFrame, 
        full_graph: bool = False
    ) -> pd.DataFrame:
    """
    Computes real arcs from segments and creates a graph with them.
    """
    # compute real arcs from segments
    # create graph with stops and real arcs

    nodes = get_dict_of_nodes(segments_df)
    list_of_nodes = list(nodes.keys())

    list_of_stops = stops_df["CODFERMATA"].unique()
    if not full_graph:
        list_of_arcs = get_arcs(list_of_stops)
        list_of_arcs = manual_arcs(list_of_arcs)
    elif full_graph:
        list_of_arcs = get_arcs(list_of_stops, all_arcs=True)

    graph = nx.DiGraph()
    for arc in list_of_arcs:
        graph.add_edge(
            segments_df.loc[segments_df["CODPRGARCOFERMATA"] == arc, "CODPUNTODA"].values[0],
            segments_df.loc[segments_df["CODPRGARCOFERMATA"] == arc, "CODPUNTOA"].values[-1],
            weight=compute_length_of_arc(arc, segments_df),
        )

    return graph

def create_visualization():
    stops_df = get_stops()
    list_of_stops = stops_df["CODFERMATA"].unique()
    list_of_arcs = get_arcs(list_of_stops)
    list_of_arcs = manual_arcs(list_of_arcs)
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

    segments_df, stops_df = filter_geographically(segments_df, stops_df)
    fig = plot_map(segments_df=segments_df, stops_df=stops_df, show=False)
    return fig


def get_shortest_path(graph, node_a, node_b, stops_df, segments_df):
    predecessors, shortest_paths = nx.floyd_warshall_predecessor_and_distance(
        graph, weight="weight"
    )

    # Convert to a fast lookup dictionary
    distance_lookup = {
        (src, dst): shortest_paths[src][dst]
        for src in graph.nodes
        for dst in graph.nodes
    }

    stops_to_node = get_dict_of_stops(stops_df, segments_df)
    nodes_to_stop = {v: k for k, v in stops_to_node.items()}
    path = nx.reconstruct_path(
        stops_to_node[node_a], stops_to_node[node_b], predecessors
    )
    path = [int(nodes_to_stop[el]) for el in path]


def manual_segments(segments_df: pd.DataFrame) -> pd.DataFrame:
    """
    Manually adds segment from stop 702 to stop 720
    """
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

    return segments_df


def setup(
        list_of_depots_stops: list, 
        stops_data: str = "./data/DataSetActvAut(Fermate).csv",
        full_arcs: bool = False,
        bigger_map: bool = True,
        show_map: bool = False):
    """
        Given stops data and list of depots stops, returns:
        - Directed graph for the stops 
        - Dataframe with all stops in Favaro Area
        - Dataframe with the list of segments in the Favaro Area
    """
    stops_df = get_stops(list_of_depots_stops=list_of_depots_stops, data=stops_data, all_stops=False)
    list_of_stops = stops_df["CODFERMATA"].unique()
    list_of_arcs = get_arcs(list_of_stops, all_arcs=full_arcs)
    list_of_arcs = manual_arcs(list_of_arcs)
    segments_df = get_segments(list_of_arcs)
    segments_df = manual_segments(segments_df)
    segments_df, stops_df = filter_geographically(segments_df, stops_df, bigger_map)
    print(f"DEBUG: len(stop_df) = {len(stops_df)}")
    plot_map(segments_df, stops_df, show=show_map)
    graph = create_graph(segments_df, stops_df, full_graph=full_arcs)
    # graph = None
    return graph, stops_df, segments_df


def main():
    graph, stops_df, segments_df = setup(
        list_of_depots_stops=[DEPOT_ID],
        full_arcs=True,
        bigger_map = False,
        show_map=False,
    )

    # predecessors, shortest_paths = nx.floyd_warshall_predecessor_and_distance(graph, weight="weight")

    # # Convert to a fast lookup dictionary
    # distance_lookup = {
    #     (src, dst): shortest_paths[src][dst] for src in graph.nodes for dst in graph.nodes
    # }

    # stops_to_node = get_dict_of_stops(stops_df, segments_df)
    # nodes_to_stop = {v: k for k, v in stops_to_node.items()}

    # node_a = 277
    # node_b = 684
    # print(f"Distance between nodes {node_a} and {node_b}:")
    # print(distance_lookup[(stops_to_node[node_a], stops_to_node[node_b])])
    # print("Path between nodes:")
    # path = nx.reconstruct_path(
    #     stops_to_node[node_a], stops_to_node[node_b], predecessors
    # )
    # path = [int(nodes_to_stop[el]) for el in path]
    # print(path)

if __name__ == "__main__":
    main()
