import numpy as np
import numpy.random as rnd
import copy
import pandas as pd
from typing import Union
import pathlib

END_OF_DAY = 1000
SEED = 1234

# macro for mapping customer to row in the dataframe where each row corresponds
# to a node, therefore each customer has 2 rows (pick up and delivery)
cust_row = lambda id, pickup: id * 2 - 1 if pickup else id * 2


def read_cordeau_data(file: str, print_data: bool = False) -> dict:
    """
    Parses a benchmark instance in the format proposed by Cordeau et al. (2001) for VRP variants.
    Information on this protocol can be found at: https://www.bernabe.dorronsoro.es/vrp/ under the 
    dedicated page for describing Cordeau's data format. More information can be found at the 
    documentation of this project (``docs`` folder).

    Parameters
    ----------
    file : str
        Path to the input `.txt` file in Cordeau's benchmark format.
    print_data : bool, optional
        If True, prints debug information including file name and parsed quantities.

    Returns
    -------
    dict
        A dictionary containing parsed problem data with the following keys:

        - `name` : str  
          File name.
        - `vehicles` : int  
          Number of vehicles.
        - `capacity` : int  
          Maximum vehicle load (assumed identical for all).
        - `dimension` : int  
          Number of customers (excluding depots).
        - `n_depots` : int  
          Number of depots.
        - `depots` : list[int]  
          Indices of depot nodes.
        - `depot_to_vehicles` : dict[int, list[int]]  
          Mapping from depot ID to list of vehicles assigned to it.
        - `vehicle_to_depot` : dict[int, int]  
          Mapping from vehicle ID to assigned depot.
        - `node_coord` : ndarray  
          Node coordinates (customers + depots).
        - `demand` : ndarray  
          Demand at each node (0 for depots).
        - `time_window` : list[list[int]]  
          Earliest and latest service times.
        - `service_time` : list[float]  
          Service duration for each node.
        - `edge_weight` : ndarray  
          Cost matrix derived from Euclidean distances.
    """
    filename = str(file).split("/")[-1]
    data_file = open(file, "r")
    if print_data:
        print("Reading data from file: ", filename)
    data = data_file.readlines()
    data_file.close()

    # Read problem type, number of vehicles, number of customers and number of days/depots/vehicle types
    # according to: https://www.bernabe.dorronsoro.es/vrp/

    type_dict = {
        0: "VRP",
        1: "PVRP",
        2: "MDVRP",
        3: "SDVRP",
        4: "VRPTW",
        5: "PVRPTW",
        6: "MDVRPTW",
        7: "SDVRPTW",
    }

    key = data[0].split()[0]
    problem_type = type_dict.get(key)  # Problem type
    m = int(data[0].split()[1])  # number of vehicles
    n = int(data[0].split()[2])  # number of customers
    t = int(data[0].split()[3])  # number of days/depots/vehicle types

    # Save depots max duration and max load in array
    depots_info = []
    for i in range(t):
        line = data[i + 1]
        depots_info.append(line.split())

    # Save customers in array
    customers = []
    for i in range(n):
        line = data[t + 1 + i]
        customers.append(line.split())

    customers = np.array(customers, dtype=np.float64)

    # Save depots in array
    depots = []
    for i in range(t):
        line = data[t + 1 + n + i]
        depots.append(line.split())

    depots = np.array(depots, dtype=np.float64)

    # Save in dict structure
    data_dict = {}
    data_dict["name"] = filename
    data_dict["vehicles"] = m   # maximum number of vehicles
    data_dict["capacity"] = int(depots_info[0][1])
    data_dict["dimension"] = n  # Number of customers only, not depots
    data_dict["n_depots"] = t
    data_dict["depot_to_vehicles"] = {}  # {depot: [vehicles]}
    data_dict["vehicle_to_depot"] = {}  # {vehicle: depot}
    data_dict["depots"] = [i for i in range(n+1, n+1 + t)]

    data_dict["node_coord"] = np.array((None, None))
    coords = np.array([rows[1:3] for rows in customers])
    data_dict["node_coord"] = np.vstack((data_dict["node_coord"], coords))
    coords = np.array([rows[1:3] for rows in depots])
    data_dict["node_coord"] = np.vstack((data_dict["node_coord"], coords))

    data_dict["demand"] = np.array([0], dtype=np.int64)
    demands = np.array([int(row[4]) for row in customers])
    data_dict["demand"] = np.concatenate((data_dict["demand"], demands))
    data_dict["demand"] = np.concatenate((data_dict["demand"], [0 for row in depots]))

    begin_times = [row[11] for row in customers]
    end_times = [row[12] for row in customers]
    data_dict["time_window"] = [[-1, -1]]
    data_dict["time_window"] += [[int(a), int(b)] for a, b in zip(begin_times, end_times)]
    data_dict["time_window"] += [[0, END_OF_DAY] for row in depots]

    data_dict["service_time"] = [None]
    data_dict["service_time"] += [int(row[3]) for row in customers]
    data_dict["service_time"] += [int(row[3]) for row in depots]
    data_dict["edge_weight"] = cost_matrix_from_coords(data_dict["node_coord"])
    data_dict["depot_to_vehicles"], data_dict["vehicle_to_depot"] = calculate_depots(
        depots=data_dict['depots'], n_vehicles=data_dict["vehicles"]
    )

    if print_data:
        print("Problem type: ", problem_type)
        print("Number of vehicles: ", m)
        print("Number of customers: ", n)
        print("Number of days/depots/vehicle types: ", t)

    return data_dict

def get_all_section_tags(file: str) -> list:
    """
    Reads a structured VRP-like file and returns a list of strings,
    each corresponding to the first word on each line (section tags, keys, etc.).
    It tests wether a list of base tags exist in the solution and otherwise raises
    an assertion error.

    Parameters
    ----------
    file : str
        Path to the input file.

    Returns
    -------
    list of str
        A list of words that appear as the first token on each line.
    """
    tags = []

    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # skip empty lines
            first_word = line.split()[0]
            if first_word.isupper():
                tags.append(first_word)

    # Check that the following tags are found
    required_tags = ["NAME", "COMMENT", "TYPE", "DIMENSION", "EDGE_WEIGHT_TYPE", "NUM_VEHICLES", "CAPACITY", "NODES_SECTION", "DEMAND_SECTION", "DEPOT_SECTION"] 

    missing = [tag for tag in required_tags if tag not in tags]
    if missing:
        raise ValueError(f"Missing required tags: {', '.join(missing)}")

    return tags


def verify_vrplib_format(file: str) -> bool:
    """
    Verifies if the file located at the given string is in VRPLIB format (see documentation)
    and is valid. It checks:
        - Existence of all problem specific information
        - Coherency in number of customers
        - Valid time windows (pick up < delivery)
    """
    with open(file, "r") as f:
        lines = f.readlines()

    # check that the file has at least the following sections 
    tags_to_check = ["NAME", "COMMENT", "TYPE", "DIMENSION", "EDGE_WEIGHT_TYPE", "NUM_VEHICLES", "CAPACITY", "NODES_SECTION", "DEMAND_SECTION", "DEPOT_SECTION"] 


def mins_since_midnight(hour: int, mins: int) -> int:
    assert 0 <= hour < 24
    assert 0 <= mins < 60 
    return hour*60 + mins

def read_vrplib_data(file: str, print_data: bool = False) -> dict:
    """
    Parses a benchmark instance in the format proposed by Gerhard Reinelt for VRP variants.
    Information on this protocol can be found at:  http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/ 
    under the `Documentation` link. More information can be found at the documentation of this project 
    (``docs`` folder) under the `Data format` chapter.

    Parameters
    ----------
    file : str
        Path to the input `.txt` file in Cordeau's benchmark format.
    print_data : bool, optional
        If True, prints debug information including file name and parsed quantities.

    Returns
    -------
    dict
        A dictionary containing parsed problem data with the following keys:

        - `name` : str  
          File name.
        - `vehicles` : int  
          Number of vehicles.
        - `capacity` : int  
          Maximum vehicle load (assumed identical for all).
        - `dimension` : int  
          Number of customers (excluding depots).
        - `n_depots` : int  
          Number of depots.
        - `depots` : list[int]  
          Indices of depot nodes.
        - `depot_to_vehicles` : dict[int, list[int]]  
          Mapping from depot ID to list of vehicles assigned to it.
        - `vehicle_to_depot` : dict[int, int]  
          Mapping from vehicle ID to assigned depot.
        - `node_coord` : ndarray  
          Node coordinates (customers + depots).
        - `demand` : ndarray  
          Demand at each node (0 for depots).
        - `pickup_time_window` : list[list[int]]  
          Earliest and latest pickup times.
        - `delivery_time_window` : list[list[int]]  
          Earliest and latest delivery times.
        - `service_time` : list[float]  
          Service duration for each node.
        - `edge_weight` : ndarray  
          Cost matrix derived from Euclidean distances.
    """
    tags = get_all_section_tags(file)

    data = {}
    section = None

    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            tokens = line.split()

            # Section change
            if tokens[0].isupper():
                section = tokens[0]
                data[section] = []
                if len(tokens) > 1:
                    data[section].append(tokens[1:])
            else:
                if section:
                    data[section].append(tokens)

    name = data['NAME'][0][-1]
    vehicles = int(data['NUM_VEHICLES'][0][-1])
    capacity = int(data['CAPACITY'][0][-1])
    dimension = int(data['DIMENSION'][0][-1])

    node_coord = np.zeros((dimension, 2))
    for row in data['NODES_SECTION']:
        idx = int(row[0]) - 1
        x, y = map(float, row[1:3])
        node_coord[idx] = [x, y]

    demand = np.zeros(dimension, dtype=int)
    for row in data['DEMAND_SECTION']:
        idx = int(row[0]) - 1
        d = int(row[1])
        demand[idx] = d

    pickup_time_window = [[None, None] for i in range(dimension)]
    for idx, row in enumerate(data['PICKUP_TIME_WINDOW_SECTION']):
        start_hours = int(row[1].split(":")[0])
        start_mins = int(row[1].split(":")[1])
        end_hours = int(row[2].split(":")[0])
        end_mins = int(row[2].split(":")[1])
        pickup_time_window[idx] = [mins_since_midnight(start_hours, start_mins), mins_since_midnight(end_hours, end_mins)]
        
    delivery_time_window = [[None, None] for i in range(dimension)]
    for idx, row in enumerate(data['DELIVERY_TIME_WINDOW_SECTION']):
        start_hours = int(row[1].split(":")[0])
        start_mins = int(row[1].split(":")[1])
        end_hours = int(row[2].split(":")[0])
        end_mins = int(row[2].split(":")[1])
        delivery_time_window[idx] = [mins_since_midnight(start_hours, start_mins), mins_since_midnight(end_hours, end_mins)]

    depots = []
    for row in data['DEPOT_SECTION']:
        if row[0] == '-1':
            break
        depots.append(int(row[0]))

    n_depots = len(depots)

    depot_to_vehicles, vehicle_to_depot = calculate_depots(depots=depots, n_vehicles=vehicles)

    service_time = [None] * dimension
    if "SERVICE_TIME_SECTION" in tags:
        for row in data['SERVICE_TIME_SECTION']:
            idx = int(row[0])-1
            s = int(row[1])
            service_time[idx] = s
    else:
        print(f"\n SERVICE TIME NOT FOUND \n")
        service_time = [None] * dimension

    edge_weight = cost_matrix_from_coords(node_coord)

    result = {
        'name': name,
        'vehicles': vehicles,
        'capacity': capacity,
        'dimension': dimension,
        'n_depots': n_depots,
        'depots': depots,
        'depot_to_vehicles': depot_to_vehicles,
        'vehicle_to_depot': vehicle_to_depot,
        'node_coord': node_coord,
        'demand': demand,
        'pickup_time_window': pickup_time_window,
        'delivery_time_window': delivery_time_window,
        'service_time': service_time,
        'edge_weight': edge_weight
    }

    if print_data:
        print(f"\nDEBUG: in read_vrplib_data():\n")
        print(f"DEBUG: passed file string: {file}")
        print(f"DEBUG: parsed data: {result}")

    return result



def read_solution_format(file: str, print_data: bool = False) -> dict:
    """
    Read a solution file with the described format.

    Args:
        file (str): Path to the file to be read.
        print_data (bool): If True, print parsed data.

    Returns:
        dict: Parsed data structured as a dictionary.
    """
    with open(file, "r") as f:
        lines = f.readlines()

    # First line contains the cost of the solution
    solution_cost = float(lines[0].strip())

    # Parse the remaining lines for route details
    routes = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 4:
            continue  # Skip malformed lines

        # Extract route details
        day = int(parts[0])
        vehicle = int(parts[1])
        duration = float(parts[2])
        load = float(parts[3])

        # Extract the sequence of customers
        customers = []
        for segment in parts[4:]:
            if '(' in segment and ')' in segment:
                customer, start_time = segment.split('(')
                start_time = float(start_time.strip(')'))
                customers.append((int(customer), start_time))

        # Append route information to the list
        routes.append({
            "day": day,
            "vehicle": vehicle,
            "duration": duration,
            "load": load,
            "customers": customers
        })

    # Compile all data into a dictionary
    data = {
        "solution_cost": solution_cost,
        "routes": routes
    }

    if print_data:
        print("Solution Cost:", solution_cost)
        for route in routes:
            print(f"Day {route['day']}, Vehicle {route['vehicle']}:")
            print(f"  Duration: {route['duration']}, Load: {route['load']}")
            print(f"  Customers: {route['customers']}")

    return data

# Example usage
# data = read_solution_format("path_to_file.txt", print_data=True)


def create_cust_stops_mapping(cust_df: pd.DataFrame, list_of_depot_stops: list) -> dict:
    """
    Creates a dict mapping customer ids to the stop ids in the dataframe.
    The expected dataframe should be in the format given by dynamic_extended_df,
    this condition is tested at the beginning.
    """
    df_cols = cust_df.columns.to_list()
    known_cols = ['cust_id', 'from_stop_id', 'demand', 'from_time_start', 'from_time_end', 'from_req_id', 'to_stop_id', 
                  'to_time_start', 'to_time_end', 'to_req_id', 'service_time', 'call_in_time_slot']
    assert all([col in df_cols for col in known_cols]), "Dataframe columns do not match the expected columns."

    # cust_to_nodes = {int(cust): [] for cust in twc_format_nodes_df["cust_id"].unique()}
    # for index, row in twc_format_nodes_df.iterrows():
    #     # print(f"DEBUG: index: {index}, row: {row}")
    #     cust_to_nodes[row["cust_id"]].append(row["node_id"])
    cust_to_nodes = {int(row["cust_id"]): [int(row["from_stop_id"]), int(row["to_stop_id"])] for _, row in cust_df.iterrows()}
    for dep in list_of_depot_stops:
        cust_to_nodes[dep] = [dep, dep]
    # Manually add second node for depots
    # depots_list = twc_format_nodes_df.loc[twc_format_nodes_df["type"] == "depot", "cust_id"].tolist()
    # for depot in depots_list:
    #     cust_to_nodes[depot].append(cust_to_nodes[depot][0])

    return cust_to_nodes


def create_cust_nodes_mapping(cust_to_stops: dict, stop_id_to_node: dict) -> dict:
    """
    Creates a dict mapping customer ids to the node ids in the dataframe.
    The expected dataframe should be in the format given by dynamic_extended_df,
    this condition is tested at the beginning.
    """

    # filter the depot (from_stop_id = 0)
    # cust_to_stops = {cust: stops for cust, stops in cust_to_stops.items() if stops[0] != 0}
    cust_to_nodes = {cust: [stop_id_to_node[stop] for stop in stops] for cust, stops in cust_to_stops.items()}
    return cust_to_nodes

def create_nodes_cust_mapping(cust_df: pd.DataFrame) -> dict:
    """
    Creates a dict mapping node_ids to the customer ids in the dataframe.
    The expected dataframe should be in the format given by dynamic_extended_df,
    this condition is tested at the beginning.
    """
    raise AssertionError("Not implemented")
    df_cols = cust_df.columns.to_list()
    known_cols = [
        "cust_id",
        "from_node",
        "demand",
        "from_time_start",
        "from_time_end",
        "to_node",
        "to_time_start",
        "to_time_end",
        "service_time",
        "call_in_time_slot",
    ]
    assert all(
        [col in df_cols for col in known_cols]
    ), "Dataframe columns do not match the expected columns."

    used_nodes = [int(node) for node in cust_df["from_node"].unique()]
    used_nodes = used_nodes.extend([int(node) for node in cust_df["to_node"].unique()])

    return nodes_to_cust


def cost_matrix_from_coords(coords: list, cordeau: bool=True) -> list:
    """
    Create a cost matrix from a list of coordinates. Uses the Euclidean distance as cost. 
    If Cordeau notation is used, the first row and column are set to None.
    """
    n = len(coords)
    cost_matrix = np.zeros((n, n))
    if cordeau:
        for i in range(1, n):
            for j in range(1, n):
                cost_matrix[i, j] = round(np.linalg.norm(coords[i] - coords[j]), 2)
        for i in range(n):
            cost_matrix[i, 0] = None
            cost_matrix[0, i] = None
        return cost_matrix
    else:
        for i in range(n):
            for j in range(n):
                cost_matrix[i, j] = round(np.linalg.norm(coords[i] - coords[j]), 2)
        return cost_matrix

def generate_twc_matrix(
    cust_df: pd.DataFrame,
    distances: dict,
    stop_to_nodes: dict,
    cordeau: bool = True,
) -> list:
    """
    Generate the time window compatability matrix matrix. If cordeau is True,
    the first row and column are set to -inf, as customer 0 is not considered
    in the matrix.
        Parameters:
            cust_df: pd.Dataframe
                Dataframe of requests
            distances: list
                List of distances between each pair of customers.
            cordeau: bool
                If True, the first row and column are set to -inf.
        Returns:
            list
                Time window compatibility matrix.
    """
    requests = pd.DataFrame(columns=["req_id", "node_id", "start", "end"])
    cust_df = cust_df.loc[cust_df["demand"] != 0]
    for _, row in cust_df.iterrows():     
        from_row = pd.Series({
            "req_id": row.get("from_req_id"),
            "node_id": stop_to_nodes[row.get("from_stop_id")],
            "start": row.get("from_time_start"),
            "end": row.get("from_time_end")
        })
        to_row = pd.Series({
            "req_id": row.get("to_req_id"),
            "node_id": stop_to_nodes[row.get("to_stop_id")],
            "start": row.get("to_time_start"),
            "end": row.get("to_time_end")
        })
        requests = pd.concat([requests, from_row.to_frame().T], ignore_index=True)
        requests = pd.concat([requests, to_row.to_frame().T], ignore_index=True)

    requests.set_index("req_id")

    # print(f"Requests = \n{requests}")
    # print(f"cust_df = {cust_df.columns}")

    # print(f"DEBUG: distances: \n{distances}")

    twc = {}
    for i, row_i in requests[["req_id", "node_id"]].iterrows():
        node_i = row_i["node_id"]
        for j, row_j in requests[["req_id", "node_id"]].iterrows():
            node_j = row_j["node_id"]
            twi = tuple(requests.loc[i][["start", "end"]])
            twj = tuple(requests.loc[j][["start", "end"]])

            twc[(i, j)] = time_window_compatibility(distances[(node_i, node_j)], twi, twj)
    return twc

# start_idx = 1 if cordeau else 0
# twc = np.zeros_like(distances)
# for i in range(start_idx, distances.shape[0]):
#     for j in range(start_idx, distances.shape[0]):
#         if i != j:
#             twc[i][j] = time_window_compatibility(
#                 distances[i, j], time_windows[i], time_windows[j]
#             )
#         else:
#             twc[i][j] = -np.inf
# if cordeau:
#     for i in range(distances.shape[0]):
#         twc[i][0] = -np.inf
#         twc[0][i] = -np.inf
# return twc


def time_window_compatibility(tij: float, twi: tuple, twj: tuple) -> float:
    """
    Time Window Compatibility (TWC) between a pair of vertices i and j. Based on eq. (12) of
    Wang et al. (2024). Returns the time window compatibility between two customers
    i and j, given their time windows and the travel time between them.
        Parameters:
            tij: float
                Travel time between the two customers.
            twi: tuple
                Time window of customer i.
            twj: tuple
                Time window of customer j.
        Returns:
            float
                Time window compatibility between the two customers.
    """
    (ai, bi) = twi
    (aj, bj) = twj

    if bj > ai + tij:
        return round(min([bi + tij, bj]) - max([ai + tij, aj]), 2)
    else:
        return -np.inf  # Incompatible time windows


def calculate_depots(
    depots: list,
    n_vehicles: int,
    rng: np.random.Generator = rnd.default_rng(SEED),
) -> tuple:
    """
    Calculates the depot assignment for each vehicle based on the number of depots and vehicles.

    - If the number of vehicles equals the number of depots, vehicles are assigned 1:1 to depots.
    - If vehicles > depots, a round-robin assignment is used.
    - If vehicles < depots, a random subset of depots is chosen, and each vehicle is assigned to one.

    The result is returned as two dictionaries:
    `depot_to_vehicles` and `vehicle_to_depot`.

    Examples
    --------
    1) One vehicle per depot (exact match):

        >>> depots = [44, 52]
        >>> n_vehicles = 2
        DEPOT   | VEHICLE
        44      | 0
        52      | 1
        depot_to_vehicles = {44: [0], 52: [1]}
        vehicle_to_depot = {0: 44, 1: 52}

    2) More vehicles than depots (round robin):

        >>> depots = [44, 52]
        >>> n_vehicles = 5
        DEPOT   | VEHICLE
        44      | 0, 2, 4
        52      | 1, 3

    3) More depots than vehicles (random assignment):

        >>> depots = [44, 52, 38, 7, 78]
        >>> n_vehicles = 2
        DEPOT   | VEHICLE
        44      | 1
        38      | 0

    Parameters
    ----------
    depots : list
        List of depot indices.

    n_vehicles : int
        Number of vehicles to assign.

    rng : np.random.Generator, optional
        Random number generator (default: seeded RNG).

    Returns
    -------
    tuple
        A tuple containing:
        
        - depot_to_vehicles : dict[int, list[int]]
        - vehicle_to_depot : dict[int, int]
    """
    # check that all depots are unique
    assert len(set(depots)) == len(depots), "Depot IDs must be unique"
    n_depots = len(depots)

    dict_depot_to_vehicles = {depot: [] for depot in depots}
    dict_vehicle_to_depot = {vehicle: None for vehicle in range(n_vehicles)}
    # vehicle i -> depot i
    if n_vehicles == n_depots:
        for i, depot in enumerate(depots):
            dict_depot_to_vehicles[depot].append(i)
            dict_vehicle_to_depot[i] = depot

    elif n_vehicles > n_depots:
        # Round robin assignment
        for vehicle in range(n_vehicles):
            depot = depots[vehicle % n_depots]
            dict_depot_to_vehicles[depot].append(vehicle)
            dict_vehicle_to_depot[vehicle] = depot
    else:
        # Random assignment
        assigned_depots = rng.choice(depots, size=n_vehicles, replace=False)
        for vehicle in range(n_vehicles):
            depot = assigned_depots[vehicle]
            dict_depot_to_vehicles[depot].append(vehicle)
            dict_vehicle_to_depot[vehicle] = int(depot)

    return dict_depot_to_vehicles, dict_vehicle_to_depot


def read_solution_format(file: str, print_data: bool = False) -> dict:
    """
    Read a solution file with the described format.

    Args:
        file (str): Path to the file to be read.
        print_data (bool): If True, print parsed data.

    Returns:
        dict: Parsed data structured as a dictionary.
    """
    with open(file, "r") as f:
        lines = f.readlines()

    # First line contains the cost of the solution
    solution_cost = float(lines[0].strip())

    # Parse the remaining lines for route details
    routes = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 4:
            continue  # Skip malformed lines

        # Extract route details
        day = int(parts[0])
        vehicle = int(parts[1])
        duration = float(parts[2])
        load = float(parts[3])
        # Extract the sequence of customers
        customers = []
        start_times = []
        for segment in parts[4:]:
            if "(" not in segment and ")" not in segment:
                customers.append(int(segment))
            if "(" in segment and ")" in segment:
                _, start_time = segment.split("(")
                start_time = float(start_time.strip(")"))
                start_times.append(start_time)

        # Append route information to the list
        routes.append(
            {
                "day": day,
                "vehicle": vehicle,
                "duration": duration,
                "load": load,
                "customers": customers,
                "star_times": start_times,
            }
        )

    # Compile all data into a dictionary
    data = {"solution_cost": solution_cost, "routes": routes, "n_vehicles": len(routes)}

    if print_data:
        print("Solution Cost:", solution_cost)
        for route in routes:
            print(f"Day {route['day']}, Vehicle {route['vehicle']}:")
            print(f"  Duration: {route['duration']}, Load: {route['load']}")
            print(f"  Customers: {route['customers']}")

    return data

def dynamic_df_from_dict(
        data: dict,
        static: bool = False, 
        n_steps:int = 20, seed: int = 0
    ) -> pd.DataFrame:
    """
    Convert a data dictionary to a pandas DataFrame for dynamic customer and depot information.
    If static attribute is True, call in time for each customer is sampled from a 
    gamma distribution. This is done to simulate dynamic customer requests, placing
    more customers in the beginning of the time horizon. Depots have call in time 0.
    Parameters:
        data (dict): Data dictionary.
        static (bool): If True, all call in times are set to 0.
        n_steps (int): Number of time steps.
        seed (int): Random seed.
    Returns:
        pd.DataFrame: DataFrame with dynamic customer and depot information.
        Columns:
            id: Customer ID.
            x: x-coordinate.
            y: y-coordinate.
            demand: Customer demand.
            start_time: Time window start.
            end_time: Time window end.
            service_time: Customer service time.
            call_in_time_slot: Time slot when customer calls in.
            route: Route customer is assigned to.
            done: Boolean indicating if customer has been satisfied.
    """
    np.random.seed(seed)
    n = data["dimension"] + data["n_depots"]+1
    
    data_df = pd.DataFrame(
        {
            "id": [i for i in range(n)],
            "x": [coord[0] for coord in data["node_coord"][:n]],
            "y": [coord[1] for coord in data["node_coord"][:n]],
            "demand": data["demand"][:n],
            "start_time": [tw[0] for tw in data["time_window"][:n]],
            "end_time": [tw[1] for tw in data["time_window"][:n]],
            "service_time": data["service_time"][:n],
            "call_in_time_slot": np.zeros(n, dtype=int),
            "route": [None]*n,
            "done": [False]*n,
        }
    )
    if static:
        return data_df
    
    # Parameters of gamma  distribution
    k = 0.5  # Shape parameter
    mean = n_steps/4  # Desired mean    
    theta = mean / k  # Scale parameter
    samples = np.random.gamma(k, scale=theta, size=data["dimension"]+data["n_depots"]+1)

    data_df["call_in_time_slot"] = samples.astype(int)
    data_df.iloc[-data["n_depots"]:, data_df.columns.get_loc("call_in_time_slot")] = 0

    # Preindexing the dataframe using 'id'
    data_df.set_index("id", inplace=True)

    return data_df

def generate_dynamic_df(file: str, static: bool = False, print_data: bool = False, n_steps = 20, seed = 0) -> pd.DataFrame:
    """
    Reads file and converts it to a dynamic customers and depots dataframe.
    Call in time for each customer is sampled from a gamma distribution, unless static is True.
    Parameters:
        file (str): Path to the file to be read.
        static (bool): If True, all call in times are set to 0.
        print_data (bool): If True, print parsed data.
        n_steps (int): Number of time steps.
        seed (int): Random seed.
    """

    data = read_cordeau_data(
            file, 
            print_data=print_data,
            # depot_ids = [1259, 259],
            )
    return dynamic_df_from_dict(
        data, static=static, n_steps=n_steps, seed=seed
    )

def get_ids_of_time_slot(customer_df: pd.DataFrame, time_slot: int) -> list:
    """
    Get the customer IDs of customers that call in the given time slot.
    """
    assert time_slot >= 0, "Time slot must be a non-negative integer."

    matching_ids = customer_df.loc[
            customer_df['call_in_time_slot'] == time_slot
    ].index.tolist()    #customer ids are the index column of the customer_df dataframe in
                        #format given by 'generate_dynamic_df'

    # depots ids are the elements that have demand = 0 and we need to remove them from the list
    depot_ids = customer_df.loc[customer_df["demand"] == 0].index.tolist()
    requested_ids = [i for i in matching_ids if i not in depot_ids]
    return requested_ids 

def get_initial_data(cust_df: pd.DataFrame) -> pd.DataFrame:
    """
        Returns subset of initial df containing customers
        and depots from the first time slot.
    """
    return cust_df.loc[cust_df["call_in_time_slot"] == 0]


def create_depots_dict(
    list_of_depot_stops: list,
    cust_df: pd.DataFrame,
    cust_to_nodes: dict,
    num_vehicles: int = None,
) -> dict:
    """
    Create a dictionary with depot information.
    list_of_depot_stop: list
        List of depot stops.
    cust_df: pd.DataFrame
        DataFrame with customer information.
    cust_to_nodes: dict
        Dictionary mapping customer ids to node ids.
    num_vehicles: int
        Number of vehicles.
    Returns:
        depots_dict: dict
    Dictionary with depot information. Attributes:
    num_depots: int
        Number of depots.
    depot_to_vehicles: dict
        Dictionary mapping depots to vehicles.
    vehicle_to_depot: dict
        Dictionary mapping vehicles to depots.
    coords: np.array
        Array with the coordinates of the depots.
    depots_indices: list
        List with the indices of the dep (customer indices)
    """
    if isinstance(cust_df, pd.DataFrame):
        # depots_cust_idx = cust_df.loc[cust_df["demand"] == 0, "cust_id"].tolist()
        depots_cust_idx = list_of_depot_stops
        depots_dict = {
            "num_depots": len(list_of_depot_stops),
            "depot_to_vehicles": {},
            "vehicle_to_depot": {},
            # "coords": df.loc[df["demand"] == 0, ["x", "y"]].values,
            "depots_indices": [cust for cust in depots_cust_idx],
        }

        depots_dict["depot_to_vehicles"], depots_dict["vehicle_to_depot"] = (
            calculate_depots(
                n_depots=len(list_of_depot_stops),
                depots=[cust for cust in depots_cust_idx],
                n_vehicles=num_vehicles,
            )
        )

        return depots_dict
    else:
        raise ValueError("Data must be a DataFrame.")

    # elif isinstance(data, dict):
    #     depots_dict = {
    #         "num_depots": data["n_depots"],
    #         "depot_to_vehicles": data["depot_to_vehicles"],
    #         "vehicle_to_depot": data["vehicle_to_depot"],
    #         "coords": data["node_coord"][data["dimension"] + 1 :],
    #         "depots_indices": data["depots"],
    #     }

    # return depots_dict


def dynamic_extended_df(data: Union[pd.DataFrame, str]) -> pd.DataFrame:
    """
    Create a df where each entry is either a pick up or delivery node.
    Parameters:
        data (pd.DataFrame or str): Dataframe or path to file.
    Returns:
        pd.DataFrame: Extended dataframe.
    """
    if isinstance(data, str):
        raise ValueError("Not implemented yet!")
        expected_columns = ['id', 'x', 'y', 'demand', 'pstart_time', 'pend_time', 'dx', 'dy', 'dstart_time', 'dend_time', 'service_time', 'call_in_time_slot', 'route', 'done', 'id.1']
        init_data = pd.read_csv(data)
        init_data.fillna(-1, inplace=True)
        if not np.array_equal(init_data.columns.to_list(), expected_columns):
            raise ValueError(f"CSV columns do not match the expected columns. Found: {init_data.columns.tolist()}")
    elif isinstance(data, pd.DataFrame):
        init_data = data

    else:
        print("Error: Data must be a DataFrame or a path to a file.")
        return None

    # exclude depots
    sub_df = init_data.loc[init_data["demand"] != 0]
    new_df_list = []
    for _, row in sub_df.iterrows():
        pickup_series = pd.Series(
            [
                row["from_stop_id"],
                row["cust_id"],
                # row["x"],
                # row["y"],
                row["demand"],
                row["from_time_start"],
                row["from_time_end"],
                row["service_time"],
                row["call_in_time_slot"],
                row["route"],
                "pickup",
            ],
            index=[
                "stop_id",
                "cust_id",
                # "x",
                # "y",
                "demand",
                "start_time",
                "end_time",
                "service_time",
                "call_in_time_slot",
                "route",
                "type",
            ],
        )

        delivery_series = pd.Series(
            [
                row["to_stop_id"],
                row["cust_id"],
                # row["dx"],
                # row["dy"],
                row["demand"],
                row["to_time_start"],
                row["to_time_end"],
                row["service_time"],
                row["call_in_time_slot"],
                row["route"],
                "delivery",
            ],
            index=[
                "stop_id",
                "cust_id",
                # "x",
                # "y",
                "demand",
                "start_time",
                "end_time",
                "service_time",
                "call_in_time_slot",
                "route",
                "type",
            ],
        )

        new_df_list.append(pickup_series)
        new_df_list.append(delivery_series)

    new_df = pd.concat(new_df_list, axis=1).T.reset_index(drop=True)
    # # depots series
    # depots_sub_df = init_data.loc[init_data["demand"] == 0]
    # depots_list = []
    # for _, row in depots_sub_df.iterrows():
    #     depot_series = pd.Series(
    #         [
    #             row["cust_id"],
    #             # row["x"],
    #             # row["y"],
    #             row["demand"],
    #             row["from_time_start"],
    #             row["from_time_end"],
    #             row["service_time"],
    #             row["call_in_time_slot"],
    #             row["route"],
    #             "depot",
    #         ],
    #         index=[
    #             "cust_id",
    #             # "x",
    #             # "y",
    #             "demand",
    #             "start_time",
    #             "end_time",
    #             "service_time",
    #             "call_in_time_slot",
    #             "route",
    #             "type",
    #         ],
    #     )
    #     depots_list.append(depot_series)

    # depots_df = pd.concat(depots_list, axis=1).T.reset_index(drop=True)

    # new_df = pd.concat([new_df, depots_df])

    # new_df.insert(0, "node_id", new_df.pop("node_id"))
    new_df.set_index("stop_id", inplace=True)
    # new_df.reset_index(drop=True, inplace=True)
    # new_df.index += 1
    # new_df.fillna(-1, inplace=True)
    new_df = new_df.infer_objects(copy=False)
    new_df["route"] = new_df["route"].fillna(-1).astype(int)
    # new_df = new_df.astype({"node_id": int, "cust_id": int, "demand": int, "start_time": int, "end_time": int, "service_time": int, "call_in_time_slot": int, "route": int})

    new_df[
        [
            "cust_id",
            "demand",
            "start_time",
            "end_time",
            "service_time",
            "call_in_time_slot",
        ]
    ] = new_df[
        [
            "cust_id",
            "demand",
            "start_time",
            "end_time",
            "service_time",
            "call_in_time_slot",
        ]
    ].astype(int)
    return new_df


this_file_path = pathlib.Path(__file__).parent.resolve()
data_file_path = this_file_path / "../../data/c-mdvrptw/pr12"

data = read_cordeau_data(
     str(data_file_path), 
     #depot_ids=[1], 
     print_data=False
)
# bks = read_solution_format("./data/c-mdvrptw-sol/pr02.res", print_data=True)
# test_data = read_cordeau_data(str(data_file_path), print_data=False)
# d_data = generate_dynamic_df(data_file_path, print_data=False, seed=0)
