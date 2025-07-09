import numpy as np
import pandas as pd
from lib.myvrplib.data_module import *

END_OF_DAY = 1000

def read_cordeau_data(file: str, print_data: bool = False) -> dict:
    """
    Read the Cordeau et al. (2001) benchmark data.
    """
    filename = file.split("/")[-1]
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
    problem_type = type_dict.get(6)
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
    depot_to_vehicles, vehicle_to_depot = calculate_depots(depots=data_dict['depots'], n_vehicles=data_dict['vehicles'])

    data_dict['depot_to_vehicles'] = depot_to_vehicles
    data_dict['vehicle_to_depot'] = vehicle_to_depot

    if print_data:
        print("Problem type: ", problem_type)
        print("Number of vehicles: ", m)
        print("Number of customers: ", n)
        print("Number of days/depots/vehicle types: ", t)

    return data_dict

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

def read_vrplib_data(file: str, print_data: bool = False, seed: int = None) -> dict:
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
    seed : int, optional
        Seed for random number generator. If None (default), no seed is used.
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
    verify_vrplib_format(file)
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

    depot_to_vehicles, vehicle_to_depot = calculate_depots(depots=depots, n_vehicles=vehicles, seed=seed)

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

    tags = get_all_section_tags(file)
    for tag in tags_to_check:
        assert tag in tags, f"Tag {tag} not found in data"




##############################################################################################################
##############################################################################################################

# Example usage
# data = read_solution_format("path_to_file.txt", print_data=True)

# Datasets
cvrptw_data = read_cordeau_data(
    "./data/c-mdvrptw/pr12", print_data=False
)

cvrppdtw_data = read_vrplib_data(
"/home/pettepiero/tirocinio/dial-a-ride/tests/cvrppdtw/vrplib_test_data.knd"
)

# bks = read_solution_format("./data/c-mdvrptw-sol/pr02.res", print_data=True)
test_data = read_cordeau_data(
    "./data/c-mdvrptw/pr12",
    print_data=False,
)
