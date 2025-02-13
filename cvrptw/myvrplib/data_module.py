import numpy as np
import numpy.random as rnd
import copy
import pandas as pd
from typing import Union


END_OF_DAY = 1000
SEED = 1234

# macro for mapping customer to row in the dataframe where each row corresponds
# to a node, therefore each customer has 2 rows (pick up and delivery)
cust_row = lambda id, pickup: id * 2 - 1 if pickup else id * 2


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
    data_dict["depot_to_vehicles"], data_dict["vehicle_to_depot"] = calculate_depots(
        data_dict, n_vehicles=data_dict["vehicles"]
    )

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

# Example usage
# data = read_solution_format("path_to_file.txt", print_data=True)


def create_cust_nodes_mapping(twc_format_nodes_df: pd.DataFrame) -> dict:
    """
    Creates a dict mapping customer ids to the node_ids in the dataframe.
    The expected dataframe should be in the format given by dynamic_extended_df,
    this condition is tested at the beginning.
    """
    df_cols = twc_format_nodes_df.columns.to_list()
    known_cols = ['node_id', 'cust_id', 'x', 'y', 'demand', 'start_time', 'end_time', 'service_time', 'call_in_time_slot', 'route', 'type']
    assert all([col in df_cols for col in known_cols]), "Dataframe columns do not match the expected columns."

    cust_to_nodes = {cust: [] for cust in twc_format_nodes_df["cust_id"].unique()}
    print(f"DEBUG: len(twc_format_nodes_df): {len(twc_format_nodes_df)}")
    for index, row in twc_format_nodes_df.iterrows():
        # print(f"DEBUG: index: {index}, row: {row}")
        cust_to_nodes[row["cust_id"]].append(row["node_id"])

    # Manually add second node for depots
    depots_list = twc_format_nodes_df.loc[twc_format_nodes_df["type"] == "depot", "cust_id"].tolist()
    for depot in depots_list:
        cust_to_nodes[depot].append(cust_to_nodes[depot][0])

    return cust_to_nodes


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
    time_windows: list,
    distances: np.ndarray,
    cordeau: bool = True,
) -> list:
    """
    Generate the time window compatability matrix matrix. If cordeau is True,
    the first row and column are set to -inf, as customer 0 is not considered
    in the matrix.
        Parameters:
            time_windows: list
                List of time windows for each customer.
            distances: list
                List of distances between each pair of customers.
            cordeau: bool
                If True, the first row and column are set to -inf.
        Returns:
            list
                Time window compatibility matrix.
    """
    start_idx = 1 if cordeau else 0
    twc = np.zeros_like(distances)
    for i in range(start_idx, distances.shape[0]):
        for j in range(start_idx, distances.shape[0]):
            if i != j:
                twc[i][j] = time_window_compatibility(
                    distances[i, j], time_windows[i], time_windows[j]
                )
            else:
                twc[i][j] = -np.inf
    if cordeau:
        for i in range(distances.shape[0]):
            twc[i][0] = -np.inf
            twc[0][i] = -np.inf
    return twc


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
    data: Union[dict, pd.DataFrame], 
    n_vehicles: int,
    rng: np.random.Generator = rnd.default_rng(SEED),
) -> tuple:
    """
    Calculate the depot index for the vehicles. If the number of vehicles is equal to the number of depots,
    then vehicle i is mapped to depot i. If the number of vehicles is greater than the number of depots, then
    round robin assignment is used. If the number of vehicles is less than the number of depots, then random
    assignment is used, but load balancing between depots is guaranteed. The mapping is stored in the data
    dictionaries "depot_to_vehicles" and "vehicle_to_depot".
    Parameters:
        data (dict or pd.DataFrame): Data dictionary or DataFrame.
        rng (np.random.Generator): Random number generator.
        n_vehicles (int): Number of vehicles.
    Returns:
        tuple: Tuple of dicts (depot_to_vehicles, vehicle_to_depot).
    """
    if isinstance(data, dict):
        n_depots = data["n_depots"]
        depots = data["depots"]
    elif isinstance(data, pd.DataFrame):
        n_depots = data.loc[data["demand"] == 0, "id"].count()
        depots = data.loc[data["demand"] == 0, "id"].tolist()

    dict_depot_to_vehicles = {depot: [] for depot in depots}
    dict_vehicle_to_depot = {vehicle: None for vehicle in range(n_vehicles)}
    
    # vehicle i -> depot i
    if n_vehicles == n_depots:
        for depot in depots:
            dict_depot_to_vehicles[depot].append(depot)
            dict_vehicle_to_depot[depot] = depot

    elif n_vehicles > n_depots:
        # Round robin assignment
        for vehicle in range(n_vehicles):
            depot = vehicle % n_depots
            dict_depot_to_vehicles[depot].append(vehicle)
            dict_vehicle_to_depot[vehicle] = depot
    else:
        # Random assignment
        depots = rng.choice(depots, size=n_vehicles, replace=False)
        for vehicle in range(n_vehicles):
            depot = depots[vehicle]
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
    data_df['id'] = data_df.index
    
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
    data = read_cordeau_data(file, print_data=print_data)
    return dynamic_df_from_dict(
        data, static=static, n_steps=n_steps, seed=seed
    )

def get_ids_of_time_slot(customer_df: pd.DataFrame, time_slot: int) -> list:
    """
    Get the customer IDs of customers that call in the given time slot.
    """
    assert time_slot >= 0, "Time slot must be a non-negative integer."

    indices = customer_df.loc[
        customer_df["call_in_time_slot"] == time_slot, "id"
    ].tolist()
    # depots ids are the elements that have demand = 0 and we need to remove them from the list
    depots = customer_df.loc[customer_df["demand"] == 0, "id"].tolist()
    indices = [i for i in indices if i not in depots]
    return indices

def get_initial_data(cust_df: pd.DataFrame) -> pd.DataFrame:
    """
        Returns subset of initial df containing customers
        and depots from the first time slot.
    """
    return cust_df.loc[cust_df["call_in_time_slot"] == 0]

def create_depots_dict(data: Union[dict, pd.DataFrame], num_vehicles: int=None) -> dict:
    """
    Create a dictionary with depot information.
    Attributes:
        num_depots: int
            Number of depots.
        depot_to_vehicles: dict
            Dictionary mapping depots to vehicles.
        vehicle_to_depot: dict
            Dictionary mapping vehicles to depots.
        coords: np.array
            Array with the coordinates of the depots.
        depots_indices: list
            List with the indices of the dep
    """
    if isinstance(data, pd.DataFrame):
        depots_dict = {
            "num_depots": data["demand"].value_counts()[0],
            "depot_to_vehicles": {},
            "vehicle_to_depot": {},
            "coords": data.loc[data["demand"] == 0, ["x", "y"]].values,
            "depots_indices": data.loc[data["demand"] == 0, "id"].tolist(),
        }

        depots_dict["depot_to_vehicles"], depots_dict["vehicle_to_depot"] = (
            calculate_depots(data, n_vehicles=num_vehicles)
        )
        return depots_dict

    elif isinstance(data, dict):
        depots_dict = {
            "num_depots": data["n_depots"],
            "depot_to_vehicles": data["depot_to_vehicles"],
            "vehicle_to_depot": data["vehicle_to_depot"],
            "coords": data["node_coord"][data["dimension"] + 1 :],
            "depots_indices": data["depots"],
        }
    return depots_dict


def dynamic_extended_df(data: Union[pd.DataFrame, str]) -> pd.DataFrame:
    """
    Create a df where each entry is either a pick up or delivery node.
    Parameters:
        data (pd.DataFrame or str): Dataframe or path to file.
    Returns:
        pd.DataFrame: Extended dataframe.
    """
    if isinstance(data, str):
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
                row["id"],
                row["x"],
                row["y"],
                row["demand"],
                row["pstart_time"],
                row["pend_time"],
                row["service_time"],
                row["call_in_time_slot"],
                row["route"],
                "pickup",
            ],
            index=[
                "cust_id",
                "x",
                "y",
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
                row["id"],
                row["dx"],
                row["dy"],
                row["demand"],
                row["dstart_time"],
                row["dend_time"],
                row["service_time"],
                row["call_in_time_slot"],
                row["route"],
                "delivery",
            ],
            index=[
                "cust_id",
                "x",
                "y",
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
    # depots series
    depots_sub_df = init_data.loc[init_data["demand"] == 0]
    depots_list = []
    for _, row in depots_sub_df.iterrows():
        depot_series = pd.Series(
            [
                row["id"],
                row["x"],
                row["y"],
                row["demand"],
                row["pstart_time"],
                row["pend_time"],
                row["service_time"],
                row["call_in_time_slot"],
                row["route"],
                "depot",
            ],
            index=[
                "cust_id",
                "x",
                "y",
                "demand",
                "start_time",
                "end_time",
                "service_time",
                "call_in_time_slot",
                "route",
                "type",
            ],
        )
        depots_list.append(depot_series)

    depots_df = pd.concat(depots_list, axis=1).T.reset_index(drop=True)

    new_df = pd.concat([new_df, depots_df])

    new_df["node_id"] = range(len(new_df))
    new_df.insert(0, "node_id", new_df.pop("node_id"))
    new_df.reset_index(drop=True, inplace=True)
    # new_df.index += 1
    # new_df.fillna(-1, inplace=True)
    new_df = new_df.infer_objects(copy=False)
    new_df["route"] = new_df["route"].fillna(-1).astype(int)
    # new_df = new_df.astype({"node_id": int, "cust_id": int, "demand": int, "start_time": int, "end_time": int, "service_time": int, "call_in_time_slot": int, "route": int})

    return new_df


data = read_cordeau_data(
    "./data/c-mdvrptw/pr12", print_data=False
)
# bks = read_solution_format("./data/c-mdvrptw-sol/pr02.res", print_data=True)
test_data = read_cordeau_data(
    "./data/c-mdvrptw/pr12",
    print_data=False,
)

d_data = generate_dynamic_df(
    "./data/c-mdvrptw/pr12", print_data=False, seed=0
)
