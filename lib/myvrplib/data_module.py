import numpy as np
import numpy.random as rnd
import copy
import pandas as pd

SEED = 1234


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

def calculate_depots(
    depots: list,
    n_vehicles: int,
    seed: int = SEED,
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

    seed : int, optional
        Seed for random number generator. If None, no fixed seed is used

    Returns
    -------
    tuple
        A tuple containing:
        
        - depot_to_vehicles : dict[int, list[int]]
        - vehicle_to_depot : dict[int, int]
    """
    if seed is None:
        rng = rnd.default_rng()
    else:
        rng = rnd.default_rng(seed)

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
    data_df['id'] = data_df.index
    
    return data_df


def generate_dynamic_df(data: dict, static: bool = False, print_data: bool = False, n_steps = 20, seed = 0) -> pd.DataFrame:
    """
    Reads data and converts it to a dynamic customers and depots dataframe.
    Call in time for each customer is sampled from a gamma distribution, unless static is True.
    Parameters:
        data (dict): data dict to convert
        static (bool): If True, all call in times are set to 0.
        print_data (bool): If True, print parsed data.
        n_steps (int): Number of time steps.
        seed (int): Random seed.
    """
    return dynamic_df_from_dict(
        data, static=static, n_steps=n_steps, seed=seed
    )

def get_ids_of_time_slot(customer_df: pd.DataFrame, time_slot: int) -> list:
    """
    Get the IDs of customers that call in the given time slot.
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

def create_depots_dict(data: dict) -> dict:
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

    depots_dict = {
        "num_depots": data["n_depots"],
        "depot_to_vehicles": data["depot_to_vehicles"],
        "vehicle_to_depot": data["vehicle_to_depot"],
        "coords": data["node_coord"][data["dimension"] + 1 :],
        "depots_indices": data["depots"],
    }

    return depots_dict

def mins_since_midnight(hour: int, mins: int) -> int:
    assert 0 <= hour < 24
    assert 0 <= mins < 60 
    return hour*60 + mins

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

