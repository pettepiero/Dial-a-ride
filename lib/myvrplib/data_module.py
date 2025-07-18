import numpy as np
import numpy.random as rnd
import copy
import pandas as pd

END_OF_DAY = 1000
SEED = 1234

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

    key = int(data[0].split()[0])
    problem_type = type_dict[key]  # Problem type
    assert problem_type == "MDVRPTW", f"Available data is for {problem_type} and not for MDVRPTW"
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
    calculate_depots(data_dict)

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
    data, rng: np.random.Generator = rnd.default_rng(SEED)
):
    """
    Calculate the depot index for the vehicles. If the number of vehicles is equal to the number of depots,
    then vehicle i is mapped to depot i. If the number of vehicles is greater than the number of depots, then
    round robin assignment is used. If the number of vehicles is less than the number of depots, then random
    assignment is used, but load balancing between depots is guaranteed. The mapping is stored in the data
    dictionaries "depot_to_vehicles" and "vehicle_to_depot".
    """
    n_customers = data["dimension"]
    n_vehicles = data["vehicles"]
    n_depots = data["n_depots"]
    depots = data["depots"]
    # print(f"Number of customers: {n_customers}")
    # print(f"Before, depot to vehicles: {data['depot_to_vehicles']}")
    # Initialization of the dictionaries
    for depot in depots:
        data["depot_to_vehicles"][depot] = []
    for vehicle in range(n_vehicles):
        data["vehicle_to_depot"][vehicle] = None
    # vehicle i -> depot i
    if n_vehicles == n_depots:
        for i, depot in enumerate(depots):
            #data["depot_to_vehicles"][depot].append(depot)
            #data["vehicle_to_depot"][depot] = depot
            data["depot_to_vehicles"][depot].append(i)
            data["vehicle_to_depot"][i] = depot

    elif n_vehicles > n_depots:
        # Round robin assignment
        for vehicle in range(n_vehicles):
            depot = depots[vehicle % n_depots]
            #data["depot_to_vehicles"][n_customers + depot].append(vehicle)
            #data["vehicle_to_depot"][vehicle] = n_customers + depot
            data["depot_to_vehicles"][depot].append(vehicle)
            data["vehicle_to_depot"][vehicle] = depot
    else:
        # Random assignment
        depots = rng.choice(depots, size=n_vehicles, replace=False)
        for vehicle in range(n_vehicles):
            depot = depots[vehicle]
            data["depot_to_vehicles"][depot].append(vehicle)
            data["vehicle_to_depot"][vehicle] = int(depot)


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

#cvrptw_data = read_cordeau_data(
#    "./data/c-mdvrptw/pr12", print_data=False
#)
## bks = read_solution_format("./data/c-mdvrptw-sol/pr02.res", print_data=True)
#test_data = read_cordeau_data(
#    "./data/c-mdvrptw/pr12",
#    print_data=False,
#)
#
#d_data = generate_dynamic_df(
#    "./data/c-mdvrptw/pr12", print_data=False, seed=0
#)
