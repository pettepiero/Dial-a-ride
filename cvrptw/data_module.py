import numpy as np
import numpy.random as rnd


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
    data_dict["vehicles"] = m
    data_dict["capacity"] = int(depots_info[0][1])
    data_dict["dimension"] = n  # Number of customers only, not depots
    data_dict["n_depots"] = t
    data_dict["depot_to_vehicles"] = {}  # {depot: [vehicles]}
    data_dict["vehicle_to_depot"] = {}  # {vehicle: depot}
    data_dict["depots"] = [i for i in range(n, n + t)]
    data_dict["node_coord"] = np.array([rows[1:3] for rows in customers])
    data_dict["node_coord"] = np.append(
        data_dict["node_coord"], np.array([rows[1:3] for rows in depots]), axis=0
    )
    data_dict["demand"] = [int(row[4]) for row in customers]
    data_dict["demand"] = np.array(data_dict["demand"] + [0 for row in depots])
    begin_times = [row[11] for row in customers]
    end_times = [row[12] for row in customers]
    data_dict["time_window"] = [[a, b] for a, b in zip(begin_times, end_times)]
    data_dict["time_window"] += [[0, END_OF_DAY] for row in depots]

    data_dict["service_time"] = [row[3] for row in customers]
    data_dict["service_time"] += [row[3] for row in depots]
    data_dict["edge_weight"] = cost_matrix_from_coords(data_dict["node_coord"])
    calculate_depots(data_dict)


    if print_data:
        print("Problem type: ", problem_type)
        print("Number of vehicles: ", m)
        print("Number of customers: ", n)
        print("Number of days/depots/vehicle types: ", t)

    return data_dict


def cost_matrix_from_coords(coords: list) -> list:
    """
    Create a cost matrix from a list of coordinates. Uses the Euclidean distance as cost.
    """
    n = len(coords)
    cost_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cost_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])
    return cost_matrix


def calculate_depots(data):
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
        for depot in depots:
            data["depot_to_vehicles"][depot].append(depot)
            data["vehicle_to_depot"][depot] = depot

    elif n_vehicles > n_depots:
        # Round robin assignment
        for vehicle in range(n_vehicles):
            depot = vehicle % n_depots
            # print(f"Vehicle {vehicle} assigned to depot {depot}.")
            # print(f"Depot to vehicles: {data['depot_to_vehicles']}")
            # print(f"After, depot to vehicles: {data['depot_to_vehicles']}")
            # print(f"n_customer + depot: {n_customers + depot}")
            data["depot_to_vehicles"][n_customers + depot].append(vehicle)
            data["vehicle_to_depot"][vehicle] = n_customers + depot
    else:
        # Random assignment
        depots = rnd.choice(depots, size=n_vehicles, replace=False)
        for vehicle in range(n_vehicles):
            depot = depots[vehicle]
            data["depot_to_vehicles"][depot].append(vehicle)
            data["vehicle_to_depot"][vehicle] = int(depot)


data = read_cordeau_data("./data/c-mdvrptw/pr02", print_data=True)
