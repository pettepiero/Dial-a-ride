import numpy.random as rnd

from alns import ALNS
from alns.accept import RecordToRecordTravel
from alns.select import *
from alns.stop import MaxIterations

from lib.myvrplib.myvrplib import LOGGING_LEVEL
from lib.myvrplib.data_module import read_cordeau_data
from lib.myvrplib.vrpstates import CvrptwState
from lib.initial_solutions.initial_solutions import nearest_neighbor_tw
from lib.operators.destroy import *
from lib.operators.repair import *
from lib.operators.wang_operators import *
from lib.output.analyze_solution import verify_time_windows
from lib.myvrplib.input_output import print_results_dict, parse_options, print_cvrptw_dataset
from lib.output.video import generate_video
#NUM_ITERATIONS = 100
NUM_ITERATIONS = 50 

# logging setup
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=LOGGING_LEVEL)

degree_of_destruction = 0.05

def main():
    args = parse_options()
    print(f"Arguments: {args}")
    
    if args.seed:
        print(f"Initializing with explicit seed: {args.seed}")
        alns = ALNS(rnd.default_rng(args.seed))
    else:
        print(f"Initializing ALNS without explicit seed")
        alns = ALNS(rnd.default_rng())

    dataset_name = args.dataset
    valid_datasets = ["pr02",  "pr04",  "pr06",  "pr08",  "pr10",  "pr12",  "pr14",  "pr16",  "pr18",  "pr20", "pr01", "pr03", "pr05", "pr07",  "pr09",  "pr11",  "pr13",  "pr15",  "pr17",  "pr19"]
    assert dataset_name in valid_datasets, f"Dataset {dataset_name} not found in ./data/c-mdvrptw"
    dataset_full_path = "./data/c-mdvrptw/" + dataset_name
    print(f"Chosen dataset: {dataset_full_path}")

    data = read_cordeau_data(
        dataset_full_path, print_data=False
    )
    print_cvrptw_dataset(data)


    # Set up ALNS operators
    alns.add_destroy_operator(random_removal)
    alns.add_destroy_operator(random_route_removal)
    alns.add_destroy_operator(cost_reducing_removal)
    alns.add_destroy_operator(worst_removal)
    alns.add_destroy_operator(exchange_reducing_removal) 
    # alns.add_destroy_operator(shaw_removal)   #to be implemented
    alns.add_repair_operator(greedy_repair_tw)
    alns.add_repair_operator(wang_greedy_repair)

    init = CvrptwState(dataset=data)
    initial_solution = nearest_neighbor_tw(state=init, initial_time_slot=False)
    select = RouletteWheel(
            scores=[25, 5, 1, 0], 
            decay=0.8, 
            num_destroy=5,
            num_repair=2
            )
    # select = RandomSelect(num_destroy=4, num_repair=2)
    accept = RecordToRecordTravel.autofit(
        initial_solution.objective(), 0.02, 0, NUM_ITERATIONS
    )
    stop = MaxIterations(NUM_ITERATIONS)

    result, *_ = (
        alns.iterate(initial_solution, select, accept, stop, data=data, save_plots=args.video)
    )    

    solution = result.best_state
    objective = round(solution.objective(), 2)
    print(f"Best heuristic objective is {objective}.")

    print(f"\nIn the INITIAL SOLUTION there were {len(initial_solution.routes)} routes")
    served_customers = 0
    for route in initial_solution.routes:
        customers = [
            cust
            for cust in route.customers_list
            if cust not in init.depots["depots_indices"]
        ]
        served_customers += len(customers)
        print(route.customers_list)

    print(f"Total number of served customers: {served_customers}")
    data_df = initial_solution.nodes_df
    initial_solution_stats = verify_time_windows(data_df, initial_solution, percentage=False)

    print(f"\nIn the HEURISTIC SOLUTION there are {len(solution.routes)} routes")
    served_customers = 0
    for route in solution.routes:
        customers = [
            cust
            for cust in route.customers_list
            if cust not in solution.depots["depots_indices"]
        ]
        served_customers += len(customers)
        print(route.customers_list)

    print(f"Total number of served customers: {served_customers}")
    # Calculating the late, early, ontime and left out customers
    solution_stats = verify_time_windows(data_df, solution, percentage=False)

    # results dict
    results_dict = {
        "Quantity": ["Total cost", "# Served customers", "# Late customers", "# Early customers", "# Ontime customers", "Sum early mins", "Sum late mins"],
        "Initial solution": [
            initial_solution.objective(), initial_solution_stats["total_served"], initial_solution_stats["late"], initial_solution_stats["early"], 
            initial_solution_stats["ontime"], initial_solution_stats["sum_early"], initial_solution_stats["sum_late"],
        ],
        "Heuristic solution": [
            solution.objective(), solution_stats["total_served"], solution_stats["late"], solution_stats["early"], 
            solution_stats["ontime"], solution_stats["sum_early"], solution_stats["sum_late"],
        ],
    }

    print_results_dict(results_dict)

    if args.video:
        generate_video(image_base_folder="./outputs/plots", default_output_folder="./outputs/videos", desidered_fps=12)


if __name__ == "__main__":
    main()
