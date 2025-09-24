import numpy.random as rnd
import os
from alns import ALNS
from alns.accept import RecordToRecordTravel
from alns.select import *
from alns.stop import MaxIterations

from lib.myvrplib.myvrplib import LOGGING_LEVEL
from lib.myvrplib.data_module import get_data_format, read_cordeau_data
from lib.myvrplib.CVRPState import CVRPState 
from lib.initial_solutions.initial_solutions import nearest_neighbor
from lib.operators.destroy import *
from lib.operators.repair import *
from lib.operators.wang_operators import *
from lib.output.analyze_solution import analyze_solution 
from lib.myvrplib.input_output import print_results_dict, parse_options, print_dataset
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
    
    if args.seed is not None:
        print(f"Initializing with explicit seed: {args.seed}")
        alns = ALNS(rnd.default_rng(args.seed))
    else:
        print(f"Initializing ALNS without explicit seed")
        alns = ALNS(rnd.default_rng())

    dataset_name = args.dataset

    if args.problem_type is None:
        # determine from file name or raise error
        ext = os.path.splitext(args.dataset)[-1]
        if ext == '.mdvrp' or ext == '.cmdvrp':
            problem_type = "MDVRP"
        elif ext == '.mdvrptw' or ext == '.cmdvrptw':
            problem_type = "MDVRPTW"
        else:
            raise ValueError(f"Unkown extension of dataset, please provide problem_type of dataset with explicit extension in order to determine problem type.")
    else:
        problem_type = args.problem_type
    #if valid choice, create dataset_full_path variable
    if problem_type in ["mdvrptw", "MDVRPTW"]:
        raise ValueError(f"This script is meant to run MDVRP instances, not MDVRPTW")
        dataset_full_path = "./data/c-mdvrptw/" + dataset_name
    elif problem_type in ["mdvrp", "MDVRP"]:
        dataset_full_path = "./data/C-mdvrp/" + dataset_name
    else:
        raise ValueError(f"Unkown extension of dataset")

    print(f"Chosen dataset: {dataset_full_path}")
    
    data_type = get_data_format(dataset_full_path)
    if data_type == 'cordeau':
        valid_datasets = ["pr02",  "pr04",  "pr06",  "pr08",  "pr10",  "pr12",  "pr14",  "pr16",  "pr18",  "pr20", "pr01", "pr03", "pr05", "pr07",  "pr09",  "pr11",  "pr13",  "pr15",  "pr17",  "pr19"]
        assert dataset_name in valid_datasets, f"Dataset {dataset_name} not found in ./data/c-mdvrptw"
    elif data_type == 'vrplib':
        # convert dataset to cordeau and then read
        new_path = dataset_full_path + "_vrplib"
        convert_vrplib_to_cordeau(input_path=dataset_full_path, output_path=new_path)
        dataset_full_path = new_path
        
    data = read_cordeau_data(dataset_full_path, print_data=False)
    print_dataset(data)

    if args.num_iterations is not None:
        num_iterations = int(args.num_iterations)
    else:
        num_iterations = NUM_ITERATIONS

    print(f"num_iterations: {num_iterations}\n")


    repair_ops = [
            greedy_repair_no_tw
            ]
    destroy_ops = [
            random_removal, 
            random_route_removal, 
            #cost_reducing_removal, 
            worst_removal, 
            #exchange_reducing_removal
            ]

    for op in destroy_ops:
        alns.add_destroy_operator(op)
    for op in repair_ops:
        alns.add_repair_operator(op)

    # Set up ALNS operators
    #alns.add_destroy_operator(random_removal)
    #alns.add_destroy_operator(random_route_removal)
    #alns.add_destroy_operator(cost_reducing_removal)
    #alns.add_destroy_operator(worst_removal)
    #alns.add_destroy_operator(exchange_reducing_removal) 
    ## alns.add_destroy_operator(shaw_removal)   #to be implemented
    #alns.add_repair_operator(greedy_repair_no_tw)
    ##alns.add_repair_operator(wang_greedy_repair)

    init = CVRPState(dataset=data)
    initial_solution = nearest_neighbor(state=init)
    print(f"Created initial solution")
    select = RouletteWheel(
            scores=[25, 5, 1, 0], 
            decay=0.8, 
            num_destroy=len(destroy_ops),
            num_repair=len(repair_ops)
            )
    # select = RandomSelect(num_destroy=4, num_repair=2)
    accept = RecordToRecordTravel.autofit(
        initial_solution.objective(), 0.02, 0, num_iterations 
    )
    stop = MaxIterations(num_iterations)

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
    initial_solution_stats = {"total_served": served_customers}

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
    solution_stats = {"total_served": served_customers}
    # results dict
    results_dict = {
        "Quantity": ["Total cost", "# Served customers"],
        "Initial solution": [
            initial_solution.objective(), initial_solution_stats["total_served"]],
        "Heuristic solution": [
            solution.objective(), solution_stats["total_served"]],
    }

    print_results_dict(results_dict)

    if args.video:
        generate_video(image_base_folder="./outputs/plots", default_output_folder="./outputs/videos", desidered_fps=12)


if __name__ == "__main__":
    main()
