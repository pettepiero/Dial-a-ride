import numpy.random as rnd
import numpy as np
import datetime
import os
from alns import ALNS
from alns.accept import RecordToRecordTravel
from alns.select import *
from alns.stop import MaxIterations, MaxRuntime
from tqdm import tqdm
from lib.myvrplib.myvrplib import LOGGING_LEVEL
from lib.myvrplib.data_module import get_data_format, read_cordeau_data, get_instance_full_path
from lib.myvrplib.CVRPState import CVRPState 
from lib.initial_solutions.initial_solutions import nearest_neighbor
from lib.operators.destroy import *
from lib.operators.repair import *
from lib.operators.wang_operators import *
from lib.output.analyze_solution import analyze_solution 
from lib.myvrplib.input_output import print_results_dict, parse_options, print_instance
from lib.output.video import generate_video
from lib.myvrplib.data_format_conversions import convert_vrplib_to_cordeau
import logging
import csv

degree_of_destruction = 0.05

def main():
    # logging setup
    run_id = np.random.randint(10000, 99999)
    current_path = os.getcwd()
    log_dir = os.path.join(current_path, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = os.path.join(log_dir, f"log_{run_id}.txt")
    logging.basicConfig(filename=log_filename, level=logging.DEBUG, force=True)
    print(f"Running cmdvrp.py with run_id {run_id}")
    print(f"\nLog of this execution is being written to {log_filename}")
    now = datetime.datetime.now()
    logging.debug(f"Log of compare_models_single_mode.py run on {now.day}/{now.month}/{now.year} at {now.hour}:{now.minute}:{now.second}")
    # results setup
    results_dir = os.path.join(current_path, "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    results_filename = os.path.join(results_dir, f"results_{run_id}.csv")
    print(f"Results of this execution are being written to {results_filename}\n")

    args = parse_options()

    logging.debug(f"Parsed args:")
    for el in vars(args):
        logging.debug(f"{el}")
    print(f"Arguments: {args}")
    
    if args.seed is not None:
        print(f"Initializing with explicit seed: {args.seed}")
        logging.debug(f"Initializing with explicit seed: {args.seed}")
        alns = ALNS(rnd.default_rng(args.seed))
    else:
        print(f"Initializing ALNS without explicit seed")
        logging.debug(f"Initializing ALNS without explicit seed")
        alns = ALNS(rnd.default_rng())

    instances_to_solve = []
    if args.mode == 'single_instance':
        assert (args.instance_path is not None) or (args.instance_code is not None), f"Selected single_instance mode but did not provide instance path nor code."
        if args.instance_path is not None:
            assert os.path.exists(args.instance_path), f"instance_path {args.instance_path} does not exist."
            instance_full_path = args.instance_path
        elif args.instance_code is not None:
            instance_full_path = get_instance_full_path(instance_name=args.instance_code, problem_type=args.problem_type)
        data = read_cordeau_data(instance_full_path, print_data=False)
        print_instance(data)
        instances_to_solve.append(data)

    elif args.mode == 'batch':
        assert args.dir is not None, f"Selected batch mode but did not provide directory"
        assert os.path.exists(args.dir), f"Provided dir {args.dir} does not exist"
        assert os.path.isdir(args.dir), f"Provided dir {args.dir} is not a directory"
        instances_names = os.listdir(args.dir)
        instances_names = [os.path.join(args.dir, inst) for inst in instances_names if inst.endswith('.mdvrp')]
        assert len(instances_names) > 0, f"Did not find any instances in provided dir {args.dir} that end with '.mdvrp'"

        logging.debug(f"Found {len(instances_names)} instances in {args.dir}")

        for instance_full_path in instances_names:
        #    instance_full_path = get_instance_full_path(instance_name=inst, problem_type=args.problem_type)
            problem_type = get_data_format(instance_full_path)
            if problem_type == 'vrplib':
                new_path = instance_full_path + "_vrplib"
                convert_vrplib_to_cordeau(input_path=instance_full_path, output_path=new_path)
                instance_full_path = new_path
            data = read_cordeau_data(instance_full_path, print_data=False)
            instances_to_solve.append(data)

    #stop_c = args.stop_criterion
    #if stop_c == 'iters':
    #    stop = MaxIterations(args.num_iters)
    #elif stop_c == 'runtime':
    #    stop = MaxRuntime(args.max_time)
    #else:
    #    raise ValueError(f"Unknown stopping criterion: {stop_c}")

    repair_ops = [
            greedy_repair_no_tw,
            regret3_insertion,
            GIN_repair_no_tw,
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
    initial_sol_costs = []
    final_costs = []
    solution = None
    initial_solution = None
    init = None
    for i, data in enumerate(tqdm(instances_to_solve)):
        if args.mode == 'batch':
            logging.debug(f"\nDoing instance {i}: {instances_names[i]}")
        init = CVRPState(instance=data)
        initial_solution = nearest_neighbor(state=init)
        #print(f"Created initial solution")
        select = RouletteWheel(
                scores=[25, 5, 1, 0], 
                decay=0.8, 
                num_destroy=len(destroy_ops),
                num_repair=len(repair_ops)
                )
        # select = RandomSelect(num_destroy=4, num_repair=2)
        accept = RecordToRecordTravel.autofit(
            initial_solution.objective(), 0.02, 0, args.RRT_num_iters 
        )
        stop_c = args.stop_criterion
        if stop_c == 'iters':
            stop = MaxIterations(args.num_iters)
        elif stop_c == 'runtime':
            stop = MaxRuntime(args.max_time)
        else:
            raise ValueError(f"Unknown stopping criterion: {stop_c}")


        initial_sol_costs.append(initial_solution.objective())

        result, *_ = (
            alns.iterate(initial_solution, select, accept, stop, data=data, save_plots=args.video)
        )    

        solution = result.best_state
        final_costs.append(solution.objective())
        initial_cost = initial_solution.objective()
        final_cost = solution.objective()
        diff = initial_cost - final_cost
        logging.debug(f"Instance {i}/{len(instances_to_solve)}: initial cost: {initial_cost} | final_cost: {final_cost} | improved by: {diff}")

    with open(results_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["instance_idx", "initial_cost", "final_cost", "diff"])
        for i, (initial_cost, final_cost) in enumerate(zip(initial_sol_costs, final_costs)):
            writer.writerow([i, round(initial_cost, 3), round(final_cost, 3), round(initial_cost - final_cost, 3)])

    if args.mode == 'single_instance' and args.show_solution:
        show_solution(solution, initial_solution, init)

    #if args.video:
    #    generate_video(image_base_folder="./outputs/plots", default_output_folder="./outputs/videos", desidered_fps=12)
    logging.debug(f"Finished.")
    logging.debug(f"Mean cost of batch: {round(np.array(final_costs).mean(), 3)}")
    logging.debug(f"Mean cost of initial solutions of batch: {round(np.array(initial_sol_costs).mean(), 3)}")
    print("Finished")


def show_solution(solution, initial_solution, init):
    print(f"\nShowing solution:")
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
        #print(route.customers_list)

    print(f"DEBUG: routes in initial_solution:")
    for el in initial_solution.routes:
        print(el.customers_list)

    #print(f"Total number of served customers: {served_customers}")
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
    results_dict = {
        "Quantity": ["Total cost", "# Served customers"],
        "Initial solution": [
            initial_solution.objective(), initial_solution_stats["total_served"]],
        "Heuristic solution": [
            solution.objective(), solution_stats["total_served"]],
    }

    print_results_dict(results_dict)



if __name__ == "__main__":
    main()
