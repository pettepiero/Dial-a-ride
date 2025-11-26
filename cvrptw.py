import numpy.random as rnd
import datetime
import os
import csv
import pickle
from alns import ALNS
from alns.accept import RecordToRecordTravel
from alns.select import *
from alns.stop import MaxIterations
from tqdm import tqdm
from lib.myvrplib.myvrplib import LOGGING_LEVEL
from lib.myvrplib.data_module import get_data_format, read_cordeau_data, get_instance_full_path
from lib.myvrplib.CVRPTWState import CVRPTWState 
from lib.initial_solutions.initial_solutions import nearest_neighbor_tw
from lib.operators.destroy import *
from lib.operators.repair import *
from lib.operators.wang_operators import *
from lib.output.analyze_solution import analyze_solution
from lib.myvrplib.input_output import print_results_dict, parse_options, print_instance
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

    run_id = rnd.randint(10000, 99999)
    current_path = os.getcwd()
    log_dir = os.path.join(current_path, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = os.path.join(log_dir, f"log_{run_id}.txt")
    logging.basicConfig(filename=log_filename, level=logging.DEBUG, force=True)
    print(f"Running cvrptw.py with run_id {run_id}")
    print(f"\nLog of this execution is being written to {log_filename}")
    now = datetime.datetime.now()
    logging.debug(f"Log of cvrptw run on {now.day}/{now.month}/{now.year} at {now.hour}:{now.minute}:{now.second}")
    logging.debug(f"Running cvrptw.py with run_id {run_id}")

    # results setup
    results_dir = os.path.join(current_path, "results")
    if args.output_path is None:
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        results_filename = os.path.join(results_dir, f"results_{run_id}.csv")
    else:
        results_filename = args.output_path
    print(f"Results of this execution are being written to {results_filename}\n")

    print(f"Arguments: {args}")
    
    if args.seed is not None:
        print(f"Initializing with explicit seed: {args.seed}")
        alns = ALNS(rnd.default_rng(args.seed))
    else:
        print(f"Initializing ALNS without explicit seed")
        alns = ALNS(rnd.default_rng())

    instances_to_solve = []
    if args.mode == 'single_instance':
        assert (args.instance_code is not None and args.instance_path is None) or (args.instance_path is not None and args.instance_code is None), f"Error in specifying either instance code or path"

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
        instances_names = [os.path.join(args.dir, inst) for inst in instances_names if inst.endswith('.cvrptw')]
        assert len(instances_names) > 0, f"Did not find any instances in provided dir {args.dir} that end with '.cvrptw'"

        logging.debug(f"Found {len(instances_names)} instances in {args.dir}")

        for instance_full_path in instances_names:
        #    instance_full_path = get_instance_full_path(instance_name=inst, problem_type=args.problem_type)
            problem_type = get_data_format(instance_full_path)
            if problem_type == 'vrplib':
                new_path = instance_full_path + "_cordeau"
                convert_vrplib_to_cordeau(input_path=instance_full_path, output_path=new_path)
                instance_full_path = new_path
            data = read_cordeau_data(instance_full_path, print_data=False)
            instances_to_solve.append(data)
    
    repair_ops = [
            greedy_repair_tw,
            wang_greedy_repair,
            ]
    destroy_ops = [
            random_removal,
            random_route_removal,
            cost_reducing_removal,
            worst_removal,
            exchange_reducing_removal
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
        init = CVRPTWState(instance=data)
        initial_solution = nearest_neighbor_tw(state=init, initial_time_slot=False)
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

        if args.cost_debug:
            #DEBUG save edge weigth and solution of last instance
            logging.debug("Saving initial solution cost, final solution cost and distances to .temp folder")
            if not os.path.exists('.temp'):
                os.mkdir('.temp')
            with open('.temp/edge_weight.pt', 'wb') as f:
                pickle.dump(data['edge_weight'], f)
            with open('.temp/solution.pt', 'wb') as f:
                pickle.dump(solution, f)
            with open('.temp/initial_solution.pt', 'wb') as f:
                pickle.dump(initial_solution, f)

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

    tw_sol_stats = analyze_solution(data=data_df, sol=solution)
    tw_init_stats = analyze_solution(data=data_df, sol=initial_solution)

    results_dict = {
        "Quantity": ["Total cost", "# Served customers", "#Late served customers", "#Early served customers", "#On time customers", "Sum late mins", "Sum early mins"],
        "Initial solution": [
            initial_solution.objective(), 
            initial_solution_stats["total_served"], 
            tw_init_stats['late'], 
            tw_init_stats['early'], 
            tw_init_stats['ontime'], 
            tw_init_stats['sum_late'], 
            tw_init_stats['sum_early']],
        "Heuristic solution": [
            solution.objective(), 
            solution_stats["total_served"],
            tw_sol_stats['late'], 
            tw_sol_stats['early'], 
            tw_sol_stats['ontime'], 
            tw_sol_stats['sum_late'], 
            tw_sol_stats['sum_early']],
    }

    print_results_dict(results_dict)

if __name__ == "__main__":
    main()
