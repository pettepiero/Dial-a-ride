import numpy.random as rnd
import numpy as np
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

    instances_to_solve = []
    if args.mode == 'single_instance':
        instance_full_path = get_instance_full_path(instance_name=args.instance, problem_type=args.problem_type)
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

        print(f"Found {len(instances_names)} instances in {args.dir}")

        for instance_full_path in instances_names:
        #    instance_full_path = get_instance_full_path(instance_name=inst, problem_type=args.problem_type)
            problem_type = get_data_format(instance_full_path)
            if problem_type == 'vrplib':
                new_path = instance_full_path + "_vrplib"
                convert_vrplib_to_cordeau(input_path=instance_full_path, output_path=new_path)
                instance_full_path = new_path
            data = read_cordeau_data(instance_full_path, print_data=False)
            instances_to_solve.append(data)

    stop_c = args.stop_criterion
    if stop_c == 'iters':
        stop = MaxIterations(args.num_iters)
    elif stop_c == 'runtime':
        stop = MaxRuntime(args.max_time)
    else:
        raise ValueError(f"Unknown stopping criterion: {stop_c}")

    repair_ops = [
            greedy_repair_no_tw,
            regret3_insertion,
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

    for data in tqdm(instances_to_solve):
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

        initial_sol_costs.append(initial_solution.objective())

        result, *_ = (
            alns.iterate(initial_solution, select, accept, stop, data=data, save_plots=args.video)
        )    

        solution = result.best_state
        #objective = round(solution.objective(), 2)
        final_costs.append(solution.objective())
        #print(f"Best heuristic objective is {objective}.")

        #print(f"\nIn the INITIAL SOLUTION there were {len(initial_solution.routes)} routes")
        #served_customers = 0
        #for route in initial_solution.routes:
        #    customers = [
        #        cust
        #        for cust in route.customers_list
        #        if cust not in init.depots["depots_indices"]
        #    ]
        #    served_customers += len(customers)
        #    #print(route.customers_list)

        ##print(f"Total number of served customers: {served_customers}")
        #data_df = initial_solution.nodes_df
        #initial_solution_stats = {"total_served": served_customers}

        ##print(f"\nIn the HEURISTIC SOLUTION there are {len(solution.routes)} routes")
        #served_customers = 0
        #for route in solution.routes:
        #    customers = [
        #        cust
        #        for cust in route.customers_list
        #        if cust not in solution.depots["depots_indices"]
        #    ]
        #    served_customers += len(customers)
        #    print(route.customers_list)

        #print(f"Total number of served customers: {served_customers}")
        #solution_stats = {"total_served": served_customers}
        # results dict
        #results_dict = {
        #    "Quantity": ["Total cost", "# Served customers"],
        #    "Initial solution": [
        #        initial_solution.objective(), initial_solution_stats["total_served"]],
        #    "Heuristic solution": [
        #        solution.objective(), solution_stats["total_served"]],
        #}

        #print_results_dict(results_dict)

    #if args.video:
    #    generate_video(image_base_folder="./outputs/plots", default_output_folder="./outputs/videos", desidered_fps=12)
    print(f"Finished.")
    print(f"Mean cost of batch: {round(np.array(final_costs).mean(), 3)}")
if __name__ == "__main__":
    main()
