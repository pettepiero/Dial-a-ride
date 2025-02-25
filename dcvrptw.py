import numpy.random as rnd
import pandas as pd

from alns import ALNS
from alns.accept import RecordToRecordTravel
from alns.select import *
from alns.stop import MaxIterations
from alns.My_plot import plot_solution

from cvrptw.myvrplib.myvrplib import LOGGING_LEVEL
from cvrptw.myvrplib.data_module import d_data as d_data
from cvrptw.myvrplib.vrpstates import CvrptwState
from cvrptw.initial_solutions.initial_solutions import nearest_neighbor
from cvrptw.operators.destroy import *
from cvrptw.operators.repair import *
from cvrptw.operators.wang_operators import *
from cvrptw.output.analyze_solution import verify_time_windows, check_solution
from cvrptw.myvrplib.input_output import print_results_dict, parse_options
from cvrptw.output.video import generate_video

SEED = 1234
NUM_ITERATIONS = 30

# logging setup
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOGGING_LEVEL)

degree_of_destruction = 0.05


def main():

    args = parse_options()
    print(f"Arguments: {args}")

    # alns = ALNS(rnd.default_rng(SEED))
    alns = ALNS(rnd.default_rng())

    alns.add_destroy_operator(random_removal)
    alns.add_destroy_operator(random_route_removal)
    alns.add_destroy_operator(cost_reducing_removal)
    alns.add_destroy_operator(worst_removal)

    # alns.add_destroy_operator(exchange_reducing_removal)  # to be implemented
    alns.add_destroy_operator(shaw_removal)   #to be implemented

    alns.add_repair_operator(greedy_repair_tw)
    # alns.add_repair_operator(wang_greedy_repair)

    data = pd.read_csv("./data/actv_dynamic_df.csv")
    data.index +=1 # align index to ids

    init = CvrptwState(dataset=data, n_vehicles=args.n_vehicles, vehicle_capacity=args.vehicle_capacity)

    initial_solution = nearest_neighbor(state=init, initial_time_slot=True)
    print(f"DEBUG: initial solution:\n")
    for route in initial_solution.routes:
        print(route.nodes_list)

    print(f"DEBUG: state.twc_format_nodes_df = \n{initial_solution.twc_format_nodes_df}")
    check_solution(initial_solution)

    select = RouletteWheel([25, 5, 1, 0], 0.8, 5, 1)
    # select = RandomSelect(num_destroy=4, num_repair=2)
    accept = RecordToRecordTravel.autofit(
        initial_solution.objective(), 0.02, 0, NUM_ITERATIONS
    )
    stop = MaxIterations(NUM_ITERATIONS)

    print(f"DEBUG: depots = {initial_solution.depots["depots_indices"]}")

    result, *_ = alns.iterate(
        initial_solution, select, accept, stop, data=data, save_plots=args.video
    )
    # Testing solution validity after iterations
    check_solution(initial_solution)

    solution = result.best_state
    objective = round(solution.objective(), 2)
    print(f"Best heuristic objective is {objective}.")

    print(f"\nIn the INITIAL SOLUTION there were {len(initial_solution.routes)} routes")

    print(f"Total number of planned customers: {initial_solution.n_planned_customers}")
    data_df = initial_solution.twc_format_nodes_df
    initial_solution_stats = verify_time_windows(
        data_df, initial_solution, percentage=False
    )

    print(f"\nIn the HEURISTIC SOLUTION there are {len(solution.routes)} routes")

    print(f"Total number of planned customers: {solution.n_planned_customers}")
    # Calculating the late, early, ontime and left out customers
    solution_stats = verify_time_windows(data_df, solution, percentage=False)

    # results dict
    results_dict = {
        "Quantity": [
            "Total cost",
            "# Served customers",
            "# Late customers",
            "# Early customers",
            "# Ontime customers",
            "Sum early mins",
            "Sum late mins",
        ],
        "Initial solution": [
            initial_solution.objective(),
            initial_solution_stats["total_served"],
            initial_solution_stats["late"],
            initial_solution_stats["early"],
            initial_solution_stats["ontime"],
            initial_solution_stats["sum_early"],
            initial_solution_stats["sum_late"],
        ],
        "Heuristic solution": [
            solution.objective(),
            solution_stats["total_served"],
            solution_stats["late"],
            solution_stats["early"],
            solution_stats["ontime"],
            solution_stats["sum_early"],
            solution_stats["sum_late"],
        ],
    }

    print_results_dict(results_dict)

    plot_solution(solution=solution, name="Final-solution", save=True, figsize=(8,8))
    plot_solution(solution=initial_solution, name="Initial-solution", save=True, figsize=(8,8))

    if args.video:
        generate_video(
            image_base_folder="./outputs/plots",
            default_output_folder="./outputs/videos",
            desidered_fps=12,
        )


if __name__ == "__main__":
    main()
