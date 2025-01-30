from typing import List
import pandas as pd
import argparse
import json
import cv2
import os
from datetime import datetime

import numpy.random as rnd

from alns import ALNS
from alns.accept import RecordToRecordTravel
from alns.select import *
from alns.stop import MaxIterations


from cvrptw.myvrplib.myvrplib import plot_solution, plot_data, solution_times_statistics, LOGGING_LEVEL
from cvrptw.myvrplib.data_module import END_OF_DAY, read_solution_format
from cvrptw.myvrplib.data_module import d_data as d_data
from cvrptw.myvrplib.vrpstates import CvrptwState
from cvrptw.initial_solutions.initial_solutions import nearest_neighbor_tw, time_neighbours
from cvrptw.operators.destroy import *
from cvrptw.operators.repair import *
from cvrptw.operators.wang_operators import *
from cvrptw.output.analyze_solution import verify_time_windows
from cvrptw.myvrplib.input_output import print_results_dict

SEED = 1234
NUM_ITERATIONS = 100

# logging setup
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=LOGGING_LEVEL)

degree_of_destruction = 0.05
customers_to_remove = int((data["dimension"] - 1) * degree_of_destruction)

# Parameters to read from user:
# seed (random or not?)
# logging level
# dataset name
# degree of destruction
# removal operators
# insertion operators
# acceptance criterion
# stop criterion
# # num iterations
# operator selection schemes

def read_json_options(config_file: str) -> dict:
    """Load parameters from a JSON or YAML file."""
    try:
        with open(config_file, 'r') as f:
            if config_file.endswith(".json"):
                return json.load(f)
            else:
                raise ValueError("Unsupported file format. Use JSON.")
    except Exception as e:
        print(f"Error loading config file: {e}")
        return {}


def parse_options():
    """
    Parse the command line options, allowing parameters from a config file and command-line overrides.
    """
    parser = argparse.ArgumentParser(
        description="Run the ALNS algorithm for the CVRPTW problem."
    )

    # Define command line arguments with sensible defaults
    parser.add_argument("--config", type=str, help="Configuration file in JSON format.")
    parser.add_argument(
        "--random",
        action="store_true",
        help="Use a random seed instead of a fixed one.",
    )
    parser.add_argument(
        "--logging",
        type=str,
        default="ERROR",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    parser.add_argument(
        "--dataset", type=str, default="pr12", help="Dataset name (e.g., pr01 to pr20)."
    )
    parser.add_argument(
        "--degree_of_destruction",
        type=float,
        default=0.05,
        help="Degree of destruction (default: 0.05).",
    )
    parser.add_argument(
        "--removal_operators",
        type=int,
        default=1,
        help="Number of removal operators (1 means all implemented).",
    )
    parser.add_argument(
        "--insertion_operators",
        type=int,
        default=1,
        help="Number of insertion operators (1 means all implemented).",
    )
    parser.add_argument(
        "--acceptance_criterion",
        type=int,
        default=1,
        help="Acceptance criterion (1 means RecordToRecordTravel).",
    )
    parser.add_argument(
        "--stop_criterion",
        type=int,
        default=1,
        help="Stop criterion (1 means MaxIterations).",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=100,
        help="Number of iterations for stopping (default: 100).",
    )
    parser.add_argument(
        "--operator_selection_schemes",
        type=int,
        default=1,
        help="Operator selection schemes (1 means RouletteWheel).",
    )
    parser.add_argument(
        "--video",
        type=bool,
        default=False,
        help="Generate video from images. Video is saved in outputs/videos, and images in outputs/images.",
    )

    # Parse initial command-line arguments
    args = parser.parse_args()
    args_dict = vars(args)

    # Load config file if provided
    config_options = {}
    if args.config:
        config_options = read_json_options(args.config)

    # Merge values:
    # - Config file values overwrite argparse defaults.
    # - Command-line values (non-default) overwrite config file values.
    final_options = {
        **config_options,
        **{k: v for k, v in args_dict.items() if v != parser.get_default(k)},
    }

    return argparse.Namespace(**final_options)


def generate_video(
    image_base_folder: str = "./outputs/plots",
    default_output_folder: str = "./outputs/videos",
    desidered_fps: int = 12,
):
    """
    Generates a video from images stored in the most recent folder inside `image_base_folder`.

    Parameters:
    - image_base_folder (str): The parent folder containing timestamped image subfolders.
    - default_output_folder (str): The folder where the output video will be saved.
    - desidered_fps (int): The frames per second for the output video.
    """

    # Ensure the base plots folder exists
    if not os.path.exists(image_base_folder):
        print(f"Folder {image_base_folder} not found.")
        return

    # Get all subdirectories sorted by name (assuming they are timestamped)
    subfolders = sorted(
        [
            f
            for f in os.listdir(image_base_folder)
            if os.path.isdir(os.path.join(image_base_folder, f))
        ],
        reverse=True,  # Latest folder first
    )

    if not subfolders:
        print(f"No subfolders found in {image_base_folder}.")
        return

    # Select the most recent folder
    latest_folder = os.path.join(image_base_folder, subfolders[0])
    print(f"Using images from: {latest_folder}")

    # Ensure output folder exists
    os.makedirs(default_output_folder, exist_ok=True)

    # Generate video name with current timestamp
    video_name = datetime.now().strftime("%Y%m%d%H%M")
    file_name = os.path.join(default_output_folder, f"{video_name}.avi")

    # Get sorted images from the latest folder
    images = sorted([img for img in os.listdir(latest_folder) if img.endswith(".png")])

    if not images:
        print(f"No images found in {latest_folder}.")
        return

    # Read first image to get dimensions
    frame = cv2.imread(os.path.join(latest_folder, images[0]))
    height, width, layers = frame.shape

    # Create video writer
    video = cv2.VideoWriter(
        file_name, cv2.VideoWriter_fourcc(*"XVID"), desidered_fps, (width, height)
    )

    # Write images to video
    for image in images:
        video.write(cv2.imread(os.path.join(latest_folder, image)))

    # Release video writer
    video.release()
    cv2.destroyAllWindows()

    print(f"Video saved at: {file_name}")


def main():

    args = parse_options()
    print(args)

    # alns = ALNS(rnd.default_rng(SEED))
    alns = ALNS(rnd.default_rng())

    alns.add_destroy_operator(random_removal)
    alns.add_destroy_operator(random_route_removal)
    alns.add_destroy_operator(cost_reducing_removal)
    alns.add_destroy_operator(worst_removal)

    alns.add_destroy_operator(exchange_reducing_removal)  #to be implemented
    # alns.add_destroy_operator(shaw_removal)   #to be implemented

    alns.add_repair_operator(greedy_repair_tw)
    alns.add_repair_operator(wang_greedy_repair)

    init = CvrptwState(dataset=data)
    initial_solution = nearest_neighbor_tw(state=init, initial_time_slot=False)
    select = RouletteWheel([25, 5, 1, 0], 0.8, 5, 2)
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
        customers = [cust for cust in route.customers_list if cust not in data["depots"]]
        served_customers += len(customers)
        print(route.customers_list)

    print(f"Total number of served customers: {served_customers}")
    initial_solution_stats = verify_time_windows(data, initial_solution, percentage=False)

    print(f"\nIn the HEURISTIC SOLUTION there are {len(solution.routes)} routes")
    served_customers = 0
    for route in solution.routes:
        customers = [cust for cust in route.customers_list if cust not in data['depots']]
        served_customers += len(customers)
        print(route.customers_list)

    print(f"Total number of served customers: {served_customers}")
    # Calculating the late, early, ontime and left out customers
    solution_stats = verify_time_windows(data, solution, percentage=False)

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
        generate_video()


if __name__ == "__main__":
    main()
