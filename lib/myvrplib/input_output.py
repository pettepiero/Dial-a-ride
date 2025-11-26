import tabulate
import argparse
import json

def print_results_dict(results_dict: dict) -> None:
    """
    Prints the results dictionary as a table.
    """
    print("\n\n")
    print(tabulate.tabulate(results_dict, headers="keys", tablefmt="fancy_grid"))


def print_instance(instance: dict) -> None:
    """
    Prints information about the cvrptw instance 
    """
    print("\n")
    print(f"Dataset name: {instance['name']}")
    print(f"Number of customers: {instance['dimension']}")
    print(f"Number of vehicles: {instance['vehicles']}")
    print(f"Number of depots: {instance['n_depots']}")
    print("\n")

def parse_options():
    """
    Parse the command line options, allowing parameters from a config file and command-line overrides.
    """
    parser = argparse.ArgumentParser(
        description="Run the ALNS algorithm for the CVRPTW problem."
    )

    parser.add_argument(
        "--mode", type=str, default="single_instance", choices=['single_instance', 'batch'],  help="Mode of operation. Options are 'single_instance, 'batch'"
    )

    parser.add_argument("--config", type=str, help="Configuration file in JSON format.")
    parser.add_argument("--problem_type", type=str, choices=["mdvrp", "MDVRP", "mdvrptw", "MDVRPTW"], help="Problem type", default=None)

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Set random seed.",
    )
    parser.add_argument(
        "--logging",
        type=str,
        default="ERROR",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    parser.add_argument(
        "--instance_code", type=str, default=None, help="Instance code (e.g., pr01 to pr20)."
    )
    parser.add_argument(
        "--instance_path", type=str, default=None, help="Instance path for single_instance mode. Expects VRPLIB formatted instances"
    )
    parser.add_argument(
        "--dir", type=str, default=None, help="Directory containing instances for batch mode."
    )
    parser.add_argument(
        "--stop_criterion", "--stop",
        type=str,
        default='iters',
        choices=['iters', 'runtime'],
        help="Stop criterion. Options are 'iters', 'runtime'.",
    )
    parser.add_argument(
        "--num_iters",
        type=int,
        default=100,
        help="Number of iterations for stopping criterion 'iters'. Default: 100 iterations",
    )
    parser.add_argument(
        "--RRT_num_iters",
        type=int,
        default=100,
        help="Number of iterations for acceptance criterion RecordToRecordTravel. Default: 100 iterations",
    )
    parser.add_argument(
        "--max_time",
        type=int,
        default=30,
        help="Maximum run time per instance for stopping criterion 'runtime'. Default: 30 seconds",
    )
    parser.add_argument(
        '--show_solution', 
        action='store_true', 
        default=False, 
        help="If used, shows solution of single instance search"
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help="Output file path for batch search results"
    )
    parser.add_argument(
            '--cost_debug',
            action='store_true',
            help="If True, saves initial solution, final solution and distances to .temp folder for debugging costs of solutions with 'verify.py'"
            )
    
    parser.add_argument('--video', action=argparse.BooleanOptionalAction, help='Generate video from images. Video is saved in outputs/videos, and images in outputs/images. \
                        Use --video to create video, --no-video otherwise.', default=False)

    # Parse initial command-line arguments
    args = parser.parse_args()
    args_dict = vars(args)


    # Load config file if provided
    config_options = {}
    if args.config:
        config_options = read_json_options(args.config)
        args = {
            **config_options,
            **{k: v for k, v in args_dict.items() if v != parser.get_default(k)},
        }
        args_dict = args
    return argparse.Namespace(**args_dict)

def read_json_options(config_file: str) -> dict:
    """Load parameters from a JSON or YAML file."""
    try:
        with open(config_file, "r") as f:
            if config_file.endswith(".json"):
                return json.load(f)
            else:
                raise ValueError("Unsupported file format. Use JSON.")
    except Exception as e:
        print(f"Error loading config file: {e}")
        return {}
