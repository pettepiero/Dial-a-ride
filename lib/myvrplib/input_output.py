import tabulate
import argparse
import json

def print_results_dict(results_dict: dict) -> None:
    """
    Prints the results dictionary as a table.
    """
    print("\n\n")
    print(tabulate.tabulate(results_dict, headers="keys", tablefmt="fancy_grid"))


def print_cvrptw_dataset(dataset: dict) -> None:
    """
    Prints information about the cvrptw dataset
    """
    print("\n")
    print(f"Dataset name: {dataset['name']}")
    print(f"Number of customers: {dataset['dimension']}")
    print(f"Number of vehicles: {dataset['vehicles']}")
    print(f"Number of depots: {dataset['n_depots']}")
    print("\n")

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
    
    parser.add_argument('--video', action=argparse.BooleanOptionalAction, help='Generate video from images. Video is saved in outputs/videos, and images in outputs/images. \
                        Use --video to create video, --no-video otherwise.', default=False)
    # parser.add_argument(
    #     "--video",
    #     type=bool,
    #     default=False,
    #     help="Generate video from images. Video is saved in outputs/videos, and images in outputs/images.",
    # )

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
