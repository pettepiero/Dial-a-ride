import tabulate
import argparse
import json
from cvrptw.operators.destroy import random_removal, random_route_removal, cost_reducing_removal, worst_removal, exchange_reducing_removal
from cvrptw.operators.repair import greedy_repair_tw
from cvrptw.operators.wang_operators import wang_greedy_repair
from alns import ALNS
def print_results_dict(results_dict: dict) -> None:
    """
    Prints the results dictionary as a table.
    """
    print("\n\n")
    print(tabulate.tabulate(results_dict, headers="keys", tablefmt="fancy_grid"))


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
        "--seed",
        type=int,
        help="Specify random seed to use. Default is 1234.",
    )
    parser.add_argument(
        "--random",
        action=argparse.BooleanOptionalAction,
        help="Use a random seed or not.",
    )
    parser.add_argument(
        "--dataset", type=str, default="pr12", help="Dataset name (pr01 to pr20)."
    )
    parser.add_argument(
        "--removal_operators",
        type=int,
        default=0,
        help="Removal operators (0 means all implemented).Otherwise: \
            1: random_removal, 2: random_route_removal, 3: cost_reducing_removal, 4: worst_removal,\
                5: exchange_reducing_removal. Example: --removal_operators 12 means random_removal and random_route_removal.",
    )
    parser.add_argument(
        "--insertion_operators",
        type=int,
        default=0,
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


def add_d_operators(ops: int, alns: ALNS):
    """Add destroy operators to the ALNS object, as described by the --help option"""
    if ops == 0:
        alns.add_destroy_operator(random_removal)
        alns.add_destroy_operator(random_route_removal)
        alns.add_destroy_operator(cost_reducing_removal)
        alns.add_destroy_operator(worst_removal)
        alns.add_destroy_operator(exchange_reducing_removal)
        print(f"Added all destroy operators")
        return
    else:
        assert isinstance(ops, int)
        ops_list = [int(digit) for digit in str(ops)]
        for op in ops_list:
            match op:
                case 1:
                    alns.add_destroy_operator(random_removal)
                    print(f"Added random removal operator")
                case 2:
                    alns.add_destroy_operator(random_route_removal)
                    print(f"Added random route removal operator")
                case 3:
                    alns.add_destroy_operator(cost_reducing_removal)
                    print(f"Added cost reducing removal operator")
                case 4:
                    alns.add_destroy_operator(worst_removal)
                    print(f"Added worst removal operator")
                case 5:
                    alns.add_destroy_operator(exchange_reducing_removal)
                    print(f"Added exchange reducing removal operator")
                case _:
                    print(f"Operator {op} not implemented. Skipping.")
        return

def add_i_operators(ops: int, alns: ALNS):
    """Add insertion operators to the ALNS object, as described by the --help option"""
    if ops == 0:
        alns.add_repair_operator(greedy_repair_tw)
        alns.add_repair_operator(wang_greedy_repair)
        print(f"Added all repair operators")
        return
    else:
        assert isinstance(ops, int)
        ops_list = [int(digit) for digit in str(ops)]
        for op in ops_list:
            match op:
                case 1:
                    alns.add_repair_operator(greedy_repair_tw)
                    print("Added greedy repair operator")
                case 2:
                    alns.add_repair_operator(wang_greedy_repair)
                    print("Added Wang greedy repair operator")
                case _:
                    print(f"Operator {op} not implemented. Skipping.")
        return