import tabulate

def print_results_dict(results_dict: dict) -> None:
    """
    Prints the results dictionary as a table.
    """
    print("\n\n")
    print(tabulate.tabulate(results_dict, headers="keys", tablefmt="fancy_grid"))
    