import argparse
import pickle
import numpy as np

def main(args):
    distances, solution = None, None
    with open(args.dist_file, 'rb') as f:
        distances = pickle.load(f)  
    with open(args.solution, 'rb') as f:
        solution = pickle.load(f)
        solution = solution.routes
    print(f"DEBUG: distances:\n{distances}")
    print(f"DEBUG: distances.shape:\n{distances.shape}")
    print(f"DEBUG: solution:")
    for el in solution:
        print(el.customers_list)
    print("\n")
    solution_dist = 0
    solution = [el for el in solution if len(el) > 1]
    for route in solution:
        route_dist = 0
        for i, el in enumerate(route.customers_list[:-1]): # or -2?
            cust = el
            next_cust = route.customers_list[i+1]
            d = distances[el, next_cust]
            route_dist += d
        solution_dist += route_dist        

    given_distance = args.distance
    # check distance
    print(f"Given distance: {args.distance}")
    print(f"Computed distance: {solution_dist}")
    diff = abs(given_distance - solution_dist)
    print(f"Difference: given distance - computed distance = {diff}")
    if diff < 1:
        print(f"\nPASSED")
    else:
        print(f"FAILED: diff = {diff}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='check some quantities in solutions')
    ap.add_argument('--solution', type=str, required=True, help="pickle file containing routes to check")
    ap.add_argument('--dist_file', type=str, required=True, help="pickle file containing routes to check")
    ap.add_argument('--distance', type=float, required=True, help="Distance to test")
    args = ap.parse_args()

    main(args)
