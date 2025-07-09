import unittest
from numpy import nan
from lib.myvrplib.data_module import *
from lib.myvrplib.dataset_readers import *
from lib.myvrplib.vrpstates import CvrptwState
from lib.myvrplib.route import Route
from lib.operators.destroy import cost_reducing_removal

class TestVRPLIBData(unittest.TestCase):
    def test_get_all_sections_tag(self):
        known_solution = ["NAME", "COMMENT", "TYPE", "DIMENSION", "EDGE_WEIGHT_TYPE", "NUM_VEHICLES", "CAPACITY", "NODES_SECTION", "DEMAND_SECTION", "DEPOT_SECTION"]

        tags = get_all_section_tags("/home/pettepiero/tirocinio/dial-a-ride/tests/cvrppdtw/vrplib_test_data.knd")
        for tag in known_solution:
            self.assertTrue(tag in tags)

    def test_read_vrplib_data(self):
        data = read_vrplib_data(
                file = "/home/pettepiero/tirocinio/dial-a-ride/tests/cvrppdtw/vrplib_test_data.knd", 
                print_data = False, 
                seed = 0
        )
        print(f"\nDEBUG: In read_vrplib_data:\n")
        print(f"data: \n{data}")

        known_solution = { # for seed 0
            'name': "vrplib_test_data.knd",
            'vehicles': 2,
            'capacity': 101,
            'dimension': 10,
            'n_depots': 4,
            'depots': [0, 49, 50, 51],
            'depot_to_vehicles': {0: [], 49: [], 50: [0], 51: [1]},
            'vehicle_to_depot': {0: 50, 1: 51},
            'node_coord': [[42, 7], [89, 22], [7, 13], [60, 41], [24, 45], [12, 23], [19, 68], [98, 33], [32, 85], [22, 6]],
            'demand': [1, 1, 2, 5, 1, 3, 1, 4, 1, 2],
            'pickup_time_window': [[525, 570], [450, 555], [465, 750], [705, 885], [705, 840], [615, 765], [615, 720], [435, 645], [435, 675], [555, 765]],
            'delivery_time_window': [[645, 690], [510, 615], [585, 930], [825, 1005], [825, 960], [735, 885], [675, 780], [555, 765], [555, 795], [615, 825]],
            'service_time': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            }

        for key in known_solution:
            self.assertTrue(np.array_equal(known_solution[key], data[key]), f"\nFailed on key: {key}")
