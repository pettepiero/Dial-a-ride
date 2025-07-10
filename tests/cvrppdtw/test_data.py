import unittest
from numpy import nan
from lib.myvrplib.data_module import *
from lib.myvrplib.dataset_readers import *
from lib.myvrplib.vrpstates import CvrptwState
from lib.myvrplib.route import Route
from lib.operators.destroy import cost_reducing_removal

class TestVRPLIBData(unittest.TestCase):
    def setUp(self):
        self.data_file = "/home/pettepiero/tirocinio/dial-a-ride/tests/cvrppdtw/vrplib_test_data.knd"
        self.map_data = "/home/pettepiero/tirocinio/dial-a-ride/tests/cvrppdtw/test_map_data.csv"
    def test_get_all_sections_tag(self):
        known_solution = ["NAME", "COMMENT", "TYPE", "DIMENSION", "EDGE_WEIGHT_TYPE", "NUM_VEHICLES", "CAPACITY", "NODES_SECTION", "DEMAND_SECTION", "DEPOT_SECTION"]

        tags = get_all_section_tags(self.data_file)
        for tag in known_solution:
            self.assertTrue(tag in tags)

    def test_read_vrplib_data(self):
        data = read_vrplib_data(
                file = self.data_file, 
                print_data = False, 
                seed = 0
        )
        known_solution = { # for seed 0
            'name': "vrplib_test_data.knd",
            'vehicles': 2,
            'capacity': 101,
            'dimension': 10,
            'customer_ids': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'n_depots': 4,
            'depots': [0, 49, 50, 51],
            'depot_to_vehicles': {0: [], 49: [], 50: [0], 51: [1]},
            'vehicle_to_depot': {0: 50, 1: 51},
            'requested_stops': {1: [42, 7], 2: [89, 22], 3: [7, 13], 4: [60, 41], 5: [24, 45], 6: [12, 23], 7: [19, 68], 8: [98, 33], 9: [32, 85], 10: [22, 6]},
            'demand': [1, 1, 2, 5, 1, 3, 1, 4, 1, 2],
            'pickup_time_window': [[525, 570], [450, 555], [465, 750], [705, 885], [705, 840], [615, 765], [615, 720], [435, 645], [435, 675], [555, 765]],
            'delivery_time_window': [[645, 690], [510, 615], [585, 930], [825, 1005], [825, 960], [735, 885], [675, 780], [555, 765], [555, 795], [615, 825]],
            'service_time': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            }

        for key in known_solution.keys():
            self.assertTrue(np.array_equal(known_solution[key], data[key]), f"\nFirst check failed on key: {key}")

    def test_insert_coords_from_ids(self):
        data = read_vrplib_data(file = self.data_file, print_data = False, seed=0)

        map_data = pd.read_csv(self.map_data)
        known_solution = data.copy()
        known_solution['node_coord'] = [[45.1485, 11.7564], 
            [45.9736, 11.0488],
            [45.3284, 11.2551],
            [45.8782, 11.2588],
            [45.205,  11.0631],
            [45.2925, 11.1899],
            [45.836,  11.5411],
            [45.6024, 11.3833],
            [45.5468, 11.2468],
            [45.1717, 11.2726]]

        known_solution['edge_weight'] = cost_matrix_from_coords(np.array(map_data[['lat', 'lon']].values.tolist()))
        insert_coords_from_ids(data_with_ids=data, map_data=map_data)

        for key in known_solution.keys():
            if isinstance(data[key], np.ndarray):
                try:
                    # convert to list and test
                    self.assertEqual(known_solution[key].tolist(), data[key].tolist(), f"\nFirst assertEqual failed on key: {key}")
                except AssertionError:
                    # for edge_weight, where nan = nan
                    self.assertTrue(np.allclose(known_solution[key], data[key], equal_nan=True), f"\nFailed on np.allclose")
            else:
                try:
                    self.assertEqual(known_solution[key], data[key], f"\nassertEqual failed on key: {key}")
                except ValueError:
                    self.assertTrue(np.array_equal(known_solution[key], data[key]), f"\nValueError failed on key: {key}")
