import unittest
from numpy import nan
from unittest.mock import Mock
from cvrptw.myvrplib.data_module import *
from cvrptw.myvrplib.vrpstates import CvrptwState
from cvrptw.myvrplib.route import Route
from cvrptw.operators.destroy import cost_reducing_removal


class TestDFConversion(unittest.TestCase):
    def setUp(self):
        self.data = read_cordeau_data(
            "/home/pettepiero/tirocinio/dial-a-ride/data/c-mdvrptw/pr12",
        )

    def test_get_ids_of_time_slot(self):
        dynamic_data = generate_dynamic_df(
            "/home/pettepiero/tirocinio/dial-a-ride/data/c-mdvrptw/pr12", seed=0
        )

        ids = get_ids_of_time_slot(dynamic_data, 0)
        # known indices for seed = 0 and time_slot = 0
        known_ids = [
            6,
            7,
            11,
            12,
            14,
            16,
            22,
            23,
            25,
            28,
            30,
            34,
            35,
            37,
            41,
            42,
            46,
            48,
            57,
            60,
            67,
            69,
            70,
            71,
            76,
            80,
            86,
            88,
            89,
            90,
            92,
            96
        ]
        self.assertEqual(ids, known_ids)

    def test_get_initial_data(self):
        dynamic_data = generate_dynamic_df(
            "/home/pettepiero/tirocinio/dial-a-ride/data/c-mdvrptw/pr12", seed=0
        )
        initial_data = get_initial_data(dynamic_data)
        # some hand picked customers checks for seed = 0
        # customer ids 11, 37 and 89
        self.assertEqual(initial_data.loc[11, 'x'], 37.085)
        self.assertEqual(initial_data.loc[37, 'end_time'], 496)
        self.assertEqual(initial_data.loc[89, 'demand'], 4)

    def test_generate_dynamic_df(self):

        dynamic_data = generate_dynamic_df(
            "/home/pettepiero/tirocinio/dial-a-ride/data/c-mdvrptw/pr12",
            seed=0, # do not remove this
        )
        # some hand picked customers checks for seed = 0
        #self.assertEqual(dynamic_data.loc[dynamic_data["id"] == 1, 'call_in_time_slot'].item(), 3)
        self.assertEqual(dynamic_data.loc[1, 'call_in_time_slot'], 3)
        self.assertEqual(dynamic_data.loc[2, 'call_in_time_slot'], 1)
        self.assertEqual(dynamic_data.loc[93, 'call_in_time_slot'], 5)

        # Test the static case (all call in times are 0)
        dynamic_data = generate_dynamic_df(
            "/home/pettepiero/tirocinio/dial-a-ride/data/c-mdvrptw/pr12",
            static=True,
            seed=0
        )

        num_custs = len(dynamic_data)
        zeros = np.zeros(num_custs).astype(int).tolist()
        self.assertEqual(
            dynamic_data["call_in_time_slot"].to_list(), zeros
        )

class TestDepotsDicts(unittest.TestCase):
    def test_calculate_depots(self):
        #################################################################
        # Testing when number of depots > number of vehicles
        dict_depot_to_vehicles, dict_vehicle_to_depot = calculate_depots(
            depots=[23, 67], 
            n_vehicles=1,
            rng = rnd.default_rng(0), #fix seed to 0 for the test 
        ) 
        self.assertEqual({23:  [], 67: [0]}, dict_depot_to_vehicles)
        self.assertEqual({0: 67}, dict_vehicle_to_depot)

        #################################################################
        # Testing when number of depots = number of vehicles
        dict_depot_to_vehicles, dict_vehicle_to_depot = calculate_depots(
            depots=[23, 67], 
            n_vehicles=2,
            rng = 0, #fix seed to 0 for the test 
        ) 
        self.assertEqual({23:  [0], 67: [1]}, dict_depot_to_vehicles)
        self.assertEqual({0: 23, 1: 67}, dict_vehicle_to_depot)

        #################################################################
        # Testing when number of depots < number of vehicles
        dict_depot_to_vehicles, dict_vehicle_to_depot = calculate_depots(
            depots=[23, 67], 
            n_vehicles=5,
            rng = 0, #fix seed to 0 for the test 
        ) 
        self.assertEqual({23:  [0,2,4], 67: [1,3]}, dict_depot_to_vehicles)
        self.assertEqual({0: 23, 1: 67, 2: 23, 3: 67, 4: 23}, dict_vehicle_to_depot)

class TestReadCordeauData(unittest.TestCase):
    def test_read_cordeau_data(self):
        #on hand made test data
        data = read_cordeau_data(
            "/home/pettepiero/tirocinio/dial-a-ride/tests/test_data",
        )
        known_cost_matrix = np.array(
                            [[np.nan,     np.nan,   np.nan, np.nan, np.nan, np.nan, np.nan],
                            [np.nan,      0,        65.19,  80.62,  31.62,  136.01, 14.14],
                            [np.nan,      65.19,	0.00,	35.36,  35.36,	79.06,	69.64],
                            [np.nan,      80.62,	35.36,	0.00,	50.00,	100.00,	90.00],
                            [np.nan,      31.62,    35.36,	50.00,	0.00,	111.80,	40.00],
                            [np.nan,      136.01,	79.06,	100.00,	111.80,	0.00,	134.54],
                            [np.nan,      14.14,	69.64,	90.00,	40.00,	134.54,	0.00]]
                            )
        known_dict = {
            'name': "test_data",
            'vehicles':             1,
            'capacity':             195,
            'dimension':            5,
            'n_depots':             1,
            'depot_to_vehicles':    {6: [0]},
            'vehicle_to_depot':     {0: 6},
            'depots':               [6],
            'node_coord':           [[None, None], 
                                    [60, -30],
                                    [25, 25],
                                    [50, 50],
                                    [50, 0],
                                    [-50, 50],
                                    [50, -40]],
            'demand':               [0, 4, 12, 3, 15, 13, 0],
            'time_window':          [[-1, -1],
                                    [10, 600],
                                    [30, 800],
                                    [100, 900],
                                    [23, 900],
                                    [40, 200],
                                    [0, 1000]],
            'service_time':         [None, 12, 1, 1, 15, 10, 0],
            'edge_weight':          known_cost_matrix, 
        }
        np.testing.assert_equal(data, known_dict)
        

class TestCoordsMatrix(unittest.TestCase):
    def test_coords_matrix(self):
        #on hand made test data
        data = read_cordeau_data(
            "/home/pettepiero/tirocinio/dial-a-ride/tests/test_data",
        )
        print(f"DEBUG: data after 'read_cordeau_data':\n{data}")

        data_df = dynamic_df_from_dict(data)
        print("In test_coord_matrix:\n")
        print(f"\nDEBUG: data_df: {data_df}")
        known_cost_matrix = np.array(
                            [[np.nan,     np.nan,   np.nan, np.nan, np.nan, np.nan, np.nan],
                            [np.nan,      0,        65.19,  80.62,  31.62,  136.01, 14.14],
                            [np.nan,      65.19,	0.00,	35.36,  35.36,	79.06,	69.64],
                            [np.nan,      80.62,	35.36,	0.00,	50.00,	100.00,	90.00],
                            [np.nan,      31.62,    35.36,	50.00,	0.00,	111.80,	40.00],
                            [np.nan,      136.01,	79.06,	100.00,	111.80,	0.00,	134.54],
                            [np.nan,      14.14,	69.64,	90.00,	40.00,	134.54,	0.00]]
                            )
        
        np.testing.assert_allclose(known_cost_matrix, data['edge_weight'])


class TestCRR(unittest.TestCase):
    """Test for cost reducing removal operator"""
    def setUp(self):
        self.data = read_cordeau_data(
            "/home/pettepiero/tirocinio/dial-a-ride/tests/test_data",
        )
        self.data_df = dynamic_df_from_dict(self.data)
        route1 = Route([6, 1, 2, 6])
        route2 = Route([7, 4, 3, 7])
        self.state = CvrptwState(
                routes=[route1, route2],
                dataset=self.data
                )
        self.nodes_dict = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 6: 'F', 7: 'G'}

    def test_crr(self):
        print(f"Starting from: {self.state.routes_cost}")

        for route in self.state.routes:
            path = []
            for cust in route.nodes_list:
                path.append(self.nodes_dict[cust])
            print(f'{path}\n')
        np.testing.assert_allclose(self.state.routes_cost, np.array([148.98, 170.71]), rtol=0.01)

        self.state = cost_reducing_removal(self.state, np.random)
        print(f"After: {self.state.routes_cost}")
        for route in self.state.routes:
            path = []
            for cust in route.nodes_list:
                path.append(self.nodes_dict[cust])
            print(f"{path}\n")

        self.state = cost_reducing_removal(self.state, np.random)
        print(f"After: {self.state.routes_cost}")
        for route in self.state.routes:
            path = []
            for cust in route.nodes_list:
                path.append(self.nodes_dict[cust])
            print(f"{path}\n")

        self.state = cost_reducing_removal(self.state, np.random)
        print(f"After: {self.state.routes_cost}")
        for route in self.state.routes:
            path = []
            for cust in route.nodes_list:
                path.append(self.nodes_dict[cust])
            print(f"{path}\n")

if __name__ == "__main__":
    unittest.main()
