import unittest
from unittest.mock import Mock
from lib.myvrplib.data_module import *
from lib.myvrplib.dataset_readers import read_cordeau_data 
from lib.myvrplib.vrpstates import CvrptwState
from lib.myvrplib.route import Route
from lib.operators.destroy import cost_reducing_removal

class TestDFConversion(unittest.TestCase):
    def setUp(self):
        self.data = read_cordeau_data(
            "/home/pettepiero/tirocinio/dial-a-ride/data/c-mdvrptw/pr12"
        )

    def test_get_ids_of_time_slot(self):
        dynamic_data = generate_dynamic_df(self.data, seed=0)

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
        dynamic_data = generate_dynamic_df(self.data, seed=0)
        initial_data = get_initial_data(dynamic_data)
        # some hand picked customers checks for seed = 0
        self.assertEqual(initial_data.loc[initial_data["id"] == 11, 'x'].item(), 37.085)
        self.assertEqual(
            initial_data.loc[initial_data["id"] == 37, 'end_time'].item(), 496
        )
        self.assertEqual(
            initial_data.loc[initial_data["id"] == 89, 'demand'].item(), 4
        )

    def test_generate_dynamic_df(self):
        dynamic_data = generate_dynamic_df(self.data, seed=0)
        # some hand picked customers checks for seed = 0

        self.assertEqual(dynamic_data.loc[dynamic_data["id"] == 1, 'call_in_time_slot'].item(), 3)
        self.assertEqual(
            dynamic_data.loc[
                dynamic_data["id"] == 2, "call_in_time_slot"
            ].item(),
            1,
        )
        self.assertEqual(dynamic_data.loc[dynamic_data["id"] == 93, 'call_in_time_slot'].item(), 5)

        # Test the static case (all call in times are 0)
        dynamic_data = generate_dynamic_df(self.data, static=True, seed=0)

        num_custs = len(dynamic_data)
        zeros = np.zeros(num_custs).astype(int).tolist()
        self.assertEqual(
            dynamic_data["call_in_time_slot"].to_list(), zeros
        )


class TestCoordsMatrix(unittest.TestCase):
    def test_coords_matrix(self):
        #on hand made test data
        data = read_cordeau_data(
            "/home/pettepiero/tirocinio/dial-a-ride/tests/cvrptw/test_data"
        )

        known_cost_matrix = np.array([
                            [np.nan,    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                            [np.nan,    0,      65.19,  80.62,  31.62,  136.01, 14.14,  67.08],
                            [np.nan,    65.19,	0.00,	35.36,  35.36,	79.06,	69.64,  35.36],
                            [np.nan,    80.62,	35.36,	0.00,	50.00,	100.00,	90.00,  70.71],
                            [np.nan,    31.62,  35.36,	50.00,	0.00,	111.80,	40.00,  50.00],
                            [np.nan,    136.01,	79.06,	100.00,	111.80,	0.00,	134.54, 70.71],
                            [np.nan,    14.14,	69.64,	90.00,	40.00,	134.54,	0.00,   64.03],
                            [np.nan,    67.08,  35.36,  70.71,  50.00,  70.71,  64.03,  0]])
        
        np.testing.assert_allclose(known_cost_matrix, data['edge_weight'])


class TestCRR(unittest.TestCase):
    """Test for cost reducing removal operator"""
    def setUp(self):
        self.data = read_cordeau_data(
            "/home/pettepiero/tirocinio/dial-a-ride/tests/cvrptw/test_data"
        )
        self.data_df = dynamic_df_from_dict(self.data)
        route1 = Route([6, 1, 2, 6])
        route2 = Route([7, 4, 3, 7])
        self.state = CvrptwState(routes=[route1, route2], dataset=self.data)
        self.nodes_dict = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 6: 'F', 7: 'G'}

    def test_crr(self):

        for route in self.state.routes:
            path = []
            for cust in route.customers_list:
                path.append(self.nodes_dict[cust])
        np.testing.assert_allclose(self.state.routes_cost, np.array([148.98, 170.71]), rtol=0.01)

        self.state = cost_reducing_removal(self.state, np.random)
        for route in self.state.routes:
            path = []
            for cust in route.customers_list:
                path.append(self.nodes_dict[cust])

        self.state = cost_reducing_removal(self.state, np.random)
        for route in self.state.routes:
            path = []
            for cust in route.customers_list:
                path.append(self.nodes_dict[cust])

        self.state = cost_reducing_removal(self.state, np.random)
        for route in self.state.routes:
            path = []
            for cust in route.customers_list:
                path.append(self.nodes_dict[cust])

if __name__ == "__main__":
    unittest.main()
