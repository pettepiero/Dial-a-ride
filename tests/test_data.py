import unittest
from unittest.mock import Mock
from cvrptw.myvrplib.data_module import *

class TestDFConversion(unittest.TestCase):
    def setUp(self):
        self.data = read_cordeau_data(
            "/home/pettepiero/tirocinio/dial-a-ride/data/c-mdvrptw/pr12"
        )

    def test_create_customers_df(self):
        customers_df = create_customers_df(self.data)
        # test dimensions
        self.assertEqual(customers_df.shape[0], self.data["dimension"] + self.data["n_depots"] +1)
        # some hand picked customers checks
        self.assertEqual(customers_df['x'].loc[customers_df["id"] == 1].item(), 33.588)
        self.assertEqual(customers_df['service_time'].loc[customers_df["id"] == 23].item(), 18)
        self.assertEqual(customers_df['start_time'].loc[customers_df["id"] == 36].item(), 269)
        # testing depots
        self.assertEqual(customers_df['x'].loc[customers_df["id"] == 97].item(), 6.229)
        self.assertEqual(customers_df['service_time'].loc[customers_df["id"] == 97].item(), 0)
        self.assertEqual(
            customers_df["start_time"].loc[customers_df["id"] == 99].item(),
            0,
        )
        self.assertEqual(
            customers_df["end_time"].loc[customers_df["id"] == 97].item(),
            END_OF_DAY,
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
        self.assertEqual(initial_data.loc[initial_data["id"] == 11, 'x'].item(), 37.085)
        self.assertEqual(
            initial_data.loc[initial_data["id"] == 37, 'end_time'].item(), 496
        )
        self.assertEqual(
            initial_data.loc[initial_data["id"] == 89, 'demand'].item(), 4
        )

    def test_generate_dynamic_df(self):

        dynamic_data = generate_dynamic_df(
            "/home/pettepiero/tirocinio/dial-a-ride/data/c-mdvrptw/pr12",
            seed=0
        )
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


class TestCoordsMatrix(unittest.TestCase):
    def test_coords_matrix(self):
        #on hand made test data
        data = read_cordeau_data(
            "/home/pettepiero/tirocinio/dial-a-ride/tests/test_data"
        )
        known_cost_matrix = np.array([[np.nan,     np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan],
                            [np.nan,      0,      65.19,  80.62,  31.62,  136.01, 14.14],
                            [np.nan,      65.19,	0.00,	35.36,  35.36,	79.06,	69.64],
                            [np.nan,      80.62,	35.36,	0.00,	50.00,	100.00,	90.00],
                            [np.nan,      31.62,  35.36,	50.00,	0.00,	111.80,	40.00],
                            [np.nan,      136.01,	79.06,	100.00,	111.80,	0.00,	134.54],
                            [np.nan,      14.14,	69.64,	90.00,	40.00,	134.54,	0.00]])
        
        np.testing.assert_allclose(known_cost_matrix, data['edge_weight'])


if __name__ == "__main__":
    unittest.main()
