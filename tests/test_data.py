import unittest
from unittest.mock import Mock
from cvrptw.myvrplib.data_module import *

class TestDFConversion(unittest.TestCase):
    def setUp(self):
        data = read_cordeau_data(
            "/home/pettepiero/tirocinio/dial-a-ride/data/c-mdvrptw/pr12"
        )

    def test_create_customers_df(self):
        data = read_cordeau_data(
            "/home/pettepiero/tirocinio/dial-a-ride/data/c-mdvrptw/pr12"
        )
        customers_df = create_customers_df(data)
        # some hand picked customers checks
        self.assertEqual(customers_df['x'].loc[customers_df["customer_id"] == 1].item(), 33.588)
        self.assertEqual(customers_df['service_time'].loc[customers_df["customer_id"] == 23].item(), 18)
        self.assertEqual(customers_df['start_time'].loc[customers_df["customer_id"] == 36].item(), 269)

    def test_get_ids_of_time_slot(self):
        data = convert_to_dynamic_data(
            "/home/pettepiero/tirocinio/dial-a-ride/data/c-mdvrptw/pr12", seed=0
        )
        ids = get_ids_of_time_slot(data, 0)
        # known indices for seed = 0 and time_slot = 0
        known_ids = [7, 8, 12, 13, 15, 17, 23, 24, 26, 29, 31, 35, 36, 38, 42, 43, 47, 49, 58, 61, 68, 70, 71, 72, 77, 81, 87, 89, 90, 91, 93]
        self.assertEqual(ids, known_ids)

    def test_get_initial_data(self):
        data = convert_to_dynamic_data(
            "/home/pettepiero/tirocinio/dial-a-ride/data/c-mdvrptw/pr12",
            seed=0
        )
        initial_data = get_initial_data(data)
        # some hand picked customers checks for seed = 0
        self.assertEqual(initial_data['x'].loc[initial_data["customer_id"] == 8].item(), -57.703)
        self.assertEqual(
            initial_data["end_time"].loc[initial_data["customer_id"] == 47].item(), 601
        )
        self.assertEqual(
            initial_data["demand"].loc[initial_data["customer_id"] == 89].item(), 4
        )

    def test_convert_to_dynamic_data(self):

        dynamic_data = convert_to_dynamic_data(
            "/home/pettepiero/tirocinio/dial-a-ride/data/c-mdvrptw/pr12",
            seed=0
        )
        # some hand picked customers checks for seed = 0
        print(f"\nDynamic data:\n")
        self.assertEqual(dynamic_data.loc[dynamic_data["customer_id"] == 1, 'call_in_time_slot'].item(), 3)
        self.assertEqual(
            dynamic_data.loc[
                dynamic_data["customer_id"] == 2, "call_in_time_slot"
            ].item(),
            3,
        )
        self.assertEqual(dynamic_data.loc[dynamic_data["customer_id"] == 93, 'call_in_time_slot'].item(), 0)


if __name__ == "__main__":
    unittest.main()
