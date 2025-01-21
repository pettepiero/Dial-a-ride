import unittest
from unittest.mock import Mock
from cvrptw.myvrplib.data_module import *

class TestDFConversion(unittest.TestCase):
    def test_create_customers_df(self):
        data = read_cordeau_data(
            "/home/pettepiero/tirocinio/dial-a-ride/data/c-mdvrptw/pr12"
        )

        customers_df = create_customers_df(data)
        # some hand picked customers checks
        self.assertEqual(customers_df['x'].loc[customers_df["customer_id"] == 1].item(), 33.588)
        self.assertEqual(customers_df['service_time'].loc[customers_df["customer_id"] == 23].item(), 18)
        self.assertEqual(customers_df['start_time'].loc[customers_df["customer_id"] == 36].item(), 269)



if __name__ == "__main__":
    unittest.main()
