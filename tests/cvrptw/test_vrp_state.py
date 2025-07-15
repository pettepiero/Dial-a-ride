import unittest
from lib.myvrplib.data_module import *
from lib.myvrplib.vrpstates import CVRPTWState
from lib.myvrplib.route import Route
from lib.myvrplib.dataset_readers import read_cordeau_data

class TestCVRPTWState(unittest.TestCase):
    def setUp(self):
        self.data = read_cordeau_data(
            "/home/pettepiero/tirocinio/dial-a-ride/data/c-mdvrptw/pr12"
        )
        self.state = CVRPTWState(dataset=self.data)

    def test_get_qmax(self):
        qmax = self.state.get_qmax()
        # knwown q max for this dataset is 25
        self.assertEqual(qmax, 25)

    def test_distances_attribute(self):
        self.assertEqual(self.state.distances.shape, self.data["edge_weight"].shape)
        self.assertTrue(np.array_equal(self.state.distances[1:,1:], self.data["edge_weight"][1:,1:]))

    def test_generate_twc_matrix(self):
        a = self.state.twc
        b = self.state.generate_twc_matrix(self.data["time_window"], self.data["edge_weight"])
        self.assertTrue(np.array_equal(a, b))


if __name__ == "__main__":
    unittest.main()
