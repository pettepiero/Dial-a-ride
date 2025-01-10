import unittest
from unittest.mock import Mock
from operators.destroy import random_removal
from vrpstates import CvrptwState
from data_module import test_data
from initial_solutions.initial_solutions import nearest_neighbor_tw

class TestRandomRemoval(unittest.TestCase):

    def setUp(self):
        data = test_data
        nn_solution = nearest_neighbor_tw()
        self.state = nn_solution
        self.state.copy = Mock(return_value=self.state)
        self.state.find_route = CvrptwState.find_route
        self.state.update_times = Mock()
        self.state.unassigned = []
        self.rng = Mock()
        self.rng.choice = Mock(return_value=[1, 2, 3])  # Example customers to remove

    def test_random_removal(self):
        # Call the function
        result = random_removal(self.state, self.rng)

        # Assertions
        self.assertEqual(self.state.unassigned, [1, 2, 3])
        self.state.update_times.assert_called_once()
        self.assertIsInstance(result, CvrptwState)

    def test_random_removal_with_routes(self):
        # Setup a route that includes the customers
        route = Mock()
        self.state.find_route = Mock(
            side_effect=lambda x: route if x in [1, 2, 3] else None
        )

        # Call the function
        result = random_removal(self.state, self.rng)

        # Assertions
        self.assertEqual(self.state.unassigned, [1, 2, 3])
        route.remove.assert_any_call(1)
        route.remove.assert_any_call(2)
        route.remove.assert_any_call(3)
        self.state.update_times.assert_called_once()
        self.assertIsInstance(result, CvrptwState)


if __name__ == "__main__":
    unittest.main()
