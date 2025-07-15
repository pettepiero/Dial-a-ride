from lib.myvrplib.vrpstates import CVRPTWState
from lib.myvrplib.data_module import data, END_OF_DAY, generate_dynamic_cust_df, get_initial_data
import pandas as pd
import numpy as np
import copy

class Customer():
    def __init__(self, id: int, x: float, y: float, demand: float, ready_time: float, due_time: float, service_time: float):
        self.id = id
        self.x = x
        self.y = y
        self.demand = demand
        self.ready_time = ready_time
        self.due_time = due_time
        self.service_time = service_time


class DynamicVRP():
    """
    Class for the dynamic CVRPTW problem.
    """
    def __init__(self, num_steps: int = 20, data: dict = data):
        self.state = CVRPTWState(data)
        self.num_steps = num_steps
        self.data = generate_dynamic_cust_df(
            "/home/pettepiero/tirocinio/dial-a-ride/data/c-mdvrptw/pr11",
            n_steps=self.num_steps,
        )
        self.planned_customers = []
        self.new_customers = []
        self.end_of_day = END_OF_DAY

        print(f"Data: {self.data}")

    def initial_requests(self):
        """
        Returns the initial requests for the first step.
        """
        initial_cust_idxs = np.where(self.data["call_in_time_slot"] == 0)


if __name__ == "__main__":
    vrp = DynamicVRP()
    print(vrp.data)

    initial_data = get_initial_data(vrp.data)
    # print(f"Initial data: {initial_data}")

    # Initial solution for call_in_time_slot = 0

