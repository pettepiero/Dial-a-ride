from cvrptw.myvrplib.vrpstates import CvrptwState
from cvrptw.myvrplib.data_module import data, END_OF_DAY, convert_to_dynamic_data

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
        self.state = CvrptwState(data)
        self.num_steps = num_steps
        self.data = convert_to_dynamic_data(
            "/home/pettepiero/tirocinio/dial-a-ride/data/c-mdvrptw/pr11",
            n_steps=self.num_steps,
        )
        self.planned_customers = []
        self.new_customers = []
        self.end_of_day = END_OF_DAY

    def initial_requests(self):
        """
        Returns the initial requests for the first step.
        """
        return self.data["requests"][0]






if __name__ == "__main__":
    vrp = DynamicVRP()
    print(vrp.data)
    print(vrp.data["dimension"])
