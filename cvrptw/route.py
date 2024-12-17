import copy

from data_module import data
from myvrplib import END_OF_DAY

class Route:
    """
    Class containing the route information and methods.
    Based on 'label' concept of: [1] An adaptive large neighborhood search for the multi-depot dynamic vehicle
    routing problem with time windows - Wang et al (2024)
    """

    def __init__(self, customers_list: list):
        self.customers_list = customers_list
        self.cost = self.calculate_cost()
        self.earliest_start_times = self.get_earliest_times()
        self.latest_start_times = self.get_latest_times()
        self.times = list(zip(self.earliest_start_times, self.latest_start_times))

    def __len__(self):
        return len(self.customers_list)

    def copy(self):
        return Route(copy.deepcopy(self.customers_list))

    def calculate_cost(self):
        cost = 0
        for i in range(len(self.customers_list) - 1):
            cost += data["edge_weight"][self.customers_list[i]][
                self.customers_list[i + 1]
            ]
        return cost

    def remove(self, customer: int):
        self.customers_list.remove(customer)
        self.cost = self.calculate_cost()

    def insert(self, position: int, customer: int):
        self.customers_list.insert(position, customer)
        self.cost = self.calculate_cost()

    def get_earliest_times(self):  # EST = Earliest Start Time
        est = []
        if self.customers_list[0] in data["depots"]:
            est.append(0)
        else:
            AssertionError("First node in route is not a depot")

        for i in range(1, len(self.customers_list)):
            current = self.customers_list[i]
            prev = self.customers_list[i - 1]
            time = max(
                est[i - 1]
                + data["service_time"][prev]
                + data["edge_weight"][current][prev],
                data["time_window"][current][0],
            )
            est.append(time)

        if len(est) != len(self):
            AssertionError("Error in calculating earliest start times")

        return est

    def get_latest_times(self):  # LST = Latest Start Time
        lst = [None] * len(self.customers_list)
        lst[-1] = END_OF_DAY
        for i in reversed(range(len(self.customers_list) - 1)):
            next = self.customers_list[i + 1]
            current = self.customers_list[i]
            time = min(
                lst[i + 1]
                - data["service_time"][current]
                - data["edge_weight"][current][next],
                data["time_window"][current][1],
            )
            lst[i] = float(time)

        if len(lst) != len(self):
            AssertionError("Error in calculating latest start times")

        return lst
