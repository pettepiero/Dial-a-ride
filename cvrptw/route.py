import copy
import logging

from data_module import data
from myvrplib import END_OF_DAY, LOGGING_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)

class Route:
    """
    Class containing the route information and methods.
    Based on 'label' concept of: Wang et al (2024)
        Attributes:
            customers_list: list of int
                List of customers in the route.
            vehicle: int
                Vehicle used in the route.
            cost: float
                Cost of the route.
            start_times: list of tuple
                List of tuples containing earliest and latest start times for each customer in the route.
            planned_windows: list of tuple
                List of tuples containing planned arrival and departure times for each customer in the route,
                including the depot.
    """

    def __init__(self, customers_list: list, vehicle: int=None, cost: float=None, start_times: list = None, planned_windows: list = [], vehicle_start_time:float = None):
        self.customers_list = customers_list
        self.vehicle = vehicle
        self.cost = self.calculate_cost()
        self.start_times = list(zip(self.get_earliest_times(), self.get_latest_times()))
        self.planned_windows = planned_windows   # List of tuples for each customer in route
        # first value is the planned arrival time
        # second value is the planned departure time
        # NOTE: maybe only planned arrival time is sufficient

    def __len__(self) -> int:
        """
        Returns the number of customers in the route, including the depot
        at the beginning and end.
        """
        return len(self.customers_list)

    def copy(self):
        """
        Returns a deep copy of the route.
        """  
        return Route(copy.deepcopy(
            self.customers_list
            ), 
            vehicle=self.vehicle, 
            cost=self.cost, 
            start_times=copy.deepcopy(self.start_times),
            planned_windows=copy.deepcopy(self.planned_windows))

    def calculate_cost(self) -> float:
        """
        Calculates the cost of the route based on the edge weights of 'data'
        from 'data_module'.
            Parameters:
                - None
            Returns:
                - cost: float
                    Cost of the route.
        """
        cost = 0
        for i in range(len(self.customers_list) - 1):
            cost += data["edge_weight"][self.customers_list[i]][
                self.customers_list[i + 1]
            ]
        return round(cost, 2)

    def remove(self, customer: int) -> None:
        """
        Removes a customer from the route and recalculates the cost.
            Parameters:
                - customer: int
                    Customer to be removed from the route.
            Returns:
                - None
        """
        self.customers_list.remove(customer)
        self.cost = self.calculate_cost()

    def insert(self, position: int, customer: int) -> None:
        """
        Inserts a customer in a given position in the route and recalculates the cost and
        the planned times.
            Parameters:
                - position: int
                    Position where the customer will be inserted.
                - customer: int
                    Customer to be inserted in the route.
            Returns:
                - None
        """
        self.customers_list.insert(position, customer)
        self.cost = self.calculate_cost()
        self.calculate_planned_times()
        est = self.get_earliest_times()
        lst = self.get_latest_times()
        self.start_times = list(zip(est, lst))

    def get_earliest_times(self) -> list:
        """
        Calculates a vector of the earliest start times (EST) for each customer in the route.
        Based on formula (3b) of Wang et al. (2024).
            Parameters:
                - None
            Returns:
                - est: List of earliest start times for each customer in the route.
        """
        est = []

        # If first element is a depot, then the earliest start time is 0
        if self.customers_list[0] in data["depots"]:
            est.append(0)
        else:
            # TODO: This will have to be changed for dynamic case
            raise AssertionError("First node in route is not a depot")

        # Implementation of formula 3b of Wang et al. (2024)
        for i in range(1, len(self.customers_list)-1):
            current = self.customers_list[i]
            prev = self.customers_list[i-1]
            time = float(round(max(
                est[i - 1]
                + data["service_time"][prev]
                + data["edge_weight"][current][prev],
                data["time_window"][current][0],
            ), 2))
            est.append(time)

        if len(est) != len(self):
            AssertionError("Error in calculating earliest start times")
        return est

    def get_latest_times(self) -> list:
        """
        Calculates a vector of the latest start times (LST) for each customer in the route.
        Based on formula (13) of Wang et al. (2024).
            Parameters:
                - None
            Returns:
                - lst: List of latest start times for each customer in the route.
        """
        lst = [None] * len(self.customers_list)
        lst[-1] = END_OF_DAY
        for i in reversed(range(len(self.customers_list) - 1)):
            next = self.customers_list[i + 1]
            current = self.customers_list[i]
            time = round(min(
                lst[i + 1]
                - data["service_time"][current]
                - data["edge_weight"][current][next],
                data["time_window"][current][1],
            ), 2)
            lst[i] = float(time)

        if len(lst) != len(self):
            AssertionError("Error in calculating latest start times")
        return lst

    # TODO: Test this method
    def calculate_planned_times(self):
        """
        Calculate the planned arrival and departure times for each customer in the route.
        """

        self.planned_windows = []
        first_customer = self.customers_list[1]
        self.planned_windows.append(
            [0, float(round(max(0, data["time_window"][first_customer][0] -
              data["edge_weight"][self.customers_list[0]][first_customer]), 2))])

        last_departure = self.planned_windows[0][1]
        last_customer = self.customers_list[0]
        for customer in self.customers_list[1:]:
            # Planned arrival time at customer idx
            # Planned departure time is the planned arrival time + service time
            arr = last_departure + data["edge_weight"][last_customer][customer]
            dep = arr + data["service_time"][customer]
            self.planned_windows.append([float(round(arr, 2)), float(round(dep, 2))])
            #debug
            # logger.debug(f"In calculate_planned_times: last_departure: {last_departure}")
            # logger.debug(f"data['edge_weight'][{last_customer}][{customer}]: {data['edge_weight'][last_customer][customer]}")
            last_departure = dep
            last_customer = customer
