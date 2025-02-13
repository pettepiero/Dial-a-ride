import copy
import logging

from cvrptw.myvrplib.data_module import data, d_data
from cvrptw.myvrplib.myvrplib import END_OF_DAY, LOGGING_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)

class Route():
    """
    Class containing the route information and methods.
    Based on 'label' concept of: Wang et al (2024)
        Attributes:
            customers_list: list of int
                List of node IDs in the route. Not customer IDs.
            vehicle: int
                Vehicle used in the route.
            start_times: list of tuple
                List of tuples containing earliest and latest start times for each customer in the route.
            planned_windows: list of tuple
                List of tuples containing planned arrival and departure times for each customer in the route,
                including the depot.
            demand: int
                Total demand of the route.
    """

    def __init__(self, customers_list: list, vehicle: int=None, cost: float=None, start_times: list = None, planned_windows: list = [], vehicle_start_time:float = None):
        self.customers_list = customers_list
        self.vehicle = vehicle
        self.start_times = start_times if start_times is not None else []
        self.planned_windows = planned_windows   # List of tuples for each customer in route
        # first value is the planned arrival time
        # second value is the planned departure time
        # NOTE: maybe only planned arrival time is sufficient
        self.demand = None

    def __str__(self):
        return f"Route(customers_list={self.customers_list}, vehicle={self.vehicle}, start_times={self.start_times}, planned_windows={self.planned_windows})"

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
            start_times=copy.deepcopy(self.start_times),
            planned_windows=copy.deepcopy(self.planned_windows))

    def remove(self, customer: int) -> None:
        """
        Removes a customer from the route.
            Parameters:
                - customer: int
                    Customer to be removed from the route.
            Returns:
                - None
        """
        self.customers_list = list(filter(lambda a: a!=customer, self.customers_list))

    def insert(self, position: int, customer: int) -> None:
        """
        Inserts a customer in a given position in the route
            Parameters:
                - position: int
                    Position where the customer will be inserted.
                - customer: int
                    Customer to be inserted in the route.
            Returns:
                - None
        """
        self.customers_list.insert(position, customer)

