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
            nodes_list: list of int
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

    def __init__(self, nodes_list: list, vehicle: int=None, cost: float=None, start_times: list = None, planned_windows: list = [], vehicle_start_time:float = None):
        self.nodes_list = nodes_list
        self.vehicle = vehicle
        self.start_times = start_times if start_times is not None else []
        self.planned_windows = planned_windows   # List of tuples for each customer in route
        # first value is the planned arrival time
        # second value is the planned departure time
        # NOTE: maybe only planned arrival time is sufficient
        self.demand = None

    def __str__(self):
        return f"Route(nodes_list={self.nodes_list}, vehicle={self.vehicle}, start_times={self.start_times}, planned_windows={self.planned_windows})"

    def __len__(self) -> int:
        """
        Returns the number of customers in the route, including the depot
        at the beginning and end.
        """
        return len(self.nodes_list)

    def copy(self):
        """
        Returns a deep copy of the route.
        """  
        return Route(copy.deepcopy(
            self.nodes_list
            ), 
            vehicle=self.vehicle, 
            start_times=copy.deepcopy(self.start_times),
            planned_windows=copy.deepcopy(self.planned_windows))

    def remove(self, node: list) -> None:
        """
        Removes a customer from the route.
            Parameters:
                - node: list
                    Customers to be removed from the route.
                    NOTE: node refers to node IDs, not customer IDs
            Returns:
                - None
        """
        for customer in node:
            self.nodes_list = list(filter(lambda a: a!=customer, self.nodes_list))

    def insert(self, position: int, node: int) -> None:
        """
        Inserts a customer in a given position in the route
            Parameters:
                - position: int
                    Position where the customer will be inserted.
                - node: int
                    node id to be inserted in the route.
            Returns:
                - None
        """
        self.nodes_list.insert(position, node)