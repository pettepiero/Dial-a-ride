import pandas as pd

class VRPProblem:
    def __init__(self, problem_type: str, geo_data: pd.DataFrame, requests_data: dict, distance_notion: str):
        """
        Base class for VRP problems.
        
        Parameters
        ----------
        problem_type: str
            string representing problem type
        geo_data: pd.DataFrame
            DataFrame containing geographical information on nodes
        requests_data: dict
            dict containing requests from users
        distance_notion: str
            string indicating distance notion. Acceptable strings are defined in 
            self.possible_distance_notions
        """
        self.type                       = problem_type
        self.geo_data                   = geo_data
        self.requests_data              = requests_data
        self.check_requests_data_format()
        self.distances                  = self.compute_cost_matrix(distance_notion)  
        self.possible_distance_notions  = ["real", "euclidian"]

    def check_geo_data(self):
        attributes_to_check = [
                'id',
                'lat',
                'lon'
        ]
        cols = list(self.requests_data.columns)
        for att in attributes_to_check:
            assert att in cols, f"Attribute {att} not found in self.geo_data"
         
    def check_requests_data_format(self):
        attributes_to_check = [
                'id',
                'pickup_node',
                'delivery_node',
                'pickup_time_window',
                'delivery_time_window',
                'demand'
        ]
        cols = list(self.requests_data.keys())
        for att in attributes_to_check:
            assert att in cols, f"Attribute {att} not found in self.requests_data"

    def compute_cost_matrix(self, distance_notion: str):
        """
        Compute distances matrix between every node in data. Uses
        the passed notion of distance, which can be 'real' 
        (considering the road network) or 'euclidian' 
        (considering the coords only).
        
        Parameters
        ----------
        distance_notion: str
            Which distance notion to use

        Returns
        -------
        np.ndarray
            Distances matrix
        """

        assert distance_notion in self.possible_distance_notions, f"Unknown distance notion: {distance_notion}"
       
        if distance_notion == "euclidian":
            return cost_matrix_from_coords(self.geo_data[['x', 'y']].values)
        else:
            raise NotImplementedError
