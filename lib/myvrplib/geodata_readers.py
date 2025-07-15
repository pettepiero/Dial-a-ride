import numpy as np
import pandas as pd
from lib.myvrplib.data_module import *

def base_geodata_reader(file: str) -> pd.DataFrame:
    df = pd.read_csv(file)
    return df



base_geo_data = base_geodata_reader('/home/pettepiero/tirocinio/dial-a-ride/tests/cvrppdtw/test_map_data.csv')
