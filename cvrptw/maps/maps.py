import plotly.express as px
import utm
import pandas as pd
import re

df = pd.read_csv("./data/DataSetActvAut(Fermate).csv")
zone = 32
# convert from utm to lat/lon
df["centroid_lat"] = df.apply(lambda x: utm.to_latlon(x["X"], x["Y"], zone, northern=True)[0], axis=1)
df["centroid_lon"] = df.apply(
    lambda x: utm.to_latlon(x["X"], x["Y"], zone, northern=True)[1], axis=1
)

# Filtering stops on Favaro
with open('./data/fermate_chiamata_actv.txt', 'r', encoding='utf-8') as file:
    content = file.readlines()
valid_stops = [int(match) for line in content for match in re.findall(r'\((\d+)\)', line)]
df = df.loc[df["CODFERMATA"].isin(valid_stops)]
# drop cols
cols_to_drop = ["CODPRGFERMATA", "NODO", "ARCO", "DENOMINAZIONE", "UBICAZIONE", "COMUNE", "TIPOLOGIA", "ZONATARIFFARIA", "INDFERMATA", "INDPROXFERMATA", "NETWORKVERSION"]
df = df.drop(columns=cols_to_drop)
df.to_csv("./data/actv.csv", index=False)


# read actv_df
actv_df = pd.read_csv("./data/actv.csv")



fig = px.scatter_map(
    df,
    lat="centroid_lat",
    lon="centroid_lon",
    hover_data=["CODFERMATA"],
    size_max=15,
    zoom=13,
)
fig.show()
