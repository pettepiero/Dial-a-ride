import plotly.express as px
import utm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

stops_df = pd.read_csv("./data/DataSetActvAut(Fermate).csv")
zone = 32
# convert from utm to lat/lon
stops_df["centroid_lat"] = stops_df.apply(
    lambda x: utm.to_latlon(x["X"], x["Y"], zone, northern=True)[0], axis=1
)
stops_df["centroid_lon"] = stops_df.apply(
    lambda x: utm.to_latlon(x["X"], x["Y"], zone, northern=True)[1], axis=1
)

# Filtering stops on Favaro dial a ride area
with open('./data/fermate_chiamata_actv.txt', 'r', encoding='utf-8') as file:
    content = file.readlines()
valid_stops = [int(match) for line in content for match in re.findall(r'\((\d+)\)', line)]
# drop cols
cols_to_drop = ["CODPRGFERMATA", "NODO", "ARCO", "DENOMINAZIONE", "UBICAZIONE", "COMUNE", "TIPOLOGIA", "ZONATARIFFARIA", "INDFERMATA", "INDPROXFERMATA", "NETWORKVERSION"]
stops_df = stops_df.drop(columns=cols_to_drop)
# filter valid stops
stops_df = stops_df.loc[stops_df["CODFERMATA"].isin(valid_stops)]
stops_df.to_csv("./data/actv_stops.csv", index=False)
list_of_stops = stops_df["CODFERMATA"].unique()
print("List of stops:")
print(list_of_stops)

arcs_df = pd.read_csv("./data/DataSetActvAut(Archi).csv", delimiter=",")

# filter arcs that only connect the stops in the list
arcs_df = arcs_df.loc[arcs_df["FERMDA"].isin(list_of_stops)]
arcs_df = arcs_df.loc[arcs_df["FERMAA"].isin(list_of_stops)]
print(arcs_df.head())

list_of_arcs = arcs_df["CODPRGARCOFERMATA"].unique()
print("List of arcs:")
print(list_of_arcs)

# Read the CSV
segments_df = pd.read_csv(
    "./data/DataSetActvAut(Segmenti).csv", delimiter=","
)  # Adjust delimiter if needed

# Filter the segments DataFrame to only include the arcs in the arcs DataFrame
segments_df = segments_df.loc[
    segments_df["CODPRGARCOFERMATA"].isin(list_of_arcs)
]


# UTM Zone (You need to specify the correct one for your region)
zone = 32  # Example for Northern Italy (Veneto)

# Convert UTM to lat/lon
segments_df["lat_da"], segments_df["lon_da"] = zip(
    *segments_df.apply(
        lambda x: utm.to_latlon(x["XDA"], x["YDA"], zone, northern=True), axis=1
    )
)
segments_df["lat_a"], segments_df["lon_a"] = zip(
    *segments_df.apply(
        lambda x: utm.to_latlon(x["XA"], x["YA"], zone, northern=True), axis=1
    )
)

print(
    f"min max lat {stops_df['centroid_lat'].min()} max lat {stops_df['centroid_lat'].max()}"
)
print(f"min max long {stops_df['centroid_lon'].min()} max long {stops_df['centroid_lon'].max()}")

# filter favaro area
# long_min = 12.25766
# long_max = 12.31817
# lat_min = 45.490074
# lat_max = 45.52111
# epsilon = 0.002

# long_min = stops_df["centroid_lon"].min() - epsilon
# long_max = stops_df["centroid_lon"].max() + epsilon
# lat_min = stops_df["centroid_lat"].min() - epsilon
# lat_max = stops_df["centroid_lat"].max() + epsilon
# df_filtered = df[
#     (df["lon_da"] >= long_min)
#     & (df["lon_da"] <= long_max)  # Start point longitude
#     & (df["lat_da"] >= lat_min)
#     & (df["lat_da"] <= lat_max)  # Start point latitude
#     & (df["lon_a"] >= long_min)
#     & (df["lon_a"] <= long_max)  # End point longitude
#     & (df["lat_a"] >= lat_min)
#     & (df["lat_a"] <= lat_max)  # End point latitude
# ]
# print(df_filtered)
# Create a long-form DataFrame for Plotly (each segment as a line)
plot_data = []
for _, row in segments_df.iterrows():
    plot_data.append(
        {"lat": row["lat_da"], "lon": row["lon_da"], "segment": row["CODSEGMENTO"], "CODPRGARCOFERMATA" : row["CODPRGARCOFERMATA"]}
    )
    plot_data.append(
        {
            "lat": row["lat_a"],
            "lon": row["lon_a"],
            "segment": row["CODSEGMENTO"],
            "CODPRGARCOFERMATA": row["CODPRGARCOFERMATA"],
        }
    )

df_plot = pd.DataFrame(plot_data)


arcs = df_plot["CODPRGARCOFERMATA"].unique()
print(df_plot.head())

# Plot with Plotly Express
fig = px.line_map(
    df_plot.loc[df_plot["CODPRGARCOFERMATA"] == 1311],
    # df_plot,
    lat="lat",
    lon="lon",
    # mapbox_style="open-street-map",
    zoom=12,
    title="Transport Network",
    hover_data=["segment"],
    line_group="CODPRGARCOFERMATA",
)

fig.add_scattermap(
    lat=stops_df["centroid_lat"],
    lon=stops_df["centroid_lon"],
    mode="markers",
    marker=dict(size=8, color="red"),
    hovertext=stops_df["CODFERMATA"],
)

fig.show()
