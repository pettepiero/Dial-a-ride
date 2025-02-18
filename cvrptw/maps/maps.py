import plotly.express as px
import utm
import pandas as pd
import matplotlib.pyplot as plt
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
# fig = px.scatter_map(
#     df,
#     lat="centroid_lat",
#     lon="centroid_lon",
#     hover_data=["CODFERMATA"],
#     size_max=15,
#     zoom=13,
# )
# fig.show()


# us_cities = pd.read_csv(
#     "https://raw.githubusercontent.com/plotly/datasets/master/us-cities-top-1k.csv"
# )
# us_cities = us_cities.query("State in ['New York', 'Ohio']")
# print(us_cities)

# fig = px.line_map(us_cities, lat="lat", lon="lon", color="State", zoom=3, height=300)


# fig.update_layout(
#     map_style="open-street-map",
#     map_zoom=4,
#     map_center_lat=41,
#     margin={"r": 0, "t": 0, "l": 0, "b": 0},
# )

# fig.show()

# Read the CSV
df = pd.read_csv(
    "./data/DataSetActvAut(Segmenti).csv", delimiter=","
)  # Adjust delimiter if needed

print(df.columns)   

# UTM Zone (You need to specify the correct one for your region)
zone = 32  # Example for Northern Italy (Veneto)

# Convert UTM to lat/lon
df["lat_da"], df["lon_da"] = zip(
    *df.apply(lambda x: utm.to_latlon(x["XDA"], x["YDA"], zone, northern=True), axis=1)
)
df["lat_a"], df["lon_a"] = zip(
    *df.apply(lambda x: utm.to_latlon(x["XA"], x["YA"], zone, northern=True), axis=1)
)

print(f"min max lat {df['lat_da'].min()} max lat {df['lat_da'].max()}")
print(f"min max long {df['lon_da'].min()} max long {df['lon_da'].max()}")

# filter favaro area
long_min = 12.25766
long_max = 12.31817
lat_min = 45.490074
lat_max = 45.52111
df_filtered = df[
    (df["lon_da"] >= long_min)
    & (df["lon_da"] <= long_max)  # Start point longitude
    & (df["lat_da"] >= lat_min)
    & (df["lat_da"] <= lat_max)  # Start point latitude
    & (df["lon_a"] >= long_min)
    & (df["lon_a"] <= long_max)  # End point longitude
    & (df["lat_a"] >= lat_min)
    & (df["lat_a"] <= lat_max)  # End point latitude
]
print(df_filtered)
# Create a long-form DataFrame for Plotly (each segment as a line)
plot_data = []
for _, row in df.iterrows():
    plot_data.append(
        {"lat": row["lat_da"], "lon": row["lon_da"], "segment": row["CODSEGMENTO"]}
    )
    plot_data.append(
        {"lat": row["lat_a"], "lon": row["lon_a"], "segment": row["CODSEGMENTO"]}
    )

df_plot = pd.DataFrame(plot_data)

# Plot with Plotly Express
fig = px.line_mapbox(
    df_plot,
    lat="lat",
    lon="lon",
    mapbox_style="open-street-map",
    zoom=12,
    title="Transport Network",
)

fig.show()
