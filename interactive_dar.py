import os
import dash
from dash import dcc, html, ctx
import dash_leaflet as dl
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import json
import cvrptw.maps.maps as mp

# Sample bus stops with coordinates
bus_stops = [
    {"name": "Stop A", "lat": 45.4642, "lon": 9.1900},
    {"name": "Stop B", "lat": 45.4662, "lon": 9.1920},
    {"name": "Stop C", "lat": 45.4682, "lon": 9.1940},
]
styles = {"pre": {"border": "thin lightgrey solid", "overflowX": "scroll"}}

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container(
    [
        html.H2("Dial-a-Ride"),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="map", style={"height": "650px"}), lg=6, sm=12),
            ],
        ),
        html.Hr(),
        dcc.Store(id="choices", storage_type="memory"),
        html.Div(children="Selected start and end points:"),
        html.Pre(id="departure", style=styles["pre"]),
        html.Pre(id="arrival", style=styles["pre"]),
        html.Pre(id="choices-text", style=styles["pre"]),
        # Buttons for submitting and clearing choices
        dbc.Button("Submit", id="submit-btn", color="primary", className="me-2"),
        dbc.Button("Clear", id="clear-btn", color="danger"),
        dbc.Button("Load dataset", id="load-btn", color="info"),
    ]
)

@app.callback(
    Output("map", "figure"),
    Input("load-btn", "n_clicks"),
    prevent_initial_call=True,
)
def load_dataset(n_clicks):
    graph, stops_df, segments_df = mp.setup()
    fig = mp.create_visualization()
    return fig


# Callback to render the map
@app.callback(
    Output("map", "figure"),
    Input("map", "id"),
    prevent_initial_call=False,
)
def display_map(*args):
    # Create the map using the create_visualization function
    fig = (
        mp.create_visualization()
    )  # Calls the create_visualization function from map_utils.py
    return fig


# Callback to update choices from map clicks
@app.callback(
    Output("departure", "children", allow_duplicate=True),
    Output("arrival", "children", allow_duplicate=True),
    Output("choices-text", "children", allow_duplicate=True),
    Input("map", "clickData"),
    State("choices-text", "children"),
    prevent_initial_call=True,
)
def update_choices(clickData, stored_choices):
    if clickData is None:
        return "Waiting for selection...", "Waiting for selection...", ""

    if "points" in clickData and len(clickData["points"]) > 0:
        stop_name = clickData["points"][0].get("hovertext", "Unknown Stop")

        # Load previous choices from State
        choices = stored_choices.split("\n") if stored_choices else []
        choices.append(stop_name)

        # Keep only the last two selections
        if len(choices) > 2:
            choices = choices[-2:]

        # Ensure all choices are strings
        choices = [str(choice) for choice in choices]

        depart = choices[0] if len(choices) > 0 else "Waiting for selection..."
        arrival = choices[1] if len(choices) > 1 else "Waiting for selection..."

        return depart, arrival, "\n".join(choices)

    return "Waiting for selection...", "Waiting for selection...", ""


# Callback to save choices to a file when clicking Submit
@app.callback(
    Output("submit-btn", "n_clicks"),
    Input("submit-btn", "n_clicks"),
    State("choices-text", "children"),
    prevent_initial_call=True,
)
def save_choices(n_clicks, choices):
    if choices:
        with open("choices.txt", "w") as f:
            f.write(choices + "\n")
        get_route()
    return n_clicks  # Dummy return to avoid duplicate Output error

def get_route():
    with open("choices.txt", "r") as f:
        choices = f.readlines()
        if len(choices) == 2:
            start = int(choices[0].strip())
            end = int(choices[1].strip())
            print(f"Start: {start}, End: {end}")
            mp.get_shortest_path(graph, start, end, stops_df, segments_df)


# Callback to clear selections
@app.callback(
    Output("departure", "children", allow_duplicate=True),
    Output("arrival", "children", allow_duplicate=True),
    Output("choices-text", "children", allow_duplicate=True),
    Input("clear-btn", "n_clicks"),
    prevent_initial_call=True
)
def clear_choices(n_clicks):
    with open("choices.txt", "w") as f:
        f.write("")  # Clear file
    return "Waiting for selection...", "Waiting for selection...", ""


if __name__ == "__main__":
    # Clear choices file on startup
    with open("choices.txt", "w") as f:
        f.write("")
    graph, stops_df, segments_df = mp.setup()

    app.run_server(debug=True)
