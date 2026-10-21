"""
06_dashboard.py — Dash application for visualizing flight delay patterns.

This is a standalone Dash app that reads data/flights_clean.csv (or
data/flights_with_weather.csv if present) and displays three sections:

1. Delay rate by departure hour (origin toggle: SFO, OAK, SJC)
2. Delay profile by destination (dropdown: top 20 destinations)
3. Overall delay distribution (histogram of ArrDelay)

Run with:  python 06_dashboard.py
"""

import os
import pandas as pd
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
DATA_PATH = "data/flights_with_weather.csv" if os.path.exists("data/flights_with_weather.csv") else "data/flights_clean.csv"
print(f"Loading data from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)

# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------
app = dash.Dash(__name__, title="Flight Delay Dashboard")

app.layout = html.Div(
    [
        html.H1("Flight Delay Dashboard – Bay Area Airports", style={"textAlign": "center", "marginBottom": "20px"}),

        # --- Section 1: Delay rate by departure hour, toggled by origin ---
        html.Div(
            [
                html.H3("Delay Rate by Departure Hour"),
                html.P(
                    "Toggle between SFO, OAK, and SJC to compare how delay risk varies "
                    "throughout the day.",
                    style={"textAlign": "center", "marginBottom": "10px"},
                ),
                dcc.RadioItems(
                    id="origin-toggle",
                    options=[
                        {"label": "SFO", "value": "SFO"},
                        {"label": "OAK", "value": "OAK"},
                        {"label": "SJC", "value": "SJC"},
                    ],
                    value="SFO",
                    inline=True,
                    style={"marginBottom": "20px", "justifyContent": "center"},
                ),
                dcc.Graph(id="hourly-delay-graph"),
            ],
            style={"marginBottom": "40px"},
        ),

        # --- Section 2: Delay profile by destination ---
        html.Div(
            [
                html.H3("Delay Profile by Destination"),
                html.P(
                    "Select a destination airport to see the delay rate distribution across "
                    "departure hours for flights going to that airport.",
                    style={"textAlign": "center", "marginBottom": "10px"},
                ),
                dcc.Dropdown(
                    id="dest-dropdown",
                    options=[
                        {"label": dest, "value": dest}
                        for dest in sorted(df["Dest"].unique())[:20]
                    ],
                    value="JFK",
                    clearable=False,
                    style={"width": "50%", "margin": "0 auto"},
                ),
                dcc.Graph(id="dest-delay-graph"),
            ],
            style={"marginBottom": "40px"},
        ),

        # --- Section 3: Overall delay distribution ---
        html.Div(
            [
                html.H3("Overall Delay Distribution"),
                html.P(
                    "Distribution of arrival delays (in minutes) across all flights. "
                    "ArrDel15 = 1 means the flight was delayed by 15+ minutes.",
                    style={"textAlign": "center", "marginBottom": "10px"},
                ),
                dcc.Graph(id="overall-dist-graph"),
            ],
            style={"marginBottom": "40px"},
        ),
    ],
    style={"maxWidth": "900", "margin": "0 auto", "padding": "20px"},
)


# ---------------------------------------------------------------------------
# Callback 1: Update hourly delay graph based on origin toggle
# ---------------------------------------------------------------------------
@app.callback(
    Output("hourly-delay-graph", "figure"),
    Input("origin-toggle", "value"),
)
def update_hourly_delay(origin):
    """Show delay rate by departure hour for the selected origin."""
    df_origin = df[df["Origin"] == origin]
    hourly = df_origin.groupby("dep_hour")["ArrDel15"].mean().reset_index()

    fig = px.line(
        hourly,
        x="dep_hour",
        y="ArrDel15",
        markers=True,
        title=f"Delay Rate by Departure Hour ({origin})",
        labels={"dep_hour": "Departure Hour (0-23)", "ArrDel15": "Delay Rate (15+ min)"},
    )

    fig.update_yaxes(tickformat=".0%")
    fig.update_xaxes(dtick=1)
    fig.add_hline(
        y=df["ArrDel15"].mean(),
        line_dash="dot",
        line_color="gray",
        annotation_text="Overall average",
        annotation_position="bottom right",
    )
    return fig


# ---------------------------------------------------------------------------
# Callback 2: Update destination delay profile based on dropdown selection
# ---------------------------------------------------------------------------
@app.callback(
    Output("dest-delay-graph", "figure"),
    Input("dest-dropdown", "value"),
)
def update_dest_delay(selected_dest):
    """Show delay rate by departure hour for flights to the selected destination."""
    df_dest = df[df["Dest"] == selected_dest]
    hourly = df_dest.groupby("dep_hour")["ArrDel15"].mean().reset_index()

    fig = px.line(
        hourly,
        x="dep_hour",
        y="ArrDel15",
        markers=True,
        title=f"Delay Rate by Departure Hour (Dest: {selected_dest})",
        labels={"dep_hour": "Departure Hour (0-23)", "ArrDel15": "Delay Rate (15+ min)"},
    )

    fig.update_yaxes(tickformat=".0%")
    fig.update_xaxes(dtick=1)
    overall_rate = df["ArrDel15"].mean()
    fig.add_hline(
        y=overall_rate,
        line_dash="dot",
        line_color="gray",
        annotation_text="Overall avg",
        annotation_position="bottom right",
    )
    return fig


# ---------------------------------------------------------------------------
# Callback 3: Update overall delay distribution (static, no inputs)
# ---------------------------------------------------------------------------
@app.callback(
    Output("overall-dist-graph", "figure"),
    []  # no inputs — static graph
)
def update_overall_dist():
    """Show the overall distribution of arrival delays."""
    fig = px.histogram(
        df,
        x="ArrDelay",
        nbins=30,
        title="Overall Arrival Delay Distribution",
        labels={"ArrDelay": "Arrival Delay (minutes)", "count": "Number of Flights"},
        marginal="rug",
    )

    # Add a vertical line at 0
    fig.add_vline(x=0, line_dash="dot", line_color="blue")

    # Add a vertical line at 15 (the 15-min threshold)
    fig.add_vline(
        x=15, line_dash="dash", line_color="red", annotation_text="15-min threshold"
    )

    return fig


# ---------------------------------------------------------------------------
# Validate interactivity: ensure all callbacks produce valid figures
# ---------------------------------------------------------------------------
def validate_app():
    """Run basic validation that all callbacks produce valid Plotly figures."""
    print("Validating dashboard callbacks...")

    # Test Callback 1: hourly delay graph
    try:
        fig1 = update_hourly_delay("SFO")
        assert isinstance(fig1, go.Figure), "Callback 1 did not return a go.Figure"
        print("  ✓ Callback 1 (hourly delay) OK")
    except Exception as e:
        print(f"  ✗ Callback 1 failed: {e}")

    # Test Callback 2: destination delay graph
    try:
        fig2 = update_dest_delay("JFK")
        assert isinstance(fig2, go.Figure), "Callback 2 did not return a go.Figure"
        print("  ✓ Callback 2 (dest delay) OK")
    except Exception as e:
        print(f"  ✗ Callback 2 failed: {e}")

    # Test Callback 3: overall distribution
    try:
        fig3 = update_overall_dist()
        assert isinstance(fig3, go.Figure), "Callback 3 did not return a go.Figure"
        print("  ✓ Callback 3 (overall dist) OK")
    except Exception as e:
        print(f"  ✗ Callback 3 failed: {e}")

    print("Validation complete.\n")


# Run validation on import
validate_app()

# ---------------------------------------------------------------------------
# Run the app
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)