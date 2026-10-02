import numpy as np
import pandas as pd
import joblib
import pickle
import os
from pathlib import Path
from dash import Dash, html, dcc, Input, Output, State
import models
from models import Normal, LinearRegression, NoPenalty
import sys
sys.modules['__main__'].Normal = Normal
sys.modules['__main__'].LinearRegression = LinearRegression
sys.modules['__main__'].NoPenalty = NoPenalty
# --------------------------------------------------
# 1. Load trained models & artifacts
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

# --- Model A1 (Scikit-Learn Pipeline) ---
MODEL_A1_PATH = BASE_DIR / "car_price_model.joblib"
model_a1 = joblib.load(MODEL_A1_PATH)

# Extract categories from A1 OneHotEncoder
encoder = (
    model_a1
    .named_steps["preprocessor"]
    .named_transformers_["cat"]
    .named_steps["encoder"]
)

brands = encoder.categories_[0].tolist()
fuels = encoder.categories_[1].tolist()
seller_types = encoder.categories_[2].tolist()
transmissions = encoder.categories_[3].tolist()

# --- Model A2 (Custom NumPy LinearRegression) ---
PREPROCESSOR_A2_PATH = BASE_DIR / "preprocessor.pkl"
MODEL_A2_PATH = BASE_DIR / "model_a2.pkl"

with open(PREPROCESSOR_A2_PATH, "rb") as f:
    preprocessor_a2 = pickle.load(f)

with open(MODEL_A2_PATH, "rb") as f:
    model_a2 = pickle.load(f)

# --------------------------------------------------
# 2. Create Dash app
# --------------------------------------------------
app = Dash(__name__)
app.title = "Car Price Prediction - A1 vs A2"

# --------------------------------------------------
# 3. Layout
# --------------------------------------------------
app.layout = html.Div(
    [
        html.H1("Car Price Prediction App"),
        
        # Section giải thích sự cải tiến của A2 theo đề bài
        html.Div(
            [
                html.H4("💡 What's new in Model A2?"),
                html.P(
                    "Model A2 uses a custom-built Linear Regression engine powered by Stochastic Gradient Descent (SGD) with Polyak Momentum. "
                    "It achieves higher generalization performance (CV R² ≈ 0.9028, Test R² = 0.8635) compared to the standard baseline model."
                )
            ],
            style={
                "backgroundColor": "#eef6ff",
                "padding": "15px",
                "borderRadius": "8px",
                "borderLeft": "5px solid #007bff",
                "marginBottom": "20px"
            }
        ),

        html.P("Select a model version and enter the car specifications below:"),

        # Selection cho Model A1 hay A2
        html.Label("Choose Model Version:", style={"fontWeight": "bold"}),
        dcc.RadioItems(
            id="model-version",
            options=[
                {"label": " Old Model (Assignment 1 - Scikit-Learn)", "value": "a1"},
                {"label": " New Model (Assignment 2 - Custom NumPy SGD + Momentum)", "value": "a2"},
            ],
            value="a2",  # Mặc định chọn A2
            labelStyle={"display": "block", "marginBottom": "5px"}
        ),

        html.Hr(),

        # Form Inputs
        html.Label("Brand"),
        dcc.Dropdown(
            id="brand",
            options=[{"label": b, "value": b} for b in brands],
            placeholder="Select brand",
        ),
        html.Br(),

        html.Label("Year"),
        dcc.Input(id="year", type="number", placeholder="e.g. 2018"),
        html.Br(), html.Br(),

        html.Label("Kilometers Driven"),
        dcc.Input(id="km_driven", type="number", placeholder="e.g. 50000"),
        html.Br(), html.Br(),

        html.Label("Fuel"),
        dcc.Dropdown(
            id="fuel",
            options=[{"label": f, "value": f} for f in fuels],
            placeholder="Select fuel",
        ),
        html.Br(),

        html.Label("Seller Type"),
        dcc.Dropdown(
            id="seller_type",
            options=[{"label": s, "value": s} for s in seller_types],
            placeholder="Select seller type",
        ),
        html.Br(),

        html.Label("Transmission"),
        dcc.Dropdown(
            id="transmission",
            options=[{"label": t, "value": t} for t in transmissions],
            placeholder="Select transmission",
        ),
        html.Br(),

        html.Label("Owner"),
        dcc.Input(id="owner", type="number", placeholder="e.g. 1"),
        html.Br(), html.Br(),

        html.Label("Mileage"),
        dcc.Input(id="mileage", type="number", placeholder="e.g. 20.5"),
        html.Br(), html.Br(),

        html.Label("Engine"),
        dcc.Input(id="engine", type="number", placeholder="e.g. 1498"),
        html.Br(), html.Br(),

        html.Label("Max Power"),
        dcc.Input(id="max_power", type="number", placeholder="e.g. 100"),
        html.Br(), html.Br(),

        html.Label("Seats"),
        dcc.Input(id="seats", type="number", placeholder="e.g. 5"),
        html.Br(), html.Br(),

        html.Button(
            "Predict Price",
            id="predict-button",
            n_clicks=0,
            style={
                "backgroundColor": "#28a745",
                "color": "white",
                "padding": "10px 20px",
                "border": "none",
                "borderRadius": "5px",
                "cursor": "pointer"
            }
        ),

        html.Br(), html.Br(),

        html.H2(id="prediction-output", style={"color": "#007bff"}),
    ],
    style={
        "maxWidth": "700px",
        "margin": "40px auto",
        "padding": "20px",
        "fontFamily": "Arial, sans-serif"
    },
)

# --------------------------------------------------
# 4. Prediction callback
# --------------------------------------------------
@app.callback(
    Output("prediction-output", "children"),
    Input("predict-button", "n_clicks"),
    State("model-version", "value"),
    State("brand", "value"),
    State("year", "value"),
    State("km_driven", "value"),
    State("fuel", "value"),
    State("seller_type", "value"),
    State("transmission", "value"),
    State("owner", "value"),
    State("mileage", "value"),
    State("engine", "value"),
    State("max_power", "value"),
    State("seats", "value"),
)
def predict_price(
    n_clicks,
    model_version,
    brand,
    year,
    km_driven,
    fuel,
    seller_type,
    transmission,
    owner,
    mileage,
    engine,
    max_power,
    seats,
):
    if n_clicks == 0:
        return "Enter the car information and click Predict Price."

    # Validate inputs
    if None in [brand, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]:
        return "⚠️ Please fill in all car specifications before predicting."

    # Prepare DataFrame
    input_data = pd.DataFrame([
        {
            "brand": brand,
            "year": float(year),
            "km_driven": float(km_driven),
            "fuel": fuel,
            "seller_type": seller_type,
            "transmission": transmission,
            "owner": float(owner),
            "mileage": float(mileage),
            "engine": float(engine),
            "max_power": float(max_power),
            "seats": float(seats),
        }
    ])

    # Perform prediction based on selected model
    if model_version == "a1":
        predicted_log = model_a1.predict(input_data)
        predicted_price = np.exp(predicted_log[0])
        model_name = "Model A1 (Baseline)"
    else:
        # Preprocess input using A2 preprocessor
        X_proc = preprocessor_a2.transform(input_data)
        # Add bias column (X_0 = 1) at index 0
        X_final = np.hstack([np.ones((X_proc.shape[0], 1)), X_proc])
        
        predicted_log = model_a2.predict(X_final)
        predicted_price = np.exp(predicted_log[0])
        model_name = "Model A2 (Custom SGD + Momentum)"

    return f"[{model_name}] Predicted Price: ${predicted_price:,.2f}"

# --------------------------------------------------
# 5. Run application
# --------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
    )