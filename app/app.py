import numpy as np
import pandas as pd
import joblib
import pickle
import os
import sys
from pathlib import Path
from dash import Dash, html, dcc, Input, Output, State

import models
from models import Normal, LinearRegression, NoPenalty, LogisticRegression

# The custom models were pickled from notebooks where these classes lived in
# __main__, so we expose them there before unpickling.
sys.modules['__main__'].Normal = Normal
sys.modules['__main__'].LinearRegression = LinearRegression
sys.modules['__main__'].NoPenalty = NoPenalty
sys.modules['__main__'].LogisticRegression = LogisticRegression

# --------------------------------------------------
# 1. Load trained models & artifacts
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

# --- Model A1 (Scikit-Learn Pipeline, regression) ---
MODEL_A1_PATH = BASE_DIR / "car_price_model.joblib"
model_a1 = joblib.load(MODEL_A1_PATH)

# Extract categories from the A1 OneHotEncoder to fill the dropdowns
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

# --- Model A2 (Custom NumPy Linear Regression, regression) ---
PREPROCESSOR_A2_PATH = BASE_DIR / "preprocessor.pkl"
MODEL_A2_PATH = BASE_DIR / "model_a2.pkl"

with open(PREPROCESSOR_A2_PATH, "rb") as f:
    preprocessor_a2 = pickle.load(f)

with open(MODEL_A2_PATH, "rb") as f:
    model_a2 = pickle.load(f)

# --- Model A3 (Custom NumPy Logistic Regression + Ridge, 4-class classification) ---
PREPROCESSOR_A3_PATH = BASE_DIR / "preprocessor_a3.pkl"
MODEL_A3_PATH = BASE_DIR / "model_a3.pkl"
PRICE_BINS_A3_PATH = BASE_DIR / "price_bins_a3.pkl"

with open(PREPROCESSOR_A3_PATH, "rb") as f:
    preprocessor_a3 = pickle.load(f)

with open(MODEL_A3_PATH, "rb") as f:
    model_a3 = pickle.load(f)

# Price bin edges from pd.qcut (5 edges -> 4 classes). Used only to describe
# what each class means; if the file is missing the app still works and
# simply shows the class number.
try:
    with open(PRICE_BINS_A3_PATH, "rb") as f:
        price_bins_a3 = np.asarray(pickle.load(f), dtype=float)
except FileNotFoundError:
    price_bins_a3 = None


def class_price_range(c):
    """Human-readable price range for class c (0-3)."""
    if price_bins_a3 is None:
        return None
    e1, e2, e3 = price_bins_a3[1], price_bins_a3[2], price_bins_a3[3]
    ranges = [
        f"below ${e1:,.0f}",
        f"${e1:,.0f} - ${e2:,.0f}",
        f"${e2:,.0f} - ${e3:,.0f}",
        f"above ${e3:,.0f}",
    ]
    return ranges[c]


CLASS_NAMES = {0: "Low", 1: "Mid-low", 2: "Mid-high", 3: "High"}

# --------------------------------------------------
# 2. Create Dash app
# --------------------------------------------------
app = Dash(__name__)
app.title = "Car Price Prediction - A1 vs A2 vs A3"

# --------------------------------------------------
# 3. Layout
# --------------------------------------------------
app.layout = html.Div(
    [
        html.H1("Car Price Prediction App"),

        # Explanation of what is new in A3
        html.Div(
            [
                html.H4("💡 What's new in Model A3?"),
                html.P(
                    "Model A3 treats car price prediction as a classification problem. "
                    "The selling price is split into 4 equal-frequency classes (0 = lowest, 3 = highest) "
                    "and predicted by a custom-built Multinomial Logistic Regression (softmax) "
                    "trained with batch gradient descent and an optional Ridge (L2) penalty."
                ),
                html.P(
                    "Best run: batch, alpha = 0.001, Ridge lambda = 0.1 "
                    "(Test Accuracy = 0.6376, Macro F1 = 0.5780, Weighted F1 = 0.5778)."
                ),
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

        # Model selection
        html.Label("Choose Model Version:", style={"fontWeight": "bold"}),
        dcc.RadioItems(
            id="model-version",
            options=[
                {"label": " Model A1 (Scikit-Learn regression)", "value": "a1"},
                {"label": " Model A2 (Custom NumPy SGD + Momentum regression)", "value": "a2"},
                {"label": " Model A3 (Custom Logistic Regression + Ridge, price class)", "value": "a3"},
            ],
            value="a3",  # A3 is the default
            labelStyle={"display": "block", "marginBottom": "5px"}
        ),

        html.Hr(),

        # Form inputs
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

        html.Div(id="prediction-output", style={"color": "#007bff"}),
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
        return html.H2("Enter the car information and click Predict Price.")

    # Validate inputs
    if None in [brand, year, km_driven, fuel, seller_type, transmission,
                owner, mileage, engine, max_power, seats]:
        return html.H2("⚠️ Please fill in all car specifications before predicting.")

    # Build the input row (same columns used during training)
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

    # ---------------- Model A1 ----------------
    if model_version == "a1":
        predicted_log = model_a1.predict(input_data)
        predicted_price = np.exp(predicted_log[0])
        return html.H2(f"[Model A1 (Baseline)] Predicted Price: ${predicted_price:,.2f}")

    # ---------------- Model A2 ----------------
    if model_version == "a2":
        X_proc = preprocessor_a2.transform(input_data)
        # Add bias column (X_0 = 1) at index 0
        X_final = np.hstack([np.ones((X_proc.shape[0], 1)), X_proc])

        predicted_log = model_a2.predict(X_final)
        predicted_price = np.exp(predicted_log[0])
        return html.H2(f"[Model A2 (Custom SGD + Momentum)] Predicted Price: ${predicted_price:,.2f}")

    # ---------------- Model A3 ----------------
    X_proc = preprocessor_a3.transform(input_data)
    # Add bias column (X_0 = 1) at index 0, same as in training
    X_final = np.hstack([np.ones((X_proc.shape[0], 1)), X_proc])

    pred_class = int(model_a3.predict(X_final)[0])
    probs = model_a3.h_theta(X_final, model_a3.W)[0]  # softmax probabilities, shape (4,)

    price_range = class_price_range(pred_class)
    headline = f"[Model A3 (Logistic Regression + Ridge)] Predicted Price Class: {pred_class} ({CLASS_NAMES[pred_class]})"

    children = [html.H2(headline)]
    if price_range is not None:
        children.append(html.H3(f"Estimated price range: {price_range}"))

    children.append(html.P("Class probabilities:", style={"fontWeight": "bold", "marginBottom": "5px"}))
    children.append(
        html.Ul(
            [
                html.Li(
                    f"Class {c} ({CLASS_NAMES[c]}): {probs[c] * 100:.1f}%",
                    style={"fontWeight": "bold" if c == pred_class else "normal"},
                )
                for c in range(len(probs))
            ]
        )
    )
    return children


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