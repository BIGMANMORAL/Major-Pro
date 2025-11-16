from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__, template_folder='.')

# --- MODEL & METADATA LOADING ---

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

model_path = os.path.join(MODEL_DIR, "champion_model.joblib")
cols_path = os.path.join(MODEL_DIR, "training_columns.joblib")

try:
    model = joblib.load(model_path)
    TRAINING_COLUMNS = joblib.load(cols_path)
    print("✅ Model and training columns loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model or training columns: {e}")
    model = None
    TRAINING_COLUMNS = None


# --- PREPROCESSING FUNCTION ---

def preprocess_input(data: dict) -> pd.DataFrame:
    """
    Prepare a single player input (from the frontend) so it matches
    the Random Forest training data structure (X_encoded).
    """

    # Create DataFrame from incoming JSON
    df = pd.DataFrame([data])

    # 1. Ensure numeric fields are numeric
    numeric_cols = [
        "age",
        "height",
        "appearance",
        "minutes_played",
        "award",
        "goals",
        "assists",
        "yellow_cards",
        "red_cards",
        "goals_conceded",
        "clean_sheets",
        "days_injured",
        # if your dataset had these, we'll fill them later if needed:
        "second_yellow_cards",
        "games_injured",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            # Create missing ones with 0 for safety
            df[col] = 0

    # 2. Rename columns to match the original DataFrame columns before get_dummies
    # (these names are what likely exist in cleaned_final_data.csv)
    df.rename(
        columns={
            "yellow_cards": "yellow cards",
            "red_cards": "red cards",
            "goals_conceded": "goals conceded",
            "clean_sheets": "clean sheets",
            "minutes_played": "minutes played",
            "second_yellow_cards": "second yellow cards",
            "days_injured": "days_injured",  # keep same if that's the column name
        },
        inplace=True,
    )

    # 3. Add categorical columns expected by training
    # Index.html doesn't ask for 'team', so we default to 'Unknown'
    if "team" not in df.columns:
        df["team"] = "Unknown"

    # Ensure 'position' exists (it comes from the form dropdown)
    if "position" not in df.columns:
        raise ValueError("Missing 'position' in input data")

    # 4. Apply one-hot encoding in the SAME way as training
    # Training did: pd.get_dummies(X, columns=['team', 'position'])
    categorical_cols = []
    for c in ["team", "position"]:
        if c in df.columns:
            categorical_cols.append(c)

    df_encoded = pd.get_dummies(df, columns=categorical_cols)

    # 5. Align with TRAINING_COLUMNS:
    #    - Add any missing training columns (fill with 0)
    #    - Drop any extra columns not seen during training
    for col in TRAINING_COLUMNS:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Keep only columns the model was trained on, in the correct order
    df_encoded = df_encoded[TRAINING_COLUMNS]

    print("Processed feature columns:", df_encoded.columns.tolist())
    print("Processed row:\n", df_encoded.head())

    return df_encoded


# --- ROUTES ---

@app.route("/")
def home():
    # index.html is already in the same folder; template_folder='.' handles it
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or TRAINING_COLUMNS is None:
        return (
            jsonify(
                {"error": "Model or training columns not loaded on the server."}
            ),
            500,
        )

    try:
        data = request.get_json(force=True)
        print("🔹 Raw incoming data:", data)

        processed = preprocess_input(data)

        # RandomForest was trained directly on 'current_value' (no log transform)
        prediction = model.predict(processed)[0]

        # Sanity: no negative values
        prediction = float(prediction)
        if prediction < 0:
            prediction = 0.0

        return jsonify({"predicted_value": prediction})

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return jsonify({"error": f"Server error during prediction: {e}"}), 500


if __name__ == "__main__":
    # In production use: app.run(host="0.0.0.0", port=8000)
    app.run(debug=True)
