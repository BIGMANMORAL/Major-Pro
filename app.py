from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# --- MODEL AND ENCODER LOADING ---
# Load the trained model and the label encoders from the 'models' folder
try:
    model_path = os.path.join('models', r'C:\Users\aruna\OneDrive\Desktop\Major-Pro\Notebooks\models\lgbm_model.pkl')
    encoders_path = os.path.join('models', r'C:\Users\aruna\OneDrive\Desktop\Major-Pro\Notebooks\models\label_encoders.pkl')
    
    model = joblib.load(model_path)
    encoders = joblib.load(encoders_path)
    print("Model and encoders loaded successfully!")

    # CORRECTED: Added 'age_bucket' to the list to match the 19 features the model was trained on.
    TRAINING_COLUMNS = [
        'team', 'position', 'height', 'age', 'appearance', 'goals', 'assists', 
        'yellow cards', 'second yellow cards', 'red cards', 'goals conceded', 
        'clean sheets', 'minutes played', 'days_injured', 'games_injured', 
        'award', 'position_encoded', 'winger', 'age_bucket'
    ]


except FileNotFoundError:
    print("Error: Model or encoder file not found in the 'models' directory.")
    model = None
    encoders = None
except Exception as e:
    print(f"An error occurred while loading files: {e}")
    model = None
    encoders = None


# --- DATA PREPROCESSING FUNCTION ---
def preprocess_input(data):
    """
    Prepares the user input from the form for the model.
    This function MUST replicate the preprocessing from your training script.
    """
    
    # 1. Create a DataFrame from the input dictionary
    df = pd.DataFrame([data])
    
    # 2. Convert data types from form to numeric
    numeric_cols = {
        'height': 0, 'age': 0, 'appearance': 0, 'goals': 0, 'assists': 0,
        'yellow_cards': 0, 'red_cards': 0, 'goals_conceded': 0, 'clean_sheets': 0,
        'minutes_played': 0, 'days_injured': 0, 'award': 0
    }
    for col, default in numeric_cols.items():
        df[col] = pd.to_numeric(df.get(col, default))

    # Rename columns to match training data
    df.rename(columns={
        'yellow_cards': 'yellow cards',
        'red_cards': 'red cards',
        'goals_conceded': 'goals conceded',
        'clean_sheets': 'clean sheets',
        'minutes_played': 'minutes played'
    }, inplace=True)

    # 3. Add placeholders for features not on the form
    df['team'] = 'Unknown'
    df['second yellow cards'] = 0
    df['games_injured'] = 0 # Form provides days_injured, so we default games_injured to 0

    # 4. --- Feature Engineering ---
    # Replicate 'position_encoded' and 'winger' logic
    position = df['position'].iloc[0]
    if position == 'Goalkeeper':
        df['position_encoded'] = 1
        df['winger'] = 0
    elif position == 'Defender':
        df['position_encoded'] = 2
        df['winger'] = 0
    elif position == 'Midfield':
        df['position_encoded'] = 3
        df['winger'] = 0
    elif position == 'Attack':
        df['position_encoded'] = 4
        df['winger'] = 1 # Assume an attacker can be a winger
    else:
        df['position_encoded'] = 0 # Default for unknown
        df['winger'] = 0

    # ADDED: Create the 'age_bucket' feature exactly as in the training notebook. This was the missing step.
    df["age_bucket"] = pd.cut(df["age"], bins=[0, 20, 25, 30, 35, 100],
                            labels=["<20", "20-25", "25-30", "30-35", "35+"], right=False)

    # 5. --- Encoding ---
    # Apply the loaded LabelEncoders for string-based categorical columns
    for col, encoder in encoders.items():
        # Check if the column exists in our dataframe before trying to encode it
        if col in df.columns:
            try:
                # Ensure the column is of type string for the encoder
                df[col] = encoder.transform(df[col].astype(str))
            except ValueError:
                print(f"Warning: Unseen label for column {col}: {df[col].iloc[0]}. Assigning -1.")
                df[col] = -1 # Use -1 for unknown categories
            
    # 6. --- Feature Ordering ---
    # Ensure the column order is IDENTICAL to the training data
    df = df.reindex(columns=TRAINING_COLUMNS)
    
    print(f"Number of features being sent to model: {len(df.columns)}")
    print("Processed DataFrame columns:", df.columns.tolist())
    print("Data going into model:\n", df.head())
    
    return df


# --- FLASK ROUTES ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or encoders is None:
        return jsonify({'error': 'Model or encoders not loaded. Check server logs.'}), 500

    try:
        data = request.get_json(force=True)
        print(f"Received data: {data}")

        processed_data = preprocess_input(data)

        # Make prediction
        prediction_log = model.predict(processed_data)[0]
        
        # --- Inverse Transform ---
        # The model predicts the log value, so we convert it back to the original scale
        prediction_eur = np.expm1(prediction_log)

        # Ensure the value is not negative
        if prediction_eur < 0:
            prediction_eur = 0

        return jsonify({'predicted_value': prediction_eur})

    except Exception as e:
        # Provide a more specific error message back to the front-end if possible
        print(f"An error occurred during prediction: {e}")
        return jsonify({'error': f'An error occurred on the server: {e}'}), 500

if __name__ == '__main__':
    app.run(debug=True)

