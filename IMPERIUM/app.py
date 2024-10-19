import sqlite3
import pandas as pd
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained machine learning model
model = joblib.load('model.pkl')

def get_location_data(location):
    # Connect to the SQLite database
    conn = sqlite3.connect(r'C:\Users\ruthv\OneDrive\Desktop\trial\B_dataset.db')  # Update with your database path
    cursor = conn.cursor()

    # Fetch average speed and congestion based on the provided location
    cursor.execute("SELECT average_speed, congestion FROM data_table WHERE junction_name = ?", (location,))
    result = cursor.fetchone()
    conn.close()

    if result:
        return result  # returns (average_speed, congestion)
    else:
        return None  # Location not found

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    location = data.get('location')
    print(f"Received location: {location}")

    # Fetch average speed and congestion from the database
    location_data = get_location_data(location)
    if location_data is None:
        return jsonify({"error": "Location data not found."}), 404

    average_speed, congestion = location_data

    # Prepare the features for prediction
    # Create a feature array with required inputs
    features = [average_speed, Road1, Road2, Road3, Road4, congestion]  # Add any other features as required for your model

    # Make prediction
    predicted_cycle_time = model.predict([features])[0]

    response = {
        "message": f"Prediction data for {location} is ready.",
        "location": location,
        "predicted_cycle_time": predicted_cycle_time
    }

    return jsonify(response), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
