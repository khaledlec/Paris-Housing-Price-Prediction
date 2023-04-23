import json
import pickle
import pandas as pd

from flask import Flask, request, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

# Load the encoder
with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Read form data into a Pandas dataframe
    data = pd.json_normalize(request.form)

    # Convert 'Voie' column to lowercase
    data['Voie'] = data['Voie'].str.lower()

    # Encode 'Voie' column
    data['Voie'] = encoder.transform(data[['Voie']])

    # Make the prediction
    prediction = model.predict(data)[0]

    return render_template("home.html", prediction_text="The house price prediction is {}".format(prediction))

if __name__ == "__main__":
    app.run(debug=True)