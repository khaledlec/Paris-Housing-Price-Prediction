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
    if prediction:
    # Format prediction value to display only the real part of the number price
        #prediction_value = '{}'.format(prediction.real)

    # French translation of the prediction text
        prediction_text = "Après avoir examiné les critères fournis, nous avons estimé que le prix de cet appartement est d'environ {:.0f} €".format(prediction.real)
    else:
        # French translation of the error text
        prediction_text = "Oops, quelque chose s'est mal passé. Veuillez réessayer."

    # Render the template with the updated prediction text
    return render_template("home.html", prediction_text=prediction_text)


if __name__ == "__main__":
    app.run(debug=True)