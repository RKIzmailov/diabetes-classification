import pickle

from flask import Flask
from flask import request
from flask import jsonify

import pandas as pd


model_file = '../bins/model.bin'

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)
    print('The model loaded.')

app = Flask('diabete')

@app.route('/predict', methods=['POST'])
def predict():
    X_dict = request.get_json()
    X = pd.DataFrame([X_dict])

    y_pred = model.predict_proba(X)[0, 1]
    y_pred_binary = y_pred >= 0.5

    result = {
        'diabete_probability': float(y_pred),
        'diabete': bool(y_pred_binary)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)