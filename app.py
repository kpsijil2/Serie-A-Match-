import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model_1.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('match.html')

@app.route('/predict', methods=['POST'])
def predict():

    int_features = [int(X) for X in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('match.html', prediction_text= "The Match Winner: {}".format(output))

@app.route('/predict_api', methods=['POST'])
def predict_api():
    match_df = request.get_json(force=True)
    prediction = model.predict([np.array(list(match_df.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
