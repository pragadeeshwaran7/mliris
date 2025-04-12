from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import numpy as np

app = Flask(__name__)
iris = load_iris()
model = RandomForestClassifier()
model.fit(iris.data, iris.target)

@app.route('/')
def home():
    return "Iris Classifier API"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    if not all(key in data for key in ["sepal_length", "sepal_width", "petal_length", "petal_width"]):
        return jsonify({'error': 'Missing features'}), 400

    features = [
        data["sepal_length"],
        data["sepal_width"],
        data["petal_length"],
        data["petal_width"]
    ]

    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    label = iris.target_names[prediction[0]]

    return jsonify({'prediction': label})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
