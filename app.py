from flask import Flask, request, jsonify
import pickle

data = pickle.load(open("model.pkl", "rb"))
model = data["model"]
class_names = data["class_names"]

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    features = request.json['features']
    prediction = model.predict([features])[0]
    return jsonify({'prediction': class_names[prediction]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
