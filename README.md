
# Iris Classifier API

This project provides a Flask-based API for predicting the Iris flower species based on input features such as sepal length, sepal width, petal length, and petal width. It uses a Random Forest Classifier trained on the Iris dataset.

## Features

- **Flask Web API**: Exposes an endpoint to make predictions based on the input features.
- **Dockerized**: The application is containerized using Docker, making it easy to deploy and run in any environment.

## Requirements

### Software Dependencies

- **Docker**: To run the application in a container.
- **Flask**: A Python web framework used to build the API.
- **Scikit-learn**: A machine learning library to train the Random Forest model.
- **Numpy**: For numerical computations.
  
You can install the dependencies using the following command:

```bash
pip install flask scikit-learn numpy
```

### Docker (Optional)

If you want to use Docker, make sure Docker is installed and running on your system.

- **Docker Documentation**: [https://docs.docker.com/get-docker/](https://docs.docker.com/get-docker/)

## Setting Up the Project

### Step 1: Clone the Repository

Clone this repository to your local machine.

```bash
git clone <repository_url>
cd <repository_name>
```

### Step 2: Build the Docker Image

If you're using Docker, navigate to the project directory and build the Docker image.

```bash
docker build -t iris-label-predictor .
```

This command will build the Docker image using the `Dockerfile` in the project directory.

### Step 3: Run the Application Locally

If you're using Docker, you can run the containerized application with the following command:

```bash
docker run -p 5000:5000 iris-label-predictor
```

If you're not using Docker, you can run the Flask app directly:

```bash
python app.py
```

The Flask app will be accessible on `http://127.0.0.1:5000`.

### Step 4: Test the API

Once the application is running, you can test the API by sending a `POST` request to the `/predict` endpoint.

You can use `curl` or any HTTP client (like Postman or Insomnia) to send the request.

#### Example Request:

```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

#### Example Response:

```json
{
    "prediction": "setosa"
}
```

This will return the predicted Iris flower species based on the features you provide.

### Step 5: Handle Errors

If there are any issues with the request, the API will return an error message in the response. For example, if you don't provide all the necessary features:

```json
{
    "error": "Missing features"
}
```

## Code Overview

### `app.py`

This file contains the Flask app that exposes the `/predict` endpoint. Here's a brief overview of its contents:

```python
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
```

- **Home Route (`/`)**: A simple route that returns a welcome message.
- **Predict Route (`/predict`)**: Accepts a `POST` request with the Iris flower features and returns a prediction of the flower species.
![Screenshot from 2025-04-12 23-27-35](https://github.com/user-attachments/assets/fbb558da-c7c9-476e-9a8d-3599b33dc78a)

### Dockerfile

This file describes how to containerize the Flask app with Docker. Here's the content:

```Dockerfile
# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "app.py"]
```
![Screenshot from 2025-04-13 00-09-48](https://github.com/user-attachments/assets/7ab17c1d-a16f-4845-96da-d2d97f9adb0f)

![Screenshot from 2025-04-13 00-13-43](https://github.com/user-attachments/assets/11215e3c-40a7-432f-b15c-f206acc0b549)


## Troubleshooting

- **500 Internal Server Error**: Check the Flask app logs for detailed error messages. You can view Docker logs by running `docker logs <container_id>`.
- **Missing Dependencies**: Ensure that you have all required dependencies installed (Flask, Scikit-learn, and Numpy). Run `pip install -r requirements.txt` if necessary.
- **Permission Issues**: If you're running Docker on Linux, make sure you're part of the Docker group or use `sudo` with Docker commands.

## Conclusion

Now you have a fully functional API for classifying Iris flowers using the Random Forest model. You can run the API both in Docker and locally, and make predictions using HTTP requests.

If you need to extend this app or modify it for a different classification problem, you can replace the Iris dataset with any other dataset and retrain the model.
