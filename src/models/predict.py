import joblib
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
import numpy as np

def load_best_model(model_name):
    """
    Loads the best performing model.
    """
    if model_name == 'RandomForest':
        model = joblib.load('models/best_model.pkl')
    elif model_name == 'TensorFlow':
        model = tf.keras.models.load_model('models/best_model.h5')
    else:
        class PyTorchNN(nn.Module):
            # Same model class definition as before
            def __init__(self, input_size):
                super(PyTorchNN, self).__init__()
                self.fc1 = nn.Linear(input_size, 16)
                self.fc2 = nn.Linear(16, 8)
                self.fc3 = nn.Linear(8, 1)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = self.sigmoid(self.fc3(x))
                return x

        input_size = 10  # Adjust based on feature count
        model = PyTorchNN(input_size)
        model.load_state_dict(torch.load('models/best_model.pth'))
        model.eval()
    return model

def make_prediction(model, model_name, input_data):
    """
    Makes a prediction using the best model.
    """
    if model_name == 'RandomForest':
        prediction = model.predict(input_data)
    elif model_name == 'TensorFlow':
        prediction = (model.predict(input_data) > 0.5).astype(int)
    else:
        input_tensor = torch.from_numpy(input_data.values).float()
        with torch.no_grad():
            outputs = model(input_tensor)
            prediction = (outputs > 0.5).float().numpy()
    return prediction
