import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim

def train_models(X, y):
    """
    Trains multiple models and selects the best one.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {}
    accuracies = {}

    # Random Forest (scikit-learn)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_preds)
    models['RandomForest'] = rf_model
    accuracies['RandomForest'] = rf_accuracy

    # Neural Network (TensorFlow)
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    tf_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    tf_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    tf_loss, tf_accuracy = tf_model.evaluate(X_test, y_test, verbose=0)
    models['TensorFlow'] = tf_model
    accuracies['TensorFlow'] = tf_accuracy

    # Neural Network (PyTorch)
    class PyTorchNN(nn.Module):
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

    input_size = X_train.shape[1]
    pt_model = PyTorchNN(input_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(pt_model.parameters(), lr=0.001)
    X_train_pt = torch.from_numpy(X_train.values).float()
    y_train_pt = torch.from_numpy(y_train.values).float().unsqueeze(1)
    X_test_pt = torch.from_numpy(X_test.values).float()
    y_test_pt = torch.from_numpy(y_test.values).float().unsqueeze(1)

    for epoch in range(10):
        pt_model.train()
        optimizer.zero_grad()
        outputs = pt_model(X_train_pt)
        loss = criterion(outputs, y_train_pt)
        loss.backward()
        optimizer.step()

    pt_model.eval()
    with torch.no_grad():
        outputs = pt_model(X_test_pt)
        predicted = (outputs > 0.5).float()
        pt_accuracy = (predicted.eq(y_test_pt).sum() / y_test_pt.shape[0]).item()
    models['PyTorch'] = pt_model
    accuracies['PyTorch'] = pt_accuracy

    # Select the best model
    best_model_name = max(accuracies, key=accuracies.get)
    best_model = models[best_model_name]
    best_accuracy = accuracies[best_model_name]

    # Save the best model
    os.makedirs('models', exist_ok=True)
    if best_model_name == 'RandomForest':
        joblib.dump(best_model, 'models/best_model.pkl')
    elif best_model_name == 'TensorFlow':
        best_model.save('models/best_model.h5')
    else:
        torch.save(best_model.state_dict(), 'models/best_model.pth')

    print(f"Best Model: {best_model_name} with accuracy {best_accuracy}")
    return best_model_name, best_accuracy
