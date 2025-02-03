import unittest
import pandas as pd
from src.features.preprocess_data import preprocess_data
from src.models.train_models import train_models
from src.data.generate_synthetic_data import generate_synthetic_data


class TestModelTraining(unittest.TestCase):
    def test_train_models(self):
        # Generate small dataset for testing
        data = generate_synthetic_data(500)
        X, y = preprocess_data(data)
        best_model_name, best_accuracy = train_models(X, y)
        self.assertTrue(best_accuracy > 0)

if __name__ == '__main__':
    unittest.main()
