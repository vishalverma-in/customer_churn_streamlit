import unittest
from src.data.generate_synthetic_data import generate_synthetic_data

class TestDataGeneration(unittest.TestCase):
    def test_generate_synthetic_data(self):
        data = generate_synthetic_data(100)
        self.assertEqual(len(data), 100)
        self.assertIn('Exited', data.columns)

if __name__ == '__main__':
    unittest.main()
