# tests/test_evaluation.py

import unittest
import numpy as np
from src.evaluation import evaluate_model, plot_confusion_matrix, plot_roc_curve
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for testing
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class TestEvaluation(unittest.TestCase):
    def setUp(self):
        # Create a simple model
        self.model = Sequential([
            Dense(1, activation='sigmoid', input_shape=(70,))
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        # Generate dummy data
        self.X_test = np.random.rand(10, 70)
        self.y_test = np.random.randint(0, 2, 10)
        # Train the model briefly
        self.model.fit(self.X_test, self.y_test, epochs=1, verbose=0)

    def test_evaluate_model(self):
        accuracy, precision, recall, f1 = evaluate_model(self.model, self.X_test, self.y_test)
        # Check that metrics are floats and within expected ranges
        self.assertIsInstance(accuracy, float)
        self.assertIsInstance(precision, float)
        self.assertIsInstance(recall, float)
        self.assertIsInstance(f1, float)
        self.assertTrue(0 <= accuracy <= 1)
        self.assertTrue(0 <= precision <= 1)
        self.assertTrue(0 <= recall <= 1)
        self.assertTrue(0 <= f1 <= 1)

    def test_plot_confusion_matrix(self):
        y_pred = (self.model.predict(self.X_test) > 0.5).astype("int32")
        # Ensure no exceptions are raised during plotting
        try:
            plot_confusion_matrix(self.y_test, y_pred)
            plt.close()
        except Exception as e:
            self.fail(f"plot_confusion_matrix raised an exception: {e}")

    def test_plot_roc_curve(self):
        y_pred_prob = self.model.predict(self.X_test)
        # Ensure no exceptions are raised during plotting
        try:
            plot_roc_curve(self.y_test, y_pred_prob)
            plt.close()
        except Exception as e:
            self.fail(f"plot_roc_curve raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()
