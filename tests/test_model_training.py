# tests/test_model_training.py

import unittest
import numpy as np
from src.model_training import (
    train_gru_attention,
    train_bilstm_attention,
    train_bilstm_dense,
    train_lstm_cnn,
)
from src.model_architectures import (
    build_gru_attention,
    build_bilstm_attention,
    build_bilstm_dense,
    build_lstm_cnn,
)

class TestModelTraining(unittest.TestCase):
    def setUp(self):
        # Create a dummy embedding matrix
        self.embedding_matrix = np.random.rand(1000, 300)
        # Dummy data: 10 samples, sequence length 70
        self.X_train = [np.random.randint(0, 1000, (10, 70)), np.random.randint(0, 1000, (10, 70))]
        self.X_test = [np.random.randint(0, 1000, (5, 70)), np.random.randint(0, 1000, (5, 70))]
        self.y_train = np.random.randint(0, 2, (10,))
        self.y_test = np.random.randint(0, 2, (5,))

    def test_train_gru_attention(self):
        model = build_gru_attention(self.embedding_matrix)
        history = model.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_test, self.y_test),
            epochs=1,
            batch_size=2,
            verbose=0
        )
        self.assertIn('accuracy', history.history)

    def test_train_bilstm_attention(self):
        model = build_bilstm_attention(self.embedding_matrix)
        history = model.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_test, self.y_test),
            epochs=1,
            batch_size=2,
            verbose=0
        )
        self.assertIn('accuracy', history.history)

    def test_train_bilstm_dense(self):
        model = build_bilstm_dense(self.embedding_matrix)
        history = model.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_test, self.y_test),
            epochs=1,
            batch_size=2,
            verbose=0
        )
        self.assertIn('accuracy', history.history)

    def test_train_lstm_cnn(self):
        model = build_lstm_cnn(self.embedding_matrix)
        history = model.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_test, self.y_test),
            epochs=1,
            batch_size=2,
            verbose=0
        )
        self.assertIn('accuracy', history.history)

if __name__ == '__main__':
    unittest.main()
