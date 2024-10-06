# tests/test_data_preprocessing.py

import unittest
import pandas as pd
from src.data_preprocessing import (
    clean_text,
    preprocess_text_series,
    tokenize_texts,
    pad_sequences_custom,
    preprocess_data,
)
import numpy as np

class TestDataPreprocessing(unittest.TestCase):
    def test_clean_text(self):
        text = "This is a sample sentence! With punctuation."
        cleaned = clean_text(text)
        # Expected output after cleaning
        expected = "sampl sentenc punctuat"
        self.assertEqual(cleaned, expected)

    def test_preprocess_text_series(self):
        series = pd.Series(["Hello World!", "Testing, one two three.", None])
        processed_series = preprocess_text_series(series)
        expected_series = pd.Series(["hello world", "test one two three", ""])
        pd.testing.assert_series_equal(processed_series, expected_series)

    def test_tokenize_texts(self):
        texts = ["hello world", "test one two three"]
        sequences, word_index, tokenizer = tokenize_texts(texts, max_num_words=5000)
        self.assertIsInstance(sequences, list)
        self.assertIsInstance(word_index, dict)
        self.assertTrue(len(word_index) <= 5000)

    def test_pad_sequences_custom(self):
        sequences = [[1, 2, 3], [4, 5]]
        padded = pad_sequences_custom(sequences, maxlen=5)
        expected = np.array([
            [1, 2, 3, 0, 0],
            [4, 5, 0, 0, 0]
        ])
        np.testing.assert_array_equal(padded, expected)

    def test_preprocess_data(self):
        df = pd.DataFrame({
            'question1': ["How are you?", "What is your name?"],
            'question2': ["I am fine.", "My name is ChatGPT."]
        })
        text_columns = ['question1', 'question2']
        X_q1, X_q2, word_index, tokenizer = preprocess_data(
            df, text_columns, max_num_words=5000, max_sequence_length=10
        )
        self.assertEqual(X_q1.shape, (2, 10))
        self.assertEqual(X_q2.shape, (2, 10))
        self.assertIsInstance(word_index, dict)
        self.assertIsNotNone(tokenizer)

if __name__ == '__main__':
    unittest.main()
