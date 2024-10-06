# tests/test_resampling.py

import unittest
import numpy as np
from src.resampling.smote import apply_smote
from src.resampling.adasyn import apply_adasyn
from src.resampling.ros import apply_ros
from src.resampling.original_imbalanced import use_original_imbalanced

class TestResamplingTechniques(unittest.TestCase):
    def setUp(self):
        # Generate imbalanced dummy data
        self.X = np.random.rand(100, 20)
        self.y = np.array([0]*90 + [1]*10)  # 90% class 0, 10% class 1

    def test_apply_smote(self):
        X_res, y_res = apply_smote(self.X, self.y)
        self.assertEqual(sum(y_res == 0), sum(y_res == 1))

    def test_apply_adasyn(self):
        X_res, y_res = apply_adasyn(self.X, self.y)
        self.assertEqual(sum(y_res == 0), sum(y_res == 1))

    def test_apply_ros(self):
        X_res, y_res = apply_ros(self.X, self.y)
        self.assertEqual(sum(y_res == 0), sum(y_res == 1))

    def test_use_original_imbalanced(self):
        X_res, y_res = use_original_imbalanced(self.X, self.y)
        self.assertEqual(len(y_res), len(self.y))

if __name__ == '__main__':
    unittest.main()
