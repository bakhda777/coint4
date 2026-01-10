import unittest
import numpy as np
import pandas as pd
from src.coint2.core.math_utils import safe_div, rolling_zscore

class TestWFMechanics(unittest.TestCase):
    def test_safe_div(self):
        """Test safe_div utility."""
        self.assertEqual(safe_div(10, 2), 5.0)
        self.assertTrue(np.isnan(safe_div(10, 0)))
        self.assertEqual(safe_div(10, 0, default=0.0), 0.0)
        self.assertTrue(np.isnan(safe_div(10, float('nan'))))
        self.assertTrue(np.isnan(safe_div(10, None)))

    def test_rolling_zscore(self):
        """Test rolling_zscore normalization."""
        # Case 1: Normal data
        data = pd.Series([1, 2, 3, 4, 5])
        z = rolling_zscore(data, window=3)
        # Window 1: [1,2,3], mean=2, std=1. (3-2)/1 = 1.0
        self.assertAlmostEqual(z.iloc[2], 1.0)
        
        # Case 2: Constant data (std=0)
        data_const = pd.Series([1, 1, 1, 1, 1])
        z_const = rolling_zscore(data_const, window=3)
        # Std is 0, replaced by epsilon. (1-1)/eps = 0
        self.assertEqual(z_const.iloc[2], 0.0)
        
        # Case 3: NaN handling
        data_nan = pd.Series([1, np.nan, 1, 1])
        z_nan = rolling_zscore(data_nan, window=2)
        # Should not crash
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
