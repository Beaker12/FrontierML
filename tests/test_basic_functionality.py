"""Basic tests for the FrontierML utility functions.

This module contains basic functionality tests to ensure all core components
are working correctly.

References:
    - pytest documentation: https://docs.pytest.org/
"""

import pytest
import numpy as np
import pandas as pd
import logging
from typing import Tuple

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_import_basic_libraries():
    """Test that all basic libraries can be imported."""
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import sklearn
        logger.info("All basic libraries imported successfully")
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import basic libraries: {e}")


def test_numpy_functionality():
    """Test basic numpy functionality."""
    arr = np.array([1, 2, 3, 4, 5])
    assert arr.mean() == 3.0
    assert arr.std() > 0


def test_pandas_functionality():
    """Test basic pandas functionality."""
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })
    assert len(df) == 3
    assert list(df.columns) == ['A', 'B']


def test_sklearn_basic_functionality():
    """Test basic sklearn functionality."""
    from sklearn.linear_model import LinearRegression
    from sklearn.datasets import make_regression
    
    X, y = make_regression(n_samples=100, n_features=1, noise=0.1)
    model = LinearRegression()
    model.fit(X, y)
    
    assert hasattr(model, 'coef_')
    assert hasattr(model, 'intercept_')


class TestDataUtils:
    """Test data utility functions."""
    
    def test_data_generation(self):
        """Test synthetic data generation."""
        # This would test our data_utils functions when they're imported
        # For now, just test that we can create basic synthetic data
        from sklearn.datasets import make_regression
        
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        assert X.shape == (100, 5)
        assert y.shape == (100,)


if __name__ == "__main__":
    pytest.main([__file__])
