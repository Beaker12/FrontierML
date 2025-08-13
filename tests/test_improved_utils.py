"""
Test suite for improved utility functions.

This module tests the enhanced functionality of the utils modules
to ensure they meet the COPILOT_RULES requirements.

References:
    - pytest documentation: https://docs.pytest.org/
    - unittest.mock documentation: https://docs.python.org/3/library/unittest.mock.html
"""

import pytest
import numpy as np
import pandas as pd
import logging
from unittest.mock import patch

# Import utility functions
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.data_utils import (
    load_sample_data, preprocess_data, generate_synthetic_data, 
    check_data_quality
)
from utils.evaluation_utils import evaluate_classification_model

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestDataUtils:
    """Test class for data utility functions."""
    
    def test_load_sample_data_valid_datasets(self):
        """Test loading valid sample datasets."""
        valid_datasets = ['iris', 'wine', 'digits']
        
        for dataset_name in valid_datasets:
            df = load_sample_data(dataset_name)
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            assert 'target' in df.columns
            logger.info(f"Successfully loaded {dataset_name} dataset: {df.shape}")
    
    def test_load_sample_data_invalid_dataset(self):
        """Test loading invalid dataset raises ValueError."""
        with pytest.raises(ValueError, match="Dataset 'invalid' not recognized"):
            load_sample_data('invalid')
    
    def test_preprocess_data_validation(self):
        """Test input validation in preprocess_data function."""
        # Create sample data
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10],
            'target': [0, 1, 0, 1, 0]
        })
        
        # Test invalid target column
        with pytest.raises(KeyError, match="Target column 'nonexistent' not found"):
            preprocess_data(df, target_column='nonexistent')
        
        # Test invalid test_size
        with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
            preprocess_data(df, target_column='target', test_size=1.5)
    
    def test_preprocess_data_functionality(self):
        """Test preprocess_data returns correct shapes and types."""
        # Create sample data
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })
        
        X_train, X_test, y_train, y_test = preprocess_data(
            df, target_column='target', test_size=0.2, random_state=42
        )
        
        # Check types
        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        assert isinstance(y_test, np.ndarray)
        
        # Check shapes
        assert X_train.shape[0] == 80  # 80% of 100
        assert X_test.shape[0] == 20   # 20% of 100
        assert X_train.shape[1] == 2   # 2 features
        assert X_test.shape[1] == 2    # 2 features
        
        logger.info(f"Preprocess test passed: train {X_train.shape}, test {X_test.shape}")
    
    def test_generate_synthetic_data_validation(self):
        """Test input validation in generate_synthetic_data."""
        # Test invalid parameters
        with pytest.raises(ValueError, match="n_samples must be positive"):
            generate_synthetic_data(n_samples=-1)
        
        with pytest.raises(ValueError, match="n_features must be positive"):
            generate_synthetic_data(n_features=0)
        
        with pytest.raises(ValueError, match="n_classes must be >= 2"):
            generate_synthetic_data(n_classes=1)
    
    def test_generate_synthetic_data_functionality(self):
        """Test generate_synthetic_data returns correct shapes."""
        X, y = generate_synthetic_data(n_samples=100, n_features=5, n_classes=3)
        
        assert X.shape == (100, 5)
        assert y.shape == (100,)
        assert len(np.unique(y)) == 3  # Should have 3 classes
        
        logger.info(f"Synthetic data generation test passed: X {X.shape}, y {y.shape}")
    
    def test_check_data_quality(self):
        """Test data quality check function."""
        # Create test data with various issues
        df = pd.DataFrame({
            'numeric1': [1, 2, np.nan, 4, 5],
            'numeric2': [1.1, 2.2, 3.3, 4.4, 5.5],
            'categorical': ['A', 'B', 'A', 'C', 'B'],
            'with_missing': [1, np.nan, 3, np.nan, 5]
        })
        # Add duplicate row
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        
        quality_report = check_data_quality(df)
        
        # Check report structure
        expected_keys = [
            'shape', 'missing_values', 'duplicate_rows', 'data_types',
            'memory_usage', 'numeric_columns', 'categorical_columns'
        ]
        for key in expected_keys:
            assert key in quality_report
        
        # Check specific values
        assert quality_report['shape'] == (6, 4)
        assert quality_report['duplicate_rows'] == 1
        assert quality_report['missing_values']['numeric1'] == 1
        assert quality_report['missing_values']['with_missing'] == 2
        
        logger.info("Data quality check test passed")


class TestEvaluationUtils:
    """Test class for evaluation utility functions."""
    
    def test_evaluate_classification_model_validation(self):
        """Test input validation in evaluate_classification_model."""
        y_true = np.array([0, 1, 0, 1])
        y_pred_wrong_length = np.array([0, 1, 0])
        y_pred_proba_wrong_length = np.array([[0.1, 0.9], [0.8, 0.2]])
        
        # Test mismatched lengths
        with pytest.raises(ValueError, match="y_true and y_pred must have same length"):
            evaluate_classification_model(y_true, y_pred_wrong_length)
        
        # Test mismatched proba lengths
        with pytest.raises(ValueError, match="y_true and y_pred_proba must have same length"):
            evaluate_classification_model(y_true, y_true, y_pred_proba_wrong_length)
    
    def test_evaluate_classification_model_functionality(self):
        """Test evaluate_classification_model returns correct metrics."""
        # Create test data
        y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1])
        y_pred_proba = np.array([
            [0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.7, 0.3],
            [0.1, 0.9], [0.9, 0.1], [0.3, 0.7], [0.4, 0.6]
        ])
        
        metrics = evaluate_classification_model(y_true, y_pred, y_pred_proba)
        
        # Check that all expected metrics are present
        expected_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score',
            'confusion_matrix', 'classification_report', 'roc_auc'
        ]
        for metric in expected_metrics:
            assert metric in metrics
        
        # Check metric types and ranges
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
        assert 0 <= metrics['roc_auc'] <= 1
        
        assert isinstance(metrics['confusion_matrix'], np.ndarray)
        assert metrics['confusion_matrix'].shape == (2, 2)
        
        logger.info("Classification evaluation test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
