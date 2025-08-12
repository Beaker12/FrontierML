"""Data loading and preprocessing utilities for AI/ML practice problems.

This module provides common data loading, preprocessing, and transformation
utilities used across multiple notebooks.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Union, List
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


def load_sample_data(dataset_name: str) -> pd.DataFrame:
    """Load sample datasets for practice problems.
    
    Args:
        dataset_name: Name of the dataset to load
        
    Returns:
        DataFrame containing the requested dataset
        
    Raises:
        ValueError: If dataset_name is not recognized
    """
    datasets = {
        'iris': _load_iris_data,
        'boston': _load_boston_data,
        'wine': _load_wine_data,
        'digits': _load_digits_data,
    }
    
    if dataset_name not in datasets:
        raise ValueError(f"Dataset '{dataset_name}' not recognized. "
                        f"Available: {list(datasets.keys())}")
    
    logger.info(f"Loading {dataset_name} dataset")
    return datasets[dataset_name]()


def _load_iris_data() -> pd.DataFrame:
    """Load the Iris dataset."""
    from sklearn.datasets import load_iris
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    df['target_name'] = [data.target_names[i] for i in data.target]
    return df


def _load_boston_data() -> pd.DataFrame:
    """Load the Boston housing dataset."""
    from sklearn.datasets import load_boston
    data = load_boston()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df


def _load_wine_data() -> pd.DataFrame:
    """Load the Wine dataset."""
    from sklearn.datasets import load_wine
    data = load_wine()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    df['target_name'] = [data.target_names[i] for i in data.target]
    return df


def _load_digits_data() -> pd.DataFrame:
    """Load the Digits dataset."""
    from sklearn.datasets import load_digits
    data = load_digits()
    df = pd.DataFrame(data.data, columns=[f'pixel_{i}' for i in range(data.data.shape[1])])
    df['target'] = data.target
    return df


def preprocess_data(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
    scale_features: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess data for machine learning models.
    
    Args:
        df: Input DataFrame
        target_column: Name of the target column
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        scale_features: Whether to standardize features
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    logger.info(f"Preprocessing data with target column: {target_column}")
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Handle categorical features
    categorical_columns = X.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features if requested
    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    logger.info(f"Data split: {X_train.shape[0]} training, {X_test.shape[0]} testing samples")
    
    return X_train, X_test, y_train, y_test


def generate_synthetic_data(
    n_samples: int = 1000,
    n_features: int = 10,
    n_classes: int = 2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data for testing algorithms.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features
        n_classes: Number of classes for classification
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X, y) arrays
    """
    from sklearn.datasets import make_classification
    
    logger.info(f"Generating synthetic data: {n_samples} samples, {n_features} features")
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_redundant=0,
        n_informative=n_features,
        random_state=random_state
    )
    
    return X, y


def check_data_quality(df: pd.DataFrame) -> dict:
    """Perform basic data quality checks.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with data quality metrics
    """
    quality_report = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
    }
    
    return quality_report


def save_data(data: Union[pd.DataFrame, dict], filepath: str, format: str = 'csv') -> str:
    """
    Save data in specified format.
    
    Parameters
    ----------
    data : pd.DataFrame or dict
        Data to save
    filepath : str
        Path to save the file
    format : str, default='csv'
        Format to save ('csv', 'json', 'parquet', 'pickle')
        
    Returns
    -------
    str
        Path where the file was saved
        
    Raises
    ------
    ValueError
        If format is not supported or data type is incompatible
    """
    import os
    import json
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if format == 'csv':
        if isinstance(data, pd.DataFrame):
            data.to_csv(filepath, index=False)
        else:
            raise ValueError("CSV format requires DataFrame input")
            
    elif format == 'json':
        if isinstance(data, pd.DataFrame):
            data.to_json(filepath, orient='records', date_format='iso')
        elif isinstance(data, dict):
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        else:
            raise ValueError("JSON format requires DataFrame or dict input")
            
    elif format == 'parquet':
        if isinstance(data, pd.DataFrame):
            data.to_parquet(filepath, index=False)
        else:
            raise ValueError("Parquet format requires DataFrame input")
            
    elif format == 'pickle':
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
            
    else:
        raise ValueError(f"Unsupported format: {format}. "
                        f"Supported formats: csv, json, parquet, pickle")
    
    logger.info(f"Data saved to: {filepath}")
    return filepath


def load_data(filepath: str, format: str = None) -> Union[pd.DataFrame, dict]:
    """
    Load data from file, auto-detecting format if not specified.
    
    Parameters
    ----------
    filepath : str
        Path to the file to load
    format : str, optional
        Format of the file ('csv', 'json', 'parquet', 'pickle')
        If None, will be inferred from file extension
        
    Returns
    -------
    pd.DataFrame or dict
        Loaded data
        
    Raises
    ------
    ValueError
        If format is not supported
    FileNotFoundError
        If file doesn't exist
    """
    import os
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Auto-detect format from extension if not provided
    if format is None:
        ext = os.path.splitext(filepath)[1].lower()
        format_map = {
            '.csv': 'csv',
            '.json': 'json',
            '.parquet': 'parquet',
            '.pkl': 'pickle',
            '.pickle': 'pickle'
        }
        format = format_map.get(ext)
        if format is None:
            raise ValueError(f"Cannot infer format from extension: {ext}")
    
    if format == 'csv':
        return pd.read_csv(filepath)
    elif format == 'json':
        try:
            return pd.read_json(filepath, orient='records')
        except ValueError:
            # Try loading as dict if pandas fails
            import json
            with open(filepath, 'r') as f:
                return json.load(f)
    elif format == 'parquet':
        return pd.read_parquet(filepath)
    elif format == 'pickle':
        import pickle
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported format: {format}")
