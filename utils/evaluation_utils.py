"""Model evaluation utilities for AI/ML practice problems.

This module provides common evaluation metrics and model assessment
utilities used across multiple notebooks.

References:
    - Pedregosa et al. (2011). Scikit-learn: Machine learning in Python. 
      Journal of Machine Learning Research, 12(Oct), 2825-2857.
    - Powers, D. M. (2011). Evaluation: from precision, recall and F-measure 
      to ROC, informedness, markedness and correlation. Journal of Machine 
      Learning Technologies, 2(1), 37-63.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)


def evaluate_classification_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None
) -> Dict[str, Union[float, np.ndarray]]:
    """Comprehensive evaluation of classification models.
    
    Computes standard classification metrics including accuracy, precision,
    recall, F1-score, confusion matrix, and ROC-AUC as recommended by
    Powers (2011) for comprehensive model evaluation.
    
    Args:
        y_true: True labels as numpy array
        y_pred: Predicted labels as numpy array  
        y_pred_proba: Predicted probabilities (optional, for ROC-AUC)
        class_names: Names of classes (optional, for reporting)
        
    Returns:
        Dictionary containing evaluation metrics:
        - accuracy: Overall accuracy score
        - precision: Weighted average precision
        - recall: Weighted average recall
        - f1_score: Weighted average F1-score
        - confusion_matrix: Confusion matrix as numpy array
        - classification_report: Detailed per-class metrics
        - roc_auc: ROC-AUC score (if probabilities provided)
        
    Raises:
        ValueError: If array shapes don't match or invalid inputs
        
    References:
        Powers, D. M. (2011). Evaluation: from precision, recall and F-measure 
        to ROC, informedness, markedness and correlation. Journal of Machine 
        Learning Technologies, 2(1), 37-63.
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        classification_report, confusion_matrix, roc_auc_score
    )
    
    # Input validation
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true and y_pred must have same length: {len(y_true)} vs {len(y_pred)}")
    if y_pred_proba is not None and len(y_true) != len(y_pred_proba):
        raise ValueError(f"y_true and y_pred_proba must have same length: {len(y_true)} vs {len(y_pred_proba)}")
    
    logger.info(f"Evaluating classification model with {len(y_true)} samples")
    
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
    metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted')
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    # Classification report
    metrics['classification_report'] = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )
    
    # ROC AUC (if probabilities provided)
    if y_pred_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:  # Binary classification
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            else:  # Multi-class
                metrics['roc_auc'] = roc_auc_score(
                    y_true, y_pred_proba, multi_class='ovr'
                )
        except Exception as e:
            logger.warning(f"Could not compute ROC AUC: {e}")
    
    return metrics


def evaluate_regression_model(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """Comprehensive evaluation of regression models.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary containing various evaluation metrics
    """
    from sklearn.metrics import (
        mean_squared_error, mean_absolute_error, r2_score,
        mean_absolute_percentage_error
    )
    
    metrics = {}
    
    # Basic metrics
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['r2_score'] = r2_score(y_true, y_pred)
    
    # Additional metrics
    try:
        metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
    except Exception:
        # Calculate MAPE manually if not available
        metrics['mape'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Residual analysis
    residuals = y_true - y_pred
    metrics['residual_mean'] = np.mean(residuals)
    metrics['residual_std'] = np.std(residuals)
    
    return metrics


def cross_validate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: str = 'accuracy',
    random_state: int = 42
) -> Dict[str, Union[float, np.ndarray]]:
    """Perform cross-validation on a model.
    
    Args:
        model: Scikit-learn model object
        X: Feature matrix
        y: Target vector
        cv: Number of cross-validation folds
        scoring: Scoring metric
        random_state: Random seed
        
    Returns:
        Dictionary with cross-validation results
    """
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    
    if scoring in ['accuracy', 'f1', 'precision', 'recall']:
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    else:
        cv_splitter = cv
    
    scores = cross_val_score(model, X, y, cv=cv_splitter, scoring=scoring)
    
    results = {
        'scores': scores,
        'mean_score': scores.mean(),
        'std_score': scores.std(),
        'min_score': scores.min(),
        'max_score': scores.max()
    }
    
    logger.info(f"Cross-validation {scoring}: {results['mean_score']:.4f} (+/- {results['std_score'] * 2:.4f})")
    
    return results


def calculate_learning_curve(
    model,
    X: np.ndarray,
    y: np.ndarray,
    train_sizes: Optional[np.ndarray] = None,
    cv: int = 5,
    scoring: str = 'accuracy'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate learning curve for a model.
    
    Args:
        model: Scikit-learn model object
        X: Feature matrix
        y: Target vector
        train_sizes: Training set sizes to evaluate
        cv: Number of cross-validation folds
        scoring: Scoring metric
        
    Returns:
        Tuple of (train_sizes, train_scores, validation_scores)
    """
    from sklearn.model_selection import learning_curve
    
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)
    
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X, y,
        train_sizes=train_sizes,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )
    
    return train_sizes_abs, train_scores, val_scores


def calculate_validation_curve(
    model,
    X: np.ndarray,
    y: np.ndarray,
    param_name: str,
    param_range: np.ndarray,
    cv: int = 5,
    scoring: str = 'accuracy'
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate validation curve for hyperparameter tuning.
    
    Args:
        model: Scikit-learn model object
        X: Feature matrix
        y: Target vector
        param_name: Name of parameter to vary
        param_range: Range of parameter values
        cv: Number of cross-validation folds
        scoring: Scoring metric
        
    Returns:
        Tuple of (train_scores, validation_scores)
    """
    from sklearn.model_selection import validation_curve
    
    train_scores, val_scores = validation_curve(
        model, X, y,
        param_name=param_name,
        param_range=param_range,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )
    
    return train_scores, val_scores


def compare_models(
    models: Dict[str, object],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    task_type: str = 'classification'
) -> pd.DataFrame:
    """Compare multiple models on the same dataset.
    
    Args:
        models: Dictionary mapping model names to model objects
        X_train: Training features
        X_test: Test features
        y_train: Training targets
        y_test: Test targets
        task_type: Type of task ('classification' or 'regression')
        
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    for name, model in models.items():
        logger.info(f"Evaluating model: {name}")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate based on task type
        if task_type == 'classification':
            y_pred_proba = None
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)
            
            metrics = evaluate_classification_model(y_test, y_pred, y_pred_proba)
            
            result = {
                'Model': name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
            }
            
            if 'roc_auc' in metrics:
                result['ROC AUC'] = metrics['roc_auc']
                
        else:  # regression
            metrics = evaluate_regression_model(y_test, y_pred)
            
            result = {
                'Model': name,
                'MSE': metrics['mse'],
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae'],
                'R²': metrics['r2_score'],
                'MAPE': metrics['mape']
            }
        
        results.append(result)
    
    return pd.DataFrame(results)
