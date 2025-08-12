"""Visualization utilities for AI/ML practice problems.

This module provides common plotting and visualization functions
used across multiple notebooks.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Union
import logging

logger = logging.getLogger(__name__)

# Set default plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """Plot a confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        title: Plot title
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    
    ax.set_title(title)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    
    plt.tight_layout()
    return fig


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    title: str = "Feature Importance",
    top_n: int = 20,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """Plot feature importance.
    
    Args:
        feature_names: Names of features
        importances: Feature importance values
        title: Plot title
        top_n: Number of top features to show
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    # Sort features by importance
    indices = np.argsort(importances)[::-1][:top_n]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.bar(range(len(indices)), importances[indices])
    ax.set_title(title)
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance')
    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
    
    plt.tight_layout()
    return fig


def plot_learning_curve(
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    train_sizes: np.ndarray,
    title: str = "Learning Curve",
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """Plot learning curves.
    
    Args:
        train_scores: Training scores
        val_scores: Validation scores
        train_sizes: Training set sizes
        title: Plot title
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate mean and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Plot curves
    ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                    alpha=0.1, color='blue')
    
    ax.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                    alpha=0.1, color='red')
    
    ax.set_title(title)
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('Score')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str = "ROC Curve",
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """Plot ROC curve.
    
    Args:
        y_true: True labels
        y_score: Predicted scores
        title: Plot title
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True)
    
    plt.tight_layout()
    return fig


def plot_distribution(
    data: Union[pd.Series, np.ndarray],
    title: str = "Data Distribution",
    bins: int = 30,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """Plot data distribution with histogram and KDE.
    
    Args:
        data: Data to plot
        title: Plot title
        bins: Number of histogram bins
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram
    ax1.hist(data, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_title(f'{title} - Histogram')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    # KDE plot
    sns.kdeplot(data=data, ax=ax2, color='darkblue')
    ax2.set_title(f'{title} - Kernel Density Estimation')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Density')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    title: str = "Correlation Matrix",
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """Plot correlation matrix heatmap.
    
    Args:
        df: DataFrame to compute correlations for
        title: Plot title
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    # Compute correlation matrix
    corr = df.corr()
    
    # Generate mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw heatmap
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        cmap='coolwarm',
        center=0,
        square=True,
        fmt='.2f',
        ax=ax
    )
    
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_time_series(
    dates: Union[pd.DatetimeIndex, np.ndarray],
    values: np.ndarray,
    title: str = "Time Series",
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """Plot time series data.
    
    Args:
        dates: Date/time values
        values: Time series values
        title: Plot title
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(dates, values, linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    fig.autofmt_xdate()
    
    plt.tight_layout()
    return fig
