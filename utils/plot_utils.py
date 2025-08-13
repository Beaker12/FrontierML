"""Visualization utilities for FrontierML course.

This module provides common plotting and visualization functions
used across multiple notebooks, with proper configuration for
Jupyter Book display.

References:
    - Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. 
      Computing in Science & Engineering, 9(3), 90-95.
    - Waskom, M. L. (2021). Seaborn: statistical data visualization. 
      Journal of Open Source Software, 6(60), 3021.

Author: Stuart W. Parkhurst
Date: August 12, 2025
"""

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Union
import logging
import warnings

logger = logging.getLogger(__name__)

# Suppress matplotlib warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


def configure_plotting(style: str = 'default', 
                      figsize: Tuple[int, int] = (10, 6),
                      dpi: int = 100,
                      inline: bool = True) -> None:
    """
    Configure matplotlib and seaborn for optimal display in Jupyter Book.
    
    Parameters
    ----------
    style : str, default='default'
        Matplotlib style to use ('default', 'seaborn-v0_8', 'ggplot', etc.)
    figsize : tuple, default=(10, 6)
        Default figure size for plots
    dpi : int, default=100
        Dots per inch for figure resolution
    inline : bool, default=True
        Whether to enable inline plotting for Jupyter
    
    Notes
    -----
    This function configures matplotlib for optimal display in both
    Jupyter notebooks and Jupyter Book builds. It avoids the 'Agg'
    backend which prevents proper plot display in books.
    
    References
    ----------
    Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. 
    Computing in Science & Engineering, 9(3), 90-95.
    """
    # Configure matplotlib backend for Jupyter Book compatibility
    if inline:
        # Use inline backend for Jupyter environments
        try:
            # Check if we're in a Jupyter environment
            from IPython import get_ipython
            ipy = get_ipython()
            if ipy is not None:
                ipy.run_line_magic('matplotlib', 'inline')
        except (ImportError, NameError):
            # Not in Jupyter, use module backend for compatibility
            try:
                matplotlib.use('module://matplotlib_inline.backend_inline')
            except ImportError:
                # Fallback to default
                pass
    
    # Set plotting style
    plt.style.use(style)
    
    # Configure default parameters for enhanced publication-quality plots
    plt.rcParams.update({
        # Figure settings
        'figure.figsize': figsize,
        'figure.dpi': dpi,
        'figure.facecolor': 'white',
        'figure.edgecolor': 'none',
        'figure.autolayout': True,
        
        # Axes settings
        'axes.facecolor': 'white',
        'axes.edgecolor': '#333333',
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'axes.axisbelow': True,
        
        # Grid settings
        'grid.color': '#E0E0E0',
        'grid.linestyle': '-',
        'grid.linewidth': 0.8,
        'grid.alpha': 0.6,
        
        # Font settings with fallback options
        'font.size': 12,
        
        # Text sizes
        'axes.labelsize': 13,
        'axes.titlesize': 15,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.titlesize': 17,
        
        # Line and marker settings
        'lines.linewidth': 2.2,
        'lines.markersize': 6,
        'patch.linewidth': 0.8,
        
        # Tick settings
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'xtick.minor.size': 3,
        'ytick.minor.size': 3,
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        
        # Save settings
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'none',
        'savefig.dpi': dpi,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        
        # Legend settings
        'legend.frameon': True,
        'legend.fancybox': True,
        'legend.shadow': False,
        'legend.framealpha': 0.95,
        'legend.edgecolor': '#CCCCCC',
        'legend.facecolor': 'white'
    })
    
    # Try to configure Computer Modern fonts with fallbacks
    try:
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Computer Modern', 'Times New Roman', 'DejaVu Serif'],
            'mathtext.fontset': 'cm',
            'mathtext.rm': 'serif',
            'mathtext.it': 'serif:italic',
            'mathtext.bf': 'serif:bold',
        })
        logger.info("Computer Modern fonts configured successfully")
    except Exception as e:
        logger.warning(f"Could not configure Computer Modern fonts: {e}")
        # Fallback to best available serif fonts
        plt.rcParams.update({
            'font.family': 'serif',
            'mathtext.fontset': 'dejavuserif',
        })
    
    # Configure seaborn with enhanced styling
    sns.set_palette("husl")
    sns.set_context("notebook", font_scale=1.1, rc={
        'lines.linewidth': 2.2,
        'grid.linewidth': 0.8,
        'patch.linewidth': 0.8
    })
    
    # Set custom color palette for better visual appeal
    custom_colors = [
        '#2E86AB',  # Blue
        '#A23B72',  # Magenta
        '#F18F01',  # Orange
        '#C73E1D',  # Red
        '#6A994E',  # Green
        '#7209B7',  # Purple
        '#F77F00',  # Amber
        '#277DA1'   # Teal
    ]
    sns.set_palette(custom_colors)


def save_and_show_plot(fig: Optional[matplotlib.figure.Figure] = None,
                       filename: Optional[str] = None,
                       show: bool = True,
                       close: bool = False) -> None:
    """
    Save and/or show a matplotlib plot with proper configuration.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure, optional
        Figure to save/show. If None, uses current figure.
    filename : str, optional
        Filename to save the plot. If None, plot is not saved.
    show : bool, default=True
        Whether to display the plot
    close : bool, default=False
        Whether to close the figure after showing/saving
    
    Notes
    -----
    This function ensures plots are properly displayed in Jupyter Book
    while optionally saving them for external use.
    """
    if fig is None:
        fig = plt.gcf()
    
    # Save plot if filename provided
    if filename:
        fig.savefig(filename, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
    
    # Show plot
    if show:
        plt.show()
    
    # Close figure if requested
    if close:
        plt.close(fig)


# Initialize default plotting configuration
configure_plotting(style='seaborn-v0_8')


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
    try:
        from sklearn.metrics import confusion_matrix
    except ImportError:
        raise ImportError("scikit-learn is required for confusion matrix plotting")
    
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
    figsize: Tuple[int, int] = (10, 8),
    color: str = '#A23B72'
) -> plt.Figure:
    """Plot feature importance with enhanced styling.
    
    Parameters
    ----------
    feature_names : List[str]
        Names of features
    importances : np.ndarray
        Feature importance values
    title : str, default="Feature Importance"
        Plot title
    top_n : int, default=20
        Number of top features to show
    figsize : tuple, default=(10, 8)
        Figure size
    color : str, default='#A23B72'
        Color for the bars
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object with feature importance plot
        
    Notes
    -----
    Creates horizontal bar plot with features sorted by importance.
    Uses Computer Modern fonts for mathematical expressions.
    """
    # Sort features by importance
    indices = np.argsort(importances)[::-1][:top_n]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create horizontal bar plot for better label readability
    y_pos = np.arange(len(indices))
    bars = ax.barh(y_pos, importances[indices], color=color, alpha=0.8, edgecolor='white', linewidth=1)
    
    # Add value labels on bars
    for i, (bar, importance) in enumerate(zip(bars, importances[indices])):
        ax.text(importance + max(importances[indices]) * 0.01, i, f'{importance:.3f}', 
                va='center', fontsize=10, fontweight='semibold')
    
    ax.set_title(title, fontweight='bold', pad=20, fontsize=15)
    ax.set_xlabel('Importance Score', fontweight='semibold')
    ax.set_ylabel('Features', fontweight='semibold')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in indices])
    
    # Invert y-axis to show most important features at top
    ax.invert_yaxis()
    
    # Add subtle background gradient
    ax.set_facecolor('#FAFAFA')
    
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
    try:
        from sklearn.metrics import roc_curve, auc
    except ImportError:
        raise ImportError("scikit-learn is required for ROC curve plotting")
    
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
    figsize: Tuple[int, int] = (12, 5),
    color: str = '#2E86AB'
) -> plt.Figure:
    """Plot data distribution with histogram and KDE using enhanced styling.
    
    Parameters
    ----------
    data : pd.Series or np.ndarray
        Data to plot
    title : str, default="Data Distribution"
        Plot title
    bins : int, default=30
        Number of histogram bins
    figsize : tuple, default=(12, 5)
        Figure size
    color : str, default='#2E86AB'
        Color for the plots
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object with distribution plots
        
    Notes
    -----
    Creates side-by-side histogram and KDE plots with consistent styling
    and Computer Modern fonts for mathematical expressions.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram with enhanced styling
    n, bins_edges, patches = ax1.hist(
        data, 
        bins=bins, 
        alpha=0.7, 
        color=color, 
        edgecolor='white',
        linewidth=1.2,
        density=True
    )
    
    # Add subtle gradient to histogram bars
    for patch, value in zip(patches, n):
        patch.set_alpha(0.6 + 0.4 * (value / max(n)))
    
    ax1.set_title(f'{title} - Histogram', fontweight='bold', pad=15)
    ax1.set_xlabel('Value', fontweight='semibold')
    ax1.set_ylabel('Density', fontweight='semibold')
    
    # Add statistics text
    mean_val = np.mean(data)
    std_val = np.std(data)
    ax1.axvline(mean_val, color='red', linestyle='--', alpha=0.8, linewidth=2, label=f'$\\mu = {mean_val:.2f}$')
    ax1.legend(frameon=True, fancybox=True)
    
    # KDE plot with enhanced styling
    sns.kdeplot(
        data=data, 
        ax=ax2, 
        color=color, 
        linewidth=2.5,
        fill=True,
        alpha=0.3
    )
    ax2.axvline(mean_val, color='red', linestyle='--', alpha=0.8, linewidth=2, label=f'$\\mu = {mean_val:.2f}$')
    ax2.set_title(f'{title} - Kernel Density Estimation', fontweight='bold', pad=15)
    ax2.set_xlabel('Value', fontweight='semibold')
    ax2.set_ylabel('Density', fontweight='semibold')
    ax2.legend(frameon=True, fancybox=True)
    
    plt.tight_layout()
    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    title: str = "Correlation Matrix",
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = 'RdBu_r'
) -> plt.Figure:
    """Plot correlation matrix heatmap with enhanced styling.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to compute correlations for
    title : str, default="Correlation Matrix"
        Plot title
    figsize : tuple, default=(12, 10)
        Figure size
    cmap : str, default='RdBu_r'
        Colormap for the heatmap
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object with the correlation matrix plot
        
    Notes
    -----
    Uses diverging colormap with Computer Modern fonts for publication-quality output.
    Correlation values are annotated with appropriate precision.
    """
    # Compute correlation matrix
    corr = df.corr()
    
    # Generate mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw heatmap with enhanced styling
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        cmap=cmap,
        center=0,
        square=True,
        fmt='.3f',
        cbar_kws={
            'shrink': 0.8,
            'label': 'Correlation Coefficient'
        },
        linewidths=0.5,
        linecolor='white',
        ax=ax
    )
    
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # Rotate labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
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


def plot_decision_boundary(
    X: np.ndarray,
    y: np.ndarray,
    model,
    title: str = "Decision Boundary",
    figsize: Tuple[int, int] = (8, 6),
    mesh_step: float = 0.02,
    alpha: float = 0.6
) -> plt.Figure:
    """Plot decision boundary for 2D classification problems.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix (must be 2D for visualization)
    y : np.ndarray
        Target labels
    model : object
        Trained model with predict_proba or predict method
    title : str, default="Decision Boundary"
        Plot title
    figsize : tuple, default=(8, 6)
        Figure size
    mesh_step : float, default=0.02
        Step size for mesh grid
    alpha : float, default=0.6
        Transparency for decision boundary
    
    Returns
    -------
    plt.Figure
        matplotlib Figure object
        
    Notes
    -----
    Requires model to have either predict_proba or predict method.
    For binary classification, uses predict_proba if available.
    """
    if X.shape[1] != 2:
        raise ValueError("Decision boundary plotting requires 2D feature space")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create mesh for decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, mesh_step),
        np.arange(y_min, y_max, mesh_step)
    )
    
    # Make predictions on mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Use predict_proba if available, otherwise predict
    if hasattr(model, 'predict_proba'):
        Z = model.predict_proba(mesh_points)
        if Z.ndim > 1 and Z.shape[1] > 1:
            Z = Z[:, 1]  # Use probability of positive class for binary
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary with contour
        contour = ax.contourf(xx, yy, Z, levels=50, alpha=alpha, cmap=plt.cm.RdYlBu)
        ax.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='--', linewidths=2)
    else:
        Z = model.predict(mesh_points)
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=alpha, cmap=plt.cm.RdYlBu)
    
    # Plot data points
    scatter = ax.scatter(
        X[:, 0], X[:, 1], c=y,
        cmap=plt.cm.RdYlBu, edgecolors='black', alpha=0.8
    )
    
    ax.set_title(title)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    
    plt.tight_layout()
    return fig


def plot_training_history(
    losses: List[float],
    title: str = "Training Loss Curve",
    figsize: Tuple[int, int] = (8, 6),
    xlabel: str = "Iteration",
    ylabel: str = "Loss"
) -> plt.Figure:
    """Plot training loss curve.
    
    Parameters
    ----------
    losses : List[float]
        List of loss values during training
    title : str, default="Training Loss Curve"
        Plot title
    figsize : tuple, default=(8, 6)
        Figure size
    xlabel : str, default="Iteration"
        X-axis label
    ylabel : str, default="Loss"
        Y-axis label
    
    Returns
    -------
    plt.Figure
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(losses, marker='o', markersize=3, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
