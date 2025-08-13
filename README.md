# FrontierML: Machine Learning with Real-World Data

An interactive **Jupyter Book** that teaches machine learning concepts through hands-on implementation with real-world data collection and analysis.

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter Book](https://img.shields.io/badge/Jupyter-Book-orange.svg)](https://jupyterbook.org/)

## Overview

This comprehensive book provides an in-depth introduction to **22 essential machine learning and AI algorithms**, combining theoretical understanding with practical implementation. What sets this book apart is its emphasis on:

- **Real-world data collection** through ethical web scraping and APIs
- **Step-by-step mathematical explanations** with proper citations
- **Interactive code examples** that you can run and modify
- **Comprehensive visualizations** to understand algorithm behavior
- **Best practices** for reproducible data science workflows
- **Complete coverage** from basic regression to deep learning

## Course Coverage

### **Supervised Learning (13 chapters)**

- Linear & Logistic Regression
- Decision Trees & Random Forest
- Support Vector Machines
- Neural Networks & Deep Learning
- Naive Bayes & K-Nearest Neighbors
- Boosting Methods (AdaBoost, Gradient Boosting)

### **Unsupervised Learning (6 chapters)**

- Clustering (K-Means, Hierarchical, DBSCAN, GMM)
- Dimensionality Reduction (PCA, LDA)

### **Specialized Techniques (3 chapters)**

- Association Rule Mining
- Reinforcement Learning (Q-Learning)
- Deep Learning (Autoencoders, CNNs)

## Repository Structure

```text
FrontierML/
├── notebooks/          # Interactive Jupyter Book chapters (22 total)
│   ├── 00_index.ipynb                      # Course overview
│   ├── 01_data_collection.ipynb            # Data collection and scraping
│   ├── 02_linear_regression.ipynb          # Linear regression
│   ├── 03_logistic_regression.ipynb        # Logistic regression
│   ├── 04_decision_trees.ipynb             # Decision trees
│   ├── 05_random_forest.ipynb              # Random forest
│   ├── 06_support_vector_machines.ipynb    # Support vector machines
│   ├── 07_neural_networks.ipynb            # Neural networks
│   ├── 08_k_means_clustering.ipynb         # K-means clustering
│   ├── 09_hierarchical_clustering.ipynb    # Hierarchical clustering
│   ├── 10_principal_component_analysis.ipynb # PCA
│   ├── 11_naive_bayes.ipynb                # Naive Bayes
│   ├── 12_k_nearest_neighbors.ipynb        # KNN
│   ├── 13_gradient_boosting.ipynb          # Gradient boosting
│   ├── 14_association_rule_mining.ipynb    # Association rules
│   ├── 15_dbscan_clustering.ipynb          # DBSCAN clustering
│   ├── 16_linear_discriminant_analysis.ipynb # LDA
│   ├── 17_gaussian_mixture_models.ipynb    # GMM
│   ├── 18_adaboost.ipynb                   # AdaBoost
│   ├── 19_q_learning.ipynb                 # Q-learning
│   ├── 20_autoencoders.ipynb               # Autoencoders
│   └── 21_convolutional_neural_networks.ipynb # CNNs
├── data/               # Datasets used in notebooks
│   ├── raw/                    # Raw scraped data
│   ├── processed/              # Cleaned and processed data
│   ├── features/               # Feature engineered datasets
│   └── samples/                # Sample datasets for examples
├── utils/              # Utility functions for notebooks
│   ├── data_utils.py           # Data processing utilities
│   ├── evaluation_utils.py     # Model evaluation utilities
│   ├── plot_utils.py           # Visualization utilities
│   └── scraping_utils.py       # Web scraping utilities
├── tests/              # Basic functionality tests
├── docs/               # Additional documentation
├── _config.yml         # Jupyter Book configuration
├── _toc.yml           # Table of contents
├── intro.md           # Book introduction
├── references.bib     # Bibliography with proper citations
└── requirements.txt    # Dependencies including deep learning libraries
```

## Prerequisites

### **Knowledge Requirements**

- **Python Programming**: Intermediate Python skills (functions, classes, NumPy basics)
- **Mathematics**: Linear algebra, calculus fundamentals, basic probability
- **Statistics**: Descriptive statistics, hypothesis testing concepts

### **Technical Requirements**

- **Python 3.8+** with pip package manager
- **Git** for version control
- **Jupyter Lab** or **Jupyter Notebook**
- **8GB+ RAM** recommended for deep learning chapters
- **GPU support** optional but recommended for Chapters 20-21

## Installation & Setup

### **Option 1: Interactive Jupyter Book (Recommended)**

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Beaker12/FrontierML.git
   cd FrontierML
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Build and serve the book**:

   ```bash
   make book    # Build the interactive book
   make serve   # Serve locally at http://localhost:8000
   ```

4. **Open in browser**: Navigate to `http://localhost:8000` or open `_build/html/index.html`

### **Option 2: Individual Notebooks**

1. **Start Jupyter Lab**:

   ```bash
   make jupyter
   ```

2. **Navigate to notebooks/**: Open individual chapters for hands-on learning

### **Option 3: Development Setup**

1. **Install development dependencies**:

   ```bash
   pip install -r requirements.txt
   pip install pre-commit black flake8 mypy
   ```

2. **Set up pre-commit hooks**:

   ```bash
   pre-commit install
   ```

## Quick Start

### **For Beginners**

1. Start with **Chapter 1** (Data Collection) to understand data gathering
2. Progress through **Chapter 2** (Linear Regression) for mathematical foundations  
3. Continue sequentially through supervised learning chapters

### **For Experienced Practitioners**

1. Review **Chapter 0** (Course Overview) for structure
2. Jump to specific algorithms of interest
3. Focus on implementation details and mathematical derivations

### **For Researchers**

1. Examine the mathematical foundations in each chapter
2. Review citations and references in `references.bib`
3. Use implementations as starting points for custom algorithms

## Book Contents

This comprehensive course covers **22 essential machine learning and AI techniques** organized into logical progressions:

### **Foundation & Data (Chapters 1-2)**

#### Chapter 1: Data Collection and Web Scraping

- Ethical web scraping principles and legal considerations
- API interactions for real-world data collection
- Data quality assessment and cleaning pipelines
- Feature engineering for machine learning applications

#### Chapter 2: Linear Regression

- Mathematical foundations with proper derivations
- Implementation from scratch using NumPy
- Real estate price prediction with actual scraped data
- Model evaluation, interpretation, and diagnostics

### **Supervised Learning - Core Algorithms (Chapters 3-7)**

#### Chapter 3: Logistic Regression

Probabilistic classification with sigmoid functions

#### Chapter 4: Decision Trees

Information theory and tree-based learning

#### Chapter 5: Random Forest

Ensemble methods and bootstrap aggregating

#### Chapter 6: Support Vector Machines

Margin maximization and kernel methods

#### Chapter 7: Neural Networks

Introduction to neural networks and deep learning fundamentals:
- **Mathematical foundations**: Perceptron algorithm and convergence theory
- **Multi-layer perceptrons**: Architecture design and backpropagation implementation
- **Activation functions**: ReLU, sigmoid, tanh with practical comparisons
- **Framework comparison**: Implementation across scikit-learn, TensorFlow, and PyTorch
- **Sports analytics application**: NFL player performance prediction and season classification
- **Real-world insights**: Pattern recognition in professional sports statistics

### **Supervised Learning - Additional Methods (Chapters 11-13, 18)**

#### Chapter 11: Naive Bayes Classification

Bayesian inference and probabilistic models

#### Chapter 12: K-Nearest Neighbors (KNN)

Instance-based learning and distance metrics

#### Chapter 13: Gradient Boosting Machines

XGBoost, LightGBM, and advanced boosting

#### Chapter 18: AdaBoost

Adaptive boosting with weak learner combinations

### **Unsupervised Learning - Clustering (Chapters 8-9, 15, 17)**

#### Chapter 8: K-Means Clustering

Centroid-based clustering and Lloyd's algorithm

#### Chapter 9: Hierarchical Clustering

Agglomerative and divisive clustering methods

#### Chapter 15: DBSCAN Clustering

Density-based clustering with noise detection

#### Chapter 17: Gaussian Mixture Models

Probabilistic clustering using EM algorithm

### **Unsupervised Learning - Dimensionality Reduction (Chapters 10, 16)**

#### Chapter 10: Principal Component Analysis (PCA)

Eigenvalue decomposition and variance preservation

#### Chapter 16: Linear Discriminant Analysis (LDA)

Supervised dimensionality reduction

### **Pattern Mining & Association Rules (Chapter 14)**

#### Chapter 14: Association Rule Mining

Market basket analysis and frequent itemset discovery

### **Reinforcement Learning (Chapter 19)**

#### Chapter 19: Q-Learning

Markov Decision Processes and value iteration methods

### **Deep Learning & Computer Vision (Chapters 20-21)**

#### Chapter 20: Autoencoders

Representation learning and variational autoencoders

#### Chapter 21: Convolutional Neural Networks

Computer vision and image processing

## Key Features

**Mathematical Rigor**: Every algorithm includes step-by-step mathematical derivations with proper citations  
**Real-World Data**: Actual data collection from websites and APIs, not toy datasets  
**Implementation Focus**: Build algorithms from scratch to understand core concepts  
**Production Ready**: Scikit-learn and TensorFlow implementations for practical use  
**Comprehensive Coverage**: 22 algorithms spanning the full ML spectrum  
**Reproducible Science**: Version-controlled code, documented methodology, proper citations

## Contributing

We welcome contributions that enhance the educational value of this course! Please follow these guidelines:

### **How to Contribute**

1. **Fork the repository** and create a feature branch
2. **Follow coding standards** 
3. **Add comprehensive tests** for any new functionality
4. **Update documentation** and citations as needed
5. **Submit a pull request** with detailed description

### **Areas for Contribution**

- **Additional datasets** for algorithm demonstrations
- **Exercise solutions** and coding challenges  
- **Mathematical clarifications** and improved derivations
- **Performance optimizations** and computational efficiency
- **Translation** to other programming languages

### **Review Process**

All contributions are reviewed for:

- **Mathematical accuracy** and proper citations
- **Code quality** and adherence to standards
- **Educational clarity** and learning progression
- **Reproducibility** and documentation completeness

## Citation

If you use this course in your research or teaching, please cite:

```bibtex
@misc{frontierml2025,
  title={FrontierML: A Comprehensive Course in Machine Learning and AI},
  author={Beaker12},
  year={2025},
  url={https://github.com/Beaker12/FrontierML}
}
```

## License

This educational resource is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Mathematical foundations** based on established academic literature
- **Implementation guidance** from scikit-learn and TensorFlow documentation  
- **Educational approach** inspired by best practices in machine learning pedagogy
- **Community contributions** from students and practitioners worldwide

---

**Start your machine learning journey today!**

Whether you're a beginner exploring your first algorithm or an expert deepening your understanding, FrontierML provides the mathematical rigor and practical implementation skills needed to excel in modern AI and machine learning.
