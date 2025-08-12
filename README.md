# FrontierML: Machine Learning with Real-World Data

An interactive **Jupyter Book** that teaches machine learning concepts through hands-on implementation with real-world data collection and analysis.

## Overview

This book provides a comprehensive introduction to machine learning algorithms, combining theoretical understanding with practical implementation. What sets this book apart is its emphasis on:

- **Real-world data collection** through web scraping and APIs
- **Step-by-step mathematical explanations** with proper citations
- **Interactive code examples** that you can run and modify
- **Comprehensive visualizations** to understand algorithm behavior
- **Best practices** for data science workflows

## Repository Structure

```
FrontierML/
├── notebooks/          # Interactive Jupyter Book chapters
│   ├── 00_index.ipynb          # Course overview
│   ├── 01_data_collection.ipynb # Data collection and scraping
│   ├── 02_linear_regression.ipynb # Linear regression
│   └── ...                     # Additional ML topics
├── data/               # Datasets used in notebooks
├── utils/              # Utility functions for notebooks
│   ├── scraping_utils.py       # Web scraping utilities
│   ├── data_utils.py           # Data processing utilities
│   ├── evaluation_utils.py     # Model evaluation utilities
│   └── plot_utils.py           # Visualization utilities
├── docs/               # Additional documentation
├── tests/              # Basic functionality tests
├── _config.yml         # Jupyter Book configuration
├── _toc.yml           # Table of contents
├── intro.md           # Book introduction
└── requirements.txt    # Dependencies including jupyter-book
```

## Getting Started

### Option 1: Interactive Jupyter Book (Recommended)

1. Clone this repository:

   ```bash
   git clone <repository-url>
   cd FrontierML
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Build the Jupyter Book:

   ```bash
   make book
   ```

4. Open `_build/html/index.html` in your browser

### Option 2: Individual Notebooks

1. Start Jupyter Lab:

   ```bash
   make jupyter
   ```

2. Navigate to the `notebooks/` directory and open individual chapters

## Book Contents

### Chapter 1: Data Collection and Web Scraping
- Web scraping fundamentals and ethics
- API interactions for real-world data
- Data quality assessment and preprocessing
- Feature engineering techniques

### Chapter 2: Linear Regression
- Mathematical foundations and derivations
- Implementation from scratch
- Real-world applications with financial data
- Model evaluation and interpretation

### Chapter 3: Logistic Regression
- Theory of classification algorithms
- Maximum likelihood estimation
- Practical implementation and evaluation
- Handling imbalanced datasets

### Chapter 4: Decision Trees
- Information theory and entropy
- Tree construction algorithms
- Overfitting and pruning techniques
- Feature importance analysis

### Chapter 5: Random Forests
- Ensemble learning principles
- Bootstrap aggregating (bagging)
- Out-of-bag error estimation
- Hyperparameter tuning

### Chapter 6: Support Vector Machines
- Geometric interpretation and margin maximization
- Kernel methods and the kernel trick
- Handling non-linearly separable data
- Multi-class classification strategies

### Chapter 7: Neural Networks
- Perceptron and multi-layer networks
- Backpropagation algorithm
- Activation functions and optimization
- Practical deep learning considerations

## Contributing

This is an educational repository. Please ensure all contributions maintain clarity and educational value.

## License

See LICENSE file for details.
