# FrontierML: AI/ML Practice Problems

A comprehensive collection of end-to-end AI/ML practice problems implemented in Python using Jupyter notebooks.

## Project Structure

```
FrontierML/
├── notebooks/          # Individual practice notebooks for 20 major AI/ML problems
├── data/              # Datasets and test data
├── src/               # Utility functions and helper modules
├── tests/             # Unit tests for utility functions
├── docs/              # Documentation
├── requirements.txt   # Python dependencies
├── Makefile          # Environment setup and common tasks
├── pyproject.toml    # Package configuration
└── .gitignore        # Git ignore rules
```

## Getting Started

### Prerequisites
- Python 3.8+
- Git

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd FrontierML
```

2. Set up the environment:
```bash
make setup
```

3. Activate the virtual environment:
```bash
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

4. Start Jupyter Lab:
```bash
make jupyter
```

## Practice Problems

This project includes 20 major AI/ML practice problems:

1. **Linear Regression** - House price prediction
2. **Logistic Regression** - Binary classification
3. **Decision Trees** - Feature importance analysis
4. **Random Forest** - Ensemble learning
5. **Support Vector Machines** - Non-linear classification
6. **K-Means Clustering** - Customer segmentation
7. **Hierarchical Clustering** - Dendrogram analysis
8. **Principal Component Analysis** - Dimensionality reduction
9. **Neural Networks** - Multi-layer perceptron
10. **Convolutional Neural Networks** - Image classification
11. **Recurrent Neural Networks** - Time series prediction
12. **LSTM Networks** - Sequential data modeling
13. **Autoencoders** - Anomaly detection
14. **Generative Adversarial Networks** - Data generation
15. **Natural Language Processing** - Text classification
16. **Recommendation Systems** - Collaborative filtering
17. **Reinforcement Learning** - Q-learning
18. **Time Series Forecasting** - ARIMA models
19. **Transfer Learning** - Pre-trained models
20. **Model Deployment** - MLOps pipeline

Each notebook includes:
- Detailed problem description
- Data exploration and visualization
- Feature engineering
- Model implementation
- Evaluation metrics
- Comprehensive comments and explanations

## Contributing

Please follow the coding standards outlined in the project guidelines:
- Use snake_case for variable names
- Include type hints for all functions
- Write comprehensive docstrings
- Follow PEP 8 conventions

## License

MIT License - see LICENSE file for details.
