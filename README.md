# NLP Interpretability and Overfitting Analysis

This project investigates the relationship between model overfitting and interpretability quality in NLP sentiment classification tasks. We demonstrate that high-accuracy overfitted models produce unreliable SHAP interpretations, highlighting the importance of considering model generalization when evaluating interpretability methods.

## 🎯 Project Overview

### Core Hypotheses
- **Overfitted models with high accuracy produce poor interpretability metrics**
- **SHAP values become non-intuitive when models memorize rather than generalize**
- **Cross-dataset validation reveals interpretation instability in overfitted models**

### Technical Approach
- Train models with intentional overfitting strategies
- Evaluate interpretability using SHAP (SHapley Additive exPlanations)
- Perform cross-dataset validation to test interpretation consistency
- Compare transformer models (BERT, RoBERTa) with baseline models (Bag-of-Words)

## 🖥️ Hardware Requirements

- **MacBook Pro M4 Pro (Apple Silicon)**
- **No CUDA support** - Uses MPS (Metal Performance Shaders) or CPU
- **Memory Management** - Implements gradient checkpointing and batch size optimization

## 🚀 Installation

### Prerequisites
- Python 3.9+
- macOS with Apple Silicon (M1/M2/M3/M4)
- Xcode Command Line Tools

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd nlp_interpretability_overfit
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "from src.utils.device_utils import get_device; print(f'Device: {get_device()}')"
   ```

### Apple Silicon Specific Notes

- PyTorch automatically detects and uses MPS (Metal Performance Shaders)
- No CUDA installation required
- Some operations may fall back to CPU for compatibility
- Memory management is optimized for Apple Silicon architecture

## 📁 Project Structure

```
nlp_interpretability_overfit/
├── README.md
├── requirements.txt
├── config.yaml                   # Unified configuration file
├── data/                          # Data storage
│   ├── raw/                       # Raw datasets
│   ├── processed/                 # Preprocessed data
│   └── cache/                     # Cached data
├── src/                           # Source code
│   ├── models/                    # Model implementations
│   ├── datasets/                  # Dataset handling
│   ├── training/                  # Training utilities
│   ├── interpretability/          # SHAP and interpretability
│   ├── evaluation/                # Evaluation metrics
│   └── utils/                     # Utility functions
├── results/                       # Experiment results
│   ├── models/                    # Saved model checkpoints
│   ├── logs/                      # Experiment logs
│   ├── figures/                   # Generated plots
│   └── tables/                    # Results tables
└── tests/                         # Unit tests
```

## 🔧 Configuration

The project uses YAML configuration files for easy parameter management:

- **`config.yaml`** - Unified configuration file containing all settings for models, datasets, training, and experiments

### Key Configuration Options

```yaml
# Overfitting strategy
training:
  overfitting:
    learning_rate: 1e-3        # High LR for faster overfitting
    num_epochs: 50             # Many epochs
    weight_decay: 0.0          # No regularization
    dropout: 0.0               # No dropout
    batch_size: 8              # Small batch size
    early_stopping: false      # Let it overfit

# Interpretability settings
interpretability:
  shap:
    max_samples: 100
    device: "cpu"              # Always use CPU for SHAP
    explainer_type: "auto"
```

## 🏃‍♂️ Quick Start

### 1. Basic Setup
```bash
# Test device detection
python -c "from src.utils.device_utils import get_device; print(get_device())"

# Test reproducibility setup
python -c "from src.utils.reproducibility import setup_reproducibility; setup_reproducibility({'seed': 42})"
```

### 2. Run Experiments
```bash
# Train models with overfitting strategy
python train.py

# Evaluate interpretability
python validation.py

# Generate visualizations
python visualization.py
```

### 3. View Results
- Check `results/logs/` for experiment logs
- View `results/figures/` for generated plots
- Examine `results/tables/` for numerical results

## 📊 Experiments

### Experiment 1: Overfitting Analysis
- Train models with intentional overfitting
- Compare interpretability metrics across different overfitting levels
- Analyze relationship between accuracy and interpretability quality

### Experiment 2: Cross-Dataset Validation
- Train on one dataset, test on another
- Evaluate interpretation consistency across domains
- Measure interpretation drift between in-domain and out-domain

### Experiment 3: Model Comparison
- Compare transformer models (BERT, RoBERTa) with baseline models
- Analyze interpretability differences between model architectures
- Evaluate computational efficiency vs. interpretability trade-offs

## 🔍 Interpretability Metrics

### Faithfulness
Measures whether important features actually impact model predictions.

### Consistency
Evaluates consistency of interpretations across similar examples.

### Sparsity
Measures concentration of importance scores.

### Intuitiveness
Compares interpretations against human intuition (when available).

## 📈 Expected Results

1. **Overfitted models show poor interpretability metrics**
2. **SHAP values become less intuitive with increased overfitting**
3. **Cross-dataset validation reveals interpretation instability**
4. **Baseline models may show more consistent interpretations**

## 🛠️ Development

### Code Standards
- Python 3.9+ with type hints
- PEP 8 style guide
- Google-style docstrings
- Comprehensive error handling

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test categories
pytest tests/test_models.py
pytest tests/test_interpretability.py
```

### Memory Management
- Gradient checkpointing for large models
- Batch size auto-tuning for Apple Silicon
- MPS fallback to CPU when needed

## 🐛 Troubleshooting

### Common Issues

1. **MPS Errors**
   ```python
   # Automatic fallback to CPU
   from src.utils.device_utils import get_device
   device = get_device()  # Handles MPS errors automatically
   ```

2. **Memory Issues**
   ```python
   # Reduce batch size
   from src.utils.device_utils import get_optimal_batch_size
   optimal_batch_size = get_optimal_batch_size(model, input_shape)
   ```

3. **SHAP Compatibility**
   ```python
   # Always use CPU for SHAP
   device = torch.device("cpu")
   ```

### Performance Optimization

- Use gradient checkpointing for large models
- Implement batch size auto-tuning
- Cache processed datasets
- Use lazy loading for large datasets

## 📚 Dependencies

### Core Libraries
- **PyTorch** (≥2.0.0) - MPS support for Apple Silicon
- **Transformers** (≥4.30.0) - HuggingFace models
- **SHAP** (≥0.42.0) - Interpretability analysis
- **Scikit-learn** (≥1.3.0) - Baseline models
- **Datasets** (≥2.14.0) - Data loading

### Visualization
- **Matplotlib** (≥3.7.0) - Plotting
- **Seaborn** (≥0.12.0) - Statistical plots
- **Plotly** (≥5.15.0) - Interactive plots

### Utilities
- **PyYAML** (≥6.0) - Configuration
- **TQDM** (≥4.65.0) - Progress bars
- **MLflow** (≥2.5.0) - Experiment tracking

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the logs in `results/logs/`
3. Open an issue on GitHub

## 🔬 Research Context

This project contributes to the growing field of interpretable machine learning by:
- Investigating the relationship between overfitting and interpretability
- Providing empirical evidence for interpretation reliability
- Offering practical guidelines for model evaluation
- Contributing to the development of robust interpretability methods

## 📖 References

- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions.
- Molnar, C. (2020). Interpretable Machine Learning.
- Ribeiro, M. T., et al. (2016). "Why should I trust you?" Explaining the predictions of any classifier.

---

**Note**: This project is specifically designed for Apple Silicon MacBooks. For other hardware configurations, modifications to the device management and memory optimization may be required.