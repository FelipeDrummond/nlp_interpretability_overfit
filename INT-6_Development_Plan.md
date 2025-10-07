# INT-6 Development Plan: SHAP Analysis Implementation

## Overview
This document outlines the development plan for implementing INT-6, which involves creating a comprehensive SHAP analysis system for evaluating model interpretability in the NLP overfitting study. The implementation will support both baseline and transformer models with proper Apple Silicon MPS optimization.

## Current State Analysis

### ✅ Already Implemented
- **Base Model Architecture**: `src/models/base.py` provides abstract base class
- **Model Implementations**: Both baseline (`src/models/baseline.py`) and transformer (`src/models/transformers.py`) models exist
- **Training Pipeline**: `train.py` supports both model types with overfitting strategy
- **Configuration**: `config.yaml` includes comprehensive interpretability settings
- **Dependencies**: SHAP library already in `requirements.txt`
- **Device Management**: `src/utils/device_utils.py` handles MPS/CPU device selection
- **Results Structure**: `results/` directory with organized subdirectories

### ❌ Missing Components
- **SHAP Analyzer**: No `src/interpretability/shap_analyzer.py` module
- **Interpretability Metrics**: No `src/interpretability/metrics.py` module
- **Visualization**: No `src/interpretability/visualization.py` module
- **Evaluation Integration**: No integration with training pipeline
- **Cross-Dataset Analysis**: No cross-dataset interpretability evaluation
- **Results Processing**: No automated results analysis and reporting

## Development Tasks

### Task 1: Create SHAP Analyzer (`src/interpretability/shap_analyzer.py`)
**Priority**: High  
**Estimated Time**: 3-4 hours

**Requirements**:
- Create `SHAPAnalyzer` class for model-agnostic SHAP analysis
- Support both baseline and transformer models
- Implement proper device handling (CPU for SHAP, MPS for training)
- Add explainer type selection (TreeExplainer, DeepExplainer, GradientExplainer)
- Implement background data sampling for efficiency

**Key Implementation Details**:
```python
class SHAPAnalyzer:
    def __init__(self, model, tokenizer=None, device='cpu', config=None):
        # Initialize with model and configuration
        # Set up appropriate explainer based on model type
        # Handle device placement (always CPU for SHAP)
    
    def analyze(self, texts, max_samples=100, background_samples=50):
        # Compute SHAP values for given texts
        # Handle batch processing for memory efficiency
        # Return SHAP values and feature names
    
    def get_explainer(self, model_type):
        # Select appropriate explainer (Tree, Deep, Gradient)
        # Handle model-specific requirements
```

### Task 2: Implement Interpretability Metrics (`src/interpretability/metrics.py`)
**Priority**: High  
**Estimated Time**: 4-5 hours

**Requirements**:
- Create `InterpretabilityMetrics` class with all required metrics
- Implement faithfulness, consistency, sparsity, and intuitiveness scores
- Add statistical significance testing
- Support both single-model and cross-model comparisons

**Key Implementation Details**:
```python
class InterpretabilityMetrics:
    @staticmethod
    def faithfulness_score(model, inputs, shap_values, method="removal"):
        # Measure if important features actually impact predictions
        # Implement feature removal and permutation methods
    
    @staticmethod
    def consistency_score(shap_values_1, shap_values_2, similarity_threshold=0.7):
        # Measure consistency across similar examples
        # Use cosine similarity or other distance metrics
    
    @staticmethod
    def sparsity_score(shap_values, threshold=0.01):
        # Measure concentration of importance scores
        # Calculate Gini coefficient or entropy-based metrics
    
    @staticmethod
    def intuitiveness_score(shap_values, human_annotations=None):
        # Compare against human intuition (when available)
        # Implement correlation with human importance ratings
```

### Task 3: Create Visualization Module (`src/interpretability/visualization.py`)
**Priority**: High  
**Estimated Time**: 3-4 hours

**Requirements**:
- Create publication-quality SHAP visualizations
- Implement summary plots, waterfall plots, and force plots
- Add cross-dataset comparison visualizations
- Support both baseline and transformer model outputs

**Key Implementation Details**:
```python
class InterpretabilityVisualizer:
    def __init__(self, config):
        # Initialize with visualization configuration
        # Set up color schemes and figure settings
    
    def create_summary_plot(self, shap_values, feature_names, max_features=20):
        # Create SHAP summary plot
        # Handle both baseline and transformer features
    
    def create_waterfall_plot(self, shap_values, feature_names, sample_idx=0):
        # Create waterfall plot for individual predictions
        # Show feature contributions step by step
    
    def create_cross_dataset_comparison(self, results_dict):
        # Compare interpretability across datasets
        # Create side-by-side visualizations
```

### Task 4: Integrate with Training Pipeline (`train.py`)
**Priority**: High  
**Estimated Time**: 2-3 hours

**Requirements**:
- Add SHAP analysis to training pipeline
- Implement post-training interpretability evaluation
- Add results saving and logging
- Handle both baseline and transformer models

**Key Implementation Details**:
```python
def run_interpretability_analysis(model, X_test, y_test, model_name, dataset_name, config):
    # Initialize SHAP analyzer
    # Compute SHAP values for test set
    # Calculate interpretability metrics
    # Generate visualizations
    # Save results to results/interpretability/
```

### Task 5: Create Cross-Dataset Evaluation (`src/evaluation/cross_validation.py`)
**Priority**: Medium  
**Estimated Time**: 3-4 hours

**Requirements**:
- Implement leave-one-dataset-out validation
- Evaluate interpretability consistency across domains
- Measure interpretation drift between datasets
- Create cross-dataset comparison reports

**Key Implementation Details**:
```python
class CrossDatasetEvaluator:
    def __init__(self, models, datasets, config):
        # Initialize with trained models and datasets
    
    def evaluate_interpretability_consistency(self):
        # Train on one dataset, evaluate on others
        # Measure interpretation consistency
        # Generate comparison reports
    
    def measure_interpretation_drift(self, model, dataset_pairs):
        # Measure how interpretations change across domains
        # Calculate drift metrics
```

### Task 6: Create Results Analysis Script (`analyze_results.py`)
**Priority**: Medium  
**Estimated Time**: 2-3 hours

**Requirements**:
- Create script to analyze all experiment results
- Generate comprehensive interpretability reports
- Create publication-ready tables and figures
- Implement statistical significance testing

**Key Implementation Details**:
```python
def analyze_all_results(results_dir="results"):
    # Load all experiment results
    # Compute aggregate interpretability metrics
    # Generate comparison tables
    # Create summary visualizations
    # Export results for publication
```

### Task 7: Add Unit Tests (`tests/test_interpretability.py`)
**Priority**: Medium  
**Estimated Time**: 2-3 hours

**Requirements**:
- Test SHAP analyzer with both model types
- Validate interpretability metrics calculations
- Test visualization generation
- Test cross-dataset evaluation

**Key Implementation Details**:
```python
def test_shap_analyzer():
    # Test SHAP computation for both baseline and transformer models
    # Validate SHAP values sum to model output difference
    # Test error handling and edge cases

def test_interpretability_metrics():
    # Test all metric calculations
    # Validate statistical significance testing
    # Test cross-model comparisons
```

## Implementation Order

1. **Phase 1**: Create core SHAP analyzer (`src/interpretability/shap_analyzer.py`)
2. **Phase 2**: Implement interpretability metrics (`src/interpretability/metrics.py`)
3. **Phase 3**: Create visualization module (`src/interpretability/visualization.py`)
4. **Phase 4**: Integrate with training pipeline (`train.py`)
5. **Phase 5**: Add cross-dataset evaluation (`src/evaluation/cross_validation.py`)
6. **Phase 6**: Create results analysis script (`analyze_results.py`)
7. **Phase 7**: Add comprehensive testing (`tests/test_interpretability.py`)

## Success Criteria

### ✅ Acceptance Criteria from INT-6
- [ ] `src/interpretability/shap_analyzer.py` created with SHAPAnalyzer class
- [ ] `src/interpretability/metrics.py` implements all interpretability metrics
- [ ] `src/interpretability/visualization.py` creates publication-quality plots
- [ ] SHAP analysis integrated into training pipeline
- [ ] Cross-dataset interpretability evaluation implemented
- [ ] Results analysis script generates comprehensive reports
- [ ] All visualizations saved to `results/interpretability/`
- [ ] Statistical significance testing implemented
- [ ] Both baseline and transformer models supported

### 🔧 Technical Requirements
- [ ] SHAP values computed on CPU (MPS compatibility)
- [ ] Background sampling for computational efficiency
- [ ] Batch processing for large datasets
- [ ] Proper error handling and logging
- [ ] Memory-efficient implementation
- [ ] Reproducible results with fixed seeds

## File Structure After Implementation

```
src/interpretability/
├── __init__.py
├── shap_analyzer.py          # 🆕 SHAP analysis core
├── metrics.py                # 🆕 Interpretability metrics
└── visualization.py          # 🆕 Visualization utilities

src/evaluation/
├── __init__.py
└── cross_validation.py       # 🆕 Cross-dataset evaluation

results/
├── models/
├── interpretability/         # 🆕 SHAP analysis results
│   ├── shap_values/
│   ├── metrics/
│   └── visualizations/
├── figures/
└── tables/

analyze_results.py            # 🆕 Results analysis script
tests/
└── test_interpretability.py  # 🆕 Interpretability tests
```

## Configuration Updates

### Additional Configuration Sections
```yaml
# Add to config.yaml
interpretability:
  # SHAP settings
  shap:
    max_samples: 100
    background_samples: 50
    explainer_type: "auto"  # auto, tree, deep, gradient
    device: "cpu"  # Always use CPU for SHAP
    
  # Metrics to compute
  metrics:
    faithfulness:
      enabled: true
      method: "removal"
      num_samples: 50
      
    consistency:
      enabled: true
      similarity_threshold: 0.7
      num_pairs: 100
      
    sparsity:
      enabled: true
      threshold: 0.01
      
    intuitiveness:
      enabled: false  # Requires human annotations
      annotation_file: null
      
  # Visualization settings
  visualization:
    max_features_display: 20
    figure_dpi: 300
    figure_format: "png"
    color_scheme: "viridis"
    save_plots: true
```

## Risk Mitigation

### Potential Issues
1. **Memory Constraints**: SHAP analysis can be memory-intensive
   - **Mitigation**: Implement batch processing and background sampling
2. **MPS Compatibility**: SHAP may not work with MPS
   - **Mitigation**: Always use CPU for SHAP analysis
3. **Computational Cost**: SHAP analysis can be slow
   - **Mitigation**: Use efficient explainers and limit sample sizes
4. **Model Compatibility**: Different models need different explainers
   - **Mitigation**: Implement automatic explainer selection

### Testing Strategy
1. **Unit Tests**: Test individual components (SHAP analyzer, metrics)
2. **Integration Tests**: Test full interpretability pipeline
3. **Performance Tests**: Test memory usage and computation time
4. **Cross-Model Tests**: Test with both baseline and transformer models

## Dependencies

### New Dependencies Required
```txt
# Already in requirements.txt
shap>=0.42.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.10.0  # For statistical tests
```

### Additional Dependencies (if needed)
```txt
plotly>=5.15.0  # For interactive visualizations
scikit-learn>=1.3.0  # For additional metrics
```

## Timeline

- **Day 1**: Tasks 1-2 (SHAP analyzer and metrics)
- **Day 2**: Tasks 3-4 (Visualization and training integration)
- **Day 3**: Tasks 5-6 (Cross-dataset evaluation and results analysis)
- **Day 4**: Task 7 (Testing and validation)

**Total Estimated Time**: 12-16 hours

## Next Steps

1. Start with Task 1: Create `src/interpretability/shap_analyzer.py`
2. Implement basic SHAP analysis for both model types
3. Test with existing trained models
4. Iterate and add metrics and visualizations
5. Integrate with training pipeline
6. Add comprehensive testing

This plan provides a comprehensive roadmap for implementing INT-6 while building on the existing codebase foundation and ensuring compatibility with the Apple Silicon M4 Pro hardware constraints. The implementation will enable comprehensive interpretability analysis for the overfitting study, supporting both baseline and transformer models with proper SHAP integration.
