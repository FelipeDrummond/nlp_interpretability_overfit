# INT-5 Development Plan: BERT Model Implementation with MPS Optimization

## Overview
This document outlines the development plan for implementing INT-5, which involves creating a BERT model implementation with Apple Silicon MPS optimization and extending the training pipeline to support transformer models.

## Current State Analysis

### ✅ Already Implemented
- **Base Model Architecture**: `src/models/base.py` provides a solid abstract base class
- **Baseline Models**: `src/models/baseline.py` implements BagOfWordsModel with TF-IDF
- **Training Pipeline**: `train.py` has a working training loop for baseline models
- **Device Management**: `src/utils/device_utils.py` handles MPS/CPU device selection
- **Configuration**: `config.yaml` includes transformer model configurations
- **Data Pipeline**: `src/prepare_data.py` handles dataset preparation and preprocessing

### ❌ Missing Components
- **Transformer Models**: No `src/models/transformers.py` module exists
- **BERT Implementation**: No BERTModel class
- **Tokenization Pipeline**: No HuggingFace tokenizer integration
- **Gradient Accumulation**: Not implemented in training loop
- **Model Factory**: No factory function for creating transformer models
- **Training Extension**: `train.py` only supports baseline models

## Development Tasks

### Task 1: Create BERTModel Class (`src/models/transformers.py`)
**Priority**: High  
**Estimated Time**: 2-3 hours

**Requirements**:
- Create `BERTModel` class inheriting from `BaseModel`
- Implement HuggingFace `bert-base-uncased` model loading
- Add proper MPS device handling with CPU fallback
- Support for tokenization with max_length=512
- Implement forward pass and training methods

**Key Implementation Details**:
```python
class BERTModel(BaseModel):
    def __init__(self, config: Dict[str, Any], model_name: str = "BERTModel"):
        # Initialize tokenizer and model
        # Set up device handling
        # Configure for overfitting (no dropout, high LR)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # Implement training loop with gradient accumulation
        # Handle MPS/CPU device switching
        # Track training history
    
    def predict(self, X):
        # Implement inference with proper tokenization
        # Handle batch processing
```

### Task 2: Implement Tokenization Pipeline
**Priority**: High  
**Estimated Time**: 1-2 hours

**Requirements**:
- Integrate HuggingFace tokenizer for BERT
- Support padding, truncation, and attention masks
- Handle batch processing efficiently
- Cache tokenized data for performance

**Key Implementation Details**:
```python
def tokenize_texts(self, texts, max_length=512):
    # Tokenize with padding and truncation
    # Return input_ids, attention_masks, token_type_ids
    # Handle device placement
```

### Task 3: Extend Training Pipeline (`train.py`)
**Priority**: High  
**Estimated Time**: 2-3 hours

**Requirements**:
- Add support for transformer models in training loop
- Implement gradient accumulation (accumulation_steps=4)
- Add model type selection via command line arguments
- Handle different model types (baseline vs transformer)

**Key Implementation Details**:
```python
def create_model(model_type: str, config: DictConfig):
    if model_type.startswith('bert'):
        return create_transformer_model(model_type, config)
    else:
        return create_baseline_model(model_type, config)

def train_transformer_model(model, X_train, y_train, X_val, y_val, config):
    # Implement training loop with gradient accumulation
    # Handle MPS/CPU device switching
    # Track training metrics
```

### Task 4: Implement MPS Optimization and Fallback
**Priority**: High  
**Estimated Time**: 1-2 hours

**Requirements**:
- Add MPS-specific optimizations for BERT
- Implement automatic fallback to CPU on MPS errors
- Add proper logging for device switching
- Handle mixed precision training if MPS supports it

**Key Implementation Details**:
```python
def setup_mps_optimization(model, device):
    if device.type == "mps":
        # Enable gradient checkpointing
        # Configure for MPS-specific optimizations
        # Set up memory management
    else:
        # CPU-specific optimizations
```

### Task 5: Add Model Factory Functions
**Priority**: Medium  
**Estimated Time**: 1 hour

**Requirements**:
- Create `create_transformer_model()` function
- Update existing `create_baseline_model()` function
- Add model type detection and routing
- Support for different transformer variants

**Key Implementation Details**:
```python
def create_transformer_model(model_type: str, config: Dict[str, Any]) -> BaseModel:
    if model_type == "bert-base-uncased":
        return BERTModel(config)
    # Add other transformer models later

def create_model(model_type: str, config: Dict[str, Any]) -> BaseModel:
    if model_type in TRANSFORMER_MODELS:
        return create_transformer_model(model_type, config)
    else:
        return create_baseline_model(model_type, config)
```

### Task 6: Update Configuration and Dependencies
**Priority**: Medium  
**Estimated Time**: 30 minutes

**Requirements**:
- Add transformer-specific configuration sections
- Update `requirements.txt` with HuggingFace dependencies
- Add model selection configuration
- Configure memory management for M4 Pro

**Key Configuration Updates**:
```yaml
models:
  transformer_models:
    bert-base-uncased:
      type: "transformer"
      model_name: "bert-base-uncased"
      tokenizer_name: "bert-base-uncased"
      max_length: 512
      batch_size: 8
      gradient_accumulation_steps: 4
```

### Task 7: Testing and Validation
**Priority**: High  
**Estimated Time**: 2-3 hours

**Requirements**:
- Test BERT model creation and initialization
- Verify tokenization pipeline works correctly
- Test training loop with gradient accumulation
- Validate MPS/CPU fallback mechanisms
- Test model saving and loading

**Test Commands**:
```bash
# Test BERT model creation
python -c "from src.models.transformers import BERTModel; print('BERT model created successfully')"

# Test training pipeline
python train.py model=bert-base-uncased dataset=imdb

# Test with different datasets
python train.py model=bert-base-uncased dataset=amazon_polarity
python train.py model=bert-base-uncased dataset=yelp_polarity
```

## Implementation Order

1. **Phase 1**: Create `src/models/transformers.py` with BERTModel class
2. **Phase 2**: Implement tokenization pipeline and device handling
3. **Phase 3**: Extend `train.py` to support transformer models
4. **Phase 4**: Add model factory functions and configuration updates
5. **Phase 5**: Testing and validation

## Success Criteria

### ✅ Acceptance Criteria from INT-5
- [ ] `models/transformers.py` created with BERTModel class inheriting from BaseModel
- [ ] HuggingFace `bert-base-uncased` loads successfully and moves to MPS device
- [ ] Tokenization implemented with proper padding, truncation, and attention masks
- [ ] `preprocess.py` extended to handle BERT-specific preprocessing and caching
- [ ] Gradient accumulation implemented in `train.py` (accumulation_steps=4 for memory efficiency)
- [ ] MPS fallback to CPU implemented with try/except blocks and proper logging
- [ ] Command works: `python train.py --model bert --dataset imdb`
- [ ] Model checkpoint saved to `results/models/bert_imdb_epoch50.pt`
- [ ] Both BoW and BERT can be trained from the same `train.py` script

### 🔧 Technical Requirements
- [ ] Tokenizer max_length: 512
- [ ] Batch size: 8 (with gradient accumulation if needed)
- [ ] Learning rate: 1e-3 (high for overfitting)
- [ ] No dropout (set to 0.0 for overfitting)
- [ ] Mixed precision training if MPS supports it

## Risk Mitigation

### Potential Issues
1. **MPS Compatibility**: Some operations may not work on MPS
   - **Mitigation**: Implement robust CPU fallback
2. **Memory Constraints**: BERT models are memory-intensive
   - **Mitigation**: Use gradient accumulation and smaller batch sizes
3. **Tokenization Performance**: Large datasets may be slow to tokenize
   - **Mitigation**: Implement caching and batch processing
4. **Model Loading**: HuggingFace models may have compatibility issues
   - **Mitigation**: Test with different model versions and add error handling

### Testing Strategy
1. **Unit Tests**: Test individual components (tokenization, model creation)
2. **Integration Tests**: Test full training pipeline
3. **Device Tests**: Test MPS and CPU fallback scenarios
4. **Memory Tests**: Test with different batch sizes and accumulation steps

## Dependencies

### New Dependencies Required
```txt
transformers>=4.30.0
torch>=2.0.0
tokenizers>=0.13.0
```

### Existing Dependencies (Already in requirements.txt)
- PyTorch (for MPS support)
- HuggingFace datasets
- NumPy, Pandas
- Scikit-learn (for baseline models)

## File Structure After Implementation

```
src/models/
├── __init__.py
├── base.py                    # ✅ Already exists
├── baseline.py               # ✅ Already exists
└── transformers.py           # 🆕 New file for INT-5

src/utils/
├── device_utils.py           # ✅ Already exists
└── ...

train.py                      # 🔄 Modified for transformer support
config.yaml                   # 🔄 Updated with transformer configs
```

## Timeline

- **Day 1**: Tasks 1-2 (BERTModel class and tokenization)
- **Day 2**: Tasks 3-4 (Training pipeline and MPS optimization)
- **Day 3**: Tasks 5-6 (Factory functions and configuration)
- **Day 4**: Task 7 (Testing and validation)

**Total Estimated Time**: 8-12 hours

## Next Steps

1. Start with Task 1: Create `src/models/transformers.py`
2. Implement BERTModel class with proper inheritance
3. Add tokenization pipeline
4. Test basic functionality before moving to training pipeline
5. Iterate and test each component as it's implemented

This plan provides a comprehensive roadmap for implementing INT-5 while building on the existing codebase foundation and ensuring compatibility with the Apple Silicon M4 Pro hardware constraints.
