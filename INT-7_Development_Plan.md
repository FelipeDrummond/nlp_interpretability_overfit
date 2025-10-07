# INT-7 Development Plan: RoBERTa and Llama Models Implementation

## Overview
This document outlines the development plan for implementing INT-7, which involves extending the existing transformer model infrastructure to support RoBERTa and Llama models. Building on the foundation established in INT-5 (BERT implementation) and INT-6 (SHAP analysis), this ticket will add two additional transformer architectures to enable comprehensive model comparison in the NLP interpretability overfitting study.

## Current State Analysis

### ✅ Already Implemented (from INT-5 and INT-6)
- **Base Model Architecture**: `src/models/base.py` provides abstract base class
- **BERT Model Implementation**: `src/models/transformers.py` contains BERTModel class
- **Training Pipeline**: `train.py` supports transformer models with overfitting strategy
- **Configuration**: `config.yaml` includes RoBERTa and DistilBERT configurations
- **Device Management**: `src/utils/device_utils.py` handles MPS/CPU device selection
- **Model Factory**: `create_transformer_model()` function exists but only supports BERT
- **Dependencies**: All required HuggingFace libraries are in `requirements.txt`

### ❌ Missing Components for INT-7
- **RoBERTa Model Class**: No `RoBERTaModel` class implementation
- **Llama Model Class**: No `LlamaModel` class implementation
- **Model Factory Extension**: Factory function doesn't support RoBERTa/Llama
- **Configuration Updates**: Need Llama-specific configurations
- **Memory Optimization**: Llama models require special memory handling
- **Testing**: No tests for new model implementations

## Development Tasks

### Task 1: Implement RoBERTa Model Class
**Priority**: High  
**Estimated Time**: 2-3 hours

**Requirements**:
- Create `RoBERTaModel` class inheriting from `BaseModel`
- Implement HuggingFace `roberta-base` model loading
- Add proper MPS device handling with CPU fallback
- Support for RoBERTa-specific tokenization (no token_type_ids)
- Implement forward pass and training methods

**Key Implementation Details**:
```python
class RoBERTaModel(BaseModel):
    def __init__(self, config: Dict[str, Any], model_name: str = "RoBERTaModel"):
        # Initialize RoBERTa tokenizer and model
        # Set up device handling
        # Configure for overfitting (no dropout)
    
    def tokenize_texts(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        # RoBERTa-specific tokenization (no token_type_ids)
        # Handle padding, truncation, and attention masks
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # Implement training loop with gradient accumulation
        # Handle MPS/CPU device switching
        # Track training history
```

### Task 2: Implement Llama Model Class
**Priority**: High  
**Estimated Time**: 3-4 hours

**Requirements**:
- Create `LlamaModel` class inheriting from `BaseModel`
- Implement HuggingFace `meta-llama/Llama-2-7b-hf` model loading
- Add special memory optimization for large Llama models
- Implement gradient checkpointing and memory-efficient training
- Handle Llama-specific tokenization and attention patterns

**Key Implementation Details**:
```python
class LlamaModel(BaseModel):
    def __init__(self, config: Dict[str, Any], model_name: str = "LlamaModel"):
        # Initialize Llama tokenizer and model
        # Set up memory optimization (gradient checkpointing)
        # Configure for overfitting with memory constraints
    
    def tokenize_texts(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        # Llama-specific tokenization
        # Handle special tokens and attention patterns
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # Implement memory-efficient training
        # Use gradient accumulation and checkpointing
        # Handle potential OOM errors gracefully
```

### Task 3: Extend Model Factory Function
**Priority**: High  
**Estimated Time**: 1 hour

**Requirements**:
- Update `create_transformer_model()` to support RoBERTa and Llama
- Add model type detection and routing
- Update `TRANSFORMER_MODELS` dictionary
- Ensure backward compatibility with existing BERT implementation

**Key Implementation Details**:
```python
def create_transformer_model(model_type: str, config: Dict[str, Any]) -> BaseModel:
    if model_type == "bert-base-uncased":
        return BERTModel(config)
    elif model_type == "roberta-base":
        return RoBERTaModel(config)
    elif model_type == "meta-llama/Llama-2-7b-hf":
        return LlamaModel(config)
    else:
        raise ValueError(f"Unknown transformer model type: {model_type}")

# Update available models
TRANSFORMER_MODELS = {
    'bert-base-uncased': BERTModel,
    'roberta-base': RoBERTaModel,
    'meta-llama/Llama-2-7b-hf': LlamaModel,
}
```

### Task 4: Update Configuration Files
**Priority**: Medium  
**Estimated Time**: 1 hour

**Requirements**:
- Add Llama-specific configuration to `config.yaml`
- Update model selection lists
- Add memory optimization settings for Llama
- Ensure all models have consistent configuration structure

**Key Configuration Updates**:
```yaml
models:
  transformer_models:
    # Existing BERT and RoBERTa configs...
    
    meta-llama/Llama-2-7b-hf:
      type: "transformer"
      model_name: "meta-llama/Llama-2-7b-hf"
      tokenizer_name: "meta-llama/Llama-2-7b-hf"
      num_labels: 2
      max_length: 512
      batch_size: 2  # Smaller batch size for memory
      learning_rate: 1e-4  # Lower LR for stability
      num_epochs: 1  # Fewer epochs due to size
      gradient_accumulation_steps: 8  # More accumulation
      gradient_checkpointing: true
      use_cache: false  # Disable KV cache for memory
      
  model_selection:
    selected_models:
      - "bert-base-uncased"
      - "roberta-base"
      - "meta-llama/Llama-2-7b-hf"
      - "bag-of-words-tfidf"
```

### Task 5: Add Memory Optimization for Large Models
**Priority**: High  
**Estimated Time**: 2-3 hours

**Requirements**:
- Implement gradient checkpointing for Llama models
- Add memory monitoring and OOM error handling
- Implement dynamic batch size adjustment
- Add memory-efficient attention mechanisms

**Key Implementation Details**:
```python
def setup_memory_optimization(model, device, config):
    if "llama" in model.config.name_or_path.lower():
        # Enable gradient checkpointing
        model.gradient_checkpointing_enable()
        
        # Configure for memory efficiency
        model.config.use_cache = False
        
        # Set up memory monitoring
        if device.type == "mps":
            torch.mps.empty_cache()
    
    return model

def handle_oom_error(model, batch_size, min_batch_size=1):
    """Handle out-of-memory errors by reducing batch size."""
    if batch_size > min_batch_size:
        new_batch_size = max(min_batch_size, batch_size // 2)
        logger.warning(f"OOM error, reducing batch size to {new_batch_size}")
        return new_batch_size
    else:
        raise RuntimeError("Cannot reduce batch size further")
```

### Task 6: Update Training Pipeline
**Priority**: Medium  
**Estimated Time**: 1-2 hours

**Requirements**:
- Update `train.py` to handle new model types
- Add model-specific training configurations
- Implement memory monitoring for large models
- Add progress tracking for long-running Llama training

**Key Implementation Details**:
```python
def train_model(model_type: str, X_train, y_train, X_val, y_val, model_config):
    # Create model with memory optimization
    model = create_model(model_type, model_config)
    
    # Apply memory optimization for large models
    if "llama" in model_type.lower():
        model = setup_memory_optimization(model, model.device, model_config)
    
    # Train with progress tracking
    training_history = model.fit(X_train, y_train, X_val, y_val)
    
    return model
```

### Task 7: Add Comprehensive Testing
**Priority**: High  
**Estimated Time**: 2-3 hours

**Requirements**:
- Test RoBERTa model creation and initialization
- Test Llama model creation (with memory constraints)
- Verify tokenization pipelines work correctly
- Test training loops with different model types
- Validate MPS/CPU fallback mechanisms

**Test Commands**:
```bash
# Test RoBERTa model creation
python -c "from src.models.transformers import RoBERTaModel; print('RoBERTa model created successfully')"

# Test Llama model creation (if memory allows)
python -c "from src.models.transformers import LlamaModel; print('Llama model created successfully')"

# Test training pipeline
python train.py model=roberta-base dataset=imdb
python train.py model=meta-llama/Llama-2-7b-hf dataset=imdb

# Test with different datasets
python train.py model=roberta-base dataset=amazon_polarity
python train.py model=roberta-base dataset=yelp_polarity
```

## Implementation Order

1. **Phase 1**: Implement RoBERTa model class (`src/models/transformers.py`)
2. **Phase 2**: Implement Llama model class with memory optimization
3. **Phase 3**: Update model factory function and configuration
4. **Phase 4**: Add memory optimization utilities
5. **Phase 5**: Update training pipeline
6. **Phase 6**: Add comprehensive testing

## Success Criteria

### ✅ Acceptance Criteria from INT-7
- [ ] `RoBERTaModel` class created inheriting from BaseModel
- [ ] `LlamaModel` class created with memory optimization
- [ ] HuggingFace `roberta-base` loads successfully and moves to MPS device
- [ ] HuggingFace `meta-llama/Llama-2-7b-hf` loads with memory optimization
- [ ] Tokenization implemented for both models (RoBERTa: no token_type_ids, Llama: special tokens)
- [ ] Model factory function supports all three transformer types
- [ ] Commands work: `python train.py model=roberta-base dataset=imdb`
- [ ] Commands work: `python train.py model=meta-llama/Llama-2-7b-hf dataset=imdb`
- [ ] Model checkpoints saved to `results/models/`
- [ ] All three transformer models can be trained from the same `train.py` script

### 🔧 Technical Requirements
- [ ] RoBERTa tokenizer max_length: 512, no token_type_ids
- [ ] Llama tokenizer max_length: 512, special token handling
- [ ] Batch size: 8 for BERT/RoBERTa, 2 for Llama (with gradient accumulation)
- [ ] Learning rate: 1e-3 for BERT/RoBERTa, 1e-4 for Llama
- [ ] Gradient checkpointing enabled for Llama models
- [ ] Memory monitoring and OOM error handling
- [ ] MPS compatibility with CPU fallback

## Risk Mitigation

### Potential Issues
1. **Memory Constraints**: Llama models are very large (7B parameters)
   - **Mitigation**: Implement gradient checkpointing, small batch sizes, memory monitoring
2. **MPS Compatibility**: Large models may not fit in MPS memory
   - **Mitigation**: Implement automatic CPU fallback, memory monitoring
3. **Training Time**: Llama models will take much longer to train
   - **Mitigation**: Use fewer epochs, gradient accumulation, progress tracking
4. **Model Loading**: Llama models require special permissions from HuggingFace
   - **Mitigation**: Add error handling, provide alternative smaller models

### Testing Strategy
1. **Unit Tests**: Test individual model classes and tokenization
2. **Integration Tests**: Test full training pipeline with each model
3. **Memory Tests**: Test memory usage and OOM handling
4. **Performance Tests**: Test training speed and convergence

## Dependencies

### New Dependencies Required
```txt
# Already in requirements.txt
transformers>=4.30.0  # Supports RoBERTa and Llama
torch>=2.0.0  # MPS support
accelerate>=0.20.0  # For memory optimization
```

### Additional Dependencies (if needed)
```txt
bitsandbytes>=0.41.0  # For quantization (optional)
```

## File Structure After Implementation

```
src/models/
├── __init__.py
├── base.py                    # ✅ Already exists
├── baseline.py               # ✅ Already exists
└── transformers.py           # 🔄 Extended with RoBERTa and Llama

config.yaml                   # 🔄 Updated with Llama configs
train.py                      # 🔄 Updated for new models
tests/
└── test_models.py            # 🆕 Tests for new models
```

## Configuration Updates

### Additional Configuration Sections
```yaml
# Add to config.yaml
models:
  transformer_models:
    # Existing BERT and RoBERTa configs...
    
    meta-llama/Llama-2-7b-hf:
      type: "transformer"
      model_name: "meta-llama/Llama-2-7b-hf"
      tokenizer_name: "meta-llama/Llama-2-7b-hf"
      num_labels: 2
      max_length: 512
      batch_size: 2
      learning_rate: 1e-4
      num_epochs: 1
      gradient_accumulation_steps: 8
      gradient_checkpointing: true
      use_cache: false
      
  # Memory optimization for large models
  memory_optimization:
    llama:
      gradient_checkpointing: true
      use_cache: false
      max_memory_usage: 0.9
      batch_size_auto_adjust: true
      min_batch_size: 1
```

## Timeline

- **Day 1**: Tasks 1-2 (RoBERTa and Llama model classes)
- **Day 2**: Tasks 3-4 (Factory function and configuration updates)
- **Day 3**: Tasks 5-6 (Memory optimization and training pipeline)
- **Day 4**: Task 7 (Testing and validation)

**Total Estimated Time**: 12-16 hours

## Next Steps

1. Start with Task 1: Implement `RoBERTaModel` class
2. Test RoBERTa implementation before moving to Llama
3. Implement Llama model with careful memory management
4. Update factory function and configuration
5. Add comprehensive testing
6. Validate all models work with existing training pipeline

## Notes for Implementation

### RoBERTa Specific Considerations
- RoBERTa doesn't use `token_type_ids` (unlike BERT)
- Uses SentencePiece tokenization
- Generally more robust than BERT for many tasks

### Llama Specific Considerations
- Very large model (7B parameters) requires careful memory management
- Uses LlamaTokenizer with special tokens
- May require HuggingFace Hub authentication
- Consider using smaller variants (e.g., Llama-2-7b-chat-hf) if memory is limited

### Memory Management Strategy
1. **Gradient Checkpointing**: Trade compute for memory
2. **Small Batch Sizes**: Use batch_size=2 with gradient accumulation
3. **CPU Offloading**: Move some operations to CPU if needed
4. **Memory Monitoring**: Track memory usage and adjust dynamically

This plan provides a comprehensive roadmap for implementing INT-7 while building on the existing codebase foundation and ensuring compatibility with the Apple Silicon M4 Pro hardware constraints. The implementation will enable comprehensive model comparison across different transformer architectures in the overfitting study.
