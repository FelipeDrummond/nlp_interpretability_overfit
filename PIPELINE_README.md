# NLP Interpretability Pipeline Runner

This directory contains scripts to run the complete NLP interpretability experiments with all models on all datasets.

## 🚀 Quick Start

### 1. Full Pipeline (All Models, All Datasets)
```bash
./run_full_pipeline.sh
```

This will run:
- **5 Models**: BERT, RoBERTa, DistilBERT, Llama-3.2-1B, Bag-of-Words
- **3 Datasets**: IMDB, Amazon Polarity, Yelp Polarity
- **Total**: 15 experiments

### 2. Quick Test (Single Model, Single Dataset)
```bash
# Test BERT on IMDB (default)
./run_quick_test.sh

# Test specific model and dataset
./run_quick_test.sh roberta-base amazon_polarity
./run_quick_test.sh meta-llama/Llama-3.2-1B yelp_polarity
```

## 📋 Available Models

| Model | Type | Parameters | Memory Usage |
|-------|------|------------|--------------|
| `bag-of-words-tfidf` | Baseline | ~1M | Low |
| `bert-base-uncased` | Transformer | 110M | Medium |
| `roberta-base` | Transformer | 125M | Medium |
| `distilbert-base-uncased` | Transformer | 66M | Low-Medium |
| `meta-llama/Llama-3.2-1B` | Transformer | 1.24B | High |

## 📊 Available Datasets

| Dataset | Train | Val | Test | Total |
|---------|-------|-----|------|-------|
| `imdb` | 25,000 | 2,500 | 2,500 | 30,000 |
| `amazon_polarity` | 3,600,000 | 400,000 | 400,000 | 4,400,000 |
| `yelp_polarity` | 560,000 | 38,000 | 38,000 | 636,000 |

## 📁 Output Structure

```
results/
├── logs/
│   ├── full_pipeline_YYYYMMDD_HHMMSS.log
│   ├── bert-base-uncased_imdb_YYYYMMDD_HHMMSS.log
│   └── ...
├── models/
│   ├── bert-base-uncased_imdb_epoch2.pt
│   ├── roberta-base_amazon_polarity_epoch2.pt
│   └── ...
├── metrics/
│   ├── bert-base-uncased_imdb_YYYYMMDD_HHMMSS_history.json
│   └── ...
├── figures/
│   ├── bert-base-uncased_imdb_YYYYMMDD_HHMMSS_curves.png
│   └── ...
└── configs/
    ├── bert-base-uncased_imdb_YYYYMMDD_HHMMSS_config.yaml
    └── ...
```

## ⏱️ Estimated Runtime

| Model | IMDB | Amazon | Yelp | Total |
|-------|------|--------|------|-------|
| Bag-of-Words | 30s | 2m | 1m | 3.5m |
| BERT | 5m | 20m | 8m | 33m |
| RoBERTa | 6m | 25m | 10m | 41m |
| DistilBERT | 3m | 12m | 5m | 20m |
| Llama-3.2-1B | 15m | 60m | 25m | 100m |
| **Total** | **29m** | **119m** | **49m** | **197m (3.3h)** |

## 🔧 Prerequisites

1. **Virtual Environment**: Make sure `venv/` exists
2. **Dependencies**: Install with `pip install -r requirements.txt`
3. **HuggingFace Auth**: For Llama models, run `huggingface-cli login`
4. **Data**: Script will auto-prepare if missing

## 📊 Monitoring Progress

The full pipeline script provides:
- ✅ Real-time progress tracking
- 📊 Success/failure counts
- ⏱️ Time estimates and actual runtime
- 📝 Individual experiment logs
- 🎯 Overall pipeline summary

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**:
   - Llama models may need smaller batch sizes
   - Script will auto-adjust batch sizes
   - Check available RAM (need ~16GB for Llama)

2. **HuggingFace Authentication**:
   ```bash
   huggingface-cli login
   # Accept Llama license at: https://huggingface.co/meta-llama/Llama-3.2-1B
   ```

3. **Missing Data**:
   - Script auto-prepares data
   - Manual: `python src/prepare_data.py`

4. **Model Not Found**:
   - Check model name spelling
   - Verify model is in `config.yaml`

### Log Files

- **Main Log**: `results/logs/full_pipeline_YYYYMMDD_HHMMSS.log`
- **Individual Logs**: `results/logs/{model}_{dataset}_YYYYMMDD_HHMMSS.log`
- **Error Details**: Check individual log files for specific errors

## 🎯 Expected Results

After successful completion, you should have:
- **15 trained models** (5 models × 3 datasets)
- **Training curves** for each experiment
- **Performance metrics** (accuracy, loss, overfitting gap)
- **Memory usage reports** for transformer models
- **Comprehensive logs** for analysis

## 🔄 Resume Failed Experiments

If some experiments fail, you can:
1. Check individual log files for errors
2. Fix the issues (auth, memory, etc.)
3. Re-run specific experiments:
   ```bash
   ./run_quick_test.sh {model} {dataset}
   ```

## 📈 Next Steps

After pipeline completion:
1. **Analyze Results**: Check `results/metrics/` for performance data
2. **Compare Models**: Use training curves in `results/figures/`
3. **Interpretability**: Run SHAP analysis on trained models
4. **Overfitting Analysis**: Compare train vs validation performance

---

**Happy Experimenting! 🧪✨**
