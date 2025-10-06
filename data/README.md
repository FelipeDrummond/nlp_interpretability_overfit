# Dataset Documentation

This directory contains the processed sentiment analysis datasets used in the NLP interpretability study. All datasets have been standardized to ensure consistent format and preprocessing across different sources.

## Dataset Format

All processed datasets follow a uniform format with the following columns:

- **`text`**: The review text (preprocessed and cleaned)
- **`label`**: Binary sentiment label (0 = negative, 1 = positive)

## Available Datasets

### 1. IMDB Movie Reviews (`imdb_*`)

**Source**: [HuggingFace Datasets - IMDB](https://huggingface.co/datasets/imdb)

**Description**: Large movie review dataset with binary sentiment classification. Contains 50,000 movie reviews from IMDB, split into 25,000 training and 25,000 test reviews.

**Preprocessing**:
- HTML tags removed
- URLs removed
- Whitespace normalized
- Text converted to lowercase
- Empty texts filtered out

**Files**:
- `imdb_train.csv`: Training set (20,000 samples after validation split)
- `imdb_val.csv`: Validation set (5,000 samples)
- `imdb_test.csv`: Test set (25,000 samples)

### 2. Amazon Product Reviews Polarity (`amazon_polarity_*`)

**Source**: [HuggingFace Datasets - Amazon Polarity](https://huggingface.co/datasets/amazon_polarity)

**Description**: Amazon product reviews with binary sentiment classification. Contains millions of Amazon product reviews with star ratings converted to binary sentiment.

**Preprocessing**:
- HTML tags removed
- URLs removed
- Whitespace normalized
- Text converted to lowercase
- Empty texts filtered out

**Files**:
- `amazon_polarity_train.csv`: Training set (3,200,000 samples after validation split)
- `amazon_polarity_val.csv`: Validation set (800,000 samples)
- `amazon_polarity_test.csv`: Test set (400,000 samples)

### 3. Yelp Review Polarity (`yelp_polarity_*`)

**Source**: [HuggingFace Datasets - Yelp Polarity](https://huggingface.co/datasets/yelp_polarity)

**Description**: Yelp business reviews with binary sentiment classification. Contains 598,000 Yelp business reviews with star ratings converted to binary sentiment.

**Preprocessing**:
- HTML tags removed
- URLs removed
- Whitespace normalized
- Text converted to lowercase
- Empty texts filtered out

**Files**:
- `yelp_polarity_train.csv`: Training set (478,400 samples after validation split)
- `yelp_polarity_val.csv`: Validation set (119,600 samples)
- `yelp_polarity_test.csv`: Test set (598,000 samples)

## Data Splits

All datasets are split using the following strategy:

1. **Training Set**: 80% of the original training data
2. **Validation Set**: 20% of the original training data (stratified by label)
3. **Test Set**: Original test set (unchanged)

The splits are created using a fixed random seed (42) to ensure reproducibility.

## Label Encoding

All datasets use consistent binary label encoding:
- `0`: Negative sentiment
- `1`: Positive sentiment

## Data Quality

The following quality checks are performed on all datasets:

- **Empty Text Removal**: Texts with zero length after preprocessing are removed
- **Duplicate Detection**: Duplicate texts are identified and logged
- **Label Distribution**: Class balance is verified and logged
- **Text Length Statistics**: Mean, median, min, and max text lengths are computed

## Usage

To load a processed dataset:

```python
import pandas as pd

# Load training data
train_data = pd.read_csv('/mnt/volume/data/processed/imdb_train.csv')

# Check the format
print(train_data.head())
print(f"Shape: {train_data.shape}")
print(f"Label distribution: {train_data['label'].value_counts()}")
```

## Reproducibility

All datasets are processed using the `src/prepare_data.py` script with the unified configuration in `config.yaml`. The processing is deterministic and reproducible across different environments.

To reprocess the datasets:

```bash
python src/prepare_data.py
```

## File Structure

```
/mnt/volume/data/
├── processed/                   # Processed datasets
│   ├── imdb_train.csv
│   ├── imdb_val.csv
│   ├── imdb_test.csv
│   ├── amazon_polarity_train.csv
│   ├── amazon_polarity_val.csv
│   ├── amazon_polarity_test.csv
│   ├── yelp_polarity_train.csv
│   ├── yelp_polarity_val.csv
│   └── yelp_polarity_test.csv
└── cache/                       # Cached raw datasets
    ├── imdb/
    ├── amazon_polarity/
    └── yelp_polarity/
```

## Notes

- All text preprocessing is designed to be minimal to preserve the original content while ensuring consistency
- The datasets are large, so consider using lazy loading or sampling for development
- For the interpretability study, smaller subsets may be used for faster experimentation
- All datasets are cached locally to avoid re-downloading on subsequent runs
