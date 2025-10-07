#!/bin/bash

# Quick Test Runner - Tests one model on one dataset
# Usage: ./run_quick_test.sh [model] [dataset]

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default values
MODEL="${1:-bert-base-uncased}"
DATASET="${2:-imdb}"

echo -e "${BLUE}🧪 Quick Test: $MODEL on $DATASET${NC}"
echo "=================================="

# Activate virtual environment
source venv/bin/activate

# Check if data exists
if [ ! -f "data/processed/${DATASET}_train.csv" ]; then
    echo "Preparing data..."
    python src/prepare_data.py
fi

# Run the experiment
echo "Running experiment..."
python train.py global.model="$MODEL" global.dataset="$DATASET"

echo -e "${GREEN}✅ Quick test completed!${NC}"
