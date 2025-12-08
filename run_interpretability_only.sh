#!/bin/bash

# Interpretability-Only Runner
# Mirrors run_full_pipeline.sh style but skips training. Uses run_interpretability_only.py.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$SCRIPT_DIR/venv"
LOG_DIR="$SCRIPT_DIR/results/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="$LOG_DIR/interpretability_only_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

print() { echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MAIN_LOG"; }

check_venv() {
    if [ ! -d "$VENV_PATH" ]; then
        print "❌ Virtual environment not found at $VENV_PATH"
        print "Create it and install deps: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
        exit 1
    fi
}

activate_venv() {
    print "Activating virtual environment..."
    # shellcheck disable=SC1090
    source "$VENV_PATH/bin/activate"
    print "✅ Virtual environment activated"
}

check_data() {
    # Ensure processed data exists for all datasets declared in config
    local data_dir="$SCRIPT_DIR/data/processed"
    local files=("imdb_test.csv" "amazon_polarity_test.csv" "yelp_polarity_test.csv")
    local missing=0
    for f in "${files[@]}"; do
        if [ ! -f "$data_dir/$f" ]; then
            print "⚠️  Missing $f in $data_dir"
            missing=1
        fi
    done
    if [ "$missing" -eq 1 ]; then
        print "Preparing data via src/prepare_data.py ..."
        python "$SCRIPT_DIR/src/prepare_data.py" | tee -a "$MAIN_LOG"
    else
        print "✅ All required processed test files present"
    fi
}

run_model() {
    local model_name="$1"
    print "Starting interpretability for: $model_name"
    if python "$SCRIPT_DIR/run_interpretability_only.py" --model "$model_name" >> "$MAIN_LOG" 2>&1; then
        print "✅ Completed: $model_name"
    else
        print "❌ Failed: $model_name (see $MAIN_LOG)"
    fi
}

main() {
    print "🚀 Interpretability-Only Pipeline"
    print "Log: $MAIN_LOG"
    cd "$SCRIPT_DIR"
    check_venv
    activate_venv
    check_data

    # Using MultiBERTs models to compare different initialization seeds
    local models=(
        "bag-of-words-tfidf"
        "multiberts-seed_0"
        "multiberts-seed_1"
        "multiberts-seed_2"
        "multiberts-seed_3"
        "multiberts-seed_4"
        # Add more seeds as needed: seed_5 through seed_24
    )

    for m in "${models[@]}"; do
        run_model "$m"
    done

    print "🎉 Interpretability-only run completed"
}

trap 'print "Interrupted"; exit 130' INT TERM
main "$@"


