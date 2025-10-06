#!/bin/bash

# Full Pipeline Runner for NLP Interpretability Experiments
# This script runs all models on all datasets with comprehensive logging

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$SCRIPT_DIR/venv"
LOG_DIR="/mnt/volume/results/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="$LOG_DIR/full_pipeline_${TIMESTAMP}.log"

# Create log directory
mkdir -p "$LOG_DIR"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$MAIN_LOG"
}

print_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ✅ $1${NC}" | tee -a "$MAIN_LOG"
}

print_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  $1${NC}" | tee -a "$MAIN_LOG"
}

print_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ❌ $1${NC}" | tee -a "$MAIN_LOG"
}

print_header() {
    echo -e "${PURPLE}$1${NC}" | tee -a "$MAIN_LOG"
}

# Function to run a single model-dataset combination
run_experiment() {
    local model_type="$1"
    local dataset_name="$2"
    local start_time=$(date +%s)
    
    print_status "Starting experiment: $model_type on $dataset_name"
    
    # Create experiment log file (sanitize model name for file path)
    local sanitized_model=$(echo "$model_type" | sed 's/[\/]/_/g')
    local exp_log="$LOG_DIR/${sanitized_model}_${dataset_name}_${TIMESTAMP}.log"
    
    # Run the experiment
    if python train.py global.model="$model_type" > "$exp_log" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        print_success "Completed $model_type on $dataset_name in ${duration}s"
        return 0
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        print_error "Failed $model_type on $dataset_name after ${duration}s"
        print_error "Check log: $exp_log"
        return 1
    fi
}

# Function to check if virtual environment exists
check_venv() {
    if [ ! -d "$VENV_PATH" ]; then
        print_error "Virtual environment not found at $VENV_PATH"
        print_error "Please create a virtual environment first:"
        print_error "  python3 -m venv venv"
        print_error "  source venv/bin/activate"
        print_error "  pip install -r requirements.txt"
        exit 1
    fi
}

# Function to activate virtual environment
activate_venv() {
    print_status "Activating virtual environment..."
    source "$VENV_PATH/bin/activate"
    print_success "Virtual environment activated"
}

# Function to check if data is prepared
check_data() {
    print_status "Checking if data is prepared..."
    
    local data_dir="/mnt/volume/data/processed"
    local required_files=(
        "imdb_train.csv"
        "imdb_val.csv" 
        "imdb_test.csv"
        "amazon_polarity_train.csv"
        "amazon_polarity_val.csv"
        "amazon_polarity_test.csv"
        "yelp_polarity_train.csv"
        "yelp_polarity_val.csv"
        "yelp_polarity_test.csv"
    )
    
    local missing_files=()
    for file in "${required_files[@]}"; do
        if [ ! -f "$data_dir/$file" ]; then
            missing_files+=("$file")
        fi
    done
    
    if [ ${#missing_files[@]} -gt 0 ]; then
        print_warning "Missing data files:"
        for file in "${missing_files[@]}"; do
            print_warning "  - $file"
        done
        print_status "Running data preparation..."
        python src/prepare_data.py
        print_success "Data preparation completed"
    else
        print_success "All required data files found"
    fi
}

# Function to show pipeline summary
show_summary() {
    local total_experiments=$1
    local successful=$2
    local failed=$3
    local total_time=$4
    
    print_header "=" * 80
    print_header "PIPELINE EXECUTION SUMMARY"
    print_header "=" * 80
    print_status "Total Experiments: $total_experiments"
    print_success "Successful: $successful"
    if [ $failed -gt 0 ]; then
        print_error "Failed: $failed"
    else
        print_success "Failed: $failed"
    fi
    print_status "Success Rate: $(( successful * 100 / total_experiments ))%"
    print_status "Total Time: ${total_time}s ($(( total_time / 60 ))m $(( total_time % 60 ))s)"
    print_status "Main Log: $MAIN_LOG"
    print_header "=" * 80
}

# Main execution
main() {
    print_header "🚀 NLP Interpretability Full Pipeline Runner"
    print_header "=" * 80
    print_status "Starting full pipeline execution at $(date)"
    print_status "Log file: $MAIN_LOG"
    
    # Change to script directory
    cd "$SCRIPT_DIR"
    
    # Pre-flight checks
    check_venv
    activate_venv
    check_data
    
    # Define models and datasets
    local models=(
        "bag-of-words-tfidf"
        "bert-base-uncased"
        "roberta-base"
        "distilbert-base-uncased"
        "meta-llama/Llama-3.2-1B"
    )
    
    local datasets=(
        "imdb"
        "amazon_polarity"
        "yelp_polarity"
    )
    
    # Calculate total experiments (one per model)
    local total_experiments=${#models[@]}
    local successful=0
    local failed=0
    local pipeline_start_time=$(date +%s)
    
    print_status "Running $total_experiments experiments across ${#models[@]} models (each training on all ${#datasets[@]} datasets)"
    print_status "Models: ${models[*]}"
    print_status "Datasets: ${datasets[*]}"
    
    # Run all experiments (one per model, each trains on all datasets)
    local experiment_num=0
    for model in "${models[@]}"; do
        experiment_num=$((experiment_num + 1))
        print_header "EXPERIMENT $experiment_num/${#models[@]}: $model on all datasets"
        
        if run_experiment "$model" "all_datasets"; then
            successful=$((successful + 1))
        else
            failed=$((failed + 1))
        fi
        
        print_status "Progress: $experiment_num/${#models[@]} completed"
        echo "" | tee -a "$MAIN_LOG"
    done
    
    # Calculate total time
    local pipeline_end_time=$(date +%s)
    local total_time=$((pipeline_end_time - pipeline_start_time))
    
    # Show summary
    show_summary $total_experiments $successful $failed $total_time
    
    # Final status
    if [ $failed -eq 0 ]; then
        print_success "🎉 All experiments completed successfully!"
        exit 0
    else
        print_warning "⚠️  Some experiments failed. Check individual log files for details."
        exit 1
    fi
}

# Handle script interruption
trap 'print_error "Pipeline interrupted by user"; exit 130' INT TERM

# Run main function
main "$@"
