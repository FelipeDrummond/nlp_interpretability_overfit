#!/bin/bash

# Training and Interpretability Analysis Script
# This script trains all models except Llama, then runs interpretability analysis

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
LOG_DIR="$SCRIPT_DIR/results/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="$LOG_DIR/training_and_analysis_${TIMESTAMP}.log"

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
    
    local data_dir="$SCRIPT_DIR/data/processed"
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

# Function to run training for a single model
run_training() {
    local model_type="$1"
    local start_time=$(date +%s)
    
    print_status "Training model: $model_type"
    
    # Create experiment log file
    local sanitized_model=$(echo "$model_type" | sed 's/[\/]/_/g')
    local exp_log="$LOG_DIR/training_${sanitized_model}_${TIMESTAMP}.log"
    
    # Run the training
    if python train.py global.model="$model_type" > "$exp_log" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        print_success "Completed training $model_type in ${duration}s"
        return 0
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        print_error "Failed training $model_type after ${duration}s"
        print_error "Check log: $exp_log"
        return 1
    fi
}

# Function to run interpretability analysis
run_interpretability_analysis() {
    print_status "Starting interpretability analysis..."
    
    local analysis_log="$LOG_DIR/interpretability_analysis_${TIMESTAMP}.log"
    
    if python run_training_and_interpretability.py > "$analysis_log" 2>&1; then
        print_success "Interpretability analysis completed"
        return 0
    else
        print_error "Interpretability analysis failed"
        print_error "Check log: $analysis_log"
        return 1
    fi
}

# Main execution
main() {
    print_header "🚀 NLP Interpretability Training and Analysis Pipeline"
    print_header "=" * 80
    print_status "Starting pipeline execution at $(date)"
    print_status "Log file: $MAIN_LOG"
    
    # Change to script directory
    cd "$SCRIPT_DIR"
    
    # Pre-flight checks
    check_venv
    activate_venv
    check_data
    
    # Define models to train (excluding Llama)
    local models_to_train=(
        "bag-of-words-tfidf"
        "bert-base-uncased"
        "roberta-base"
        "distilbert-base-uncased"
    )
    
    # Phase 1: Training
    print_header "PHASE 1: TRAINING MODELS"
    print_header "=" * 50
    
    local total_models=${#models_to_train[@]}
    local successful=0
    local failed=0
    local training_start_time=$(date +%s)
    
    print_status "Training $total_models models (excluding Llama)"
    print_status "Models: ${models_to_train[*]}"
    
    local model_num=0
    for model in "${models_to_train[@]}"; do
        model_num=$((model_num + 1))
        print_header "TRAINING $model_num/$total_models: $model"
        
        if run_training "$model"; then
            successful=$((successful + 1))
        else
            failed=$((failed + 1))
        fi
        
        print_status "Progress: $model_num/$total_models completed"
        echo "" | tee -a "$MAIN_LOG"
    done
    
    # Training phase summary
    local training_end_time=$(date +%s)
    local training_duration=$((training_end_time - training_start_time))
    
    print_header "TRAINING PHASE SUMMARY"
    print_header "=" * 30
    print_status "Total Models: $total_models"
    print_success "Successful: $successful"
    if [ $failed -gt 0 ]; then
        print_error "Failed: $failed"
    else
        print_success "Failed: $failed"
    fi
    print_status "Success Rate: $(( successful * 100 / total_models ))%"
    print_status "Duration: ${training_duration}s ($(( training_duration / 60 ))m $(( training_duration % 60 ))s)"
    
    # Phase 2: Interpretability Analysis
    if [ $successful -gt 0 ]; then
        print_header "PHASE 2: INTERPRETABILITY ANALYSIS"
        print_header "=" * 50
        
        local analysis_start_time=$(date +%s)
        
        if run_interpretability_analysis; then
            local analysis_end_time=$(date +%s)
            local analysis_duration=$((analysis_end_time - analysis_start_time))
            print_success "Interpretability analysis completed in ${analysis_duration}s"
        else
            print_error "Interpretability analysis failed"
        fi
    else
        print_warning "Skipping interpretability analysis - no models were successfully trained"
    fi
    
    # Final summary
    local pipeline_end_time=$(date +%s)
    local total_duration=$((pipeline_end_time - training_start_time))
    
    print_header "PIPELINE EXECUTION SUMMARY"
    print_header "=" * 40
    print_status "Total Duration: ${total_duration}s ($(( total_duration / 60 ))m $(( total_duration % 60 ))s)"
    print_status "Models Trained: $successful/$total_models"
    print_status "Main Log: $MAIN_LOG"
    print_status "Results Directory: $SCRIPT_DIR/results"
    
    # Final status
    if [ $failed -eq 0 ]; then
        print_success "🎉 All phases completed successfully!"
        exit 0
    else
        print_warning "⚠️  Some phases had issues. Check individual log files for details."
        exit 1
    fi
}

# Handle script interruption
trap 'print_error "Pipeline interrupted by user"; exit 130' INT TERM

# Run main function
main "$@"
