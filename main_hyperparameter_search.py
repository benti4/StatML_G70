"""
Main script with hyperparameter search functionality.

This demonstrates how to easily use different search strategies:
1. Grid Search - exhaustive search over specified values
2. Random Search - random sampling from parameter distributions
3. Manual Search - test specific combinations you define

Simply change the SEARCH_STRATEGY variable to switch between methods.
"""

import numpy as np

from data_preprocessing_functionality.data_processing_logistic_regression import process_data_logistic_regression
from model_functionality.logistic_regression import logistic_regression
from eval import eval_avg_f1

from config.hyperparameter_config import SEARCH_SPACES
from utils.hyperparameter_search import (
    grid_search, random_search, manual_search,
    filter_conditional_params, parallel_hyperparameter_search
)
from utils.cross_validation import k_fold_cross_validation, print_cv_results


# ============================================================================
# CONFIGURATION
# ============================================================================

# Model configuration
MODEL_TYPE = 'logistic_regression'
process_data = process_data_logistic_regression
model_train_evaluate = logistic_regression
eval_func = eval_avg_f1

# Search strategy: 'grid', 'random', 'manual', or 'single'
SEARCH_STRATEGY = 'random'

# Cross-validation settings
K_FOLDS = 10
print_shit = True

# Parallel execution settings
# n_jobs: Number of parallel workers (-1 = all cores, 1 = sequential, N = N workers)
N_JOBS = -1

# Random search settings (only used if SEARCH_STRATEGY == 'random')
RANDOM_SEARCH_ITERATIONS = 100
RANDOM_SEED = 42

# Manual search combinations (only used if SEARCH_STRATEGY == 'manual')
MANUAL_COMBINATIONS = [
    {'regularization_method': 'l2', 'C': 1.0},
    {'regularization_method': 'l2', 'C': 10.0},
    {'regularization_method': 'l1', 'C': 1.0},
    {'regularization_method': 'elastic', 'C': 1.0, 'l1_ratio': 0.5},
]

# Single hyperparameter set (only used if SEARCH_STRATEGY == 'single')
SINGLE_HYPERPARAMETERS = {'regularization_method': 'elastic', 'C': 100, 'l1_ratio': 1.0}

# ============================================================================


def run_hyperparameter_search(data_file: str):
    """
    Run hyperparameter search with the specified strategy.
    """
    print(f"\n{'='*80}")
    print(f"HYPERPARAMETER SEARCH - {SEARCH_STRATEGY.upper()} STRATEGY")
    print(f"Model: {MODEL_TYPE}")
    print(f"{'='*80}\n")

    # Process the data
    print("Loading and processing data...")
    data = process_data(data_file)
    print(f"Data loaded: {len(data)} samples\n")

    # Get search space for the model
    search_space = SEARCH_SPACES.get(MODEL_TYPE, {})

    # Generate parameter combinations based on strategy
    if SEARCH_STRATEGY == 'grid':
        print("Generating grid search combinations...")
        param_combinations = grid_search(search_space)
        print(f"Testing {len(param_combinations)} combinations\n")

    elif SEARCH_STRATEGY == 'random':
        print(f"Generating {RANDOM_SEARCH_ITERATIONS} random combinations...")
        param_combinations = random_search(search_space, n_samples=RANDOM_SEARCH_ITERATIONS, seed=RANDOM_SEED)
        print(f"Testing {len(param_combinations)} combinations\n")

    elif SEARCH_STRATEGY == 'manual':
        print("Using manually specified combinations...")
        param_combinations = manual_search(MANUAL_COMBINATIONS)
        print(f"Testing {len(param_combinations)} combinations\n")

    elif SEARCH_STRATEGY == 'single':
        print("Using single hyperparameter configuration...")
        param_combinations = [SINGLE_HYPERPARAMETERS]

    else:
        raise ValueError(f"Unknown search strategy: {SEARCH_STRATEGY}")

    # Track best results
    best_score = -np.inf
    best_params = None
    best_results = None

    # Evaluate all combinations (in parallel if N_JOBS != 1)
    if SEARCH_STRATEGY == 'single' or len(param_combinations) == 1:
        # For single configuration, no need for parallelization
        all_search_results = []
        for idx, hyperparameters in enumerate(param_combinations):
            hyperparameters = filter_conditional_params(hyperparameters, search_space)
            print(f"\nTesting: {hyperparameters}")

            cv_results = k_fold_cross_validation(
                data=data,
                model_train_evaluate=model_train_evaluate,
                hyperparameters=hyperparameters,
                eval_func=eval_func,
                k=K_FOLDS,
                verbose=False
            )

            score = cv_results.get('average_f1_score', {}).get('mean', -np.inf)
            score_std = cv_results.get('average_f1_score', {}).get('std', 0)
            print(f"   → Average F1 Score: {score:.4f} ± {score_std:.4f}")

            all_search_results.append({
                'hyperparameters': hyperparameters.copy(),
                'score': score,
                'score_std': score_std,
                'cv_results': cv_results
            })
    else:
        # Use parallel execution for multiple combinations
        print(f"\nEvaluating {len(param_combinations)} combinations with {N_JOBS if N_JOBS > 0 else 'all'} workers...")
        all_search_results = parallel_hyperparameter_search(
            param_combinations=param_combinations,
            data=data,
            model_train_evaluate=model_train_evaluate,
            eval_func=eval_func,
            search_space=search_space,
            k_folds=K_FOLDS,
            n_jobs=N_JOBS,
            verbose=True
        )

    # Find best results
    for result in all_search_results:
        if result['score'] > best_score:
            best_score = result['score']
            best_params = result['hyperparameters']
            best_results = result['cv_results']

    # Print final results
    print(f"\n{'='*80}")
    print("SEARCH RESULTS")
    print(f"{'='*80}\n")

    # Sort results by score
    all_search_results.sort(key=lambda x: x['score'], reverse=True)

    print("Top 5 Configurations:")
    for idx, result in enumerate(all_search_results[:5], 1):
        print(f"\n{idx}. Score: {result['score']:.4f} ± {result['score_std']:.4f}")
        print(f"   Params: {result['hyperparameters']}")

    print(f"\n{'='*80}")
    print("BEST CONFIGURATION")
    print(f"{'='*80}")
    print(f"\nBest Hyperparameters: {best_params}")
    print(f"Best F1 Score: {best_score:.4f}")

    if print_shit and best_results:
        print_cv_results(best_results, best_params)

    return {
        'best_params': best_params,
        'best_score': best_score,
        'best_results': best_results,
        'all_results': all_search_results
    }


if __name__ == "__main__":
    results = run_hyperparameter_search("training_data_ht2025.csv")
