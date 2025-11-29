"""
Main script for nested cross-validation evaluation.

This script demonstrates how to use nested cross-validation to get an unbiased
estimate of a model's performance. Nested CV is the gold standard for model
evaluation when you also need to tune hyperparameters.

Nested CV structure:
- Outer loop (K_OUTER folds): Provides unbiased performance estimate
- Inner loop (K_INNER folds): Selects best hyperparameters for each outer fold

The reported performance is from the outer loop and represents how the model
would perform on truly unseen data.
"""

from data_preprocessing_functionality.data_processing_logistic_regression import process_data_logistic_regression
from model_functionality.logistic_regression import logistic_regression
from eval import eval_avg_f1
from config.hyperparameter_config import SEARCH_SPACES
from utils.nested_cross_validation import evaluate_method_with_nested_cv


# ============================================================================
# CONFIGURATION
# ============================================================================

# Data file
DATA_FILE = "training_data_ht2025.csv"

# Model configuration
MODEL_TYPE = 'logistic_regression'
process_data = process_data_logistic_regression
model_train_evaluate = logistic_regression
eval_func = eval_avg_f1
search_space = SEARCH_SPACES[MODEL_TYPE]

# Nested CV settings
K_OUTER = 10  # Outer folds for performance estimation
K_INNER = 5   # Inner folds for hyperparameter tuning

# Search strategy: 'grid' or 'random'
SEARCH_STRATEGY = 'random'

# Random search settings (only used if SEARCH_STRATEGY == 'random')
RANDOM_SEARCH_ITERATIONS = 50
RANDOM_SEED = 42

# Parallel execution settings
# n_jobs: Number of parallel workers (-1 = all cores, 1 = sequential, N = N workers)
N_JOBS = -1

# Verbosity
VERBOSE = True

# ============================================================================


def main():
    """
    Run nested cross-validation to evaluate logistic regression.
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING {MODEL_TYPE.upper()} WITH NESTED CROSS-VALIDATION")
    print(f"{'='*80}")
    print(f"\nConfiguration:")
    print(f"  - Data file: {DATA_FILE}")
    print(f"  - Outer folds: {K_OUTER} (for performance estimation)")
    print(f"  - Inner folds: {K_INNER} (for hyperparameter tuning)")
    print(f"  - Search strategy: {SEARCH_STRATEGY}")
    if SEARCH_STRATEGY == 'random':
        print(f"  - Random search iterations: {RANDOM_SEARCH_ITERATIONS}")
    print(f"  - Random seed: {RANDOM_SEED}")
    print(f"  - Parallel workers: {N_JOBS if N_JOBS > 0 else 'all cores'}")
    print(f"{'='*80}\n")

    # Run nested cross-validation
    results = evaluate_method_with_nested_cv(
        data_file=DATA_FILE,
        process_data_func=process_data,
        model_train_evaluate=model_train_evaluate,
        search_space=search_space,
        eval_func=eval_func,
        outer_k=K_OUTER,
        inner_k=K_INNER,
        search_strategy=SEARCH_STRATEGY,
        random_search_iterations=RANDOM_SEARCH_ITERATIONS,
        random_seed=RANDOM_SEED,
        n_jobs=N_JOBS,
        verbose=VERBOSE
    )

    # Print final summary
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS")
    print(f"{'='*80}")
    print(f"\nModel: {MODEL_TYPE}")
    print(f"Estimated Performance (from outer CV):")
    print(f"  Mean F1 Score: {results['mean_score']:.4f} ± {results['std_score']:.4f}")
    print(f"\nIndividual Outer Fold Scores:")
    for i, score in enumerate(results['outer_fold_scores'], 1):
        print(f"  Fold {i}: {score:.4f}")

    # Analyze hyperparameter consistency across folds
    print(f"\nHyperparameter Consistency Analysis:")
    all_params = results['best_hyperparameters_per_fold']

    # Check which hyperparameters are most common
    if all_params:
        # For each hyperparameter, show the distribution of values
        param_keys = set()
        for params in all_params:
            param_keys.update(params.keys())

        for key in sorted(param_keys):
            values = [params.get(key) for params in all_params if key in params]
            unique_values = list(set(values))
            print(f"  {key}:")
            for val in unique_values:
                count = values.count(val)
                print(f"    {val}: {count}/{len(all_params)} folds")

    print(f"\n{'='*80}")
    print(f"INTERPRETATION")
    print(f"{'='*80}")
    print(f"""
The reported mean F1 score of {results['mean_score']:.4f} ± {results['std_score']:.4f}
is an UNBIASED estimate of how well {MODEL_TYPE} would perform on new,
unseen data.

This is more reliable than the score from regular cross-validation with
hyperparameter tuning, which would be optimistically biased.

The hyperparameter values that were selected in each outer fold are shown
above. If they vary significantly across folds, it suggests the optimal
hyperparameters are sensitive to the training data.
    """)
    print(f"{'='*80}\n")

    return results


if __name__ == "__main__":
    results = main()

