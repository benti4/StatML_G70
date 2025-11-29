"""
Nested cross-validation utilities for unbiased model evaluation.

Nested CV consists of:
- Outer loop: Evaluates the model's generalization performance
- Inner loop: Selects optimal hyperparameters for each outer fold

This provides an unbiased estimate of model performance.
"""

import pandas as pd
import numpy as np
from typing import Callable, Dict, Any, Optional

from utils.hyperparameter_search import grid_search, random_search, filter_conditional_params
from utils.cross_validation import k_fold_cross_validation


def nested_cross_validation(
    data: pd.DataFrame,
    model_train_evaluate: Callable,
    search_space: Dict,
    eval_func: Callable,
    outer_k: int = 10,
    inner_k: int = 5,
    search_strategy: str = 'grid',
    random_search_iterations: int = 20,
    random_seed: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Perform nested cross-validation for unbiased model evaluation.

    Parameters:
    - data: The full dataset
    - model_train_evaluate: Function to train and evaluate model
    - search_space: Dictionary defining hyperparameter search space
    - eval_func: Evaluation function
    - outer_k: Number of folds in outer loop (for performance estimation)
    - inner_k: Number of folds in inner loop (for hyperparameter selection)
    - search_strategy: 'grid' or 'random' search for hyperparameters
    - random_search_iterations: Number of iterations if using random search
    - random_seed: Random seed for reproducibility
    - verbose: Whether to print progress

    Returns:
    - Dictionary containing:
        - outer_fold_scores: List of scores from each outer fold
        - best_hyperparameters_per_fold: Best hyperparameters found in each outer fold
        - mean_score: Mean performance across outer folds
        - std_score: Standard deviation of performance across outer folds
        - all_fold_results: Detailed results from each outer fold
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"NESTED CROSS-VALIDATION")
        print(f"Outer folds: {outer_k}, Inner folds: {inner_k}")
        print(f"Search strategy: {search_strategy}")
        print(f"{'='*80}\n")

    fold_size = len(data) // outer_k
    outer_fold_scores = []
    best_hyperparameters_per_fold = []
    all_fold_results = []

    # Outer loop: For performance estimation
    for outer_fold in range(outer_k):
        if verbose:
            print(f"\n{'='*80}")
            print(f"OUTER FOLD {outer_fold + 1}/{outer_k}")
            print(f"{'='*80}")

        # Split data into outer train and test
        start = outer_fold * fold_size
        end = (outer_fold + 1) * fold_size if outer_fold != outer_k - 1 else len(data)

        outer_test_data = data.iloc[start:end]
        outer_train_data = pd.concat([data.iloc[:start], data.iloc[end:]])

        if verbose:
            print(f"Outer train size: {len(outer_train_data)}, Outer test size: {len(outer_test_data)}")

        # Inner loop: Hyperparameter selection using only outer_train_data
        if verbose:
            print(f"\nStarting inner loop hyperparameter search...")

        # Generate parameter combinations
        if search_strategy == 'grid':
            param_combinations = grid_search(search_space)
        elif search_strategy == 'random':
            param_combinations = random_search(
                search_space,
                n_samples=random_search_iterations,
                seed=random_seed
            )
        else:
            raise ValueError(f"Unknown search strategy: {search_strategy}")

        if verbose:
            print(f"Testing {len(param_combinations)} hyperparameter combinations in inner loop...")

        # Evaluate each hyperparameter combination on inner folds
        best_inner_score = -np.inf
        best_inner_params = None

        for idx, hyperparameters in enumerate(param_combinations):
            # Filter conditional parameters
            hyperparameters = filter_conditional_params(hyperparameters, search_space)

            # Use inner k-fold CV on outer_train_data only
            inner_cv_results = k_fold_cross_validation(
                data=outer_train_data,
                model_train_evaluate=model_train_evaluate,
                hyperparameters=hyperparameters,
                eval_func=eval_func,
                k=inner_k,
                verbose=False
            )

            # Get the score (assuming eval_func returns average_f1_score)
            inner_score = inner_cv_results.get('average_f1_score', {}).get('mean', -np.inf)

            if verbose and (idx + 1) % max(1, len(param_combinations) // 10) == 0:
                print(f"  [{idx + 1}/{len(param_combinations)}] Evaluated hyperparameters...")

            # Track best hyperparameters
            if inner_score > best_inner_score:
                best_inner_score = inner_score
                best_inner_params = hyperparameters.copy()

        if verbose:
            print(f"\nBest hyperparameters for outer fold {outer_fold + 1}: {best_inner_params}")
            print(f"Inner CV score: {best_inner_score:.4f}")

        best_hyperparameters_per_fold.append(best_inner_params)

        # Train final model on entire outer_train_data with best hyperparameters
        # and evaluate on outer_test_data
        outer_results = model_train_evaluate(
            outer_train_data,
            outer_test_data,
            best_inner_params,
            eval_func
        )

        # Extract the score
        outer_score = outer_results.get('average_f1_score', -np.inf)
        outer_fold_scores.append(outer_score)

        if verbose:
            print(f"Outer fold {outer_fold + 1} test score: {outer_score:.4f}")

        # Store detailed results
        all_fold_results.append({
            'fold': outer_fold + 1,
            'best_hyperparameters': best_inner_params,
            'inner_cv_score': best_inner_score,
            'outer_test_score': outer_score,
            'outer_test_results': outer_results
        })

    # Calculate summary statistics
    mean_score = np.mean(outer_fold_scores)
    std_score = np.std(outer_fold_scores)

    if verbose:
        print(f"\n{'='*80}")
        print(f"NESTED CROSS-VALIDATION SUMMARY")
        print(f"{'='*80}")
        print(f"\nOuter fold scores: {[f'{score:.4f}' for score in outer_fold_scores]}")
        print(f"Mean performance: {mean_score:.4f} ± {std_score:.4f}")
        print(f"\nBest hyperparameters per fold:")
        for i, params in enumerate(best_hyperparameters_per_fold, 1):
            print(f"  Fold {i}: {params}")
        print(f"{'='*80}\n")

    return {
        'outer_fold_scores': outer_fold_scores,
        'best_hyperparameters_per_fold': best_hyperparameters_per_fold,
        'mean_score': mean_score,
        'std_score': std_score,
        'all_fold_results': all_fold_results
    }


def evaluate_method_with_nested_cv(
    data_file: str,
    process_data_func: Callable,
    model_train_evaluate: Callable,
    search_space: Dict,
    eval_func: Callable,
    outer_k: int = 10,
    inner_k: int = 5,
    search_strategy: str = 'grid',
    random_search_iterations: int = 20,
    random_seed: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to evaluate a method using nested cross-validation.

    This combines data loading, processing, and nested CV in one function.

    Parameters:
    - data_file: Path to the data file
    - process_data_func: Function to process/load data
    - model_train_evaluate: Function to train and evaluate model
    - search_space: Dictionary defining hyperparameter search space
    - eval_func: Evaluation function
    - outer_k: Number of folds in outer loop
    - inner_k: Number of folds in inner loop
    - search_strategy: 'grid' or 'random' search
    - random_search_iterations: Number of iterations if using random search
    - random_seed: Random seed for reproducibility
    - verbose: Whether to print progress

    Returns:
    - Nested CV results dictionary
    """
    if verbose:
        print(f"Loading and processing data from {data_file}...")

    data = process_data_func(data_file)

    if verbose:
        print(f"Data loaded: {len(data)} samples\n")

    return nested_cross_validation(
        data=data,
        model_train_evaluate=model_train_evaluate,
        search_space=search_space,
        eval_func=eval_func,
        outer_k=outer_k,
        inner_k=inner_k,
        search_strategy=search_strategy,
        random_search_iterations=random_search_iterations,
        random_seed=random_seed,
        verbose=verbose
    )

