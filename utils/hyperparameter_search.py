"""
Utilities for hyperparameter search strategies.

Supports:
- Grid Search: exhaustive search over all combinations
- Random Search: random sampling from parameter distributions
- Manual Search: specify your own list of parameter combinations
- Parallel Search: parallel evaluation of hyperparameter combinations
"""

import itertools
import numpy as np
from typing import Dict, List, Optional, Callable, Any
from joblib import Parallel, delayed
import pandas as pd


def grid_search(search_space: Dict, max_combinations: Optional[int] = None) -> List[Dict]:
    """
    Generate all combinations for grid search.

    Parameters:
    - search_space: Dictionary defining the search space for each parameter
    - max_combinations: Maximum number of combinations to return (optional)

    Returns:
    - List of dictionaries, each containing a hyperparameter combination
    """
    # Extract parameters that don't have conditions
    unconditional_params = {k: v for k, v in search_space.items() if 'condition' not in v}
    conditional_params = {k: v for k, v in search_space.items() if 'condition' in v}

    param_names = list(unconditional_params.keys())
    param_grids = []

    for param_name in param_names:
        param_config = unconditional_params[param_name]
        param_grids.append(param_config['grid_values'])

    combinations = list(itertools.product(*param_grids))

    # Generate all combinations and filter based on conditions
    all_combos = []
    for combo in combinations:
        param_dict = dict(zip(param_names, combo))

        # Add conditional parameters if their condition is met
        for cond_param_name, cond_param_config in conditional_params.items():
            if cond_param_config['condition'](param_dict):
                # Need to expand combinations with this parameter
                for val in cond_param_config['grid_values']:
                    combo_with_cond = param_dict.copy()
                    combo_with_cond[cond_param_name] = val
                    all_combos.append(combo_with_cond)
            else:
                all_combos.append(param_dict.copy())

        # If no conditional parameters, just add the combination
        if not conditional_params:
            all_combos.append(param_dict)

    # Remove duplicates
    unique_combos = []
    seen = set()
    for combo in all_combos:
        combo_tuple = tuple(sorted(combo.items()))
        if combo_tuple not in seen:
            seen.add(combo_tuple)
            unique_combos.append(combo)

    if max_combinations:
        unique_combos = unique_combos[:max_combinations]

    return unique_combos


def random_search(search_space: Dict, n_samples: int = 20, seed: Optional[int] = None) -> List[Dict]:
    """
    Generate random hyperparameter combinations.

    Parameters:
    - search_space: Dictionary defining the search space for each parameter
    - n_samples: Number of random combinations to generate
    - seed: Random seed for reproducibility

    Returns:
    - List of dictionaries, each containing a hyperparameter combination
    """
    if seed is not None:
        np.random.seed(seed)

    samples = []
    max_attempts = n_samples * 10  # Prevent infinite loop
    attempts = 0

    while len(samples) < n_samples and attempts < max_attempts:
        attempts += 1
        sample = {}

        # First, sample unconditional parameters
        for param_name, param_config in search_space.items():
            if 'condition' in param_config:
                continue  # Handle conditional parameters later

            if param_config['type'] == 'categorical':
                sample[param_name] = np.random.choice(param_config['values'])
            elif param_config['type'] == 'continuous':
                low, high = param_config['range']
                if param_config.get('log_scale', False):
                    sample[param_name] = 10 ** np.random.uniform(np.log10(low), np.log10(high))
                else:
                    sample[param_name] = np.random.uniform(low, high)
            elif param_config['type'] == 'discrete':
                low, high = param_config['range']
                sample[param_name] = int(np.random.randint(low, high + 1))

        # Now handle conditional parameters
        for param_name, param_config in search_space.items():
            if 'condition' not in param_config:
                continue

            if param_config['condition'](sample):
                if param_config['type'] == 'categorical':
                    sample[param_name] = np.random.choice(param_config['values'])
                elif param_config['type'] == 'continuous':
                    low, high = param_config['range']
                    if param_config.get('log_scale', False):
                        sample[param_name] = 10 ** np.random.uniform(np.log10(low), np.log10(high))
                    else:
                        sample[param_name] = np.random.uniform(low, high)
                elif param_config['type'] == 'discrete':
                    low, high = param_config['range']
                    sample[param_name] = int(np.random.randint(low, high + 1))

        samples.append(sample)

    return samples


def filter_conditional_params(params: Dict, search_space: Dict) -> Dict:
    """
    Remove parameters that don't meet their conditions.

    Parameters:
    - params: Dictionary of hyperparameters
    - search_space: Dictionary defining the search space

    Returns:
    - Filtered dictionary with only valid parameters
    """
    filtered = {}
    for param_name, value in params.items():
        param_config = search_space.get(param_name, {})
        condition = param_config.get('condition')
        if condition is None or condition(params):
            filtered[param_name] = value
    return filtered


def manual_search(param_combinations: List[Dict]) -> List[Dict]:
    """
    Use manually specified parameter combinations.

    Parameters:
    - param_combinations: List of dictionaries with hyperparameter combinations

    Returns:
    - The same list (for consistency with other search functions)
    """
    return param_combinations


def evaluate_single_hyperparameter(
    hyperparameters: Dict,
    data: pd.DataFrame,
    model_train_evaluate: Callable,
    eval_func: Callable,
    search_space: Dict,
    k_folds: int = 10,
    idx: int = 0,
    total: int = 1,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Evaluate a single hyperparameter combination using k-fold cross-validation.
    This is a helper function designed to be run in parallel.

    Parameters:
    - hyperparameters: Dictionary of hyperparameters to evaluate
    - data: The dataset
    - model_train_evaluate: Function to train and evaluate model
    - eval_func: Evaluation function
    - search_space: Dictionary defining the search space
    - k_folds: Number of folds for cross-validation
    - idx: Index of this combination (for progress tracking)
    - total: Total number of combinations (for progress tracking)
    - verbose: Whether to print progress

    Returns:
    - Dictionary with hyperparameters, score, score_std, and cv_results
    """
    from utils.cross_validation import k_fold_cross_validation

    # Filter conditional parameters
    hyperparameters = filter_conditional_params(hyperparameters, search_space)

    if verbose:
        print(f"[{idx + 1}/{total}] Evaluating: {hyperparameters}")

    # Run K-fold cross-validation
    cv_results = k_fold_cross_validation(
        data=data,
        model_train_evaluate=model_train_evaluate,
        hyperparameters=hyperparameters,
        eval_func=eval_func,
        k=k_folds,
        verbose=False
    )

    # Get the score (assuming eval_func returns average_f1_score)
    score = cv_results.get('average_f1_score', {}).get('mean', -np.inf)
    score_std = cv_results.get('average_f1_score', {}).get('std', 0)

    if verbose:
        print(f"[{idx + 1}/{total}] Score: {score:.4f} ± {score_std:.4f}")

    return {
        'hyperparameters': hyperparameters.copy(),
        'score': score,
        'score_std': score_std,
        'cv_results': cv_results
    }


def parallel_hyperparameter_search(
    param_combinations: List[Dict],
    data: pd.DataFrame,
    model_train_evaluate: Callable,
    eval_func: Callable,
    search_space: Dict,
    k_folds: int = 10,
    n_jobs: int = -1,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Evaluate multiple hyperparameter combinations in parallel.

    Parameters:
    - param_combinations: List of hyperparameter dictionaries to evaluate
    - data: The dataset
    - model_train_evaluate: Function to train and evaluate model
    - eval_func: Evaluation function
    - search_space: Dictionary defining the search space
    - k_folds: Number of folds for cross-validation
    - n_jobs: Number of parallel jobs to run (-1 uses all available cores, 1 disables parallelization)
    - verbose: Whether to print progress

    Returns:
    - List of result dictionaries, each containing hyperparameters, score, score_std, and cv_results
    """
    if verbose:
        if n_jobs == -1:
            print(f"Running parallel hyperparameter search with all available cores...")
        elif n_jobs == 1:
            print(f"Running sequential hyperparameter search...")
        else:
            print(f"Running parallel hyperparameter search with {n_jobs} workers...")

    # Use joblib's Parallel to evaluate combinations in parallel
    results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
        delayed(evaluate_single_hyperparameter)(
            hyperparameters=params,
            data=data,
            model_train_evaluate=model_train_evaluate,
            eval_func=eval_func,
            search_space=search_space,
            k_folds=k_folds,
            idx=idx,
            total=len(param_combinations),
            verbose=False  # Disable verbose in worker to avoid cluttered output
        )
        for idx, params in enumerate(param_combinations)
    )

    return results

