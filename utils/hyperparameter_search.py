# utils/hyperparameter_search.py
"""
Utilities for hyperparameter search strategies.

Supports:
- Grid Search: exhaustive search over all combinations
- Random Search: random sampling from parameter distributions
- Manual Search: specify your own list of parameter combinations
"""

import itertools
import numpy as np
from typing import Dict, List, Optional


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
