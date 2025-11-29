# utils/cross_validation.py
"""
Cross-validation utilities for model evaluation.
"""

import pandas as pd
import numpy as np
from typing import Callable, Dict, Any


def k_fold_cross_validation(
    data: pd.DataFrame,
    model_train_evaluate: Callable,
    hyperparameters: Dict,
    eval_func: Callable,
    k: int = 10,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Perform K-Fold Cross Validation.

    Parameters:
    - data: The full dataset
    - model_train_evaluate: Function to train and evaluate model
    - hyperparameters: Dictionary of hyperparameters
    - eval_func: Evaluation function
    - k: Number of folds
    - verbose: Whether to print progress

    Returns:
    - Dictionary with aggregated results including mean and std of metrics
    """
    fold_size = len(data) // k
    all_results = []

    for fold in range(k):
        if verbose:
            print(f"Processing fold {fold + 1}/{k}...")

        start = fold * fold_size
        end = (fold + 1) * fold_size if fold != k - 1 else len(data)

        test_data = data.iloc[start:end]
        train_data = pd.concat([data.iloc[:start], data.iloc[end:]])

        results = model_train_evaluate(train_data, test_data, hyperparameters, eval_func)
        all_results.append(results)

    # Aggregate results
    agg_results = {key: [] for key in all_results[0].keys()}
    for result in all_results:
        for key, value in result.items():
            agg_results[key].append(value)

    # Calculate mean and std for each metric
    summary = {}
    for key, values in agg_results.items():
        if key == "classification_report":
            # Average each metric in the classification report
            avg_report = {}
            for label in values[0].keys():
                if isinstance(values[0][label], dict):
                    avg_report[label] = {}
                    for metric in values[0][label].keys():
                        metric_values = [v[label][metric] for v in values]
                        avg_report[label][metric] = {
                            'mean': np.mean(metric_values),
                            'std': np.std(metric_values)
                        }
                else:
                    metric_values = [v[label] for v in values]
                    avg_report[label] = {
                        'mean': np.mean(metric_values),
                        'std': np.std(metric_values)
                    }
            summary[key] = avg_report
        elif key == "confusion_matrix":
            summary[key] = {
                'mean': np.mean(values, axis=0),
                'std': np.std(values, axis=0)
            }
        else:
            summary[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'all_folds': values
            }

    return summary


def print_cv_results(cv_results: Dict, hyperparameters: Dict):
    """Print cross-validation results in a formatted way."""
    print("\n" + "="*60)
    print(f"Hyperparameters: {hyperparameters}")
    print("="*60)

    for key, value in cv_results.items():
        if key == "classification_report":
            print("\nClassification Report (averaged across folds):")
            for label, metrics in value.items():
                if isinstance(metrics, dict) and 'mean' in metrics:
                    print(f"  {label}: {metrics['mean']:.4f} ± {metrics['std']:.4f}")
                elif isinstance(metrics, dict):
                    print(f"  {label}:")
                    for metric, stats in metrics.items():
                        print(f"    {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        elif key == "confusion_matrix":
            print(f"\nAverage Confusion Matrix:")
            print(value['mean'])
        else:
            if isinstance(value, dict) and 'mean' in value:
                print(f"{key}: {value['mean']:.4f} ± {value['std']:.4f}")
            else:
                print(f"{key}: {value}")
    print()
