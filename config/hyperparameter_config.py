"""
Configuration file for hyperparameter search spaces.

Each model has a search space dictionary where you can define:
- 'type': 'categorical', 'continuous', or 'discrete'
- 'values': list of values for categorical parameters
- 'range': [min, max] for continuous/discrete parameters
- 'log_scale': True/False for continuous parameters (useful for C, learning_rate, etc.)
- 'grid_values': specific values to try in grid search
- 'condition': lambda function to conditionally include parameter
"""

SEARCH_SPACES = {
    'logistic_regression': {
        'regularization_method': {
            'type': 'categorical',
            'values': ['l1', 'l2', 'elastic'],
            'grid_values': ['l1', 'l2', 'elastic']
        },
        'C': {
            'type': 'continuous',
            'range': [0.001, 1000],
            'log_scale': True,
            'grid_values': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        },
        'l1_ratio': {
            'type': 'continuous',
            'range': [0.0, 1.0],
            'grid_values': [0.0, 0.25, 0.5, 0.75, 1.0],
            'condition': lambda params: params.get('regularization_method') == 'elastic'
        }
    },

    # Example for future models you might add
    'random_forest': {
        'n_estimators': {
            'type': 'discrete',
            'range': [10, 500],
            'grid_values': [10, 50, 100, 200, 500]
        },
        'max_depth': {
            'type': 'discrete',
            'range': [1, 50],
            'grid_values': [5, 10, 20, 30, None]
        },
        'min_samples_split': {
            'type': 'discrete',
            'range': [2, 20],
            'grid_values': [2, 5, 10, 15]
        },
        'min_samples_leaf': {
            'type': 'discrete',
            'range': [1, 10],
            'grid_values': [1, 2, 4, 8]
        }
    },

    'svm': {
        'C': {
            'type': 'continuous',
            'range': [0.001, 1000],
            'log_scale': True,
            'grid_values': [0.01, 0.1, 1, 10, 100]
        },
        'kernel': {
            'type': 'categorical',
            'values': ['linear', 'rbf', 'poly'],
            'grid_values': ['linear', 'rbf', 'poly']
        },
        'gamma': {
            'type': 'continuous',
            'range': [0.0001, 10],
            'log_scale': True,
            'grid_values': [0.001, 0.01, 0.1, 1, 'scale', 'auto'],
            'condition': lambda params: params.get('kernel') in ['rbf', 'poly']
        }
    }
}

# Default hyperparameters for each model (used as baseline)
DEFAULT_HYPERPARAMETERS = {
    'logistic_regression': {
        'regularization_method': 'l2',
        'C': 1.0
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1
    },
    'svm': {
        'C': 1.0,
        'kernel': 'rbf',
        'gamma': 'scale'
    }
}
