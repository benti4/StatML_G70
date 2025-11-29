# Hyperparameter Optimization System

This system provides a flexible and easy-to-use hyperparameter optimization framework for machine learning models.

## 🚀 Quick Start

### Option 1: Use the Hyperparameter Search Script

Run the automated hyperparameter search:

```bash
python main_hyperparameter_search.py
```

### Option 2: Use the Original Main Script

Run with a single hyperparameter configuration:

```bash
python main.py
```

## 📝 How to Define Search Spaces

### 1. Edit the Configuration File

Open `config/hyperparameter_config.py` and add or modify search spaces:

```python
SEARCH_SPACES = {
    'your_model_name': {
        'parameter_name': {
            'type': 'categorical',  # or 'continuous', 'discrete'
            'values': [value1, value2, ...],  # for categorical
            'grid_values': [val1, val2, ...]  # values to try in grid search
        },
        'another_parameter': {
            'type': 'continuous',
            'range': [min_value, max_value],
            'log_scale': True,  # use logarithmic scale
            'grid_values': [0.001, 0.01, 0.1, 1, 10]
        },
        'conditional_parameter': {
            'type': 'continuous',
            'range': [0.0, 1.0],
            'grid_values': [0.0, 0.5, 1.0],
            'condition': lambda params: params.get('other_param') == 'elastic'
        }
    }
}
```

### 2. Parameter Types

- **categorical**: Discrete choices (e.g., 'l1', 'l2', 'elastic')
- **continuous**: Floating-point values (e.g., learning rate, C)
- **discrete**: Integer values (e.g., number of trees, depth)

### 3. Parameter Options

- `type`: Required. One of 'categorical', 'continuous', 'discrete'
- `values`: For categorical - list of possible values
- `range`: For continuous/discrete - [min, max]
- `log_scale`: For continuous - use logarithmic scale (good for C, learning_rate)
- `grid_values`: Specific values to try in grid search
- `condition`: Lambda function to conditionally include parameter

## 🔍 Search Strategies

Edit `main_hyperparameter_search.py` and change `SEARCH_STRATEGY`:

### Grid Search
Tests all combinations of specified values.

```python
SEARCH_STRATEGY = 'grid'
```

**Pros**: Exhaustive, guaranteed to find best in search space
**Cons**: Expensive for large search spaces

### Random Search
Randomly samples from parameter distributions.

```python
SEARCH_STRATEGY = 'random'
RANDOM_SEARCH_ITERATIONS = 20  # number of combinations to try
RANDOM_SEED = 42  # for reproducibility
```

**Pros**: More efficient, good for large search spaces
**Cons**: May miss optimal combination

### Manual Search
Test specific combinations you define.

```python
SEARCH_STRATEGY = 'manual'
MANUAL_COMBINATIONS = [
    {'regularization_method': 'l2', 'C': 1.0},
    {'regularization_method': 'l1', 'C': 10.0},
    {'regularization_method': 'elastic', 'C': 1.0, 'l1_ratio': 0.5},
]
```

**Pros**: Test specific hypotheses, fast
**Cons**: Requires domain knowledge

### Single Configuration
Test just one configuration (like original main.py).

```python
SEARCH_STRATEGY = 'single'
SINGLE_HYPERPARAMETERS = {'regularization_method': 'elastic', 'C': 100, 'l1_ratio': 1.0}
```

## 📊 Example: Logistic Regression

Current search space for logistic regression:

```python
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
}
```

This will test:
- 3 regularization methods × 7 C values = 21 base combinations
- Plus 5 l1_ratio values when regularization_method='elastic'
- Total: ~30 combinations in grid search

## 🛠️ Adding a New Model

### 1. Define the search space in `config/hyperparameter_config.py`:

```python
SEARCH_SPACES = {
    # ...existing models...
    'my_new_model': {
        'param1': {
            'type': 'continuous',
            'range': [0.1, 10],
            'log_scale': True,
            'grid_values': [0.1, 1, 10]
        },
        'param2': {
            'type': 'categorical',
            'values': ['option1', 'option2'],
            'grid_values': ['option1', 'option2']
        }
    }
}
```

### 2. Update `main_hyperparameter_search.py`:

```python
from model_functionality.my_new_model import my_new_model

MODEL_TYPE = 'my_new_model'
model_train_evaluate = my_new_model
```

### 3. Run the search:

```bash
python main_hyperparameter_search.py
```

## 📁 File Structure

```
StatML_G70/
├── config/
│   └── hyperparameter_config.py      # Define search spaces here
├── utils/
│   ├── hyperparameter_search.py      # Search strategy implementations
│   └── cross_validation.py           # CV utilities
├── main_hyperparameter_search.py     # Run hyperparameter search
└── main.py                            # Original simple script
```

## 💡 Tips

1. **Start with random search** - It's usually more efficient than grid search
2. **Use log scale for C** - Regularization strength often works best on log scale
3. **Narrow down the range** - After initial search, focus on promising regions
4. **Use conditional parameters** - Save computation by only testing relevant combinations
5. **Check the output** - The script shows top 5 configurations for comparison

## 🎯 Example Output

```
================================================================================
HYPERPARAMETER SEARCH - GRID STRATEGY
Model: logistic_regression
================================================================================

Testing 21 combinations

[1/21] Testing: {'regularization_method': 'l1', 'C': 0.001}
   → Average F1 Score: 0.7234 ± 0.0156

[2/21] Testing: {'regularization_method': 'l1', 'C': 0.01}
   → Average F1 Score: 0.7456 ± 0.0142

...

================================================================================
SEARCH RESULTS
================================================================================

Top 5 Configurations:

1. Score: 0.7891 ± 0.0123
   Params: {'regularization_method': 'elastic', 'C': 10, 'l1_ratio': 0.5}

2. Score: 0.7856 ± 0.0134
   Params: {'regularization_method': 'l2', 'C': 10}

...
```

