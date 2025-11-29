from sklearn.tree import DecisionTreeClassifier
from model_functionality.bagging import bagging_classifier
import warnings
import numpy as np

def classification_tree(train, test, hyperparams, eval_func):
    # Separate features and target variable
    X_train = train.drop('increase_stock', axis=1).to_numpy().astype(np.float64)
    y_train = train['increase_stock'].to_numpy().astype(np.int64)

    X_test = test.drop('increase_stock', axis=1).to_numpy().astype(np.float64)
    y_test = test['increase_stock'].to_numpy().astype(np.int64)

    max_depth = hyperparams.get('max_depth', None)
    min_samples_split = hyperparams.get('min_samples_split', 2)
    min_samples_leaf = hyperparams.get('min_samples_leaf', 1)
    criterion = hyperparams.get('criterion', 'gini')
    max_features = hyperparams.get('max_features', None) 

    bagging = hyperparams.get('bagging', False)
    n_estimators = hyperparams.get('n_estimators', 100)
    
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        max_features=max_features,
        random_state=12345,
    )

    if bagging:
        # If bagging is True, pass the configured base model to the bagging function
        return bagging_classifier(X_train, y_train, X_test, y_test, model, n_estimators, eval_func)
    
    else:
        # Standard classification tree process
        print("--- Running Single Decision Tree ---")
        
        # Train the single model
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=Warning)
            model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        # Evaluate the model
        results = eval_func(predictions, y_test)
        results['model_type'] = 'Decision Tree'
        return results
