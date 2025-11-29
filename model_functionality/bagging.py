from sklearn.ensemble import BaggingClassifier
import warnings

def bagging_classifier(X_train, y_train, X_test, y_test, base_model, n_estimators, eval_func):
    print(f"--- Running Bagging with {n_estimators} estimators ---")
    
    # Initialize the BaggingClassifier
    bagging_model = BaggingClassifier(
        estimator=base_model,
        n_estimators=n_estimators,
        max_samples=1.0,
        max_features=1.0,
        random_state=12345
    )

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=Warning)
        bagging_model.fit(X_train, y_train)

    # Make predictions using the ensemble
    predictions = bagging_model.predict(X_test)

    # Evaluate the ensemble model
    results = eval_func(predictions, y_test)
    return results