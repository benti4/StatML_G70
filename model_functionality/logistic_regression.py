from sklearn.linear_model import LogisticRegression

def logistic_regression(train, test, eval_func=None):
    # Separate features and target variable
    X_train = train.drop('increase_stock', axis=1).to_numpy()
    y_train = train['increase_stock'].to_numpy()

    X_test = test.drop('increase_stock', axis=1).to_numpy()
    y_test = test['increase_stock'].to_numpy()
    # Initialize and train the Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Evaluate the model if an evaluation function is provided
    if eval_func:
        results = eval_func(predictions, y_test)
        return results
    else:
        # compute accuracy as default evaluation
        accuracy = (predictions == y_test).mean()
        return {"accuracy": accuracy}
