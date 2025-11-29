def naive_method(train, test, eval_func):
    # Separate features and target variable
    X_train = train.drop('increase_stock', axis=1).to_numpy()
    y_train = train['increase_stock'].to_numpy()

    X_test = test.drop('increase_stock', axis=1).to_numpy()
    y_test = test['increase_stock'].to_numpy()

    # Naive method: predict the majority class from the training set
    majority_class = max(set(y_train), key=list(y_train).count)
    predictions = [majority_class] * len(y_test)

    # Evaluate the model if an evaluation function is provided
    if eval_func:
        results = eval_func(predictions, y_test)
        return results
    else:
        # compute accuracy as default evaluation
        accuracy = (predictions == y_test).mean()
        return {"accuracy": accuracy}
