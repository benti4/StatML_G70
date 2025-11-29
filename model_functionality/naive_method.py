def naive_method(train, test, hyperparams, eval_func):
    # Separate features and target variable
    X_train = train.drop('increase_stock', axis=1).to_numpy()
    y_train = train['increase_stock'].to_numpy()

    X_test = test.drop('increase_stock', axis=1).to_numpy()
    y_test = test['increase_stock'].to_numpy()

    # Naive method: predict the majority class from the training set
    majority_class = max(set(y_train), key=list(y_train).count)
    predictions = [majority_class] * len(y_test)

    # Evaluate the model
    results = eval_func(predictions, y_test)
    return results
