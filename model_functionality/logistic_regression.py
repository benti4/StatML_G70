from sklearn.linear_model import LogisticRegression
import warnings

def logistic_regression(train, test, hyperparams, eval_func):
    # Separate features and target variable
    X_train = train.drop('increase_stock', axis=1).to_numpy()
    y_train = train['increase_stock'].to_numpy()

    X_test = test.drop('increase_stock', axis=1).to_numpy()
    y_test = test['increase_stock'].to_numpy()

    # Initialize and train the Logistic Regression model
    if hyperparams['regularization_method'] == 'l2':
        model = LogisticRegression(penalty='l2', C=hyperparams['C'], max_iter=1000)
    elif hyperparams['regularization_method'] == 'l1':
        model = LogisticRegression(penalty='l1', solver='liblinear', C=hyperparams['C'], max_iter=1000)
    elif hyperparams['regularization_method'] == 'elastic':
        model = LogisticRegression(penalty='elasticnet', solver='saga', C=hyperparams['C'],
                                   l1_ratio=hyperparams['l1_ratio'], max_iter=1000)
    else:
        model = LogisticRegression(max_iter=1000)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=Warning)
        model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Evaluate the model
    results = eval_func(predictions, y_test)
    return results
