from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(preds, labels):
    """
    Evaluate the model's predictions using classification report and confusion matrix.

    Parameters:
    preds (list or array-like): The predicted labels from the model.
    labels (list or array-like): The true labels.

    Returns:
    dict: A dictionary containing the classification report and confusion matrix.
    """
    report = classification_report(labels, preds, output_dict=True)
    matrix = confusion_matrix(labels, preds)

    return {
        "classification_report": report,
        "confusion_matrix": matrix
    }
