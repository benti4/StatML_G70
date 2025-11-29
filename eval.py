from sklearn.metrics import classification_report, confusion_matrix

def verbose_evaluate_model(preds, labels):
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


def eval_avg_f1(preds, labels):
    """
    Evaluate the model's predictions by calculating the average F1 score.

    Parameters:
    preds (list or array-like): The predicted labels from the model.
    labels (list or array-like): The true labels.

    Returns:
    dict: A dictionary containing the average F1 score.
    """
    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    matrix = confusion_matrix(labels, preds)
    avg_f1 = report['weighted avg']['f1-score']

    return {
        "average_f1_score": avg_f1
    }
