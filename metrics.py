from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def calculate_metrics(y_true, preds):
    """
    Calculate metrics for models.
    :param y_true: True labels
    :param preds: Predictions of model
    :return: All metrics
    """
    accuracy = accuracy_score(y_true, preds)
    precision = precision_score(y_true, preds)
    recall = recall_score(y_true, preds)
    f1 = f1_score(y_true, preds)
    return accuracy, precision, recall, f1
