from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import mlflow


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


def log_result(model, estimator_name, X_train, y_train, X_test, y_test, experiment="all_models"):
    """
    Logs result of experiment
    :param experiment: Set experiment name.
    :param tag: Set tag of experiment
    :param model: Model used for prediction. Have to be with parameters.
    :param X_train: Train data
    :param y_train: Label of train data
    :param X_test: Test data
    :param y_test: Label of test data
    :return:
    """
    mlflow.set_experiment(experiment)
    mlflow.end_run()
    with mlflow.start_run() as run:
        mlflow.set_tag("estimator_name", estimator_name)
        model = model

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        accuracy, precision, recall, f1 = calculate_metrics(y_test, preds)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
