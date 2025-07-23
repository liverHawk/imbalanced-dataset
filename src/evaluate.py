import yaml
import os
import sys
import pandas as pd
from lib.util import setup_logging
from classifier.improved_c45 import ImprovedC45
from dvclive import Live

import sklearn.metrics as metrics


def save_evaluation_results(live: Live, predict_probs, test_labels):
    predict_actions = [prop.argmax() for prop in predict_probs]

    # predict_probs -> predict_actions vs test_label
    macro_prec = metrics.precision_score(test_labels, predict_actions, average='macro')
    macro_f1 = metrics.f1_score(test_labels, predict_actions, average='macro')
    emr = metrics.balanced_accuracy_score(test_labels, predict_actions)

    live.summary = {
        "macro_precision": macro_prec,
        "macro_f1": macro_f1,
        "balanced_accuracy": emr,
    }

    live.log_sklearn_plot(
        "roc",
        test_labels,
        predict_probs,
        title="ROC Curve",
    )
    live.log_sklearn_plot(
        "precision_recall",
        test_labels,
        predict_probs,
        title="Precision-Recall Curve",
    )
    live.log_sklearn_plot(
        "confusion_matrix",
        test_labels,
        predict_actions,
        title="Confusion Matrix",
    )


def evaluate_process(evaluate_path):
    log_path = os.path.join("logs", "evaluate.log")
    logger = setup_logging(log_path)

    logger.info("Loading model...")
    model = ImprovedC45(
        load_path=os.path.join("data", "models", "improved_c45_model.joblib")
    )

    logger.info("Loading test data...")
    test_df = pd.read_csv(os.path.join("data", "prepared", "test.csv"))

    logger.info("Evaluating model...")
    predict_probs = model.predict_proba(test_df.drop("Label", axis=1))

    with Live(evaluate_path) as live:
        save_evaluation_results(live, predict_probs, test_df["Label"])


def main():
    evaluate_result_path = os.path.join("evaluate")
    evaluate_process(evaluate_result_path)


if __name__ == "__main__":
    main()