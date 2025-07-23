import yaml
import os
import sys
import pandas as pd
from lib.util import setup_logging
from classifier.improved_c45 import ImprovedC45


def train_process(model, model_params):
    log_path = os.path.join("logs", "train.log")
    logger = setup_logging(log_path)

    if model == "ImprovedC45":
        logger.info("Using ImprovedC45 classifier.")
        model = ImprovedC45(
            max_depth=model_params["max_depth"],
        )
    else:
        raise ValueError(f"Unsupported classifier: {model_params['classifier']}")

    logger.info("Loading training data...")
    train_df = pd.read_csv(os.path.join("data", "prepared", "train.csv"))
    logger.info("Training model...")
    model.fit(train_df.drop("label", axis=1), train_df["label"])

    logger.info("Saving model...")
    model.save(os.path.join("data", "models", "improved_c45_model.joblib"))
    logger.info("Model training complete.")


def main():
    os.makedirs(os.path.join("logs"), exist_ok=True)
    params = yaml.safe_load(open("params.yaml", "r"))["train"]
    print(params)
    os.makedirs(os.path.join("data", "models"), exist_ok=True)

    train_process(params["classifier"], params["classifier_params"])


if __name__ == "__main__":
    main()