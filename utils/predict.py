"""Unearthed Prediction Template"""
import logging
import argparse
from os import getenv
from os.path import join
import pandas as pd

from preprocess import preprocess
from train import model_fn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# WARNING: This function simulates how the remote pipeline will make predictions using
# the private and validation datasets. It should not need to be modified.
if __name__ == "__main__":
    """Prediction.

    The main function is only used by the Unearthed CLI.

    When a submission is made online AWS SageMaker Processing Jobs are used to perform
    preprocessing and Batch Transform Jobs are used to pass the result of preprocessing
    to the trained model.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir", type=str, default=getenv("SM_MODEL_DIR", "/opt/ml/models")
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=getenv("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"),
    )
    args, _ = parser.parse_known_args()

    # call preprocessing on the data
    df = preprocess(join(args.data_dir, "public.csv.gz"), True)

    # load the model
    model = model_fn(args.model_dir)

    # pass the model the preprocessed data
    logger.info("creating predictions")
    predictions = model.predict(df)
    logger.info(f"predictions have shape of {predictions.shape}")

    # save the predictions
    pd.DataFrame(predictions).to_csv(
        "/opt/ml/output/public.csv.out", index=False, header=False
    )
