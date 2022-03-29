"""Unearthed Training Template"""

import argparse
import logging
import pickle
import sys
from io import StringIO
from os import getenv
from os.path import abspath, join
import pandas as pd
from fastai.tabular.all import *
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor
from ensemble_model import *
from preprocess import preprocess

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Work around for a SageMaker path issue
# (see https://github.com/aws/sagemaker-python-sdk/issues/648)
# WARNING - removing this may cause the submission process to fail
if abspath("/opt/ml/code") not in sys.path:
    sys.path.append(abspath("/opt/ml/code"))


def train(args):
    """Train

    Your model code goes here.
    """
    logger.info("calling training function")

    # If you require any particular preprocessing to create features then this
    # *must* be contained in the preprocessing function for the Unearthed pipeline
    # apply it to the private data
    df = preprocess(join(args.data_dir, "public.csv.gz"), False)
    logger.info("printing columns before training function")
    logger.info(f"{len(df.columns)}")

    y_train = df['Downhole Gauge Pressure']
    logger.info(f"training target shape is {y_train.shape}")

    X_train = df.drop(columns=['Downhole Gauge Pressure'])
    logger.info(f"training input shape is {X_train.shape}")

    model = RandomForestRegressor(
        n_jobs=-1, n_estimators=55,
        max_depth=15,
        max_samples=200_000, max_features=0.5,
        min_samples_leaf=5, oob_score=True
    )
    #model = DecisionTreeRegressor(min_samples_leaf=25)
    model.fit(X_train, y_train)

    logger.info(f' {"=====" *10}')
    logger.info(f' model name is ===={type(model).__name__}')
    logger.info(f' {"=====" *10}')

    logger.info(f' {"====="*10}')
    logger.info(f' training score = {mean_absolute_error(model.predict(X_train), y_train)}')
    logger.info(f' {"====="*10}')

    logger.info(f' {"====="*10}')
    logger.info(
        f'training oob score = {mean_absolute_error(model.oob_prediction_, y_train)}')
    logger.info(f' {"====="*10}')


    # save the model to disk
    save_model(model, args.model_dir)


def save_model(model_comp, model_dir):
    """Save model to a binary file.

    This function must write the model to disk in a format that can
    be loaded from the model_fn.

    WARNING - modifying this function may cause the submission process to fail.
    """
    logger.info(f"saving model to {model_dir}")

    with open(join(model_dir, "model.pkl"), "wb") as model_file:
        pickle.dump(model_comp, model_file)
    logger.info(f"model saved to {model_dir}")


def model_fn(model_dir):
    """Load model from binary file.

    This function loads the model from disk. It is called by SageMaker.

    WARNING - modifying this function may case the submission process to fail.
    """
    logger.info("loading model")
    with open(join(model_dir, "model.pkl"), "rb") as file:
        return pickle.load(file)


def input_fn(input_data, content_type):
    """Take request data and de-serialize the data into an object for prediction.

    In the Unearthed submission pipeline the data is passed as "text/csv". This
    function reads the CSV into a Pandas dataframe ready to be passed to the model.

    WARNING - modifying this function may cause the submission process to fail.
    """
    logger.info("receiving preprocessed input")

    # this call must result in a dataframe or nparray that matches your model
    input = pd.read_csv(StringIO(input_data), index_col=0, parse_dates=True)
    logger.info(f"preprocessed input has shape {input.shape}")
    return input


class PretrainerLoading():
    def __init__(self, filename):

        self.filename = filename

    def saving_preprocessor(self, to):

        return to.export(self.filename)

    def loading_preprocessor(self, df):

        to_load = load_pandas(self.filename)
        to_new = to_load.train.new(df)
        to_new.process()
        return to_new.xs


@patch
def export(self: TabularPandas, fname='export.pkl', pickle_protocol=2):
    "Export the contents of `self` without the items"
    old_to = self
    self = self.new_empty()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pickle.dump(self, open(Path(fname), 'wb'), protocol=pickle_protocol)
        self = old_to


def load_pandas(fname):
    "Load in a `TabularPandas` object from `fname`"
    distrib_barrier()
    res = pickle.load(open(fname, 'rb'))
    return res


if __name__ == "__main__":
    """Training Main

    The main function is called by both Unearthed's SageMaker pipeline and the
    Unearthed CLI's "unearthed train" command.

    WARNING - modifying this function may cause the submission process to fail.

    The main function must call preprocess, arrange th
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
    train(parser.parse_args())
