import logging
import argparse
import pandas as pd
from os.path import join

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--actual', type=str, default='/opt/ml/processing/input/public/val.csv.gz')
    parser.add_argument('--predicted', type=str, default='/opt/ml/processing/input/predictions/public/public.csv.out')
    parser.add_argument('--output', type=str, default='/opt/ml/processing/output/graph/private/predictions.csv')
    args = parser.parse_args()

    # Load the data and bring back the header.
    target_columns = []
    df_pred = pd.read_csv(args.predicted, header=None, names=target_columns)
    target = pd.read_csv(args.actual)

    # Insert a common "index" column
    df_pred.insert(0, 'index', target['timestamp'])

    logger.info(f"Reading predictions with shape {df_pred.shape}")

    print("Saving to" + args.output)
    df_pred.to_csv(args.output, index=False)
