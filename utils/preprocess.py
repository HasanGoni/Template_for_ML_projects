"""Preprocess Template.

This script will be invoked in two ways during the Unearthed scoring pipeline:
 - first during model training on the 'public' dataset
 - secondly during generation of predictions on the 'private' dataset
"""

from sklearn.preprocessing import OrdinalEncoder
from os import getenv
import pandas as pd
import logging
import argparse
import numpy as np
from pathlib import Path
import os
from tkinter import W
os.system('pip install fastai')
from fastai.tabular.all import *

#from ensemble_model import *
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

seed_number = 42


def categorize_and_cont(df, dep_var):
    """
    Convert string columns to categorical
    columns and also cat.codes will be 
    available in new columns
    Args:
        df (pd.DataFrame): input dataframe
        dep_var (str): name of dependent
                       variable
    """
    cont, cat = cont_cat_split(df, dep_var=dep_var)
    # as we are changing some part
    # of a dataframe in case of training
    # data or validaiton data
    df = df.copy()
    cat_num_columns = []
    for i in cat:
        new_name = f'{i}_num'
        df[i] = df[i].astype('category')
        df[new_name] = df[i].cat.codes + 1
        cat_num_columns.append(new_name)
    return df, cat_num_columns


target_columns = [
    'Downhole Gauge Pressure'
]

train_columns = [
    'Casing Pressure',
    'Tubing Pressure',
    'Gas Gathering Pressure',
    'Tubing Flow Meter',
    'Water Flow Mag from Separator',
    'Pump Torque',
    'Pump Speed Actual',
    'WEC PCP Efficiency',
    'Separator Gas Pressure',
    'FCV Position Feedback',
    'WEC PCP Theoretical Pump Displacement',
    'pump_bottom_depth',
    'sensor_depth',
    'Operating_Area_Name'
]

fi_common_cols = [
    'sensor_depth',
    'Water Gathering Pressure',
    'Pump Speed Actual Max',
    'Well_Surface_Latitude',
    'Tubing Flow Meter',
    'pump_bottom_depth',
    'Pump Speed Actual Min',
    'Tubing Pressure',
    'Elapsed_num',
    'Base_Depth_m_TVD_min',
    'Net_Pay_m_mean',
    'Tubing Flow Meter Scale High|Meter Type_num',
    'Casing Pressure',
    'Cumulative_Net_Pay_m_By_Depth_max',
    'PBTD_Bottom_Depth_m_KB',
    'FCV Position Feedback',
    'Top_Depth_m_TVD_min',
    'Perforation_Max_Bottom_Depth_m_KB',
    'Latest_Pump_Bottom_Depth_m',
    'Cumulative_Net_Pay_m_By_Depth_mean',
    'Pump Speed Actual',
    'Total_Net_Pay',
    'Pump Torque',
    'Base_Depth_m_TVD_max',
    'Well_Surface_Longitude',
    'Gas Gathering Pressure',
    'WEC PCP Theoretical Pump Displacement',
    'Water Flow Mag from Separator',
    'Downhole_Gauge_Sensor_Depth_from_Rotary_Table_m',
    'Top_Depth_m_TVD_max',
    'Well_Operator_Run_num',
    'Base_Depth_m_TVD_mean',
    'WEC PCP Efficiency',
    'GL_MKB',
    'Gas Flow (Energy)',
    'Top_Depth_m_TVD_mean',
    'Well_Number_num',
    'Perforation_Min_Top_Depth_m_KB',
    'Downhole Gauge Type_num',
    'Cumulative_Net_Pay_m_By_Depth_min',
    'Net_Pay_m_max',
    'Downhole Gauge Pressure']


median_info = {'sensor_depth': 660.0,
               'Water Gathering Pressure': 155.69389303525287,
               'Pump Speed Actual Max': 76.03378295898438,
               'Well_Surface_Latitude': -0.5619353068411037,
               'Tubing Flow Meter': 10.335853251318138,
               'pump_bottom_depth': 694.71,
               'Pump Speed Actual Min': 68.0,
               'Tubing Pressure': 319.9164962768555,
               'Elapsed_num': 589.0,
               'Base_Depth_m_TVD_min': 156.94,
               'Net_Pay_m_mean': 0.3817808219178082,
               'Tubing Flow Meter Scale High|Meter Type_num': 4.0,
               'Casing Pressure': 350.16974369684857,
               'Cumulative_Net_Pay_m_By_Depth_max': 28.63,
               'PBTD_Bottom_Depth_m_KB': 733.8,
               'FCV Position Feedback': 99.89084438482922,
               'Top_Depth_m_TVD_min': 156.88,
               'Perforation_Max_Bottom_Depth_m_KB': 690.9000298999999,
               'Latest_Pump_Bottom_Depth_m': 694.71,
               'Cumulative_Net_Pay_m_By_Depth_mean': 13.5334126984127,
               'Pump Speed Actual': 72.19791666666667,
               'Total_Net_Pay': 26.73,
               'Pump Torque': 199.76513020197552,
               'Base_Depth_m_TVD_max': 703.633,
               'Well_Surface_Longitude': -2.5988617806839898,
               'Gas Gathering Pressure': 289.69443464279175,
               'WEC PCP Theoretical Pump Displacement': 24.0,
               'Water Flow Mag from Separator': 14.321628747388482,
               'Downhole_Gauge_Sensor_Depth_from_Rotary_Table_m': 670.25,
               'Top_Depth_m_TVD_max': 701.49,
               'Well_Operator_Run_num': 9.0,
               'Base_Depth_m_TVD_mean': 463.26863768115936,
               'WEC PCP Efficiency': 54.93847805835928,
               'GL_MKB': 322.82,
               'Gas Flow (Energy)': 380.89769490804906,
               'Top_Depth_m_TVD_mean': 460.80426966292123,
               'Well_Number_num': 402.0,
               'Perforation_Min_Top_Depth_m_KB': 372.400027,
               'Downhole Gauge Type_num': 4.0,
               'Cumulative_Net_Pay_m_By_Depth_min': 0.172,
               'Net_Pay_m_max': 2.28,
               'Downhole Gauge Pressure': 841.5683081944783}

median_info = pd.Series(median_info)
median_info_without_target = median_info.drop(
    target_columns[0]
).copy()


def check_limit_for_inputs(df):
    """Check manually limit from 
    plot and change the limit based on that

    Args:
        df (pd.DataFrame): input dataframe
    """

    df['Pump Speed Actual Max'] = np.where(
        df['Pump Speed Actual Max'] > 450, 450, df['Pump Speed Actual Max'])

    df['Pump Speed Actual Min'] = np.where(
        df['Pump Speed Actual Min'] > 450, 450, df['Pump Speed Actual Min'])

    df['Net_Pay_m_mean'] = np.where(
        df['Net_Pay_m_mean'] > 4, 4, df['Net_Pay_m_mean'])

    df['Casing Pressure'] = np.where(
        df['Casing Pressure'] > 6000, 6000, df['Casing Pressure'])

    df['Cumulative_Net_Pay_m_By_Depth_max'] = np.where(
        df['Cumulative_Net_Pay_m_By_Depth_max'] > 200, 200, df['Cumulative_Net_Pay_m_By_Depth_max'])

    df['FCV Position Feedback'] = np.where(
        df['FCV Position Feedback'] > 120, 120, df['FCV Position Feedback'])

    df['Cumulative_Net_Pay_m_By_Depth_mean'] = np.where(
        df['Cumulative_Net_Pay_m_By_Depth_mean'] > 100, 100, df['Cumulative_Net_Pay_m_By_Depth_mean'])

    df['Pump Speed Actual'] = np.where(
        df['Pump Speed Actual'] > 500, 500, df['Pump Speed Actual'])

    df['WEC PCP Theoretical Pump Displacement'] = np.where(
        df['WEC PCP Theoretical Pump Displacement'] > 200, 200, df['WEC PCP Theoretical Pump Displacement'])

    df['Gas Flow (Energy)'] = np.where(
        df['Gas Flow (Energy)'] > 6000, 6000, df['Gas Flow (Energy)'])

    df['Cumulative_Net_Pay_m_By_Depth_min'] = np.where(
        df['Cumulative_Net_Pay_m_By_Depth_min'] > 5, 5, df['Cumulative_Net_Pay_m_By_Depth_min'])

    df['Tubing Pressure'] = np.where(
        df['Tubing Pressure'] > 5900, 5900, df['Tubing Pressure'])
    df['Gas Gathering Pressure'] = np.where(
        df['Gas Gathering Pressure'] > 800, 800, df['Gas Gathering Pressure'])
    df['Tubing Flow Meter'] = np.where(
        df['Tubing Flow Meter'] > 400, 400, df['Tubing Flow Meter'])
    df['Water Flow Mag from Separator'] = np.where(
        df['Water Flow Mag from Separator'] > 500, 500, df['Water Flow Mag from Separator'])
    df['Pump Torque'] = np.where(
        df['Pump Torque'] > 1300, 1300,
        df['Pump Torque'])

    df['Pump Torque'] = np.where(
        df['Pump Torque'] < 0, 0,
        df['Pump Torque'])
    df['WEC PCP Efficiency'] = np.where(
        df['WEC PCP Efficiency'] < 0, 0,
        df['WEC PCP Efficiency'])

    df['WEC PCP Efficiency'] = np.where(
        df['WEC PCP Efficiency'] > 300, 300,
        df['WEC PCP Efficiency'])  # 600 possible
    return df


def preprocess(data_file, drop_targets):
    """Apply preprocessing and featurization steps to each file in the data directory.

    Your preprocessing and feature generation goes here.
    """
    logger.info(f"running preprocess on {data_file}")

    # read the data file
    df = pd.read_csv(data_file, parse_dates=True)
    #df = df.drop(columns=target_columns[0]).copy()
    logger.info(f"data read from {data_file} has shape of {df.shape}")

    # add preprocessing here
    operating_area_name_encoder = OrdinalEncoder(categories=[['05074bd887151b4dccb470dd3d26faad', '6bd116d685f1a5f6b4b773faf4212114', '86114fa7492257e065abca01e4753bba',
                                                              '00f9e34b8bdaaf007602def09837636b', '3edf8ca26a1ec14dd6e91dd277ae1de6']]).fit(df['Operating_Area_Name'].to_numpy()[:, None])
    df['Operating_Area_Name'] = operating_area_name_encoder.transform(
        df['Operating_Area_Name'].to_numpy()[:, None])
    #df.fillna(-1, inplace=True)

    logger.info(f"data after preprocessing has shape of {df.shape}")
    df['date'] = pd.to_datetime(df['date'])

    df = check_limit_for_inputs(df)

    df = add_datepart(df, 'date', drop=True)

    df, cat_cols = categorize_and_cont(
        df, dep_var=target_columns)
    logger.info(f"{'###'*20}")
    logger.info(f"data after adding datepart has shape of {df.shape}")
    logger.info(f"{'####'*20}")

    # Removing outlier (just the data which has a limit more than normal in visualization)
    logger.info(f"{'###'*20}")
    logger.info(f"data after checking limit has shape of {df.shape}")
    logger.info(f"{'###'*20}")
    df = df.filter(fi_common_cols)

    logger.info(f"fi_common_cols len == {len(fi_common_cols)}")
    logger.info(f"{'###'*20}")
    logger.info(f"data after filter columns has shape of {df.shape}")
    logger.info(f"{'#####'*20}")

    logger.info(f"printing na information ")
    logger.info(f"{df.isna().sum()}")  
    logger.info(f"printing dataframe informaiton")
    logger.info(f"{df.describe()}")
    logger.info(f"{'###'*20}")
    logger.info(f"data after filter columns has shape of {df.shape}")
    logger.info(f"{'#####'*20}")

    if target_columns[0] in df.columns:
        df_train = df.copy()
        df_train = df_train.fillna(median_info)
    else:
        df_train = df.fillna(
            median_info_without_target).copy()

    #procs = [Categorify, FillMissing]

    # if target_columns in df.columns:
        # splits = RandomSplitter(
        # valid_pct=0.1,
        # seed=42)(range_of(df))

    # print(splits)

        # cont, cat = cont_cat_split(
        #df, 1,
        # dep_var=target_columns)

        #logger.info('printing columns')

        #logger.info('converting fastai dataframe')

        # to = TabularPandas(
        # df,
        # procs,
        # cat,
        # cont,
        #y_names='Downhole Gauge Pressure',
        # splits=splits)

        #df_train = to.train.xs
        #df_train[target_columns] = to.train.y
        # to.export('to.pkl')

    #logger.info('for test data part')

    #to_load = load_pandas('to.pkl')

    #to_new = to_load.train.new(df)
    # to_new.process()
    #df_train = to_new.xs

    # Optionally drop target columns depending on the context.
    try:
        if drop_targets:
            df_train.drop(columns=target_columns, inplace=True)
    except KeyError:
        pass

    logger.info(f"data after preprocessing has shape of {df_train.shape}")
    return df_train


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
    """Preprocess Main

    The main function is called by both Unearthed's SageMaker pipeline and the
    Unearthed CLI's "unearthed preprocess" command.

    WARNING - modifying this file may cause the submission process to fail.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, default="/opt/ml/processing/input/public/public.csv.gz"
    )
    parser.add_argument(
        "--output", type=str, default="/opt/ml/processing/output/preprocess/public.csv"
    )
    args, _ = parser.parse_known_args()

    # call preprocessing on private data
    df = preprocess(args.input, True)

    logger.info(f"preprocessed result shape is {df.shape}")

    # write to the output location
    df.to_csv(args.output)
