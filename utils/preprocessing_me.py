

from fastai.tabular.all import *
from pathlib import Path
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


from fastai.tabular.all import *
from pathlib import Path
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.inspection import permutation_importance

from sklearn.feature_selection import (RFE,
                                       SelectKBest,
                                       SelectFromModel,
                                       mutual_info_regression,
                                       f_regression
                                         )
from sklearn.inspection import permutation_importance
import xgboost
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from preprocess import (check_limit_for_inputs,
                        categorize_and_cont,
                        target_columns)


seed_number = 42

def fill_missing(df,
               fill_type):
    """Filling missing numeric values
     with fill_type and also return
     missing statistics

    Args:
        df (pd.DataFrmae): DataFrame which needs to be
                           filled
        fill_type (str): median or mean 

    Returns:
        df_fill(pd.Sereis): missing Statistics
                            which column is filled 
                            with whic value
        new_df(pd.DataFrame):Filled dataFrame
    """


    if fill_type == 'median':
        df_fill = df.median()
        new_df = df.fillna(df.median()).copy()
    
    elif fill_type == 'mean':
        df_fill = df.mean()
        new_df = df.fillna(df.mean()).copy()
    else:
        df_fill = df.mode()
        new_df = df.fillna(df.mode()).copy()

    return df_fill, new_df

def train_test_data_creation(path,
                            fi_common_cols):
    """Create train test split only form this competition
    read data from path, change one ordinal value
    to number. convert to datetime to the columns
    date, add-datepart to date column, split data
    based on date,
    then converting numeric value to all columns



    Args:
        path (pathlib.Path): path of the csv file
    
    
    """
    df = pd.read_csv(path, parse_dates=True)
    operating_area_name_encoder = OrdinalEncoder(categories=[['05074bd887151b4dccb470dd3d26faad', '6bd116d685f1a5f6b4b773faf4212114', '86114fa7492257e065abca01e4753bba', '00f9e34b8bdaaf007602def09837636b', '3edf8ca26a1ec14dd6e91dd277ae1de6']]).fit(df['Operating_Area_Name'].to_numpy()[:, None])
    df['Operating_Area_Name'] = operating_area_name_encoder.transform(df['Operating_Area_Name'].to_numpy()[:, None])
    df['date'] = pd.to_datetime(df['date'])
    df = check_limit_for_inputs(df)
    df_new = add_datepart(df,'date',drop=False)
    logic = df['date'] <= pd.to_datetime('2021-01-01')
    df_train = df.loc[logic,:]
    df_valid =  df.loc[~logic,:]

    # categorize and also change to numbers
    df_train = categorize_and_continuous(
        df_train, dep_var=target_columns)
    df_valid = categorize_and_continuous(
        df_valid,
        dep_var=target_columns)

    df_fill, df_train = fill_missing(
                              df_train,
                              fill_type='median')
    df_train = df_train.drop(columns = ['date_num', 'Gas_Day_Date_num'])
    df_valid = df_valid.drop(columns=['date_num', 'Gas_Day_Date_num'])

    df_valid = df_valid.fillna(df_fill).copy()

    ### ####
    # now creating y_train, and y_valid

    y_train = df_train[target_columns[0]]
    X_train = df_train[fi_common_cols].drop(columns=target_columns[0])
    y_valid = df_valid[target_columns[0]]
    X_valid = df_valid[fi_common_cols].drop(columns=[target_columns[0]])
    return X_train, X_valid, y_train, y_valid



def rf_model(n_estimator, 
             max_depth,
             X_train, 
             y_train,
             X_valid=None,
             y_valid=None):

    model = RandomForestRegressor(
        n_jobs=-1, n_estimators=n_estimator,
        max_depth=max_depth,
        max_samples=200_000, max_features=0.5,
        min_samples_leaf=5, oob_score=True, 
        random_state=seed_number
    )

    #model = DecisionTreeRegressor(min_samples_leaf=25)
    model.fit(X_train, y_train)

    logger.info(f' {"=====" *10}')
    logger.info(f' model name is ===={type(model).__name__}')
    logger.info(f' {"=====" *10}')

    logger.info(f' {"=====" *10}')
    logger.info(f' number of trees = {n_estimator}')
    logger.info(f' {"=====" *10}')

    logger.info(f' {"====="*10}')

    logger.info(f' training score = {mean_absolute_error(model.predict(X_train), y_train)}')
    logger.info(f' {"====="*10}')

    logger.info(f' {"====="*10}')
    logger.info(
        f'training oob score = {mean_absolute_error(model.oob_prediction_, y_train)}')
    logger.info(f' {"====="*10}')

    if X_valid is not None:

        predictions = model.predict(X_valid)

        logger.info(f' {"====="*10}')

        logger.info(f' Valdation score = {mean_absolute_error(predictions, y_valid)}')

        logger.info(f' {"====="*10}')

        fig, ax = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(18,5), )
        axes = ax.ravel()
        axes[0].plot(y_valid.reset_index(drop=True),'bo', label='actual')
        axes[0].plot(model.predict(X_valid),'ro',label='prediction')
        axes[1].plot(model.predict(X_valid),y_valid,'bo')
        plt.legend()
    return model, predictions, y_valid.reset_index(drop=True)


def rf_feat_importance(m, df):
    fi=  pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False)

    print(f'total fi columns are == {len(fi)}')
    to_keep = fi[fi.imp > 0.005].cols

    print(f'removed no of columns are == {len(to_keep) - len(fi.cols)}')
    new_fi = fi[fi.imp > 0.005]
    new_fi.plot(
        'cols',
        'imp',
        'barh',
        figsize=(12,7),
        legend=False)
    return to_keep



def select_features_from_model(model, df):
    
    model = SelectFromModel(model,
                            prefit=True, 
                            threshold=0.005)
    feature_idx = model.get_support()
    feature_names = df.columns[feature_idx]
        
    return feature_names



def feature_importance_pi(clf, X, y, top_limit=None):

    # Retrieve the Bunch object after 50 repeats
    # n_repeats is the number of times that each feature was permuted to compute the final score
    bunch = permutation_importance(clf, X, y,
                                    n_repeats=50, random_state=42)

    # Average feature importance
    imp_means = bunch.importances_mean

    # List that contains the index of each feature in descending order of importance
    ordered_imp_means_args = np.argsort(imp_means)[::-1]

    # If no limit print all features
    if top_limit is None:
        top_limit = len(ordered_imp_means_args)

    whole_imp = []
    # Print relevant information

    for idx, (i, _) in enumerate(zip(ordered_imp_means_args, range(top_limit))):
        pi = pd.DataFrame()
        name = X.columns[i]
        imp_score = imp_means[i]
        imp_std = bunch.importances_std[i]
        pi.loc[idx, 'name'] = name
        pi.loc[idx, 'Score'] = imp_score
        pi.loc[idx, 'Score_std'] = imp_std 
    
        whole_imp.append(pi)
        print(f"Feature {name} with index {i} has an average importance score of {imp_score:.3f} +/- {imp_std:.3f}\n")
    return pd.concat(whole_imp)



def categorize_and_continuous(df, dep_var):
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
    return df.filter(cont + cat_num_columns + dep_var)
