import pandas as pd
import numpy as np

import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from itertools import product

from tqdm import tqdm

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import OneHotEncoder

from category_encoders import TargetEncoder

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

import warnings
warnings.filterwarnings('ignore')

def normalize_scaler(data):
    data_copy = data.copy()
    for col in data.columns:
        mean_data = data_copy[col].mean()
        range_data = data_copy[col].max()-data_copy[col].min()
        data_copy[col] = data_copy[col].apply(lambda x: (x-mean_data)/range_data)
    return data_copy

def percent_outs(array):
    length = len(array)
    neg_count = sum(array==-1)
    p_outs = neg_count/length*100
    return p_outs

def metricas(y_train, y_train_pred, y_test, y_test_pred):
    metricas = {
    'train': {
        'r2_score': r2_score(y_train, y_train_pred),
        'MAE': mean_absolute_error(y_train, y_train_pred),
        'MSE': mean_squared_error(y_train, y_train_pred),
        'RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred))
    },
    'test': {
        'r2_score': r2_score(y_test, y_test_pred),
        'MAE': mean_absolute_error(y_test, y_test_pred),
        'MSE': mean_squared_error(y_test, y_test_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred))

    }
}
    return metricas

def impute_nulls(data, method = "rf", neighbors = 5):
    df_numeric = data.select_dtypes('number')
    if method == "knn":
        imputer_knn = KNNImputer(n_neighbors=neighbors, verbose = 1)
        df_imput = pd.DataFrame(imputer_knn.fit_transform(df_numeric), columns=df_numeric.columns, index=data.index)
    elif method == "base":
        imputer_it = IterativeImputer(verbose = 1)
        df_imput = pd.DataFrame(imputer_it.fit_transform(df_numeric), columns=df_numeric.columns, index=data.index)
    elif method == "rf":
        imputer_forest = IterativeImputer(estimator=RandomForestRegressor(n_jobs=-1), verbose = 2)
        df_imput = pd.DataFrame(imputer_forest.fit_transform(df_numeric), columns=df_numeric.columns, index=data.index)
    return df_imput

def normalize_scaler(data):
    data_copy = data.copy()
    for col in data.columns:
        mean_data = data_copy[col].mean()
        range_data = data_copy[col].max()-data_copy[col].min()
        data_copy[col] = data_copy[col].apply(lambda x: (x-mean_data)/range_data)
    return data_copy

def scale_data(data, columns, method = "robust"):
    if method == "minmax":
        scaler = MinMaxScaler()
    elif method == "robust":
        scaler = RobustScaler()
    elif method == "standard":
        scaler = StandardScaler()
    elif method == "norm":
        df_scaled = normalize_scaler(data[columns])
    df_scaled = pd.DataFrame(scaler.fit_transform(data[columns]), columns=columns)    
    return df_scaled

def percent_outs(array):
    length = len(array)
    neg_count = sum(array==-1)
    p_outs = neg_count/length*100
    return p_outs

def find_outliers(data, columns, method = "ifo", random_state = 42, threshold = 70): 
    df = data.copy()
    selected_data = df[columns]
    ests = np.linspace(1,1000, 5, dtype = int)
    conts = np.linspace(0.01,0.2,5)
    neighs = np.linspace(15,45,5, dtype=int)
    if method == "ifo":   
        forest_arg_combis = list(product(ests, conts))
        for n,m in tqdm(forest_arg_combis):
            iforest = IsolationForest(random_state=random_state, n_estimators=n, contamination=m, n_jobs=-1)
            df[f"iforest_{n}_{m:.3f}"] = iforest.fit_predict(X=selected_data)
        df_detected = df.filter(like="iforest")
    elif method == "lof":
        lof_combis = list(product(neighs, conts))
        for neighbour, contaminacion in tqdm(lof_combis):
            lof = LocalOutlierFactor(n_neighbors=neighbour, contamination=contaminacion, n_jobs=-1)
            df[f"lof_{neighbour}_{contaminacion:.3f}"] = lof.fit_predict(X = selected_data)
        df_detected = df.filter(like="lof_")

    percentages = df_detected.apply(percent_outs, axis=1)
    df_outliers = df[percentages>threshold]
    return df_outliers

def get_metrics(y_train, y_train_pred,y_test, y_test_pred):
    metricas = {
    'train': {
        'r2_score': r2_score(y_train, y_train_pred),
        'MAE': mean_absolute_error(y_train, y_train_pred),
        'MSE': mean_squared_error(y_train, y_train_pred),
        'RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred))
    },
    'test': {
        'r2_score': r2_score(y_test, y_test_pred),
        'MAE': mean_absolute_error(y_test, y_test_pred),
        'MSE': mean_squared_error(y_test, y_test_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred))

    }
    }
    return pd.DataFrame(metricas).T

def encode_onehot(data, columns):
    onehot = OneHotEncoder()
    trans_one_hot = onehot.fit_transform(data[columns])
    oh_df = pd.DataFrame(trans_one_hot.toarray(), columns=onehot.get_feature_names_out())
    return oh_df

def encode_target(data, columns, response_var):
    encoder = TargetEncoder(cols = columns)
    df_encoded = encoder.fit_transform(X = data, y = data[response_var])
    return df_encoded

def create_model(params, X_train, y_train, method = DecisionTreeRegressor(), cv= 5, scoring = "neg_mean_squared_error"):
    grid_search = GridSearchCV(estimator = method, param_grid=params, cv = cv, scoring = scoring, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    return grid_search