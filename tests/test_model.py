"""
Unit test of model.py using pytest
Author: Aditya Kushwaha
date: May 5, 2024
"""
import os, sys

from datetime import datetime

import pytest, pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError


current_dir = os.path.dirname(os.path.abspath(__file__))
# Adding parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from general_logger import get_general_logger
from general_config import TEST_PATH, DATA_PATH, MODEL_PATH

from ml.data import process_data
from ml.model import compute_model_metrics, inference, get_confusion_matrix



date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
logger = get_general_logger('test_model', date_str)

logger.debug('Starting')


@pytest.fixture(scope="module")
def data():
    #load data
    data_file = os.path.join(DATA_PATH,'census.csv')
    data_df = pd.read_csv(data_file, header=0)
    data_df.columns = [col.strip() for col in data_df.columns]
    data_df = data_df.applymap(lambda x : x.strip() if isinstance(x,str) else x )
    return data_df


@pytest.fixture(scope='module')
def data_path():
    return os.path.join(DATA_PATH,'census.csv')


@pytest.fixture(scope="module")
def features():
    """
    Get all categorical features
    """
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"]
    
    return cat_features
    
"""
testcases
"""


@pytest.fixture(scope='module')
def train_data(data, features):
    train, test = train_test_split(data, test_size=0.20, random_state=20, stratify=data['salary'])
    
    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=features,
        label="salary",
        training=True
        )
    
    return X_train, y_train



def test_get_data(data_path):
    """
    validate if data file exist and its shape
    """
    try:
        with open(data_path,'r') as file:
            df = pd.read_csv(file, header=0)
    except FileNotFoundError as err:
        logger.error(f"Could not find file : {data_path}")
        raise err
    
    try:
        assert df.shape[0]>0
        assert df.shape[1]>0
        
    except AssertionError as err:
        logger.error("Validating data file : The file is empty")
        raise err
    

def test_features(data, features):
    """
    test for presence of categorical features and target feature i.e. salary
    """
    try:
        assert sorted(set(data.columns).intersection(features))==sorted(features)
        assert 'salary' in list(data.columns)
        
    except AssertionError as err:
        logger.error("Testing features: Features are missing from dataset")
        raise err
    
    
def test_process_data(data,features):
    """
        Test for data.process_data
    """
    print(data.columns)
    try:
        X, y, encoder, lb = process_data(data,features, label="salary", training=True)
        
        assert X.shape[0] == data.shape[0]
        assert  len(y) == len(data['salary'])
    except Exception as err:
        logger.error("Test process_data failed")
        raise err

def test_is_saved_model():
    model_path = MODEL_PATH
    try:
        with open(os.path.join(model_path,'trained_model.pkl'),'rb') as file:
            _ = pickle.load(file)
    except Exception as err:
        logger.error("Testing saved model failed")
        raise err


def test_inference(train_data):
    X_train, y_train = train_data
    model_path = MODEL_PATH
    
    try:
        with open(os.path.join(model_path,'trained_model.pkl'),'rb') as file:
            model = pickle.load(file)
            
            try:
                preds = inference(model, X_train)
            
            except Exception as err:
                logger.error("Testing inference failed on trained data")
                raise err
    except Exception as err:
        logger.error("Testing inference failed to load saved model")
        raise err
    
    
def test_compute_model_metrics(train_data):
    X_train, y_train = train_data
    model_path = MODEL_PATH
    
    try:
        with open(os.path.join(model_path,'trained_model.pkl'),'rb') as file:
            model = pickle.load(file)
            preds = inference(model, X_train)
            try:
                precision, recall, fbeta = compute_model_metrics(y_train, preds)
            except Exception as err:
                logger.error("performance metrics can not be computed on trained data")
                raise err
    except Exception as err:
        logger.error("Testing inference failed to load saved model / getting inference")
        raise err
    

def test_get_confusion_matrix(train_data):
    X_train, y_train = train_data
    model_path = MODEL_PATH
    
    try:
        with open(os.path.join(model_path,'trained_model.pkl'),'rb') as file:
            model = pickle.load(file)
            preds = inference(model, X_train)
            try:
                cm = get_confusion_matrix(y_train, preds)
            except Exception as err:
                logger.error("Confusion matrix can not be computed on trained data")
                raise err
    except Exception as err:
        logger.error("Testing inference failed to load saved model / getting inference")
        raise err