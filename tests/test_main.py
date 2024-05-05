""" 
pytest Unit test of main.py API module
Author : Aditya Kushwaha 
Date: May 05, 2024
"""


import json, os, sys
from datetime import datetime

from fastapi.testclient import TestClient

current_dir = os.path.dirname(os.path.abspath(__file__))
# Adding parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

#print(sys.path)
from main import app
from general_logger import get_general_logger


date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
logger = get_general_logger('test_main', date_str)

logger.debug('Starting')


client = TestClient(app)

def test_home():
    r = client.get("/")
    
    assert r.status_code == 200
    assert r.json() == "Welcome to inference API for census data"
    

def test_inference():
    """ 
    test Model inference o/p
    """
    
    sample = {
    'age': 55,
    'workclass': "Private",
    'fnlgt': 255123,
    'education': "Masters",
    'education_num': 15,
    'marital_status': "Separated",
    'occupation': "Exec-managerial",
    'relationship': "Not-in-family",
    'race': "Black",
    'sex': "Female",
    'capital_gain': 0,
    'capital_loss': 0,
    'hours_per_week': 45,
    'native_country': "United-States"}
    
    data = json.dumps(sample)
    r = client.post("/inference/", data=data)
    
    assert r.status_code == 200
    assert r.json()['age'] == 55
    assert r.json()['fnlgt'] == 255123
    
    logger.info(f'Prediction = {r.json()["prediction"]}')


def test_inference_2():
    """ 
    test Model inference o/p
    """
    
    sample =  { 'age':53,
            'workclass':"Private", 
            'fnlgt':234721,
            'education':"11th",
            'education_num':7,
            'marital_status':"Married-civ-spouse",
            'occupation':"Handlers-cleaners",
            'relationship':"Husband",
            'race':"Black",
            'sex':"Male",
            'capital_gain':0,
            'capital_loss':0,
            'hours_per_week':40,
            'native_country':"United-States"
            }
    
    data = json.dumps(sample)
    r = client.post("/inference/", data=data)
    
    assert r.status_code == 200
    assert r.json()['age'] == 53
    assert r.json()['fnlgt'] == 234721
    
    logger.info(f'Prediction = {r.json()["prediction"]}')
    
    
def test_incorrect_inference_query():
    """
    Test incomplete sample does not generate prediction
    """
    sample =  {  'age':50,
                'workclass':"Private", 
                'fnlgt':234721,
            }

    data = json.dumps(sample)
    r = client.post("/inference/", data=data )

    assert 'prediction' not in r.json().keys()
    logger.warning(f"The sample has {len(sample)} features.")



if '__name__' == '__main__':
    test_home()
    test_inference()
    test_inference_2()
    test_incorrect_inference_query()