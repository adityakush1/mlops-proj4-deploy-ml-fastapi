"""
Script to post to FastAPI instance for model inference

Author: Aditya Kushwaha
Date: May 05, 2024
"""

import json
import requests


url = "http://127.0.0.1:8000/inference/"

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

response = requests.post(url, data=data)

print(f"status code : {response.status_code}")
print(f"content: \n{response.json()}")