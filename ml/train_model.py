"""
Script to train machine learning model.
Author: Aditya Kushwaha
Date : May 4 2024
"""

import sys
import os
from datetime import datetime


from sklearn.model_selection import train_test_split
import pickle
import pandas as pd

from data import process_data, preprocess_remove_space
from model import train_model, compute_model_metrics, get_confusion_matrix, inference, compute_slices


from general_logger import get_general_logger
from general_config import DATA_PATH, MODEL_PATH

date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
logger = get_general_logger('train_model', date_str)

logger.debug('Starting')

# Add the necessary imports for the starter code.

# load in the data.
datapath = os.path.join(DATA_PATH,'census.csv')

data = pd.read_csv(datapath, header=0)
data = preprocess_remove_space(data)
print(data.columns)
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=20, stratify=data['salary'])

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
# setting train = False , we will use encoding from training set
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", 
    training=False, encoder=encoder, lb=lb
)

# saved models
model_path = os.path.join(MODEL_PATH)

model_names = ['trained_model.pkl', 'encoder.pkl', 'labelizer.pkl']

#check if saved models exists then load saved model else train and save model
if os.path.isfile(os.path.join(model_path,model_names[0])):
    with open(os.path.join(model_path,model_names[0]),'rb') as file:
        model = pickle.load(file)
    with open(os.path.join(model_path,model_names[1]),'rb') as file:
        encoder = pickle.load(file)
    with open(os.path.join(model_path,model_names[2]),'rb') as file:
        lb = pickle.load(file)
else:
    model = train_model(X_train,y_train)
    #save model to ../model
    with open(os.path.join(model_path,model_names[0]), 'wb') as file:
        pickle.dump(model, file)
    with open(os.path.join(model_path,model_names[1]),'wb') as file:
        pickle.dump(encoder, file)
    with open(os.path.join(model_path,model_names[2]),'wb') as file:
        pickle.dump(lb, file)


# Evaluate trained model on test set
pred = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test,pred)

logger.debug(f"Classification target labels: {list(lb.classes_)}")
logger.debug(f"precision:{precision} , recall:{recall} , fbeta:{fbeta}")

conf_mat = get_confusion_matrix(y_test, pred)

logger.info(f"Confusion matrix :\n {conf_mat}")

#compute performance on slices
#save results
slice_path = os.path.join(os.getcwd())

with open(os.path.join(slice_path,'slice_output.txt'),'a') as file:
    for feature in cat_features:

        perf_df = compute_slices(test, feature, y_test, pred)
        perf_df.to_csv(file,mode='a', index=False)
        
        logger.info(f"Performance on slice {feature}")
        logger.info(perf_df)
