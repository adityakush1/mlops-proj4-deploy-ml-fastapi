"""
Summary: API code goues here
Author: Aditya Kushwaha
Date: May 5, 2024
"""
import os
import pickle

from fastapi import FastAPI

# BaseModel from Pydantic is used to define data objects
from pydantic import BaseModel
import pandas as pd

from ml.data import process_data
from general_config import MODEL_PATH

saved_model_path = os.path.join(MODEL_PATH)

model_names = ['trained_model.pkl', 'encoder.pkl', 'labelizer.pkl']


# Declare the data object with its components and their type.
class QuerySample(BaseModel):
	age: int
	workclass: str 
	fnlgt: int
	education: str
	education_num: int
	marital_status: str
	occupation: str
	relationship: str
	race: str
	sex: str
	capital_gain: int
	capital_loss: int
	hours_per_week: int
	native_country: str

	class Config:
		schema_extra = {
						"example": {
									'age': 50,
									'workclass': "Private",
									'fnlgt': 234721,
									'education': "Doctorate",
									'education_num': 16,
									'marital_status': "Separated",
									'occupation': "Exec-managerial",
									'relationship': "Not-in-family",
									'race': "Black",
									'sex': "Female",
									'capital_gain': 0,
									'capital_loss': 0,
									'hours_per_week': 50,
									'native_country': "United-States"
									}
		}


# instantiate FastAPI app
app = FastAPI()


# Load saved models
def load_models(savepath, filename):
	model = pickle.load(open(os.path.join(savepath, filename[0]), "rb"))
	encoder = pickle.load(open(os.path.join(savepath, filename[1]), "rb"))
	lb = pickle.load(open(os.path.join(savepath, filename[2]), "rb"))
	return model, encoder, lb


# Prepare data for inference
def prepare_data(inference_data, cat_features, encoder, lb):
	sample, _, _, _ = process_data(
		inference_data,
		categorical_features=cat_features,
		training=False,
		encoder=encoder,
		lb=lb
	)
	return sample


@app.get("/")
async def home():
	return "Welcome to inference API for census data"


@app.post("/inference/")
async def ingest_data(inference: QuerySample):
	data = {'age': inference.age,
         'workclass': inference.workclass,
         'fnlgt': inference.fnlgt,
			'education': inference.education,
			'education-num': inference.education_num,
			'marital-status': inference.marital_status,
			'occupation': inference.occupation,
			'relationship': inference.relationship,
			'race': inference.race,
			'sex': inference.sex,
			'capital-gain': inference.capital_gain,
			'capital-loss': inference.capital_loss,
			'hours-per-week': inference.hours_per_week,
			'native-country': inference.native_country,
			}

	data_df = pd.DataFrame(data, index=[0])
	# Define categorical features
	cat_features = ["workclass", "education", "marital-status",
		"occupation", "relationship", "race", "sex", "native-country"]
 
	if all(os.path.isfile(os.path.join(saved_model_path, f)) for f in model_names):
		model, encoder, lb = load_models(saved_model_path, model_names)
	else:
		return {"message": "Saved models not found."}

	# Prepare data for inference
	sample = prepare_data(data_df, cat_features, encoder, lb)

	# Get model prediction
	prediction = model.predict(sample)

	# Convert prediction to label
	prediction_label = '>50K' if prediction[0] > 0.5 else '<=50K'

	# Add prediction label to data output
	data['prediction'] = prediction_label

	return data
