from pathlib import Path
from customtkinter import CTkImage
from PIL import Image
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import GridSearchCV
import subprocess as sp
import pandas as pd
import numpy as np
import pickle
import os

global training_path, inference_path, results_path, models_path, training_path_case

root_path = Path(os.getcwd()).resolve()

logs_path = root_path / os.pardir / "Logs"

training_path = root_path / os.pardir / "Data" / "Training"

inference_path = root_path / os.pardir / 'Data' / 'Inference'

config_path = root_path / os.pardir / "Config"

models_path = root_path / "Models"

params_path = root_path / os.pardir / "Params"

results_path = root_path / os.pardir / "Results"

def save_param(model, model_name,year,model_type ):
    if not os.path.exists(params_path):
        os.mkdir(params_path)
    if not os.path.exists(params_path / model_name):
        os.mkdir(params_path / model_name)
    if not os.path.exists(params_path / model_name / model_type):
        os.mkdir(params_path / model_name / model_type)
    if not os.path.exists(params_path / model_name/ model_type / year):
        os.mkdir(params_path / model_name/ model_type / year)
    pickle.dump(model, open( params_path / model_name / model_type / year / 'model.pickle', 'wb'))

def load_param(model_name,year,model_type ):
    return pickle.load(open( params_path / model_name / model_type / year / 'model.pickle', 'rb'))
