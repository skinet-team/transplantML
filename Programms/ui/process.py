from utils import results_path
from csv_resume import csv_resume_file
import threading
from utils import training_path, inference_path, config_path, results_path
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score, ConfusionMatrixDisplay, confusion_matrix
from performances_code import save_performance_results
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
import matplotlib
import models.m_Lasso
import models.elastic_net
import models.Bayesian_Regression
import models.kernel_ridge_regression
import models.random_forest
import models.svr_rbf
import models.lineaire
import models.Ridge_Regression
import models.omp
import models.polynomial_regression
import models.GradientBR_RKFold


BINS = 5
CLASSES = np.array([0, 15, 30, 45, 60, 90, np.inf])
LABELS  = np.array(["V", "IV", "IIIb", "IIIa", "II", "I"])


matplotlib.use('Agg')


def evaluate(mae_list, mse_list, rmse_list, r2_list, y_test_years, predictions_years) :
    mae_years = mean_absolute_error(y_test_years, predictions_years)
    mse_years = mean_squared_error(y_test_years, predictions_years)
    rmse_years = root_mean_squared_error(y_test_years, predictions_years)
    r2_years = r2_score(y_test_years, predictions_years)

    mae_list.append(mae_years)
    mse_list.append(mse_years)
    rmse_list.append(rmse_years)
    r2_list.append(r2_years)

    return(mae_list, mse_list, rmse_list, r2_list)


def mm(mode, pred, cv):

    print(f"INFO : Running condition \"{cv}\"")
    
    results_base = results_path / mode
    if cv == "Histological & clinical values" :
        if not os.path.exists(results_base / f'{pred[0]}_years' / "Histological & clinical values"):
            os.mkdir(results_base / f'{pred[0]}_years' / "Histological & clinical values")
        if not os.path.exists(results_base / f'{pred[0]}_years' / "Histological & clinical values" / 'Confusion Matrix'):
            os.mkdir(results_base / f'{pred[0]}_years' / "Histological & clinical values" / 'Confusion Matrix')
            
    if cv == "Histological values & DFG at 3 months":
        if not os.path.exists(results_base / f'{pred[0]}_years' / "Histological values & DFG at 3 months"):
            os.mkdir(results_base / f'{pred[0]}_years' / "Histological values & DFG at 3 months")
        if not os.path.exists(results_base / f'{pred[0]}_years' / "Histological values & DFG at 3 months" / 'Confusion Matrix'):
            os.mkdir(results_base / f'{pred[0]}_years' / "Histological values & DFG at 3 months" / 'Confusion Matrix')
            
    if cv == "Histological values & Age":
        if not os.path.exists(results_base / f'{pred[0]}_years' / "Histological values & Age"):
            os.mkdir(results_base / f'{pred[0]}_years' / "Histological values & Age")
        if not os.path.exists(results_base / f'{pred[0]}_years' / "Histological values & Age" / 'Confusion Matrix'):
            os.mkdir(results_base / f'{pred[0]}_years' / "Histological values & Age" / 'Confusion Matrix')

    if cv == "No clinical values & No DFG" :
        if not os.path.exists(results_base / f'{pred[0]}_years' / 'No clinical values & No DFG'):
            os.mkdir(results_base /
                     f'{pred[0]}_years' / 'No clinical values & No DFG')
        if not os.path.exists(results_base / f'{pred[0]}_years' / 'No clinical values & No DFG' / 'Confusion Matrix'):
            os.mkdir(results_base /
                     f'{pred[0]}_years' / 'No clinical values & No DFG' / 'Confusion Matrix') 

    if cv == "Clinical values only" :
        if not os.path.exists(results_base / f'{pred[0]}_years' / "Clinical values only"):
            os.mkdir(results_base /
                     f'{pred[0]}_years' / "Clinical values only")
        if not os.path.exists(results_base / f'{pred[0]}_years' / "Clinical values only" / 'Confusion Matrix'):
            os.mkdir(results_base /
                     f'{pred[0]}_years' / "Clinical values only" / 'Confusion Matrix')      
    
    if mode == "Training" :
        for item in os.scandir(training_path) :
            if os.path.splitext(item)[-1] in [".xlsx"]:
                training_file = item.path
                break
        df = pd.read_excel(training_file)
        print("INFO: Training data found at", training_path)
    else :
        for item in os.scandir(inference_path) :
            if os.path.splitext(item)[-1] in [".xlsx"]:
                inference_file = item.path
                break
        df = pd.read_excel(inference_file)
        print("INFO: Inferencing data found at", inference_path)
    print(f"INFO : {df.shape[0]} rows and {df.shape[1]} columns loaded")

    if os.path.exists(config_path / mode / 'excluded.txt'):
        with open(config_path / mode / 'excluded.txt', 'r') as read_file:
            for row in read_file :
                df = df.drop(df[df['N° Biopsie'] == row.replace('\n', '')].index)

    cols_to_ignore = ["1 an Clair CKDEPI en ml/min/1.73m2", "3 an Clair CKDEPI en ml/min/1.73m2",
                        '5 an Clair CKDEPI en ml/min/1.73m2', '7 an Clair CKDEPI en ml/min/1.73m2']
    cols_to_check = [col for col in df.columns if col not in cols_to_ignore]
    df = df.dropna(subset=cols_to_check)
    df = df.dropna(subset=[pred])

    print(f"INFO : {df.shape[0]} rows and {df.shape[1]} columns to process")
    
    if cv == "Histological & clinical values":
        x_years  =  df.iloc[:, np.r_[1:8, 15:df.shape[1]]]
    if cv == "Histological values & DFG at 3 months":
        x_years  =  df.iloc[:, np.r_[7, 15:df.shape[1]]]
    if cv == "Histological values & Age":
        x_years  =  df.iloc[:, np.r_[2, 15:df.shape[1]]]
    if cv == "No clinical values & No DFG":
        x_years  =  df.iloc[:, np.r_[15:df.shape[1]]]
    if cv == "Clinical values only":
        x_years  =  df.iloc[:, np.r_[1:7]]
    y_years  =  df[pred]

    column_names_path = results_base / f'{pred[0]}_years' / cv / 'used_column_names.txt'
    with open(column_names_path, 'w') as f:
        for column in x_years.columns:
            f.write(column + '\n\n')

    try:
        if (mode == "Training" and not os.path.exists(training_path)) or (mode == "Inference" and not os.path.exists(inference_path)) :
            raise FileExistsError
        else:
            rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)

            liste_noms_fonctions = ["m_Lasso", "elastic_net", "Bayesian_Regression", "kernel_ridge_regression",
                                    "random_forest", "svr_rbf", "lineaire", "Ridge_Regression", "omp", "polynomial_regression", "GradientBR_RKFold"]
            liste_fonctions = []
            liste_fonctions.append(getattr(models.m_Lasso, "m_Lasso"))
            liste_fonctions.append(getattr(models.elastic_net, "elastic_net"))
            liste_fonctions.append(getattr(models.Bayesian_Regression, "Bayesian_Regression"))
            liste_fonctions.append(getattr(models.kernel_ridge_regression, "kernel_ridge_regression"))
            liste_fonctions.append(getattr(models.random_forest, "random_forest"))
            liste_fonctions.append(getattr(models.svr_rbf, "svr_rbf"))
            liste_fonctions.append(getattr(models.lineaire, "lineaire"))
            liste_fonctions.append(getattr(models.Ridge_Regression, "Ridge_Regression"))
            liste_fonctions.append(getattr(models.omp, "omp"))
            liste_fonctions.append(getattr(models.polynomial_regression, "polynomial_regression"))
            liste_fonctions.append(getattr(models.GradientBR_RKFold, "GradientBR_RKFold"))  
            
            index_modele = 0
            for reference_fonction in liste_fonctions:
                mae_list = []
                mse_list = []
                rmse_list = []
                r2_list = []

                y_true_all = []
                y_pred_all = []

                if mode == "Training" :

                    for train_index, test_index in rkf.split(x_years) :

                        x_train_years, x_test_years = x_years.iloc[train_index, :], x_years.iloc[test_index, :]
                        y_train_years, y_test_years = y_years.iloc[train_index], y_years.iloc[test_index]

                        predictions_years = reference_fonction(mode, x_train_years, x_test_years, y_train_years,f'{pred[0]}_years',cv)

                        mae_list, mse_list, rmse_list, r2_list = evaluate(mae_list, mse_list, rmse_list, r2_list, y_test_years, predictions_years)
                        
                        y_test_years_ = np.digitize(y_test_years, CLASSES)

                        predictions_years_ = np.digitize(np.where(predictions_years < 0, 0, predictions_years), CLASSES)

                        y_true_all.extend(y_test_years_)
                        y_pred_all.extend(predictions_years_)

                    cm = confusion_matrix(y_true_all, y_pred_all)

                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
                    disp.plot()
                    plt.title(f'Confusion Matrix for {liste_noms_fonctions[index_modele]} - {pred[0]}_years')
                    plt.xlabel('Predicted classes')
                    plt.ylabel('True classes')

                else :
                    
                    predictions_years = reference_fonction(mode, x_years, x_years, y_years, f'{pred[0]}_years',cv)
                    
                    df[f"Prediction [{liste_noms_fonctions[index_modele]}]"] = predictions_years
                    with pd.ExcelWriter(results_base / f'{pred[0]}_years' / f"{cv}" / "predictions.xlsx") as writer :  
                        df.to_excel(writer, sheet_name='transplantML')
                    
                    mae_list, mse_list, rmse_list, r2_list = evaluate(mae_list, mse_list, rmse_list, r2_list, y_years, predictions_years)
                    
                    y_years_ = np.digitize(y_years, CLASSES)
                    predictions_years_ = np.digitize(np.where(predictions_years < 0, 0, predictions_years), CLASSES)
                    
                    cm = confusion_matrix(y_years_, predictions_years_)               

                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
                    disp.plot()
                    plt.title(f'Confusion Matrix for {liste_noms_fonctions[index_modele]} - {pred[0]}_years')
                    plt.xlabel('Predicted classes')
                    plt.ylabel('True classes')
                    
                save_performance_results(mode, mae_list, mse_list, rmse_list, r2_list, liste_noms_fonctions[index_modele], f"{pred[0]}_years", cv)
                plt.savefig(results_base / f'{pred[0]}_years' / f"{cv}" / 'Confusion Matrix' / f'confusion_matrix_{liste_noms_fonctions[index_modele]}_{pred[0]}_years.png')
                
                
                plt.close()

                index_modele += 1
        
        csv_resume_file(mode, cv, f"{pred[0]}_years", df.shape[0])
        
        print(f"INFO : Condition \"{cv}\" completed")

    except FileExistsError:
        print("Missing training file in the data folder", ':' , training_path)
