
import numpy as np
import pandas as pd
import os

from utils import results_path


def save_performance_results(mode, mae_list: list, mse_list: list, rmse_list: list, r2_list: list, nom_modele: str, annee: str, cv: str):
    """
    Creates a persormnace csv file with all the mae, mse, rmse and r2 values.
    Makes also a avergae value of all the parameters

    Args:
        mae_list (list): list of mean absolute error
        mse_list (list): list of mean square error
        rmse_list (list): list of mean square root error
        r2_list (list): list of coefficient of dertermination
        nom_modele (str): name of the model
        annee (str): test year
        cv (str): traning mode
    """
    mae_mean = np.mean(mae_list)
    mse_mean = np.mean(mse_list)
    rmse_mean = np.mean(rmse_list)
    r2_mean = np.mean(r2_list)

    mae_std = np.std(mae_list)

    performance_dict = {
        'Split': list(range(1, len(mae_list) + 1)),
        'MAE': mae_list,
        'MSE': mse_list,
        'RMSE': rmse_list,
        'R2': r2_list
    }

    performance_df = pd.DataFrame(performance_dict)

    performance_df = pd.concat([performance_df, pd.DataFrame(index=[None])])

    average_performance_dict = {
        'Split': ['Moyenne'],
        'MAE': [mae_mean],
        'MSE': [mse_mean],
        'RMSE': [rmse_mean],
        'R2': [r2_mean]
    }

    average_performance_df = pd.DataFrame(average_performance_dict)

    average_performance_df = pd.concat(
        [average_performance_df, pd.DataFrame(index=[None])])

    std_performance_dict = {
        'Split': ['écart type (std) MAE'],
        'MAE': [mae_std],
        'MSE': [np.nan],
        'RMSE': [np.nan],
        'R2': [np.nan]
    }

    std_performance_df = pd.DataFrame(std_performance_dict)

    final_performance_df = pd.concat(
        [performance_df, average_performance_df, std_performance_df], ignore_index=True)

    results_base = results_path / mode
    if cv == 'Histological & clinical values':
        if not os.path.exists(results_base / f'{annee}' / 'Histological & clinical values' / 'models'):
            os.mkdir(results_base / f'{annee}' /
                     'Histological & clinical values' / 'models')
        final_performance_df.to_csv(results_base / f'{annee}' / 'Histological & clinical values' / 'models' / f'Models_performances_{
                                    nom_modele}_{annee}.csv', index=False, lineterminator='\n\n')
    if cv == "Histological values & DFG at 3 months":
        if not os.path.exists(results_base / f'{annee}' / "Histological values & DFG at 3 months" / 'models'):
            os.mkdir(results_base / f'{annee}' /
                     "Histological values & DFG at 3 months" / 'models')
        final_performance_df.to_csv(results_base / f'{annee}' / "Histological values & DFG at 3 months" / 'models' / f'Models_performances_{
                                    nom_modele}_{annee}.csv', index=False, lineterminator='\n\n')
    if cv == "Histological values & Age":
        if not os.path.exists(results_base / f'{annee}' / "Histological values & Age" / 'models'):
            os.mkdir(results_base / f'{annee}' / "Histological values & Age" / 'models')
        final_performance_df.to_csv(results_base / f'{annee}' / "Histological values & Age" / 'models' / f'Models_performances_{
                                    nom_modele}_{annee}.csv', index=False, lineterminator='\n\n')
    if cv == 'No clinical values & No DFG':
        if not os.path.exists(results_base / f'{annee}' / 'No clinical values & No DFG' / 'models'):
            os.mkdir(
                results_base / f'{annee}' / 'No clinical values & No DFG' / 'models')
        final_performance_df.to_csv(results_base / f'{annee}' / 'No clinical values & No DFG' /
                                    'models' / f'Models_performances_{nom_modele}_{annee}.csv', index=False, lineterminator='\n\n')
    if cv == "Clinical values only":
        if not os.path.exists(results_base / f'{annee}' / "Clinical values only" / 'models'):
            os.mkdir(
                results_base / f'{annee}' / "Clinical values only" / 'models')
        final_performance_df.to_csv(results_base / f'{annee}' / "Clinical values only" /
                                    'models' / f'Models_performances_{nom_modele}_{annee}.csv', index=False, lineterminator='\n\n')
    
