import pandas as pd
import os
from utils import results_path

def csv_resume_file(mode, dir : str, year : str, nb) -> None:
    """
    Take all csv files in the directory and put them together in one csv file

    Args:
        results_base (Path): Path of the directory
        dir (str): The directory to search the xlsx files
        year (str): Year directory

    Returns:
       None

    """
    results_base = results_path / mode
    resultats = {'Models': [],
            'MAE ± standard deviation': [],
            'MSE' : [],
            'R2' : [],
            'nb_patients': ''}
    for file in os.listdir(results_base / year / dir / 'models'):
        if file.endswith('.csv'):
            df = pd.read_csv(results_base / year / dir / 'models' / file)
            if mode == "Training" :
                MAE = str(df['MAE'][len(df['MAE'])-3]) + ' +/- ' + str(df['MAE'][len(df['MAE'])-1])
            else:
                MAE = str(df['MAE'][len(df['MAE'])-3])
            MSE = str(df['MSE'][len(df['MSE'])-3])
            R2 = str(df['R2'][len(df['R2'])-3])
                           
            """if mode == "Training" :
                df_data = pd.read_excel(training_path)
            else :
                df_data = pd.read_excel(inference_path)
            df_data = df_data.dropna(subset=["3 mois Clair CKDEPI en ml/min/1.73m3"])
            if os.path.exists(config_path / "Training" / 'exclude.txt'):
                with open(config_path / "Training" / 'exclude.txt', 'r') as read_file:
                    for row in read_file:
                        df_data = df_data.drop(df_data[df_data['N° Biopsie'] == row.replace('\n','')].index)
            cols_to_ignore = ["1 an Clair CKDEPI en ml/min/1.73m2", "3 an Clair CKDEPI en ml/min/1.73m2",
                              '5 an Clair CKDEPI en ml/min/1.73m2', '7 an Clair CKDEPI en ml/min/1.73m2']
            cols_to_check = [col for col in df_data.columns if col not in cols_to_ignore]
            df_data = df_data.dropna(subset=cols_to_check)
            df_data = df_data.dropna(subset=[str(year).replace('_years',' an')+' Clair CKDEPI en ml/min/1.73m2'])
            nb_patients = df_data.shape[0]"""
            
            resultats['Models'].append(str(file).replace('Models_performances_','').replace('_'+year+'.csv',''))
            resultats['MAE ± standard deviation'].append(MAE)
            resultats['MSE'].append(MSE)
            resultats['R2'].append(R2)
            
            resultats['nb_patients'] = nb
            
    data = pd.DataFrame(resultats)
    if not os.path.exists(results_base / year / dir / 'resume'):
        os.mkdir(results_base / year / dir / 'resume')
    data.to_csv(results_base / year / dir / 'resume' / f'Models_comparison_{year}.csv', index=False)
    

