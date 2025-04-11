import pandas as pd
from sklearn.linear_model import BayesianRidge
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from utils import save_param, load_param

def Bayesian_Regression(mode, X_train, X_test, y_train, year, cvo):

    if mode == "Training" :
    
        # Définir les valeurs d'alpha_1 à tester
        alpha_1_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

        # Définir les valeurs d'alpha_2 à tester
        alpha_2_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

        # Définir les valeurs de lambda_1 à tester
        lambda_1_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

        # Définir les valeurs de lambda_2 à tester
        lambda_2_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

        # Créer un dictionnaire des paramètres à tester
        param_grid = {'alpha_1': alpha_1_values, 'alpha_2': alpha_2_values, 'lambda_1': lambda_1_values, 'lambda_2': lambda_2_values}

        # Initialiser le modèle de régression bayésienne
        bayesian_model = BayesianRidge()

        # Utiliser GridSearchCV pour trouver les meilleures valeurs d'hyperparamètres
        grid_search = GridSearchCV(bayesian_model, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)

        # Obtenir les meilleures valeurs d'hyperparamètres
        best_params = grid_search.best_params_
        best_alpha_1 = grid_search.best_params_['alpha_1']
        best_alpha_2 = grid_search.best_params_['alpha_2']
        best_lambda_1 = grid_search.best_params_['lambda_1']
        best_lambda_2 = grid_search.best_params_['lambda_2']

        # Initialiser le modèle de régression bayésienne avec les meilleures valeurs d'hyperparamètres
        model = BayesianRidge(alpha_1=best_alpha_1, alpha_2=best_alpha_2, lambda_1=best_lambda_1, lambda_2=best_lambda_2)

        # Entraîner le modèle
        model.fit(X_train, y_train)

        save_param(model,'BayesianRidge', year, cvo)
        
    else :

        model = load_param('BayesianRidge', year, cvo)
    
    # Prédire sur l'ensemble de test
    y_pred = model.predict(X_test)
    
    return y_pred

