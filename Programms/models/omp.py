import pandas as pd
from sklearn.linear_model import OrthogonalMatchingPursuit
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from utils import save_param, load_param

def omp(mode, X_train, X_test, y_train, year, cvo):

    if mode == "Training" :

        # Définir les valeurs de n_nonzero_coefs à tester (nombre maximal de coefficients non nuls)
        n_nonzero_coefs_values = [2, 5, 10, 20, 30, 40, 150]

        # Créer un dictionnaire des paramètres à tester
        param_grid = {'n_nonzero_coefs': n_nonzero_coefs_values}

        # Initialiser le modèle OMP
        omp_model = OrthogonalMatchingPursuit()

        # Utiliser GridSearchCV pour trouver la meilleure valeur de n_nonzero_coefs
        grid_search = GridSearchCV(omp_model, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)

        # Obtenir la meilleure valeur de n_nonzero_coefs
        best_param = grid_search.best_params_
        best_n_nonzero_coefs = grid_search.best_params_['n_nonzero_coefs']
        
        # Initialiser le modèle OMP avec la meilleure valeur de n_nonzero_coefs
        model = OrthogonalMatchingPursuit(n_nonzero_coefs=best_n_nonzero_coefs)

        # Entraîner le modèle
        model.fit(X_train, y_train)

        save_param(model, 'OrthogonalMatchingPursuit', year, cvo)

    else :

        model = load_param('OrthogonalMatchingPursuit', year,cvo)

    # Prédire sur l'ensemble de test
    y_pred = model.predict(X_test)

    return y_pred
