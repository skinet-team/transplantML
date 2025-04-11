import pandas as pd
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
import importlib
from sklearn.model_selection import GridSearchCV
from utils import save_param, load_param

def m_Lasso(mode, X_train, X_test, y_train, year, cvo):

    if mode == "Training" :

        # Définir les valeurs d'alpha à tester
        alphas = [0.1, 0.5, 1.0, 5.0, 10.0]

        # Créer un dictionnaire des paramètres à tester
        param_grid = {'alpha': alphas} 

        # Initialiser le modèle Lasso
        lasso_model = Lasso()

        # Utiliser GridSearchCV pour trouver la meilleure valeur d'alpha
        grid_search = GridSearchCV(lasso_model, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)

        # Obtenir la meilleure valeur d'alpha
        best_param = grid_search.best_params_
        best_alpha = grid_search.best_params_['alpha']

        # Initialiser le modèle de régression Lasso
        model = Lasso(alpha=best_alpha)

        # Entraîner le modèle
        model.fit(X_train, y_train)

        save_param(model, 'Lasso', year,cvo)

    else :

        model = load_param('Lasso', year,cvo)

    # Prédire sur l'ensemble de test
    y_pred = model.predict(X_test)

    return y_pred
