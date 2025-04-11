import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.datasets import make_friedman1
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils import save_param, load_param

def GradientBR_RKFold(mode, X_train, X_test, y_train, year, cvo):
    
    if mode == 'Training':

        # Définir les hyperparamètres à tester
        param_grid = {
            'n_estimators': [140, 150, 160],
            'learning_rate': [0.001, 0.005, 0.01],
            'max_depth': [1,2,3]
        }

        # Initialiser le modèle GradientBoostingRegressor
        model = GradientBoostingRegressor()

        # Utiliser GridSearchCV avec validation croisée pour rechercher les meilleurs hyperparamètres
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
        grid_search.fit(X_train, y_train)

        # Extraire le meilleur modèle avec les meilleurs hyperparamètres
        best_model = grid_search.best_estimator_

        # Entraîner le modèle sur l'ensemble d'entraînement pour la première cible
        best_model.fit(X_train, y_train)

        save_param(best_model, 'GradientBoostingRegressor', year, cvo)

    else :

        best_model = load_param('GradientBoostingRegressor', year, cvo)
    
    # Prédiction d'1 an
    y_pred = best_model.predict(X_test)

    return y_pred