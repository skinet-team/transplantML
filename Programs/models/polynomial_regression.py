import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from utils import save_param, load_param

def polynomial_regression(mode, X_train, X_test, y_train, year, cvo):

    if mode == "Training" :

        # Créer un modèle de régression linéaire avec caractéristiques polynomiales
        model = make_pipeline(PolynomialFeatures(), LinearRegression())

        # Définir la grille des paramètres à tester (degrés du polynôme)
        degrees=[1, 2, 3, 4, 5]
        param_grid = {'polynomialfeatures__degree': degrees}

        # Utiliser GridSearchCV pour la recherche des meilleurs paramètres avec validation croisée
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)

        # Obtenir la meilleure valeur de degré
        best_degree = grid_search.best_params_['polynomialfeatures__degree']
        
        # Créer un modèle de régression linéaire avec la meilleure valeur de degré
        best_model = make_pipeline(PolynomialFeatures(degree=best_degree), LinearRegression())

        # Entraîner le modèle sur l'ensemble d'entraînement pour la première cible
        best_model.fit(X_train, y_train)
    
        save_param(best_model, 'polynomial_regression', year, cvo)
        
    else :

        best_model = load_param('polynomial_regression', year, cvo)

    # Prédiction d'1 an
    y_pred = best_model.predict(X_test)

    return y_pred

