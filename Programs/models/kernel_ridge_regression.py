import pandas as pd
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
import importlib
from sklearn.model_selection import GridSearchCV
from utils import save_param, load_param

def kernel_ridge_regression(mode, X_train, X_test, y_train, year, cvo):

    if mode == "Training" :

        # Définir les valeurs du paramètre alpha à tester
        alphas = [0.1, 0.5, 1.0, 5.0, 10.0]

        # Définir les valeurs du paramètre kernel à tester
        kernels = ['linear', 'rbf', 'poly']

        # Créer un dictionnaire des paramètres à tester
        param_grid = {'alpha': alphas, 'kernel': kernels}

        # Initialiser le modèle Kernel Ridge Regression
        kernel_ridge_model = KernelRidge()

        # Utiliser GridSearchCV pour trouver les meilleures valeurs d'alpha et de kernel
        grid_search = GridSearchCV(kernel_ridge_model, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)

        # Obtenir les meilleures valeurs d'alpha et de kernel
        best_param = grid_search.best_params_
        best_alpha = grid_search.best_params_['alpha']
        best_kernel = grid_search.best_params_['kernel']
        
        # Initialiser le modèle Kernel Ridge Regression avec les meilleurs paramètres
        model = KernelRidge(alpha=best_alpha, kernel=best_kernel)

        # Entraîner le modèle
        model.fit(X_train, y_train)

        save_param(model, 'KernelRidge', year, cvo)
        
    else :

        model = load_param('KernelRidge', year, cvo)
    
    # Prédire sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Entraîner le modèle sur l'ensemble d'entraînement pour la deuxième cible
    
    return y_pred

