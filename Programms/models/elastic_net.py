import pandas as pd
from sklearn.linear_model import ElasticNet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
import importlib
from sklearn.model_selection import GridSearchCV
from utils import save_param, load_param

def elastic_net(mode, X_train, X_test, y_train, year, cvo):

    if mode == "Training" :

        # Définir les valeurs d'alpha à tester
        alphas = [0.1, 0.5, 1.0, 5.0, 10.0]

        # Définir les valeurs de rapport l1_ratio à tester
        l1_ratios = [0.01, 0.1, 0.3, 0.5, 0.9]

        # Créer un dictionnaire des paramètres à tester
        param_grid = {'alpha': alphas, 'l1_ratio': l1_ratios}
        
        # Initialiser le modèle ElasticNet
        elastic_net_model = ElasticNet()

        # Utiliser GridSearchCV pour trouver les meilleures valeurs d'alpha et de l1_ratio
        grid_search = GridSearchCV(elastic_net_model, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)

        # Obtenir les meilleures valeurs d'alpha et de l1_ratio
        best_params = grid_search.best_params_
        best_alpha = grid_search.best_params_['alpha']
        best_l1_ratio = grid_search.best_params_['l1_ratio']

        # Initialiser le modèle ElasticNet
        model = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio)

        # Entraîner le modèle
        model.fit(X_train, y_train)

        save_param(model, 'ElasticNet', year, cvo)

    else :
        
        model = load_param('ElasticNet', year, cvo)
    
    # Prédire sur l'ensemble de test
    y_pred = model.predict(X_test)

    return y_pred

