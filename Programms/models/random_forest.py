import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from utils import save_param, load_param

def random_forest(mode, X_train, X_test, y_train, year, cvo):

    if mode == "Training" :

        # Définir les hyperparamètres à tester
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 3, 10],
            'min_samples_split': [10, 15]
        }

        # Initialiser le modèle Random Forest Regressor
        model = RandomForestRegressor(random_state=42)

        # Utiliser GridSearchCV avec validation croisée pour rechercher les meilleurs hyperparamètres
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
        grid_search.fit(X_train, y_train)

        # Extraire le meilleur modèle avec les meilleurs hyperparamètres
        best_model = grid_search.best_estimator_
        
        # Entraîner le modèle sur l'ensemble d'entraînement pour la première cible
        best_model.fit(X_train, y_train)

        save_param(best_model,'RandomForestRegressor', year, cvo)

    else :

       best_model = load_param('RandomForestRegressor', year, cvo)

    # Prédiction d'1 an
    y_pred = best_model.predict(X_test)

    return y_pred

