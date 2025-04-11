import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from utils import save_param, load_param

def lineaire (mode, X_train, X_test, y_train, year, cvo):
    # Initialiser le modèle de régression linéaire
    model = LinearRegression()

    if mode == "Training":

        # Entraîner le modèle
        model.fit(X_train, y_train)
        model.get_params()
    
        save_param(model, 'lineaire', year, cvo)

    else:

        model = load_param('lineaire', year, cvo)
    
    # Prédire sur l'ensemble de test
    y_pred = model.predict(X_test)

    return y_pred


