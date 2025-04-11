import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from utils import save_param, load_param

def svr_rbf(mode, X_train, X_test, y_train, year, cvo):
    
    if mode == "Training" :
    
        # Mise à l'échelle des caractéristiques avec StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Définir les hyperparamètres à rechercher
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': [0.01, 0.1, 1],
            'epsilon': [0.1, 0.2, 0.5]
        }

    
        kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    

        best_model = None
        best_score = float('-inf')

        # Boucle sur les différents noyaux
        for current_kernel in kernels:
            # Initialiser le modèle Support Vector Regressor (SVR) avec le noyau courant
            model = SVR(kernel=current_kernel)

            # Utiliser GridSearchCV pour trouver les meilleurs hyperparamètres
            grid_search = GridSearchCV(model, param_grid, cv=5)
            grid_search.fit(X_train_scaled, y_train)

            # Obtenir le meilleur modèle et score
            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                best_model = grid_search.best_estimator_
        
        save_param(scaler,'StandardScaler', year, cvo)
        save_param(best_model,'SVR', year, cvo)

    else :

        scaler = load_param('StandardScaler', year, cvo)
        best_model = load_param('SVR', year, cvo)

        X_test_scaled = scaler.transform(X_test)

    # Prédiction d'1 an
    y_pred = best_model.predict(X_test_scaled)

    return y_pred
