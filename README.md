# Automatic prediction of graft lifetime before degradation

## Introduction

The aim is to develop a program that will read and extract several numerical data related to graft function, stored in an Excel file. This file also contains columns dedicated to the graft's glomerular filtration rate (GFR) score taken after one, three, five and seven years post-operation. Our Machine Learning program is supposed to train on the data in this Excel file and give a prediction of the glomerular filtration rate (GFR) score, which represents the graft's lifespan.

## Explanation of how the project directory is organized

The directory is made up of several folders, some of which are created automatically :
- Code folder containing all the python files to make our test
- Data folder containing the Excel files for all of our tests
- Result folder containing all the files created by the program

## Python files and code execution

The main file for executing Machine Learning models is the ''main.py'' file which, as its name suggests, is used to launch everyother program that we need to this tests :

````python
if __name__ == '__main__':

    required_packages = ["sklearn", "matplotlib", "tabulate",
                         "pandas", "numpy", "threading", "customtkinter", "tkinter"]
    check_and_install_packages(required_packages) # Check that all the libraries are installed, if not install them.

    tkapp() #Tkinter application to simplify test set-up
````

### Definitions of model functions

In general, the function of each model is as follows:
1. Acquisition of train and test inputs for one-year and three-year dataframes, etc.
2. Define the model.
3. `GridSearchCV` to find the best hyperparameters for the model.
4. Train the model on one year's data.
5. Predict one year's values.
6. Model training on three years of data.
7. Prediction of three years values.
8. Same for 5 and 7 years.
9. Returns both sets of predictions

### Other Python files

- `executive_Feature_Importance.py` : Train the models using the resulting files from the Feature Importance technique.
- `Feature_importance_GBR.py` : Generate the Feature Importance file for the `GradientBoostingRegressor` model.
- `Feature Importance_RF_display.py` : Generate the Feature Importance file for the `Random Forest` model + display the best input variables using the MDI and Permutation Importance technique.
- `performances_code.py` : Called by the `exec file` to record the performances of each model in a separate CSV file.
- `prediction_app.py` : Generates a small TKinter interface for using the software. The GUI allows the user to import an Excel file containing a patient's data and return an estimate of the graft's lifespan.
- `Tableau.py`: Allows the means of the performance metrics of all models to be arranged in a single table.