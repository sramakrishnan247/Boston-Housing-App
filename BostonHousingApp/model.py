import matplotlib
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit, train_test_split,learning_curve, validation_curve
import seaborn as sns
import matplotlib.pyplot as plt

# Pretty display for notebooks
#%matplotlib inline
sns.set_style("whitegrid")

# Load the Boston housing dataset
data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)

print(data.head())
print(prices.head())
print(features.head())
print(data.shape)
# Success
print("Boston housing dataset has {} data points with {} variables each.".format(*data.shape))

#Performance Metric

from sklearn.metrics import r2_score
def performance_metric(y_true, y_pred):
  return r2_score(y_true,y_pred)

#Shuffle and Split Data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
                                          features,
                                          prices,
                                          test_size = 0.20,
                                          random_state = 42)

print(X_train.head())

# Method for Fitting the Model

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer


def fit_model(X, y):
    # Create CV Sets from training data
    cv_sets = ShuffleSplit(n_splits=10, test_size=0.20, random_state=0)
    # Create decision tree regressor object
    regressor = DecisionTreeRegressor()

    params = {'max_depth': list(range(1, 11))}
    scoring_fnc = make_scorer(performance_metric)

    # Create GridSearchCV object and Fit the model
    grid = GridSearchCV(estimator=regressor,
                        param_grid=params,
                        scoring=scoring_fnc,
                        cv=cv_sets)

    grid = grid.fit(X, y)

    # Grid Search Table
    depths = [d['max_depth'] for d in grid.cv_results_["params"]]
    scores = grid.cv_results_["mean_test_score"]
    df = pd.DataFrame({"max_depth": depths, "mean_test_score": scores},
                      columns=["max_depth", "mean_test_score"])

    return grid.best_estimator_, df


reg, grid_table = fit_model(X_train, y_train)
print(reg)


#Save model using pickle
import pickle
filename = "model_final.sav"
pickle.dump(reg, open(filename,"wb"))

