# Importing the libraries
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

def backwardElimination(X, arr, sm):
    X_opt = X[:, arr]
    regressor_OLS = sm.OLS(y, X_opt).fit()
    pvalues_array = list(regressor_OLS.pvalues)
    if max(pvalues_array) > 0.05:
        index_temp = pvalues_array.index(max(pvalues_array))
        return backwardElimination(X, np.delete(arr, index_temp, 0), sm)
    else:
        return arr

# Returns the index of the significant independent variables
import statsmodels.formula.api as sm
X = np.append(np.ones((50,1)).astype(int), values = X, axis = 1)
print("\nSelected independent variables {}\n".format(backwardElimination(X, np.arange(len(X[0] - 1)), sm)))

