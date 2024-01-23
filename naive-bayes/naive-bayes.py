####################
# Naive Bayes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

def load_enrollment_data():
    """
    Load and return the enrollment forecast data.

    Returns:
    -------
    pd.DataFrame
        DataFrame containing enrollment forecast data.
    """
    enroll = pd.read_csv("enrollment_forecast.csv")
    enroll.columns = ["year", "roll", "unem", "hgrad", "inc"]
    return enroll

def visualize_data(enroll):
    """
    Visualize the data using pairplot and display the correlation matrix.

    Parameters:
    -----------
    enroll : pd.DataFrame
        DataFrame containing the enrollment forecast data.
    """
    sns.pairplot(enroll)
    plt.show()

    # Check for correlation
    print(enroll.corr())

def prepare_regression_data(enroll):
    """
    Prepare data for regression.

    Parameters:
    -----------
    enroll : pd.DataFrame
        DataFrame containing the enrollment forecast data.

    Returns:
    --------
    tuple
        Tuple containing predictors and target arrays.
    """
    enroll_data = enroll.iloc[:, [2, 3]].values
    enroll_target = enroll.iloc[:, 1].values
    return enroll_data, enroll_target

def perform_linear_regression(X, y):
    """
    Perform linear regression.

    Parameters:
    -----------
    X : np.ndarray
        Array of predictors.
    y : np.ndarray
        Array of target values.

    Returns:
    --------
    sklearn.linear_model.LinearRegression
        Trained linear regression model.
    """
    LinReg = LinearRegression()
    LinReg.fit(X, y)
    return LinReg

def evaluate_regression_model(model, X, y):
    """
    Evaluate the linear regression model and display diagnostic measures.

    Parameters:
    -----------
    model : sklearn.linear_model.LinearRegression
        Trained linear regression model.
    X : np.ndarray
        Array of predictors.
    y : np.ndarray
        Array of target values.
    """
    # R² value
    r2 = model.score(X, y)
    print(f'R² value: {r2}')

    # Adjusted R²
    n = len(y)
    p = X.shape[1]
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    print(f'Adjusted R²: {adjusted_r2}')

    # Residual Analysis
    y_pred = model.predict(X)
    residuals = y - y_pred

    # Residual plot
    plt.figure(figsize=(8, 4))
    plt.scatter(y_pred, residuals)
    plt.title('Residual Plot')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.show()

    # QQ-plot of residuals
    sm.qqplot(residuals, line='s')
    plt.show()

if __name__ == "__main__":
    # Load data
    enroll = load_enrollment_data()

    # Visualize data
    visualize_data(enroll)

    # Prepare data
    enroll_data, enroll_target = prepare_regression_data(enroll)

    # Scale the data
    X, y = scale(enroll_data), enroll_target

    # Perform linear regression
    LinReg = perform_linear_regression(X, y)

    # Evaluate the model
    evaluate_regression_model(LinReg, X, y)
