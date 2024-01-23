####################
# Linear regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
from sklearn.metrics import mean_squared_error

sns.set_style("whitegrid")

def load_and_prepare_data(file_path):
    """
    Load and prepare the enrollment data.

    Parameters:
    - file_path (str): Path to the CSV file containing enrollment data.

    Returns:
    - pd.DataFrame: Prepared enrollment DataFrame.
    """
    enroll = pd.read_csv(file_path)
    enroll.columns = ["year", "roll", "unem", "hgrad", "inc"]
    return enroll

def visualize_data(enroll):
    """
    Visualize the structure of the enrollment data.

    Parameters:
    - enroll (pd.DataFrame): Enrollment DataFrame.
    """
    # Check the structure of the data
    print(enroll.describe())

    # Pairplot for quick visual inspection
    sns.pairplot(enroll)
    plt.show()

    # Correlation matrix
    print(enroll.corr())

def prepare_regression_data(enroll):
    """
    Prepare data for linear regression.

    Parameters:
    - enroll (pd.DataFrame): Enrollment DataFrame.

    Returns:
    - np.ndarray: Scaled features (X).
    - np.ndarray: Target variable (y).
    """
    enroll_data = enroll.iloc[:, [2, 3]].values
    enroll_target = enroll.iloc[:, 1].values
    return scale(enroll_data), enroll_target

def perform_linear_regression(X, y):
    """
    Perform linear regression.

    Parameters:
    - X (np.ndarray): Scaled features.
    - y (np.ndarray): Target variable.

    Returns:
    - LinearRegression: Trained linear regression model.
    """
    LinReg = LinearRegression()
    LinReg.fit(X, y)
    return LinReg

def calculate_adjusted_r2(model, X, y):
    """
    Calculate adjusted R² for the linear regression model.

    Parameters:
    - model (LinearRegression): Trained linear regression model.
    - X (np.ndarray): Scaled features.
    - y (np.ndarray): Target variable.

    Returns:
    - float: Adjusted R² value.
    """
    n = len(y)
    p = X.shape[1]
    adjusted_r2 = 1 - (1 - model.score(X, y)) * (n - 1) / (n - p - 1)
    return adjusted_r2

def residual_analysis(model, X, y):
    """
    Perform residual analysis for the linear regression model.

    Parameters:
    - model (LinearRegression): Trained linear regression model.
    - X (np.ndarray): Scaled features.
    - y (np.ndarray): Target variable.
    """
    # Make predictions
    y_pred = model.predict(X)

    # Calculate residuals
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
    # Load and prepare data
    enroll = load_and_prepare_data("enrollment_forecast.csv")

    # Visualize data
    visualize_data(enroll)

    # Prepare data for regression
    X, y = prepare_regression_data(enroll)

    # Perform linear regression
    LinReg = perform_linear_regression(X, y)

    ##### Diagnostic measures #####

    # R² value
    r2_value = LinReg.score(X, y)
    print(f'R² Value: {r2_value}')  # Max is 1, so this is a good score

    # Adjusted R²
    adjusted_r2 = calculate_adjusted_r2(LinReg, X, y)
    print(f'Adjusted R²: {adjusted_r2}')

    # Residual Analysis
    residual_analysis(LinReg, X, y)
