####################
# Logistic regression
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def load_and_prepare_cars_data(file_path):
    """
    Load and prepare the cars data.

    Parameters:
    - file_path (str): Path to the CSV file containing cars data.

    Returns:
    - pd.DataFrame: Prepared cars DataFrame.
    """
    cars = pd.read_csv(file_path)
    cars.columns = [
        "car_names",
        "mpg",
        "cyl",
        "disp",
        "hp",
        "drat",
        "wt",
        "qsec",
        "vs",
        "am",
        "gear",
        "carb",
    ]
    return cars

def visualize_cars_data_relationship(cars):
    """
    Visualize the relationship between predictors.

    Parameters:
    - cars (pd.DataFrame): Cars DataFrame.
    """
    # Check for independence between predictors
    sns.regplot(x="drat", y="carb", data=cars, scatter=True)
    drat = cars["drat"]
    carb = cars["carb"]
    spearmanr_coefficient, p_value = spearmanr(drat, carb)
    print("Spearman Rank Correlation Coefficient %0.3f" % (spearmanr_coefficient))

def check_for_missing_values(cars):
    """
    Check for missing values in the cars DataFrame.

    Parameters:
    - cars (pd.DataFrame): Cars DataFrame.
    """
    print(cars.isnull().sum())

def check_target_variable(cars):
    """
    Check if the target variable is binary or ordinal.

    Parameters:
    - cars (pd.DataFrame): Cars DataFrame.
    """
    # Check if the target variable is binary or ordinal
    sns.countplot(x="am", data=cars, palette="hls")  # Two values, so binary

def check_dataset_size(cars):
    """
    Check if the dataset size is sufficient.

    Parameters:
    - cars (pd.DataFrame): Cars DataFrame.
    """
    # Check if the dataset size is sufficient
    cars.info()  # 32 observations, so sufficient (50 per predictor is good)

def perform_logistic_regression(cars):
    """
    Perform logistic regression on the cars data.

    Parameters:
    - cars (pd.DataFrame): Cars DataFrame.
    """
    # Set up the data for regression
    cars_data = cars.iloc[:, [5, 11]].values  # Predictors
    cars_data_names = ["drat", "carb"]  # Predictor names
    y = cars.iloc[:, 9].values  # Target

    # Scale the data
    X = scale(cars_data)

    # Perform logistic regression
    LogReg = LogisticRegression()
    LogReg.fit(X, y)

    # Model accuracy
    accuracy = LogReg.score(X, y)
    print(f'Model Accuracy: {accuracy}')  # Max is 1, so this is a good score

    # Check the model accuracy
    y_pred = LogReg.predict(X)
    print(classification_report(y, y_pred))

if __name__ == "__main__":
    # Load and prepare cars data
    cars = load_and_prepare_cars_data("mtcars.csv")

    # Visualize data relationships
    visualize_cars_data_relationship(cars)

    # Check for missing values
    check_for_missing_values(cars)

    # Check target variable
    check_target_variable(cars)

    # Check dataset size
    check_dataset_size(cars)

    # Perform logistic regression
    perform_logistic_regression(cars)
