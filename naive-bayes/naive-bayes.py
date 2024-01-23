####################
# Naive Bayes
import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

def load_spambase_data():
    """
    Load and return the Spambase dataset.

    Returns:
    -------
    pd.DataFrame
        DataFrame containing the Spambase dataset.
    """
    dataset = np.loadtxt("spambase.csv", delimiter=",")
    return pd.DataFrame(dataset)

def prepare_spambase_data(df):
    """
    Prepare Spambase data for classification.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the Spambase dataset.

    Returns:
    --------
    tuple
        Tuple containing predictors and target arrays.
    """
    X = df.iloc[:, 0:48].values
    y = df.iloc[:, -1].values
    dfX = pd.DataFrame(X)
    dfy = pd.DataFrame(y)
    return X, y, dfX, dfy

def split_data(X, y):
    """
    Split the data into training and test sets.

    Parameters:
    -----------
    X : np.ndarray
        Array of predictors.
    y : np.ndarray
        Array of target values.

    Returns:
    --------
    tuple
        Tuple containing training and test sets.
    """
    return train_test_split(X, y, test_size=0.33, random_state=42)

def fit_and_predict(model, X_train, y_train, X_test):
    """
    Fit the Naive Bayes model and make predictions.

    Parameters:
    -----------
    model : sklearn.naive_bayes
        Naive Bayes model (e.g., BernoulliNB, MultinomialNB, GaussianNB).
    X_train : np.ndarray
        Array of predictors for training.
    y_train : np.ndarray
        Array of target values for training.
    X_test : np.ndarray
        Array of predictors for testing.

    Returns:
    --------
    tuple
        Tuple containing predicted values and the trained model.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred, model

def evaluate_accuracy(y_expect, y_pred):
    """
    Evaluate and print the accuracy score.

    Parameters:
    -----------
    y_expect : np.ndarray
        Array of expected target values.
    y_pred : np.ndarray
        Array of predicted target values.
    """
    accuracy = metrics.accuracy_score(y_expect, y_pred)
    print(f'Accuracy: {accuracy:.4f}')

def additional_diagnostics(y_expect, y_pred_proba):
    """
    Print additional diagnostic measures (Confusion Matrix, Classification Report, ROC Curve, AUC).

    Parameters:
    -----------
    y_expect : np.ndarray
        Array of expected target values.
    y_pred_proba : np.ndarray
        Array of predicted probabilities.
    """
    # Confusion Matrix and Classification Report
    print("Confusion Matrix:")
    print(metrics.confusion_matrix(y_expect, y_pred))
    print("\nClassification Report:")
    print(metrics.classification_report(y_expect, y_pred))

    # ROC Curve and AUC
    fpr, tpr, thresholds = metrics.roc_curve(y_expect, y_pred_proba)
    auc = metrics.auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Load data
    spambase_data = load_spambase_data()

    # Prepare data
    X, y, dfX, dfy = prepare_spambase_data(spambase_data)

    # Split the data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Create and fit binary model BernoulliNB
    BernNB = BernoulliNB(binarize=True)
    y_pred, _ = fit_and_predict(BernNB, X_train, y_train, X_test)

    # Evaluate accuracy
    evaluate_accuracy(y_test, y_pred)

    # Create and fit multinomial model
    MultiNB = MultinomialNB()
    y_pred, _ = fit_and_predict(MultiNB, X_train, y_train, X_test)

    # Evaluate accuracy
    evaluate_accuracy(y_test, y_pred)

    # Create and fit Gaussian model
    GausNB = GaussianNB()
    y_pred, _ = fit_and_predict(GausNB, X_train, y_train, X_test)

    # Evaluate accuracy
    evaluate_accuracy(y_test, y_pred)

    # Create and fit binary model BernoulliNB with different binarize value
    BernNB = BernoulliNB(binarize=0.1)
    y_pred, _ = fit_and_predict(BernNB, X_train, y_train, X_test)

    # Evaluate accuracy
    evaluate_accuracy(y_test, y_pred)

    # Additional Diagnostics
    y_pred_proba = BernNB.predict_proba(X_test)[:, 1]
    additional_diagnostics(y_test, y_pred_proba)
