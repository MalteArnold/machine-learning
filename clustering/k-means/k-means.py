import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import silhouette_score


def visualize_clusters(iris_df, clustering, true_labels):
    """
    Visualize the ground truth and K-Means cluster assignments.

    Parameters:
    - iris_df (pd.DataFrame): DataFrame containing iris data.
    - clustering (KMeans): Fitted KMeans model.
    - true_labels (numpy.ndarray): Ground truth labels.
    """
    relabel = np.choose(clustering.labels_, [2, 0, 1]).astype(np.int64)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(
        x=iris_df["petal length (cm)"],
        y=iris_df["petal width (cm)"],
        c=true_labels,
        s=20,
        cmap="viridis",
    )
    plt.title("Ground Truth Classification")
    plt.xlabel("Petal Length")
    plt.ylabel("Petal Width")

    plt.subplot(1, 2, 2)
    plt.scatter(
        x=iris_df["petal length (cm)"],
        y=iris_df["petal width (cm)"],
        c=relabel,
        s=20,
        cmap="viridis",
    )
    plt.title("K-Means Classification")
    plt.xlabel("Petal Length")
    plt.ylabel("Petal Width")

    plt.show()

if __name__ == "__main__":
    """
    Execute K-Means clustering on the iris dataset and visualize the results.
    """
    iris = datasets.load_iris()
    X = scale(iris.data)
    y = pd.DataFrame(iris.target)
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)

    # Initialize and fit KMeans model
    clustering = KMeans(n_clusters=3, random_state=5)
    clustering.fit(X)

    color_theme = np.array(["darkgray", "lightsalmon", "powderblue"])

    # Visualize clusters and ground truth
    visualize_clusters(iris_df, clustering, y.values.flatten())

    # Relabel clusters for comparison
    relabel = np.choose(clustering.labels_, [2, 0, 1]).astype(np.int64)

    # Display classification report
    print(classification_report(y, relabel))
    
    # Display confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y, relabel))

    # Display Silhouette Score
    silhouette_avg = silhouette_score(X, clustering.labels_)
    print(f"Silhouette Score: {silhouette_avg}")
