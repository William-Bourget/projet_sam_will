from import_data import import_training_data
from sklearn.model_selection import train_test_split
from KNN_model import calculate_knn_metrics
from run_config import run_config
import numpy as np


def main():
    X_full_data, y_full_data = import_training_data()

    # Separation of full dataset
    X_train, X_test, y_train, y_test = train_test_split(X_full_data, y_full_data, test_size = 0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2)

    print("Proportion de toits verts dans l'échantillon de test: {}".format(np.mean(y_full_data[:, 1])))

    # Fitting KNN model
    if run_config["KNN"]:
        knn_tested_n_neighbors = [1, 3, 5, 10, 20]
        knn_tested_weights = ["uniform", "distance"]
        knn_tested_metric = ["manhattan", "euclidean", "mahalanobis"]
        knn_metrics = calculate_knn_metrics(X_test,
                                            y_test,
                                            X_val,
                                            y_val,
                                            knn_tested_n_neighbors,
                                            knn_tested_weights,
                                            knn_tested_metric)


if __name__ == "__main__":
    main()