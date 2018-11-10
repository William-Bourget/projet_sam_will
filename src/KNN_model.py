from sklearn.neighbors import KNeighborsClassifier
import itertools

def calculate_knn_metrics(X_test,
                          y_test,
                          X_val,
                          y_val,
                          tested_n_neighbors,
                          tested_weights,
                          tested_metric):
    """
    Returns a dictionary of wanted metrics for KNN models
    :param X_test: X test features
    :type X_test: numpy.array
    :param y_test: y test labels
    :type y_test: numpy.array
    :param X_val: X val features
    :type X_val: numpy.array
    :param y_val: y val labels
    :type y_val: numpy.array
    :param tested_n_neighbors: list of tested number of neighbors
    :type tested_n_neighbors: list
    :param tested_weights: list of tested weights methods
    :type tested_n_neighbors: list
    :param tested_p: list of tested p for Minkowsky distance
    :type tested_n_neighbors: list
    :return: precisions dictionary
    """
    params = list(itertools.product(tested_n_neighbors,
                                    tested_weights,
                                    tested_metric))

    precisions_dict = {}
    predictions_dict = {}

    for param in params:
        knn_model = KNeighborsClassifier(n_neighbors=param[0],
                                         weights=param[1],
                                         metric=param[2])
        knn_model.fit(X_test, y_test)
        precisions_dict[param] = knn_model.score(X_val, y_val)
        predictions_dict[params] = knn_model.predict(X_val)

    metrics_dict = {"precisions": precisions_dict,
                    "predictions": predictions_dict}

    return metrics_dict



if __name__ == "__main__":
    knn_tested_n_neighbors = [1, 3, 5, 10, 20]
    knn_tested_weights = ["uniform", "distance"]
    knn_tested_p = [1, 2, 3, 20]
