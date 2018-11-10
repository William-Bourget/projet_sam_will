from import_data import import_training_data
#from sklearn import train_test_split


def main():
    X_full_data, y_full_data = import_training_data()

    print(X_full_data, y_full_data)

    # Separation of full dataset
    #X_train, X_test, y_train, y_test = train_test_split(X_full_data, y_full_data, test_size = 0.2)
    #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2)



if __name__ == "__main__":
    main()