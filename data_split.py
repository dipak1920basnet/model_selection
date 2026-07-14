from sklearn.model_selection import train_test_split
import numpy as np 

def main():
    ...


def first_split(X, y, test_size):
    X_train_dev, X_test, y_train_dev, y_test = train_test_split(
    X,
    y,
    test_size=test_size, 
    random_state=42,
    stratify=y)
    return X_train_dev, X_test, y_train_dev, y_test

def split_data(X,y, dev_size=0.2, test_size=0.2):
    """
    returns X_train,X_dev,X_test, y_train, y_dev, y_test
    """

    """
    if test_size is 0.2 and dev_size is 0.2
    here since the first split will give X_test, y_test as 20% of 100% split
    but second split will give 20% of 80% 
    but we need 60% train, 20% dev and 20% test so 
    we need to adjust dev_size accordingly
    """
    dev_size = dev_size/(1-test_size) # adjusting dev size
    X_train_dev,X_test, y_train_dev, y_test = first_split(X,y)
    X_train, X_dev, y_train, y_dev = first_split(X_train_dev, y_train_dev, test_size=dev_size)
    return X_train,X_dev,X_test, y_train, y_dev, y_test

def combine_data(X1, X2, y1, y2):
    """
    returns X,y 
    were X is (X1 and X2) combined 
    and  y is (y1 and y2) combined 
    """
    X_ = np.vstack((X1, X2))
    y_ = np.concatenate((y1, y2))

    return X_, y_

if __name__ == "__main__":
    main()