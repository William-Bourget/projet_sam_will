#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 09:33:22 2018

@author: Samuel_Levesque
"""
import csv
import os
import numpy as np
import pandas as pd


def import_training_data():
    X_full_data = np.genfromtxt("../data/train-features.csv", delimiter=",", skip_header=True)
    y_full_data = np.genfromtxt("../data/data-id-train.csv", delimiter=",", skip_header=True)
    X_data_frame=pd.DataFrame(X_full_data)

    X_full_data_sort=X_data_frame.sort_values(by=X_data_frame.columns[0]).values
    return X_full_data_sort[:,1:], y_full_data[:,1]


if __name__ == "__main__":
    X, y = import_training_data()
    print(X,y)

