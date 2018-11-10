#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 09:33:22 2018

@author: Samuel_Levesque
"""
import numpy as np


def import_training_data():
    X_full_data = np.genfromtxt("../data/train-features.csv", delimiter=",", skip_header=True)
    y_full_data = np.genfromtxt("../data/data-id-train.csv", delimiter=",", skip_header=True)

    return X_full_data, y_full_data


if __name__ == "__main__":
    X, y = import_training_data()
    print(X, y)
