#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 09:33:22 2018

@author: Samuel_Levesque
"""
import csv
import os


def import_training_data():
    my_path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(my_path, "data/train-features.csv")
    with open(path, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            print(', '.join(row))

    return 0


if __name__ == "__main__":
#    test = import_training_data()

    my_path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(my_path, "data/train_features.csv")
    with open(path) as f:
        test = list(csv.reader(f))