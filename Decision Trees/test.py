import unittest
import numpy as np
import data_storage as ds
import decision_trees as dt

class TestMethods(unittest.TestCase):
    # tests building numpy array
    def test_build_nparray(self):
        file_name = "data1.csv"
        testData = np.genfromtxt(file_name, dtype=str, delimiter=',')
        self.assertTrue(ds.build_nparray(testData))

    # tests building list
    def test_build_list(self):
        file_name = "data1.csv"
        testData = np.genfromtxt(file_name, dtype=str, delimiter=',')
        self.assertTrue(ds.build_list(testData))

    # tests building dict
    def test_build_dict(self):
        file_name = "data1.csv"
        testData = np.genfromtxt(file_name, dtype=str, delimiter=',')
        self.assertTrue(ds.build_dict(testData))

    # tests training data
    def test_train_data(self):
        file_name = "data1.csv"
        testData = np.genfromtxt(file_name, dtype=str, delimiter=',')
        max_depth = 3
        X,Y = ds.build_nparray(testData)
        self.assertTrue(dt.DT_train_binary(X,Y,max_depth))

    # tests tree result
    def test_tree(self):
        file_name = "data1.csv"
        testData = np.genfromtxt(file_name, dtype=str, delimiter=',')
        max_depth = 3
        X,Y = ds.build_nparray(testData)
        DT = dt.DT_train_binary(X,Y,max_depth)
        self.assertTrue(dt.DT_test_binary(X,Y,DT))

    # tests making prediction
    def test_make_prediction(self):
        file_name = "data1.csv"
        testData = np.genfromtxt(file_name, dtype=str, delimiter=',')
        max_depth = 3
        X,Y = ds.build_nparray(testData)
        DT = dt.DT_train_binary(X,Y,max_depth)
        self.assertTrue(dt.DT_make_prediction(Y,DT))

    # tests building tree
    def test_build_tree(self):
        file_name = "data1.csv"
        testData = np.genfromtxt(file_name, dtype=str, delimiter=',')
        max_depth = 3
        X,Y = ds.build_nparray(testData)
        DT = dt.DT_train_binary(X,Y,max_depth)
        self.assertTrue(dt.build_tree(X,Y,max_depth,DT))

    # tests building tree with data for different edge cases
    def test_build_tree_data2(self):
        file_name = "data2.csv"
        testData = np.genfromtxt(file_name, dtype=str, delimiter=',')
        max_depth = 3
        X,Y = ds.build_nparray(testData)
        DT = dt.DT_train_binary(X,Y,max_depth)
        self.assertTrue(dt.build_tree(X,Y,max_depth,DT))

    # tests building tree with data for different edge cases
    def test_build_tree_data3(self):
        file_name = "data3.csv"
        testData = np.genfromtxt(file_name, dtype=str, delimiter=',')
        max_depth = 3
        X,Y = ds.build_nparray(testData)
        DT = dt.DT_train_binary(X,Y,max_depth)
        self.assertTrue(dt.build_tree(X,Y,max_depth,DT))

    # tests building tree with data for different edge cases
    def test_build_tree_data3(self):
        file_name = "data4.csv"
        testData = np.genfromtxt(file_name, dtype=str, delimiter=',')
        max_depth = 3
        X,Y = ds.build_nparray(testData)
        DT = dt.DT_train_binary(X,Y,max_depth)
        self.assertTrue(dt.build_tree(X,Y,max_depth,DT))

    # tests building forest
    def test_build_forest(self):
        file_name = "data1.csv"
        testData = np.genfromtxt(file_name, dtype=str, delimiter=',')
        max_depth = 3
        tree_count = 11
        X,Y = ds.build_nparray(testData)
        DT = dt.DT_train_binary(X,Y,max_depth)
        self.assertTrue(dt.RF_build_random_forest(X,Y,max_depth,tree_count))

    # tests forest result
    def test_forest(self):
        file_name = "data1.csv"
        testData = np.genfromtxt(file_name, dtype=str, delimiter=',')
        max_depth = 3
        tree_count = 11
        X,Y = ds.build_nparray(testData)
        RF = dt.RF_build_random_forest(X,Y,max_depth,tree_count)
        self.assertTrue(dt.RF_test_random_forest(X,Y,RF))
