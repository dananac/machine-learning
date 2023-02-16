# decision trees
# cs 422 p1
# dana conley

import os
import numpy as np
from math import log2
from numpy import random

#DT_train_binary
def DT_train_binary(X,Y,max_depth):
    dt = []
    dt = build_tree(X,Y,max_depth,dt)
    return(dt)

#DT_test_binary
def DT_test_binary(X,Y,DT):
    correct = 0
    total = len(Y)
    for row in range(len(X)):
        yn = DT_make_prediction(X[row], DT)
        if yn == Y[row]:
            correct = correct + 1
    acc = correct / total
    return(acc)

#DT_make_prediction
def DT_make_prediction(X,DT):
    dt = DT
    feat = dt[-1]

    if X[feat] == 0 and isinstance(dt[0], list):
        return DT_make_prediction(X, dt[0])
    elif X[feat] == 1 and isinstance(dt[1], list):
        return DT_make_prediction(X, dt[1])

    if X[feat] == 0 and not isinstance(dt[0], list):
        return dt[0]
    if X[feat] == 1 and not isinstance(dt[1], list):
        return dt[1]

def build_tree(X, Y, max_depth, dt):
    depth = max_depth
    l1 = []
    l2 = []
    feats = []
    ig = 0
    new_array = []
    new_lbl_array = []
    for row in X:
        l1.append(row)
    for row in Y:
        l2.append(row)
    sample_array1 = np.array(l1)
    sample_array2 = np.array(l2)

    data_array = sample_array1.astype(np.float)
    label_array = sample_array2.astype(np.int)
    #print(data_array)
    #print(label_array)

    for n in range(len(data_array[0])):
        feats.append(n)
    feat = feats[n]
    # find number of 1(yes) and 0(no) labels
    yes = 0
    no = 0
    num = len(label_array)
    for index in range(len(label_array)):
        if label_array[index] == 1:
            yes = yes + 1   # update yes label
        elif label_array[index] == 0:
            no = no + 1     # update no label

    #calculates entropy
    if no > 0:
        entropy1 = (no/num)
        ent1 = log2(entropy1)
    elif no == 0:
        entropy1 = 0
        ent1 = 0
    if yes > 0:
        entropy2 = (yes/num)
        ent2 = log2(entropy2)
    elif yes == 0:
        entropy2 = 0
        ent2 = 0
    entropy = - entropy1 * ent1 - entropy2 * ent2   #total entropy
    #print(entropy)

    for ind in range((len(data_array[row]))):
        y = 0
        y1 = 0
        y0 = 0
        n = 0
        n1 = 0
        n0 = 0
        for row in range(len(data_array)):
            if data_array[row][ind] == 1.:
                y = y + 1   # update number of samples at yes branch
                if label_array[row] == 1:
                    y1 = y1 + 1   # update number of samples on yes branch with label 1
                elif label_array[row] == 0:
                    y0 = y0 + 1   # update number of samples on yes branch with label 0

            elif data_array[row][ind] == 0.:
                n = n + 1   # update number of samples at no branch
                if label_array[row] == 1:
                    n1 = n1 + 1   # update number of samples on no branch with label 1
                elif label_array[row] == 0:
                    n0 = n0 + 1   # update number of samples on no branch with label 0

        frac_y = y / num    # frac of samples at yes branch
        frac_n = n / num    # frac of samples at no branch

        # calculate entropy for both branches
        if y1 > 0:
            py1 = y1 / y
            ent_y1 = log2(py1)
        elif y1 == 0:
            py1 = 0
            ent_y1 = 0
        if y0 > 0:
            py0 = y0 / y
            ent_y0 = log2(py0)
        elif y0 == 0:
            py0 = 0
            ent_y0 = 0
        if n1 > 0:
            pn1 = n1 / n
            ent_n1 = log2(pn1)
        elif n1 == 0:
            pn1 = 0
            ent_n1 = 0
        if n0 > 0:
            pn0 = n0 / n
            ent_n0 = log2(pn0)
        elif n0 == 0:
            pn0 = 0
            ent_n0 = 0

        ent_n = - pn0 * ent_n0 - pn1 * ent_n1
        ent_y = - py0 * ent_y0 - py1 * ent_y1
        #calculate information gain
        new_ent_n = frac_n * ent_n
        new_ent_y = frac_y * ent_y
        new_ent = new_ent_n + new_ent_y
        temp_ig = entropy - new_ent      # information gain for depth 0
        if temp_ig > ig:        #replace with highest IG found
            ig = temp_ig
            feat = feats[ind]
    for r in range(len(data_array)):
        if data_array[r][feat] == 0.:
            new_array.append(data_array[r])         # remove data samples on completed leaf
            new_lbl_array.append(label_array[r])    # remove corresponding label for samples
            lr = r
        if data_array[r][ind] == 1:
            lr = r
    new_array = np.array(new_array)
    new_lbl_array = np.array(new_lbl_array)
    depth = depth - 1
    if ig == 0:         # stop recursion if IG is zero.
        if new_ent_n < new_ent_y:       # check if left branch is ending
            left = label_array[lr]      # return label at removed row as left
            right = label_array[0]      # return remaining label as right
        else:                           # else right branch is ending
            left = label_array[0]       # return remaining label as left
            right = label_array[lr]     # return label at removed row as right
        dt = [left, right, feat]
        return (dt)
    elif depth >= 1:        # stop recursion at max depth
        if new_ent_n < new_ent_y:       # check if left branch is ending
            left = label_array[lr]     # return label at removed row as left
            right = build_tree(new_array, new_lbl_array, depth, dt)
        else:                           # else right branch is ending
            right = label_array[lr]     # return label at removed row as right
            left = build_tree(new_array, new_lbl_array, depth, dt)
        dt = [left, right, feat]
    else:
        if new_ent_n < new_ent_y:       # check if left branch is ending
            left = label_array[lr]      # return label at removed row as left
            right = label_array[0]      # return remaining label as right
        else:                           # else right branch is ending
            left = label_array[0]       # return remaining label as left
            right = label_array[lr]     # return label at removed row as right
        dt = [left, right, feat]
        return(dt)
    return(dt)


#random forests
def RF_build_random_forest(X,Y,max_depth,num_of_trees):
    n = 0
    rf = []
    l1 = []
    l2 = []
    new_data_array = []
    new_label_array = []
    for row in X:
        l1.append(row)
    for row in Y:
        l2.append(row)
    sample_array1 = np.array(l1)
    sample_array2 = np.array(l2)
    data_array = sample_array1.astype(np.float)
    label_array = sample_array2.astype(np.int)

    num_samples = len(label_array)
    samples_per_tree = num_samples * .10
    samples_per_tree = round(samples_per_tree)
    for n in range(num_of_trees):
        for i in range(samples_per_tree):
            x = random.randint(num_samples)
            new_data_array.append(data_array[x])
            new_label_array.append(label_array[x])
        rf.append(DT_train_binary(new_data_array, new_label_array, max_depth))
    return(rf)

# test random forests
def RF_test_random_forest(X,Y,RF):
    rf = RF
    rf_test = 0
    for i in range(len(rf)):
        dt = rf[i]
        dt_test = DT_test_binary(X,Y,dt)
        print("DT: ", dt_test)
        if rf_test < dt_test:
            rf_test = dt_test
    return(rf_test)
