# data storage
# cs 422 p1
# dana conley

import os
import numpy as np

#build_nparray
def build_nparray(data):
    l1 = []
    l2 = []
    for row in data:
        l1.append(row[:-1])
        l2.append(row[-1])
    sample_array1 = np.array(l1[1:])
    sample_array2 = np.array(l2[1:])

    data_array = sample_array1.astype(np.float)
    label_array = sample_array2.astype(np.int)

    return(data_array,label_array)

#build_list
def build_list(data):
    l1 = []
    l2 = []
    data_list = []
    label_list = []

    for row in data[1:]:
        l1.append(row[:-1])
        l2.append(row[-1])

    for i in range(len(l1)):
        temp = []
        for k in range((len(l1[i]))):
            temp.append(float(l1[i][k]))
        data_list.append(temp)
    for j in l2:
        label_list.append(int(j))

    return(data_list,label_list)

#build_dict
def build_dict(data):
    data_dict = {}
    samples = {}
    label_dict = {}
    l1 = []
    l2 = []
    w1 = []
    data_list = []
    label_list = []

    for row in data:
        l1.append(row[:-1])
        l2.append(row[-1])
        w1.append(row)
    l1 = l1[1:]
    l2 = l2[1:]
    w = w1[0]
    w2 = w[:-1]

    for j in l2:
        label_list.append(int(j))

    for i in range(len(l1)):
        temp = []
        for k in range((len(l1[i]))):
            temp.append(float(l1[i][k]))
        data_list.append(temp)

    ii = []
    index = 0
    for row in range(len(data_list)):
        temp_d = {w2[ind]: data_list[row][ind] for ind in range(len(w2))}
        samples[index] = temp_d
        ii.append(index)
        index = index +1

    keys = ii
    values = label_list
    for key, value in zip(keys, values):
        label_dict[key] = value

    return(samples, label_dict)