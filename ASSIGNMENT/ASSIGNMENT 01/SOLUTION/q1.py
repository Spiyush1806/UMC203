import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the 'oracle' directory to the Python path
sys.path.append(os.path.join(os.getcwd(), 'oracle'))
import oracle as oracle

mysrn = 23801

res = oracle.q1_fish_train_test_data(23801)
attributes = res[0]
train_img = res[1]
train_labels = res[2]
test_img = res[3]
test_labels = res[4]
train_img = np.array(train_img)
train_labels = np.array(train_labels)
test_img = np.array(test_img)
test_labels = np.array(test_labels)
print(attributes)
print(train_img.shape)
print(train_labels.shape)
print(test_img.shape)
print(test_labels.shape)

