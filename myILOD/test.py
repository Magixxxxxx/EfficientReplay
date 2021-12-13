
import torch
import numpy as np
label1 = np.zeros((4, 1))
lambd = 0.5
y1 = np.hstack((label1, np.full((label1.shape[0], 1), lambd)))

label2 = np.zeros((4, 1))
y2 = np.hstack((label2, np.full((label2.shape[0], 1), 1. - lambd)))

a = np.vstack((y1, y2))
a = np.random.beta(2,2)
print(a)