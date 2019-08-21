import numpy as np
import cv2
from load_svrt import load_svrt,load_data


rel_train, rel_test = load_svrt()
print(len(rel_train),len(rel_test))
img = np.array(rel_train[0][0])
qst = rel_train[0][1]
ans = rel_train[0][2]
print(rel_train[0][0])
print(type(rel_train[0][0]))
print("Image size = ", img.shape)
print("Word embed = ", qst)
print("Class labl = ", ans)

### Correct output:
# loading data...
# processing data...
# Image size =  (3, 75, 75)
# Word embed =  [1. 0. 0. 0. 0. 0. 0. 1. 1. 0. 0.]
# Class labl =  2
###

img = np.swapaxes(img,0,2)
cv2.imshow('image',img)
cv2.waitKey(3000)
