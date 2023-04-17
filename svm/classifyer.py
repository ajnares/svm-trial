import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

dir = 'C:\\Users\\Admin\\Desktop\\school works\\beans\\data\\train'

data=[]

features = []
labels = []

for feature, label in data:
    features.append(feature)
    labels.append(label)


xtrain, xtest, ytrain,ytest = train_test_split(features, labels, test_size = 0.25)

model = SVC(C=1,kernel='poly', gamma='auto')
model.fit(xtrain,ytrain)

categories = ['grade1, grade2, grade3, grade4']
prediction = model.predict(xtest)
accuracy = model.score(xtest,ytest)

print('Accuracy: ', accuracy)
print('prediction: ',categories[prediction[0]])

bean = xtest[0].reshape(50,50)
plt.imshow(bean, cmap='gray')
plt.show()