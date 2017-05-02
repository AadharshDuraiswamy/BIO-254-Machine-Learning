#Start python
python

#Load SciPy libraries
import sys
import scipy
import numpy
import pandas
import sklearn

#Import all the modules that we will use
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#Load your training dataset
#This file contained our training data in a CSV and was hosted on the local computer
data = "Training.csv"
#Name your attributes according to the CSV file
names = ['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave points', 'Symmetry', 'Fractal dimension', 'Outcome']
#Define your dataset
dataset = pandas.read_csv(data, names=names)

#Create a validation datset
array = dataset.values
#Define X as all the variables
X = array[:,0:10]
#Define Y as the outcome variable (benign, non-recurring or recurring)
Y = array[:,10]
validation size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
scoring = 'accuracy'

#Build your machine learning models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
  
#Make predictions on the Validation dataset using the most accurate model
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

#Now that the model is trained and validated, set up the Testing dataset
data2 = "Testing.csv"
names2 = ['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave points', 'Symmetry', 'Fractal dimension', 'Outcome']
dataset2 = pandes.read_csv(data2, names=names2)
array2 = dataset2.values
X2 = array2[:,0:10]
Y2 = array2[:,10]
validation size = 1.00
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X2, Y2, test_size=validation_size, random_state=seed)
scoring = 'accuracy'

#Make predictions on the Testing dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

