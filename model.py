import pickle

from joblib import dump, load

from sklearn import linear_model, svm
from sklearn.datasets import load_diabetes

X, y = load_diabetes(return_X_y=True)
LRG = linear_model.LogisticRegression(
   random_state=0, solver='liblinear').fit(X, y)

dump(LRG, "linear.ml")

LRG = load('linear.ml')
print(LRG.predict([[0.2, 0.05, -0.01, 0.02, -0.04, 0.2, 0.05, -0.01, 0.02, -0.04]]))

SVM = svm.SVC()
X, y = load_diabetes(return_X_y=True)
SVM.fit(X, y)

dump(SVM, "svm.ml")

SVM = load('svm.ml')
print(SVM.predict([[0.2, 0.05, -0.01, 0.02, -0.04, 0.2, 0.05, -0.01, 0.02, -0.04]]))