import sklearn
from sklearn import datasets
from sklearn import svm, metrics
cancer = datasets.load_breast_cancer()
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.neighbors import KNeighborsClassifier

# print("features: ", cancer.feature_names)
# print("labels: ", cancer.target_names)

X = cancer.data
Y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)
print(x_train, y_train)
classes = ['malignant', 'benign']
print("--------------")

# clf = svm.SVC(kernel='linear')
# clf.fit(x_train, y_train)
# print(clf)
#
# y_predict = clf.predict(x_test)
# print(y_predict)
# accuracy = metrics.accuracy_score(y_test, y_predict)
# print(accuracy)
# print("------------")

clf = svm.SVC(kernel='linear')
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)
print(accuracy)
print(y_test)
predictions = clf.predict(x_test)
print(predictions)

for x in range(len(predictions)):
    print("predicted: ", predictions[x], "Data: ", x_test[x], "Actual", y_test[x])
