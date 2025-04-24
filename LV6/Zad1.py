import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn . neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn . model_selection import cross_val_score

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl)


# ucitaj podatke
data = pd.read_csv("Social_Network_Ads.csv")
print(data.info())

data.hist()
plt.show()

# dataframe u numpy
X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

# podijeli podatke u omjeru 80-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)

# skaliraj ulazne velicine
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

# Model logisticke regresije
LogReg_model = LogisticRegression(penalty=None) 
LogReg_model.fit(X_train_n, y_train)

# Evaluacija modela logisticke regresije
y_train_p = LogReg_model.predict(X_train_n)
y_test_p = LogReg_model.predict(X_test_n)

print("Logisticka regresija: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))

# granica odluke pomocu logisticke regresije
plot_decision_regions(X_train_n, y_train, classifier=LogReg_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()
plt.show()

#1. zad
KNN_model = KNeighborsClassifier ( n_neighbors = 5 )
KNN_model.fit( X_train_n , y_train )
y_train_p = KNN_model.predict(X_train_n)
y_test_p = KNN_model.predict(X_test_n)
print("KNN Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("KNN Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))
plot_decision_regions(X_train_n, y_train, classifier=KNN_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()
plt.show()

KNN_model = KNeighborsClassifier ( n_neighbors = 1 )
KNN_model.fit( X_train_n , y_train )
y_train_p = KNN_model.predict(X_train_n)
y_test_p = KNN_model.predict(X_test_n)
print("KNN Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("KNN Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))
plot_decision_regions(X_train_n, y_train, classifier=KNN_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()
plt.show()

KNN_model = KNeighborsClassifier ( n_neighbors = 100 )
KNN_model.fit( X_train_n , y_train )
y_train_p = KNN_model.predict(X_train_n)
y_test_p = KNN_model.predict(X_test_n)
print("KNN Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("KNN Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))
plot_decision_regions(X_train_n, y_train, classifier=KNN_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()
plt.show()

#2
KNN_model = KNeighborsClassifier()
param_grid = {'n_neighbors': [1, 3, 5, 7, 10, 15, 20] }
knn_gscv = GridSearchCV( estimator = KNN_model, param_grid  = param_grid , cv =5 , scoring = 'accuracy', n_jobs = -1)
knn_gscv.fit(X_train_n, y_train)
print("Najbolji parametar: ", knn_gscv.best_params_)

#3
SVM_model = svm.SVC(kernel = 'rbf', gamma = 1 , C=0.1 )
SVM_model.fit( X_train_n , y_train )
y_train_p_SVM = SVM_model.predict(X_train_n)
y_test_p_SVM = SVM_model.predict(X_test_n)
print("SVM Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p_SVM))))
plot_decision_regions(X_train_n, y_train, classifier=SVM_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.tight_layout()
plt.title("C=0.1, gamma=1")
plt.show()


SVM_model = svm.SVC(kernel = 'rbf', gamma = 1 , C=1 )
SVM_model.fit( X_train_n , y_train )
y_train_p_SVM = SVM_model.predict(X_train_n)
y_test_p_SVM = SVM_model.predict(X_test_n)
print("SVM Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p_SVM))))
plot_decision_regions(X_train_n, y_train, classifier=SVM_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.tight_layout()
plt.title("C=1, gamma=1")
plt.show()

SVM_model = svm.SVC(kernel = 'rbf', gamma = 1 , C=10 )
SVM_model.fit( X_train_n , y_train )
y_train_p_SVM = SVM_model.predict(X_train_n)
y_test_p_SVM = SVM_model.predict(X_test_n)
print("SVM Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p_SVM))))
plot_decision_regions(X_train_n, y_train, classifier=SVM_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.tight_layout()
plt.title("C=10, gamma=1")
plt.show()

SVM_model = svm.SVC(kernel = 'rbf', gamma = 0.01 , C=1 )
SVM_model.fit( X_train_n , y_train )
y_train_p_SVM = SVM_model.predict(X_train_n)
y_test_p_SVM = SVM_model.predict(X_test_n)
print("SVM Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p_SVM))))
plot_decision_regions(X_train_n, y_train, classifier=SVM_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.tight_layout()
plt.title("C=1, gamma=0.01")
plt.show()

SVM_model = svm.SVC(kernel = 'rbf', gamma = 0.01 , C=10 )
SVM_model.fit( X_train_n , y_train )
y_train_p_SVM = SVM_model.predict(X_train_n)
y_test_p_SVM = SVM_model.predict(X_test_n)
print("SVM Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p_SVM))))
plot_decision_regions(X_train_n, y_train, classifier=SVM_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.tight_layout()
plt.title("C=10, gamma=0.01")
plt.show()

#4
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 0.01],
              'gamma': [0.001, 0.01, 0.1, 1, 10, 100, 10]}
svm_gscv = GridSearchCV( estimator = svm.SVC().fit(X_train_n, y_train) , param_grid  = param_grid , cv =5 , scoring = 'accuracy', n_jobs = -1)
svm_gscv.fit(X_train_n, y_train)
print(svm_gscv.best_params_)
print(svm_gscv.best_score_)
plt.show()