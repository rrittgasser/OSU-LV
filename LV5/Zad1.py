import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# a)
plt.scatter(X_train[:,0], X_train[:,1] ,c=y_train, label="Train")
plt.scatter(X_test[:,0], X_test[:,1], c=y_test, label="Test", marker="x")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.show()

# b)
LogRegression_model = LogisticRegression()
LogRegression_model.fit(X_train, y_train)

# c)
theta0, theta1, theta2 = LogRegression_model.intercept_[0], LogRegression_model.coef_[0, 0], LogRegression_model.coef_[0, 1] 

x1_vals = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
x2_vals = - (theta0 + theta1 * x1_vals) / theta2
plt.scatter(X_train[:,0], X_train[:,1] ,c=y_train, label="Train")
plt.plot(x1_vals, x2_vals, 'k-', label="Granica odluke")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.show()

# d)
y_true = y_test
y_pred = LogRegression_model.predict(X_test)
print("Tocnost: ", accuracy_score(y_true, y_pred))
cm = confusion_matrix(y_true, y_pred)
print("Matrica zabune: ", cm)
disp = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred))
disp.plot()
plt.show()
print(classification_report(y_true, y_pred))

# e)
plt.scatter(X_test[:,0], X_test[:,1], c=y_test, label="Test", marker="x")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.show()