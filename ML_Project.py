import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


# Importing the dataset
dataset = pd.read_csv('Wine.csv')
X = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13].values



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Applying PCA (Feature Extraction)
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)



def plot_graph(classifier):

    # Visualising the Test set results
    from matplotlib.colors import ListedColormap
    X_set, y_set = X_test, y_test
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.75, cmap = ListedColormap(('red', 'green','blue')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green','blue'))(i), label = j)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()


print("Training SVM Linear")
SVMlclassifier = SVC(kernel = 'linear', random_state = 0)
SVMlclassifier.fit(X_train, y_train)

y_pred = SVMlclassifier.predict(X_test)
cmSVMl = confusion_matrix(y_test, y_pred)
print("="*60)
print("Accuracy of SVM Classifier = ", accuracy_score(y_test, y_pred))
print("Confusion Matrix")
print(cmSVMl)
print("="*60)
time.sleep(3)
plot_graph(SVMlclassifier)


print("Training Decision Tree Classifier")
DTClassifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
DTClassifier.fit(X_train, y_train)

y_pred = DTClassifier.predict(X_test)
cmDT = confusion_matrix(y_test, y_pred)
print("="*60)
print("Accuracy of Decision Tree Classifier = ", accuracy_score(y_test, y_pred))
print("Confusion Matrix ")
print(cmDT)
print("="*60)
time.sleep(3)
plot_graph(DTClassifier)











