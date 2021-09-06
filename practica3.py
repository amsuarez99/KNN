import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, plot_roc_curve
from sklearn.linear_model import LogisticRegression
from numpy.linalg import norm

"""Globals"""
k_vals = [1, 2, 3, 5, 10, 15, 20, 50, 75, 100]
"""Utils"""


def get_accuracy_scores(k_values, X_train, y_train, X_test):
    scores = []
    predictions = []
    confusion_matrices = []
    for k, i in zip(k_values, range(len(k_values))):
        knn = KNeighborsClassifier(algorithm='brute', n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        predictions.append(y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        scores.append(accuracy)
        conf_mat = confusion_matrix(y_test, y_pred)
        confusion_matrices.append(conf_mat)
        print('Confusion matrix for k: ', k)
        print(conf_mat)
        print("accurracy_score: ", accuracy)
    max_score = max(scores)
    max_index = scores.index(max_score)
    best_prediction = predictions[max_index]
    best_k = k_values[max_index]
    best_conf_mat = confusion_matrices[max_index]
    return scores, best_prediction, best_k, best_conf_mat


def plot_predictions(predictions, X_test, y_test, best_k):
    test_set = np.concatenate((X_test, y_test.reshape(-1, 1)), axis=1)
    test_set = pd.DataFrame(data=test_set, columns=[
                            'Height', 'Weight', 'Gender'])
    sns.scatterplot(x='Height', y='Weight', data=test_set,
                    hue='Gender', ec=None, palette='Set1', legend=False)
    plt.title('Real Values')
    plt.show()

    predicted_set = np.concatenate(
        (X_test, predictions.reshape(-1, 1)), axis=1)
    predicted_set = pd.DataFrame(data=predicted_set, columns=[
                                 'Height', 'Weight', 'Gender'])
    sns.scatterplot(x='Height', y='Weight', data=predicted_set,
                    hue='Gender', ec=None, palette='Set1', legend=False)
    plt.title(f"Predicted Values, k = {best_k}")
    plt.show()


def plot_accuracy(k_values, scores, dataset_name):
    a = np.arange(len(k_values) + 1)[1:]
    plt.title(f"Accuracy per k val: {dataset_name}")
    plt.xlabel('k values')
    plt.ylabel('Accuracy')
    plt.plot(a, scores, '--Dy')
    plt.grid()
    plt.xticks(a, k_values)
    plt.ylim(0.8, 1)
    plt.show()


def plot_rocs(conf_mats, labels, title):
    for i, conf_mat in enumerate(conf_mats):
        tp = conf_mat[0][0]
        fp = conf_mat[0][1]
        fn = conf_mat[1][0]
        tn = conf_mat[1][1]
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        x = [0, fpr, 1]
        y = [0, tpr, 1]
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        auc = ((fpr*tpr)/2) + ((1-fpr)*(1-tpr)/2) + ((1-fpr)*(tpr))
        plt.plot(x, y, label=f"{labels[i]} (AUC = {auc})")
    plt.title(f'ROC: {title}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()


"""# Default Dataset
Prepare the data
"""

data = pd.read_csv('data/default.txt', sep='\t')
data = data[['default', 'student', 'balance', 'income']]
data.head()
# Change 'True' to 1 and 'False' to 0 in student and default columns
data['student'] = data['student'].map(dict(Yes=1, No=0))
data['default'] = data['default'].map(dict(Yes=1, No=0))
X = data[['student', 'balance', 'income']].to_numpy()
y = data.default.to_numpy()

"""Create split"""
# 0.80 for the training set
# 0.20 for the test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, shuffle=True)

"""## Logistic Regression
Create the Model
"""

lr = LogisticRegression()

"""Train the Model"""

lr.fit(X_train, y_train)

"""Create Predictions"""

y_prediction = lr.predict(X_test)
"""### Evaluation"""
confusion_matrices = []
print("------ DEFAULT DATASET -------")
print("With Sklearn LogisticRegression:")
conf_mat = confusion_matrix(y_test, y_prediction)
confusion_matrices.append(conf_mat)
print(conf_mat)
print("accurracy_score: ", accuracy_score(y_test, y_prediction))
print('\n\n')

"""## K-NN"""

print("With KNN Algorithm:")
accuracy_scores, _, _, best_confusion_mat = get_accuracy_scores(
    k_vals, X_train, y_train, X_test)
confusion_matrices.append(best_confusion_mat)
plot_accuracy(k_vals, accuracy_scores, 'Default')

"""Model Comparison"""
plot_rocs(confusion_matrices, ['LogisticRegression', 'KNN'], 'Default')

"""# Genero Dataset
Prepare the data
"""

data = pd.read_csv('data/genero.txt')

data['Gender'] = data['Gender'].map(dict(Male=1, Female=0))

X = data[['Height', 'Weight']].to_numpy()
y = data.Gender.to_numpy()

"""Create split"""
# 0.80 for the training set
# 0.20 for the test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, shuffle=True)

"""## Logistic Regression
Create the Model
"""

lr = LogisticRegression()

"""Train the Model"""

lr.fit(X_train, y_train)

"""Create Predictions"""
y_prediction = lr.predict(X_test)

"""### Evaluation"""
confusion_matrices = []
print("------ GENERO DATASET -------")
print("With Sklearn LogisticRegression:")
conf_mat = confusion_matrix(y_test, y_prediction)
confusion_matrices.append(conf_mat)
print(conf_mat)
print("accurracy_score: ", accuracy_score(y_test, y_prediction))
print('\n\n')

"""## K-NN"""
print("With KNN Algorithm:")
accuracy_scores, best_prediction, best_k, best_confusion_mat = get_accuracy_scores(
    k_vals, X_train, y_train, X_test)
plot_accuracy(k_vals, accuracy_scores, 'Gender')
plot_predictions(best_prediction, X_test, y_test, best_k)
confusion_matrices.append(best_confusion_mat)

"""Model Comparison"""
plot_rocs(confusion_matrices, ['LogisticRegression', 'KNN'], 'Genero')
