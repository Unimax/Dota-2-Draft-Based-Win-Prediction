import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
import seaborn as sn
import requests
from matplotlib import pyplot
from pandas import DataFrame
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas_profiling
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.utils.multiclass import unique_labels

from plotmatrix import pretty_plot_confusion_matrix

data = pd.read_csv('CombinedOneHotData.csv', header=None)

# data["H1"] = data["H1"].astype('category')
X = data.iloc[:, 0:-1].values
Y = np.squeeze(data.iloc[:, -1:].values)
# Feature eng.
# myOneHotEncoder = preprocessing.OneHotEncoder(categories='auto',dtype =np.int8)
# myOneHotEncoder.fit(X)
# X = myOneHotEncoder.transform(X).toarray()

print(X.shape, Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

models = []
models.append(('Logistic Regression', LogisticRegression(solver='liblinear')))
models.append(('Ada Boost Classifier', AdaBoostClassifier(n_estimators=1200)))
models.append(('Linear Discriminant Analysis', LinearDiscriminantAnalysis()))
# models.append(('RFC', RandomForestClassifier(n_estimators=200, max_features=3)))
# models.append(('XTC', ExtraTreesClassifier(n_estimators=200, max_features=3)))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))

# Precision-Recall Plot
for name, model in models:
    model.fit(X_train,Y_train)
    y_predict = model.predict(X_test)
    print(name," acc :",auc(Y_test,y_predict))
# Precision-Recall Plot
# for name, model in models:
#     model.fit(X_train,Y_train)
#     # predict probabilities
#     lr_probs = model.predict_proba(X_test)
#     # keep probabilities for the positive outcome only
#     lr_probs = lr_probs[:, 1]
#     # predict class values
#     yhat = model.predict(X_test)
#     lr_precision, lr_recall, _ = precision_recall_curve(Y_test, lr_probs)
#     lr_f1, lr_auc = f1_score(Y_test, yhat), auc(lr_recall, lr_precision)
#     # summarize scores
#     print(name,': f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
#     # plot the precision-recall curves
#     no_skill = len(Y_test[Y_test == 1]) / len(Y_test)
#     pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
#     pyplot.plot(lr_recall, lr_precision, marker='.', label=name)
#     # axis labels
#     pyplot.xlabel('Recall')
#     pyplot.ylabel('Precision')
#     # show the legend
#     pyplot.legend()
#     # show the plot
#     pyplot.show()

# ROC Recall Plot
# for name, model in models:
#     model.fit(X_train,Y_train)
#     # generate a no skill prediction (majority class)
#     ns_probs = [0 for _ in range(len(Y_test))]
#     # predict probabilities
#     lr_probs = model.predict_proba(X_test)
#     # keep probabilities for the positive outcome only
#     lr_probs = lr_probs[:, 1]
#     # calculate scores
#     ns_auc = roc_auc_score(Y_test, ns_probs)
#     lr_auc = roc_auc_score(Y_test, lr_probs)
#     # summarize scores
#     print('No Skill: ROC AUC=%.3f' % (ns_auc))
#     print(name,': ROC AUC=%.3f' % (lr_auc))
#     # calculate roc curves
#     ns_fpr, ns_tpr, _ = roc_curve(Y_test, ns_probs)
#     lr_fpr, lr_tpr, _ = roc_curve(Y_test, lr_probs)
#     # plot the roc curve for the model
#     pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
#     pyplot.plot(lr_fpr, lr_tpr, marker='.', label=name)
#     # axis labels
#     pyplot.xlabel('False Positive Rate')
#     pyplot.ylabel('True Positive Rate')
#     # show the legend
#     pyplot.legend()
#     # show the plot
#     pyplot.show()


# for name, model in models:
#     model.fit(X_train,Y_train)
#     y_pred = model.predict(X_test)
#
#
#     df_cm = DataFrame(confusion_matrix(Y_test,y_pred), index=range(0, 2), columns=range(0, 2))
#     # colormap: see this and choose your more dear
#     cmap = 'PuRd'
#     pretty_plot_confusion_matrix(df_cm, cmap=cmap)
#     plt.show()
