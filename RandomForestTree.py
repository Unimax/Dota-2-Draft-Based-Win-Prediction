from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('intitialData.csv', header=None,
                   names=['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10',  'WIN'])
X = data.iloc[:, 0:-1].values
Y = np.squeeze(data.iloc[:, -1:].values)

model = RandomForestClassifier(n_estimators=10)
model.fit(X, Y)

estimator = model.estimators_[10]
# Export as dot file
export_graphviz(estimator,
                out_file='tree.dot',
                feature_names = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10'],
                class_names = ['Win','Lose'],
                rounded = True, proportion = False,
                precision = 2, filled = True)

from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

# Display in jupyter notebook
from IPython.display import Image
Image(filename = 'tree.png')