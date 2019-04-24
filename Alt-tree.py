# 1.import packages
import numpy as np
import pandas as pd

# 2 Read CSV
dataset=pd.read_csv("buys.csv")
x=dataset.iloc[:,:-1]
y=dataset.iloc[:,4].values

# 3 Label Encoding and Decision Classifier
from sklearn.preprocessing import LabelEncoder
lab_x=LabelEncoder()
x=x.apply(LabelEncoder().fit_transform)

from sklearn import tree
reg=tree.DecisionTreeClassifier()
reg.fit(x,y)
reg.predict(x)

# 4 Predict Value for given condition
xp=np.array([1,1,0,0])
yy=reg.predict([xp])
print(yy)

# 5 plot
dot_data=tree.export_graphviz(reg,out_file='imagetree.dot')


#terminal cmds....
# 1. sudo apt install graphviz
# 2. dot imagetree.dot -Tpng -o imagetree.png