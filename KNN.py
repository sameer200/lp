import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier  

dataset = pd.read_csv("Data.csv")
#print(dataset.head())

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:, 2].values

#print(X)
#print(Y) 
trainSize = X.shape[0]
X_train = X[0:trainSize-1]
X_test = X[trainSize-1:trainSize]
Y_train = Y[0:trainSize-1]

#print(X_train)
#print(X_test)
#print(Y_train)

indicesOfBlue = np.where( Y_train == 'B' )[0]
indicesOfYellow = np.where( Y_train == 'Y' )[0]

plt.scatter(X_train[indicesOfBlue,0], X_train[indicesOfBlue,1], color='b')
plt.scatter(X_train[indicesOfYellow,0], X_train[indicesOfYellow,1], color='y')

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, Y_train)  

y_pred = classifier.predict(X_test)
print(y_pred)

if y_pred == 'Y' :
    plt.scatter(X_test[:,0], X_test[:,1], color='y')
else :
    plt.scatter(X_test[:,0], X_test[:,1], color='b')

plt.show()


