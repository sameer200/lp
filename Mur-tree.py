import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import graphviz


#1.reading dataset
data=pd.read_csv("Data.csv")
features=data.columns
features=features[1:-1]
class_names=list(set(data.iloc[1:,-1]))

#2.printing the columns which are categorical string types
#for i in data.columns.values:
    #print(data[i].dtypes)

#3.Converting string categorical columns into numerical categorical columns
for i in data.columns.values:
    le=preprocessing.LabelEncoder()
    if data[i].dtypes=="object":
        #le.fit(data[i])
        data[i]=le.fit_transform(data[i])

X_train , X_test, Y_train, Y_test = train_test_split(data.iloc[:,1:5],\
                                                     data.iloc[:,-1],\
                                                     train_size=0.7,\
                                                     random_state = 100,\
                                                     shuffle=True)

#4.Splitting data into training and testing
'''
X_train=data.values[0:12,1:5]
#print(X_train)
Y_train=data.values[0:12,5]
#print(Y_train)
X_test=data.values[12:,1:5]
#print(X_test)
'''
#5.DecisionTree using gini index
# clf_gini=DecisionTreeClassifier(criterion="gini")
# clf_gini.fit(X_train,Y_train)
# output1=clf_gini.predict([X_test])
# print(output1)
# dot_data1=tree.export_graphviz(clf_gini,out_file=None,feature_names=features,class_names=class_names,filled=True,rounded=True,special_characters=True)
# graph=graphviz.Source(dot_data1)
# graph.render('DecisionTreeGini',view=True)

#DecisionTree using entropy and IG
clf_entropy=DecisionTreeClassifier(criterion="entropy")
clf_entropy.fit(X_train,Y_train)
output2=clf_entropy.predict(X_test)
print(output2)
print( accuracy_score(Y_test , output2) )
print( classification_report(Y_test , output2) )
print( confusion_matrix(Y_test , output2) )

dot_data2=tree.export_graphviz(clf_entropy,out_file=None,feature_names=features,class_names=class_names,filled=True,rounded=True,special_characters=True)
graph=graphviz.Source(dot_data2)
graph.render('DecisionTreeEntropy',view=True)


# INFO GAIN
'''
    For output var :
        -sum(plogp)
        p -> represents probability of that cateogry of that class

    Entropy For every feature :
        all classes of each column
            find count of categories of ouput class for each category of that attribute

    Gain for attribute :
        Entropy of output - entropy of output with attribute
                        
'''
