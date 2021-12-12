import numpy as np 
import pandas as pd 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection  import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt

##############################################################################
def importdata():
    
    df1 = pd.read_csv("pol-fake.csv")
    
    df1["spam"] =0
 
    df2 = pd.read_csv("pol-real.csv")
    
    df2["spam"] =1

    #merge data
    balance_data=pd.concat([df2, df1])
    
    #shuffle data
    balance_data=balance_data.sample(frac = 1)
    
    # Printing the dataswet shape 
    print ("Dataset Length: ", len(balance_data)) 
    print ("Dataset Shape: ", balance_data.shape) 
    	
    return balance_data 

###############################################################################
def splitdataset(balance_data):
    
    # Separating the target variable
    X = balance_data.values[:, 0:35]
    Y = balance_data.values[:, 35]
    
    #Splitting the 80% of dataset into train and %20 to test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 100)
    
    return X, Y, X_train, X_test, y_train, y_test 

###############################################################################
def tarin_using_entropy(X_train, y_train): 

	# Decision tree with entropy 
	clf_entropy = DecisionTreeClassifier(criterion = "entropy") 

	# Performing training 
	clf_entropy=clf_entropy.fit(X_train, y_train) 
    
	return clf_entropy 

###############################################################################
def prediction(X_test, clf_object):
    
    y_pred = clf_object.predict(X_test)
 
    return y_pred 

###############################################################################
def cal_accuracy(y_test, y_pred):
    
    print("Confusion Matrix: ")
    print(confusion_matrix(y_test, y_pred))
    
    print ("Accuracy : ", 
	accuracy_score(y_test,y_pred)*100)
    
    print("Report : ")
    print(classification_report(y_test, y_pred)) 

###############################################################################
def main():
    
    data = importdata()
    
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
    
    clf_entropy = tarin_using_entropy(X_train, y_train)

    print("Results Using Entropy:")
    
    y_pred_entropy = prediction(X_test, clf_entropy)
    
    cal_accuracy(y_test, y_pred_entropy)
    
    tree.plot_tree(clf_entropy)
    
if __name__=="__main__":
    main()