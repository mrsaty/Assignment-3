#!/usr/bin/env python
# coding: utf-8

# In[225]:


import numpy as np
import pandas as pd
import seaborn as sn


# # Data Extraction

# In[97]:


data = pd.read_csv('/home/satish/Downloads/sonar.all-data', header = None)
data.head()


# In[98]:


x = data.values[:,:60]
#x.shape      #(208, 60)

y = data.values[:,60:]
#y.shape      #(208, 1)


# In[237]:


#convert R and M into numeric values 
#coz Random Forest works only on numeric values

_list = []
for i in y:
    if i == 'M':
        _list.append(1)
    else:
        _list.append(0)
y1 = np.asarray(_list)
#y1


# In[233]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y1, test_size = 0.2, random_state = 100)


# In[234]:


#x_train.shape   #(166, 60)
#x_test.shape   #(42, 60)


# # Random Forest

# In[315]:


#algo for random forest classification

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 20, criterion = 'gini', random_state = 100)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
#y_pred


# In[316]:


#checking model usability

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
sn.heatmap(confusion_matrix(y_test, y_pred), annot = True)


# In[317]:


accuracy = accuracy_score(y_test, y_pred)
print('accuracy = ', accuracy)

class_report = classification_report(y_test, y_pred, target_names = ['Rocks-0', 'Mines-1'])
print('\nclassification_report : ')
print(class_report)


# In[318]:


#plot of accuracy vs no. of trees

n_estimators = [5, 10, 15, 20, 30, 40, 50, 60, 70]
accuracy = [0.809, 0.857, 0.857, 0.809, 0.857, 0.857, 0.881, 0.857, 0.857]

import matplotlib.pyplot as plt
plt.plot(n_estimators, accuracy)
plt.xlabel('Number of trees')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of trees')


# # Hyperparameter tuning in Random Forest

# In[338]:


#find out all the parameters with their default values

from pprint import pprint
pprint(classifier.get_params())


# In[320]:


from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(2, 20, num = 10)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)


# In[321]:


# Use the random grid to search for best hyperparameters
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
classifier_random = RandomizedSearchCV(estimator = classifier, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
classifier_random.fit(x_train, y_train)


# In[322]:


#prints the best parameters which should be used for training the model
classifier_random.best_params_


# In[323]:


#function to calculate model performance

def evaluate(model, x_test, y_test):
    y_pred = model.predict(x_test)
    errors = abs(y_pred - y_test)
    accuracy = accuracy_score(y_test, y_pred)
    mean_error = np.mean(errors)
    print('Model Performance')
    print('Average Error: {:0.4f}'.format(mean_error))
    print('Accuracy = {:0.2f}'.format(accuracy))
    print()
    
    return accuracy

base_model = RandomForestClassifier(n_estimators = 20, random_state = 100)
base_model.fit(x_train, y_train)
base_accuracy = evaluate(base_model, x_test, y_test)

best_model = classifier_random.best_estimator_
best_accuracy = evaluate(best_model, x_test, y_test)

print('Improvement of {:0.2f}'.format((best_accuracy - base_accuracy) / base_accuracy))


# # Support Vector Machine

# In[324]:


#algo used for svm

from sklearn import svm
model_svm = svm.SVC(kernel="linear",gamma = 'scale')
model_svm.fit(x_train,y_train)


# In[326]:


#values predicted by the model

y_pred = model_svm.predict(x_test)
y_pred


# In[332]:


#determine the usability of the svm model

acc_model_svm = accuracy_score(y_test, y_pred)
print('accuracy_score_svm = ',acc_model_svm)


# In[333]:


sn.heatmap(confusion_matrix(y_test, y_pred), annot = True)


# In[334]:


print(classification_report(y_test, y_pred))


# # Hyperparameter tuning in SVM

# In[339]:


#prints all the parameters used in the model with their default values

from pprint import pprint
pprint(model_svm.get_params())


# In[336]:


#to avoid the warning message

import warnings
warnings.filterwarnings("ignore")


# In[340]:


#find out the best parameters to be used with their best values

from sklearn.model_selection import RandomizedSearchCV

Cs = [ 0.1, 1, 10,100]
gammas = [ 0.1,1,10]
kernel=["rbf","poly","linear"]
random_search = {'C': Cs, 'gamma' : gammas,"kernel":kernel}
grid_search = RandomizedSearchCV(svm.SVC(), param_grid, cv=5,verbose=0,random_state=0)
grid_search.fit(x, y)
grid_search.best_params_


# In[341]:


#train the model using best parameters

svc_best=svm.SVC(kernel="rbf",C=100,gamma=0.1)
svc_best.fit(x_train,y_train)
svc_predict=svc_best.predict(x_test)


# In[342]:


acc_model_svm = accuracy_score(y_test,svc_predict)
print('accuracy_score = ',acc_model_svm)


# # Decision Trees

# In[343]:


#algo used to train the Decision Tree Model

from sklearn import tree
model_tree = tree.DecisionTreeClassifier()
model_tree = model_tree.fit(x_train, y_train)


# In[344]:


y_pred = model_tree.predict(x_test)
y_pred


# In[345]:


print('Accuracy_score = ', accuracy_score(y_test, y_pred))


# In[227]:


sn.heatmap(confusion_matrix(y_test, y_pred), annot = True)


# In[220]:


print(classification_report(y_test, y_pred))


# # Hyperparameter tuning in Decision Tree

# In[346]:


dc=tree.DecisionTreeClassifier()
#maximum depth
max_depth=[4,6,8,10]
#min number of samples required for split
min_samples_split=[4,8,10,12]
#min number of samples required as each leaf node
min_samples_leaf=[2,3,4,5,6]


# In[347]:


param_grid_dc={"max_depth":max_depth,"min_samples_split":min_samples_split,"min_samples_leaf":min_samples_leaf}


# In[348]:


dc_random=RandomizedSearchCV(estimator=dc,param_distributions=param_grid_dc,cv=5)


# In[349]:


dc_random.fit(x_train,y_train)


# In[355]:


#best parameters to be used
dc_random.best_params_


# In[356]:


#train the model using best parameters
dc_best=tree.DecisionTreeClassifier(min_samples_split=5,min_samples_leaf=4,max_depth=4)
dc_best.fit(x_train,y_train)


# In[357]:


#predict the result using the best model
dc_predict=dc_best.predict(x_test)
dc_predict


# In[358]:


sn.heatmap(confusion_matrix(y_test,dc_predict),annot = True)


# In[359]:


accuracy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
precision = cm[0][0]/(cm[0][0]+cm[0][1])
recall = cm[0][0]/(cm[0][0]+cm[1][0])
print("Accuracy: ",accuracy)
print("Precision: ",precision)
print("Recall :",recall)


# In[ ]:




