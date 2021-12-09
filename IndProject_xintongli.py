# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 15:33:29 2021

@author: xintong
"""

##########import data###########
import pandas as pd
import numpy as np
raw_data = pd.read_excel('Kickstarter.xlsx', index_col = None)

##########data cleaning################

#1. drop invalid features (columns that can only appear after the project is launched)
#invalid_features = ['pledged', 'staff_pick', 'backers_count','state_changed_at','launched_at','usd_pledged', 'state_changed_at_weekday','launched_at_weekday','state_changed_at_month','state_changed_at_day','state_changed_at_yr','state_changed_at_hr','launched_at_month','launched_at_day','launched_at_yr','launched_at_hr','create_to_launch_days','launch_to_deadline_days','launch_to_state_change_days']

invalid_features = ['pledged', 'staff_pick', 'backers_count','state_changed_at','usd_pledged', 'state_changed_at_weekday','state_changed_at_month','state_changed_at_day','state_changed_at_yr','state_changed_at_hr','launch_to_state_change_days']


for item in invalid_features:
    raw_data.drop(item, axis=1, inplace=True)

#2. drop the rows which contains missing values
#first, check if there are some
raw_data.isna()
raw_data.isna().sum()
#drop those rows with missing values
raw_data = raw_data.dropna()

#3. drop the column which is perfectly correlated with the target
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

#convert to numerical data so we can draw heatmap
from sklearn.preprocessing import *
def prepare_inputs(X):
	oe = OrdinalEncoder()
	oe.fit(X)
	X_enc = oe.transform(X)
	return X_enc

column_list = raw_data.columns
df_enc = prepare_inputs(raw_data)
df_enc = pd.DataFrame(data=df_enc, columns=column_list)

plt.figure(figsize=(20,20))
cor = df_enc.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show() #as we can see from the heatmap, spotlight column is highly correlated with state column, so we should not include it

raw_data.drop('spotlight', axis=1, inplace=True)

#4. drop rows whose state column's value is not fail/success
options = ['failed', 'successful']
cleaned_data = raw_data[raw_data['state'].isin(options)]

#5. drop columns which intuitively does not make sense
cleaned_data.drop('project_id', axis=1, inplace=True)

#########data preprocessing################
#1. dummify categorical variables
#first, remove those non-numerical columns who are not available to dummify
cleaned_data.drop('name', axis=1, inplace=True)

cleaned_data[['created_at','deadline']] = cleaned_data[['created_at','deadline']].apply(pd.to_datetime) #if conversion required
cleaned_data['time_available'] = (cleaned_data['deadline'] - cleaned_data['created_at']).dt.days
cleaned_data.drop('created_at', axis=1, inplace=True)
cleaned_data.drop('deadline', axis=1, inplace=True)
cleaned_data.drop('launched_at', axis=1, inplace=True)
#second, now we can dummify the other non-numerical variables
non_numerical = ['disable_communication','country','currency','category','deadline_weekday','created_at_weekday','launched_at_weekday']
processed_data = pd.get_dummies(cleaned_data, columns = non_numerical)
#thirdly convert the target variable also to a numercial one
processed_data.loc[(processed_data.state == 'failed'),'state']=0
processed_data.loc[(processed_data.state == 'successful'),'state']=1
processed_data['state'] = processed_data['state'].astype('int') 

###########save the cleaned&processed data locally##############
#processed_data.to_csv('cleaned_processed_data.csv')

#1. split target and predictor
X = processed_data.drop(columns = ['state'])
y = processed_data['state']

#3. standardize predictors
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_std = scaler.fit_transform(X)


########$$$$############## Part 1: classification models #########$$$$################

#############try different models to find the one with highest accuracy#############

###################################
#      Model 1                    #
###################################

#USE LASSO TO FEATURE SELECTION
from sklearn.linear_model import Lasso
model = Lasso(alpha=0.05)
model.fit(X_std, y)
model.coef_

lasso_result = pd.DataFrame(list(zip(X.columns, model.coef_)), columns = ['predictor','coefficient'])

#remove the features selected by lasso
useless = lasso_result[lasso_result['coefficient'] == 0]
useful = lasso_result[lasso_result['coefficient'] != 0]
useful_list = useful['predictor'].tolist()

X = processed_data[processed_data.columns.intersection(useful_list)]
y = processed_data['state']
#y=y.astype('int') 

#standardize X
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

#split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_std,y, test_size = 0.3, random_state=5)

#build KNN model(record time during this process)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
model2 = knn.fit(X_train, y_train)

#gernerate classification report and confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
y_test_pred = model2.predict(X_test)
print(classification_report(y_test, y_test_pred))
from sklearn.metrics import accuracy_score
accuracy1 = accuracy_score(y_test, y_test_pred)
print("accuracy of first model: " + str(accuracy1))

##################################
#      Model 2                   #
##################################

#still use the lasso's selection result
#but this time, use GBT model as our classification model
from sklearn.ensemble import GradientBoostingClassifier
gbt = GradientBoostingClassifier()

model = gbt.fit(X_train, y_train)
y_test_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy1 = accuracy_score(y_test, y_test_pred)
print("accuracy of second model: " + str(accuracy1))


##############################
#   Model 3                  #
##############################

#this time, use Random forest to do feature selection and then use GBT model to classify

#set target and predictors
X = processed_data.drop(columns = ['state'])
y = processed_data['state']

#use random forest to select features
from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier(random_state=0)
model = randomforest.fit(X_std,y)

model.feature_importances_
rf_result = pd.DataFrame(list(zip(X.columns, model.feature_importances_)), columns = ['predictor','feature importance'])

threshold = 0.04

useful = rf_result[rf_result['feature importance'] >= 0.04]
useful_list = useful['predictor'].tolist()

X = processed_data[processed_data.columns.intersection(useful_list)]
y = processed_data['state']

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

#build GBT model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_std,y, test_size = 0.3, random_state=5)

from sklearn.ensemble import GradientBoostingClassifier
gbt = GradientBoostingClassifier()

model = gbt.fit(X_train, y_train)
model.feature_importances_
rf_result = pd.DataFrame(list(zip(X.columns, model.feature_importances_)), columns = ['predictor','feature importance'])

y_test_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy1 = accuracy_score(y_test, y_test_pred)
print("accuracy of third model: " + str(accuracy1))



###########################
#   Model 4              #
###########################

#this time, use GBT to do feature selection and then use GBT model to classify

#set target and predictors
X = processed_data.drop(columns = ['state'])
y = processed_data['state']

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

#use GBT to select features
from sklearn.ensemble import GradientBoostingClassifier
gbt = GradientBoostingClassifier()
model = gbt.fit(X_std, y)
model.feature_importances_
rf_result = pd.DataFrame(list(zip(X.columns, model.feature_importances_)), columns = ['predictor','feature importance'])

threshold = 0.05

useful = rf_result[rf_result['feature importance'] >= 0.05]
useful_list = useful['predictor'].tolist()

#since there is a category value in esful list, we need to add the rest category value, too
columns = processed_data.columns 
category_columns = []
for item in columns:
    if 'category' in item:
        category_columns.append(item)
        
useful_list_4 = useful_list+category_columns

#build GBT model
X = processed_data[processed_data.columns.intersection(useful_list_4)]
y = processed_data['state']

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_std,y, test_size = 0.3, random_state=5)

from sklearn.ensemble import GradientBoostingClassifier
gbt = GradientBoostingClassifier()
model4 = gbt.fit(X_train, y_train)

y_test_pred = model4.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy1 = accuracy_score(y_test, y_test_pred)
print("accuracy of fourth model: " + str(accuracy1))

#Use grid search to find best parameter of our GBT model
from sklearn.model_selection import cross_val_score
import numpy
scope = [120, 130, 140, 150]
result = []
for i in range(2,5):
    for j in scope:
        model2 = GradientBoostingClassifier(random_state=0, min_samples_split=i, n_estimators = j)
        scores = cross_val_score(estimator = model2, X=X_std, y=y, cv=5)
        result.append(((i,j),numpy.average(scores)))
        print(i,j,':', numpy.average(scores))

def maxTuple(listOfTuple):
    maxTuple = 0
    leftTuple = (0,0)
    for eachTuple in listOfTuple:
        if eachTuple[1] >= maxTuple:
            maxTuple = eachTuple[1]
            leftTuple = eachTuple[0]
    return leftTuple
    

(optimal_min_samples_split, optimal_n_estimators) = maxTuple(result)

#use the grid search result to refactor our GBT model
from sklearn.ensemble import GradientBoostingClassifier
gbt_optimal = GradientBoostingClassifier(random_state = 0, min_samples_split = optimal_min_samples_split, n_estimators = optimal_n_estimators)

model_optimal = gbt_optimal.fit(X_train, y_train)
y_test_pred = model_optimal.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy1 = accuracy_score(y_test, y_test_pred)
print("accuracy of fourth model after hyperparameter tuning: " + str(accuracy1))


###########################
#   Model 5               #
###########################

#this time, use PCA to do feature selection and then use GBT model to classify
#Based on the selection result made by random forest, I would like our PCA to choose 6 predictors this time
number_of_components = 6

#define target & predictors
X = processed_data.drop(columns = ['state'])
y = processed_data['state']

#Use PCA to do dimension reduction
from sklearn.decomposition import PCA
pca = PCA(n_components=number_of_components)
pca.fit(X)
X_new = pca.transform(X)

pca.explained_variance_ratio_

#standardize X_new
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_std = scaler.fit_transform(X_new)

#split train test dataset using results from PCA
X_train,X_test, y_train,y_test = train_test_split(X_std, y, test_size = 0.30, random_state=5)

#build gbt model
from sklearn.ensemble import GradientBoostingClassifier
gbt = GradientBoostingClassifier()

model = gbt.fit(X_train, y_train)
y_test_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy1 = accuracy_score(y_test, y_test_pred)
print("accuracy of second model: " + str(accuracy1))




###################$$$$############ Extra Part: To help TA grading my classificaion model #########$$$########
print("Dear TA, the part below is for you to test my model accuracy")

##########import data###########
import pandas as pd
import numpy as np
raw_data = pd.read_excel('Kickstarter-Grading-Sample.xlsx', index_col = None)

##########data cleaning################

#1. drop invalid features (columns that can only appear after the project is launched)
#invalid_features = ['pledged', 'staff_pick', 'backers_count','state_changed_at','launched_at','usd_pledged', 'state_changed_at_weekday','launched_at_weekday','state_changed_at_month','state_changed_at_day','state_changed_at_yr','state_changed_at_hr','launched_at_month','launched_at_day','launched_at_yr','launched_at_hr','create_to_launch_days','launch_to_deadline_days','launch_to_state_change_days']

invalid_features = ['pledged', 'staff_pick', 'backers_count','state_changed_at','usd_pledged', 'state_changed_at_weekday','state_changed_at_month','state_changed_at_day','state_changed_at_yr','state_changed_at_hr','launch_to_state_change_days']


for item in invalid_features:
    raw_data.drop(item, axis=1, inplace=True)

#2. drop the rows which contains missing values
#first, check if there are some
raw_data.isna()
raw_data.isna().sum()
#drop those rows with missing values
raw_data = raw_data.dropna()

#3. drop the column which is perfectly correlated with the target
raw_data.drop('spotlight', axis=1, inplace=True)

#4. drop rows whose state column's value is not fail/success
options = ['failed', 'successful']
cleaned_data = raw_data[raw_data['state'].isin(options)]

#5. drop columns which intuitively does not make sense
cleaned_data.drop('project_id', axis=1, inplace=True)

#########data preprocessing################
#1. dummify categorical variables
#first, remove those non-numerical columns who are not available to dummify
cleaned_data.drop('name', axis=1, inplace=True)

cleaned_data[['created_at','deadline']] = cleaned_data[['created_at','deadline']].apply(pd.to_datetime) #if conversion required
cleaned_data['time_available'] = (cleaned_data['deadline'] - cleaned_data['created_at']).dt.days
cleaned_data.drop('created_at', axis=1, inplace=True)
cleaned_data.drop('deadline', axis=1, inplace=True)
cleaned_data.drop('launched_at', axis=1, inplace=True)
#second, now we can dummify the other non-numerical variables
non_numerical = ['disable_communication','country','currency','category','deadline_weekday','created_at_weekday','launched_at_weekday']
processed_data = pd.get_dummies(cleaned_data, columns = non_numerical)
#thirdly convert the target variable also to a numercial one
processed_data.loc[(processed_data.state == 'failed'),'state']=0
processed_data.loc[(processed_data.state == 'successful'),'state']=1
processed_data['state'] = processed_data['state'].astype('int') 

###########save the cleaned&processed data locally##############
#processed_data.to_csv('cleaned_processed_data.csv')

#1. split target and predictor
X = processed_data.drop(columns = ['state'])
y = processed_data['state']

#3. standardize predictors
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_std = scaler.fit_transform(X)


############ select out those important features using the feature selection result ###########

X = processed_data[processed_data.columns.intersection(useful_list_4)]
y_grading = processed_data['state']

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

############# fit to model and get accuracy result ##############

y_test_pred = model_optimal.predict(X_std) 

from sklearn.metrics import accuracy_score
accuracy1 = accuracy_score(y_grading, y_test_pred)
print("accuracy of Xintong's model after hyperparameter tuning: " + str(accuracy1))
print("\n")
print("\n")









########$$$$############## Part 2: clustering models #########$$$$################

######################
#    Model 1         #
######################
##########import data###########
import pandas as pd
import numpy as np
raw_data = pd.read_excel('Kickstarter.xlsx', index_col = None)

#0. drop rows whose state column's value is not fail/success
options = ['failed', 'successful']
raw_data = raw_data[raw_data['state'].isin(options)]

#1. drop meaningless features (columns that does not make sense for clustering)
meaningless_features = ['project_id','name','state','deadline','created_at','state_changed_at','launched_at']

for item in meaningless_features:
    raw_data.drop(item, axis=1, inplace=True)
    
#2. drop columns which contains too many missing values
raw_data.drop('launch_to_state_change_days', axis=1, inplace=True)

#3. drop the rows which contains missing values
#first, check if there are some
raw_data.isna()
raw_data.isna().sum()
#drop those rows with missing values
cleaned_data_for_cluster = raw_data.dropna()



#5. dummify the non-numerical variable
non_numerical = ['disable_communication','country','currency','staff_pick','category','spotlight','deadline_weekday','state_changed_at_weekday','created_at_weekday','launched_at_weekday']
processed_data_for_cluster = pd.get_dummies(cleaned_data_for_cluster, columns = non_numerical)


import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score

def ConvertToArr(l):
    l = l.tolist()
    arr = np.asarray(l).reshape(-1,1)
    return arr

def MeasurePerformance(Arr):
    kmeans = KMeans(n_clusters = 2)
    model = kmeans.fit(Arr)
    labels = model.labels_
    performance = silhouette_score(Arr,labels)
    return performance

features = processed_data_for_cluster.columns
print("Numer of 0's in column disable communicationFalse is:", processed_data_for_cluster['disable_communication_False'].tolist().count(0), ", this means it cannot work with Kmeans model with K=2 beacuse it only has one unique value")
print("Since the other dummy variables all have a possibility to cause the same error as column disable_communication_False, and they for sure cannot work with Kmeans model with K=3 that I also want to try later on, so I won't consider these columns when selecting features for my cluster model")
print("\n")
#dropping dummy columns from features
features = features[0:27]
l = [] 

from tqdm import tqdm
def FeatureCombos():
    for i in tqdm(range(len(features))):
        name = features[i]
        series = processed_data_for_cluster[name]
        arr = ConvertToArr(series)
        performance = MeasurePerformance(arr)
        t = (name, performance)
        l.append(t)
        l.sort(key=lambda x:x[1]) #sort the list of tupples by its second element, which is the silhouette score
    return l

#this line of code below might take some time to run because it will loop through almost all columns, don't worry ;)
l_k2_sorted = FeatureCombos()

#then we select those features whose silhoutte scores are obviously higher, this part is subjective so I didn't write a python function to select
#we select the goal, time_available and static_usd_rate columns
#extract useful features
X = processed_data_for_cluster[['goal','backers_count','pledged','usd_pledged','create_to_launch_days','static_usd_rate']]
#standardize input
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

#use selected features to build k-Means model
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 2)
model = kmeans.fit(X_std)
labels = model.labels_
labels

#measure model performance
from sklearn.metrics import silhouette_samples
silhouette_samples(X_std, labels)

from sklearn.metrics import silhouette_score
silhouette_score(X_std, labels)
print("first clustering model's silhouette score is: " + str(silhouette_score(X_std, labels)))

#count how many are in first cluster and how many in second
l1 = []
l0 = []
for i in range(len(labels)):
    if labels[i] == 1:
        l1.append(labels[i])
    else:
        l0.append(labels[i])
        
print("There are "+ str(len(l1)) + " numbers in the first cluster")
print("There are "+ str(len(l0))+" numbers in the second cluster")

#find the cluster center's characteristics
centers = model.cluster_centers_
print(centers)


#####################
#   Model 2         #
#####################

#in this model, we will use all the continuous&categorical columns
#but we will still drop out those which does not make sense

##########import data###########
import pandas as pd
import numpy as np
raw_data = pd.read_excel('Kickstarter.xlsx', index_col = None)

#0. drop rows whose state column's value is not fail/success
options = ['failed', 'successful']
raw_data = raw_data[raw_data['state'].isin(options)]

#1. drop meaningless features (columns that does not make sense for clustering)
meaningless_features = ['project_id','name','state','deadline','created_at','state_changed_at','launched_at']

for item in meaningless_features:
    raw_data.drop(item, axis=1, inplace=True)
    
#2. drop columns which contains too many missing values
raw_data.drop('launch_to_state_change_days', axis=1, inplace=True)

#3. drop the rows which contains missing values
#first, check if there are some
raw_data.isna()
raw_data.isna().sum()
#drop those rows with missing values
cleaned_data_for_cluster = raw_data.dropna()


#5. dummify the non-numerical variable
non_numerical = ['disable_communication','country','currency','staff_pick','category','spotlight','deadline_weekday','state_changed_at_weekday','created_at_weekday','launched_at_weekday']
processed_data_for_cluster = pd.get_dummies(cleaned_data_for_cluster, columns = non_numerical)

##########build model###############
X = processed_data_for_cluster
#standardize input
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

############use Elbow method to find optimal k##########
from tqdm import tqdm
withinss = []
results = []
for i in tqdm(range(2,8)):
    kmeans = KMeans(n_clusters = i)
    model = kmeans.fit(X_std)
    labels = model.labels_
    withinss.append(model.inertia_)
    results.append(silhouette_score(X_std, labels))

from matplotlib import pyplot
print("Hi, please run the following two lines sepearately!!!!!!!  ;)")
#1
pyplot.plot([2,3,4,5,6,7],withinss)
print("Dear TA, please run these following two lines sepearately, otherwise its hard to see the second picture's shapem!!!!!!")
#2
pyplot.plot([2,3,4,5,6,7],results) 
print("From the second graph we can see, Silhouette score drops dramatically after k=2, so I decided to use k=2")
#use selected features to build k-Means model
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 2)
model = kmeans.fit(X_std)
labels = model.labels_
labels

from sklearn.metrics import silhouette_samples
silhouette_samples(X_std, labels)

from sklearn.metrics import silhouette_score
silhouette_score(X_std, labels)

#count how many are in first cluster and how many in second
l1 = []
l0 = []
for i in range(len(labels)):
    if labels[i] == 1:
        l1.append(labels[i])
    else:
        l0.append(labels[i])
        
print("There are "+ str(len(l1)) + " numbers in the first cluster")
print("There are "+ str(len(l0))+" numbers in the second cluster")

#find the cluster center's characteristics
centers = model.cluster_centers_
print(centers)

