#!/usr/bin/env python
# coding: utf-8

# # Set Up

# In[286]:


import numpy as np
import pandas as pd
import sklearn.linear_model 
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score  
from math import sqrt  # for root mean-squared error calculation
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

# ignore warnings
import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

plt.rcParams.update({'figure.max_open_warning': 0})

# seaborn plot settings
sns.set(style="ticks", rc={'figure.figsize':(12,8)})
sns.set_palette("coolwarm")

# seed value for random number generators to obtain reproducible results
RANDOM_SEED = 1

# although we standardize X and y variables on input,
# we will fit the intercept term in the models
# Expect fitted values to be close to zero
SET_FIT_INTERCEPT = True

# read data for the Boston Housing Study
boston_input = pd.read_csv('boston.csv')


# # Data Preparation

# In[287]:


print('\nGeneral Description of the Boston DataFrame:')
print(boston_input.info())


# In[288]:


# drop neighborhood from the data being considered
boston = boston_input.drop('neighborhood', 1)

print('\nGeneral description of the Boston DataFrame:')
print(boston.info())


# # Data Exploration

# In[289]:


print('First Five Rows of the Boston DataFrame:')
boston.head()


# In[290]:


print('Descriptive statistics of the Boston DataFrame:')
boston.describe()


# In[291]:


# correlation matrix/heat map to examine correlations among housing variables
def corr_chart(df_corr):
    corr=df_corr.corr()
    #screen top half to get a triangle
    top = np.zeros_like(corr, dtype=np.bool)
    top[np.triu_indices_from(top)] = True
    fig=plt.figure()
    fig, ax = plt.subplots(figsize=(12,12))
    sns.heatmap(corr, mask=top, cmap='coolwarm', 
        center = 0, square=True, 
        linewidths=.5, cbar_kws={'shrink':.5}, 
        annot = True, annot_kws={'size': 9}, fmt = '.3f')           
    plt.xticks(rotation=45) # rotate variable labels on columns (x axis)
    plt.yticks(rotation=0) # use horizontal variable labels on rows (y axis)
    plt.title('Correlation Heat Map', fontsize=16)   
    plt.savefig('plot-corr-map.pdf', 
        bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
        orientation='portrait', papertype=None, format=None, 
        transparent=True, pad_inches=0.25, frameon=None)     
    
corr_chart(df_corr = boston)


# Notes:
# Based off of this heatmap, it seems like the median price has positive correlations with:
# - the number of rooms - rooms (0.696)
# - percentage of land zoned for lots - zn (0.360) 
# - the weighted distance to employment centers - dis (0.249) 
# - whether or not the house is located on the Charles River - chas (0.176)
# 
# The median price also has a very negative correlation with the percentage of the population being of a lower socioeconomic status - istat (-0.741)
# 
# In addition, the tax rate (tax) and accessibility to the radial highway has a strong correlation (0.910)

# In[292]:


# view the distribution of the mv variable

sns.distplot(boston["mv"])

plt.title("1970's Median Housing Price Distribution", fontsize = 16)
plt.xlabel("Price (in thousands)", fontsize = 14)
plt.ylabel("Distribution (in decimals)", fontsize = 14)
plt.show()

plt.savefig('mvdist.pdf')


# In[293]:


# view the skew of the data
print("Skewness of variable mv: {}".format(boston['mv'].skew()))


# Note: Because mv has a skewness of > 1, it is considered substantially skewed

# In[294]:


# normalize the variable mv by taking the log of its values
logmv = np.log1p(boston['mv'])    # use log1p for correct output when x is a small value

boston['logmv'] = logmv

print("Skewness of log variable mv: {}".format(logmv.skew()))


# Note: normalizing mv by performing a log transformation has reduced the skewness substantially 

# In[295]:


# view distribution of log transformed mv

sns.distplot(logmv)

plt.title("1970's Median Housing Price Distribution (Natural Log)", fontsize = 16)
plt.xlabel("Price (in thousands)", fontsize = 14)
plt.ylabel("Distribution (in decimals)", fontsize = 14)
plt.show()

plt.savefig('logmvdist.pdf')


# In[296]:


sns.pairplot(boston)

plt.show()

plt.savefig('pairplot.pdf')


# # Model Set Up

# <b>Model Set-Up - Defining Model Data</b>

# In[297]:


# set up linear models and rmse metric
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet

names = ["OLS", "Ridge Regression", "Lasso Regression", "ElasticNet"]
linmodel = [LinearRegression(), 
            Ridge(alpha=0.1, fit_intercept=True, normalize=False, 
                  copy_X=True, max_iter=None, tol=0.001, solver='auto', 
                  random_state=RANDOM_SEED), 
            Lasso(alpha=0.1, fit_intercept=True, normalize=False, 
                  precompute=False, copy_X=True, max_iter=1000, tol=0.0001, 
                  warm_start=False, positive=False, random_state=RANDOM_SEED, 
                  selection='cyclic'), 
            ElasticNet(alpha=0.1, l1_ratio=0.5, fit_intercept=True, 
                       normalize=False, precompute=False, max_iter=1000, 
                       copy_X=True, tol=0.0001, warm_start=False, 
                       positive=False, random_state=RANDOM_SEED, 
                       selection='cyclic')]


# In[332]:


# set up preliminary data for data for fitting the models 
# the first column is the median housing value response
# the remaining columns are the explanatory variables
prelim_model_data = np.array([boston.mv,    boston.crim,    boston.zn,    boston.indus,    boston.chas,    boston.nox,    boston.rooms,    boston.age,    boston.dis,    boston.rad,    boston.tax,    boston.ptratio,    boston.lstat]).T

prelim_log_model_data = np.array([boston.logmv,    boston.crim,    boston.zn,    boston.indus,    boston.chas,    boston.nox,    boston.rooms,    boston.age,    boston.dis,    boston.rad,    boston.tax,    boston.ptratio,    boston.lstat]).T
# dimensions of the polynomial model X input and y response
# preliminary data before standardization
print('Data dimensions:', prelim_model_data.shape)


# In[333]:


# standard scores for the columns... along axis 0
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
print(scaler.fit(prelim_model_data))


# In[334]:


# show standardization constants being employed
print(scaler.mean_)
print(scaler.scale_)


# In[335]:


# the model data will be standardized form of preliminary model data
model_data = scaler.fit_transform(prelim_model_data)

log_model_data = scaler.fit_transform(prelim_log_model_data)

# dimensions of the polynomial model X input and y response
# all in standardized units of measure
print('Dimensions for model_data:', model_data.shape)


# <b>K-Fold Cross Validation & Linear Model Set Up</b>

# In[338]:


from sklearn.model_selection import KFold

# define number of folds for cross validation - 5 folds
n_folds = 5

# set up numpy array for storing results
rmse_results = np.zeros((n_folds, len(linmodel)))

kf = KFold(n_splits = n_folds, shuffle=False, random_state = RANDOM_SEED)

#initialize fold count
fold_count = 0

for train_index, test_index in kf.split(model_data):
    print("\n---------------- Fold Count: {} ----------------".format(fold_count))
    # define test and train variables
    # x represents the explanatory variables
    X_train = model_data[train_index, 1:model_data.shape[1]] # 1 to last
    X_test = model_data[test_index, 1:model_data.shape[1]] # 1 to last
    
    # model_data.shape[1]-1 is the response - predictive variable
    y_train = model_data[train_index,0] # 0
    y_test = model_data[test_index,0] # 0
   

    
    # provides info on the data shape for each fold
    print("\nShape of data for fold {}:".format(fold_count))
    print("Data Set: (Observations, Features)")
    print('X_train:', X_train.shape)
    print('X_test:',X_test.shape)
    print('y_train:', y_train.shape)
    print('y_test:',y_test.shape)
    
    model_count = 0
    
    for name, linear in zip(names, linmodel):
        # fit the model to the training set
        linear.fit(X_train, y_train)
        pred = linear.predict(X_test)
        train_score = linear.score(X_train, y_train)
        test_score = linear.score(X_test, y_test)
        rmse = sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)
        
        rmse_results[fold_count, model_count] = rmse
        
        print("\n{}".format(name))
        print("Train Set Fit: {}".format(round(train_score, 2)))
        print("Test Set Fit: {}".format(round(test_score, 2)))
        print("RMSE: {}".format(round(rmse, 2)))
        print("R^2: {}".format(round(r2, 2)))
        
        #plot the outputs
        plt.scatter(y_test, pred) 
        plt.show()
        
        model_count += 1
        
    fold_count += 1
        


# <b>K-Fold Cross Validation & Log-Linear Model Set Up</b>

# In[336]:


from sklearn.model_selection import KFold

# define number of folds for cross validation - 5 folds
n_folds = 5

# set up numpy array for storing results
log_rmse_results = np.zeros((n_folds, len(linmodel)))

kf = KFold(n_splits = n_folds, shuffle=False, random_state = RANDOM_SEED)

#initialize fold count
fold_count = 0

for train_index, test_index in kf.split(log_model_data):
    print("\n---------------- Fold Count: {} ----------------".format(fold_count))
    # define test and train variables
    # x represents the explanatory variables
    X_train = log_model_data[train_index, 1:model_data.shape[1]] # 1 to last
    X_test = log_model_data[test_index, 1:model_data.shape[1]] # 1 to last
    
    # model_data.shape[1]-1 is the response - predictive variable
    y_train = log_model_data[train_index,0] # 0
    y_test = log_model_data[test_index,0] # 0
   

    
    # provides info on the data shape for each fold
    print("\nShape of data for fold {}:".format(fold_count))
    print("Data Set: (Observations, Features)")
    print('X_train:', X_train.shape)
    print('X_test:',X_test.shape)
    print('y_train:', y_train.shape)
    print('y_test:',y_test.shape)
    
    model_count = 0
    
    for name, linear in zip(names, linmodel):
        # fit the model to the training set
        linear.fit(X_train, y_train)
        pred = linear.predict(X_test)
        train_score = linear.score(X_train, y_train)
        test_score = linear.score(X_test, y_test)
        rmse = sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)
        
        log_rmse_results[fold_count, model_count] = rmse
        
        print("\n{}".format(name))
        print("Train Set Fit: {}".format(round(train_score, 2)))
        print("Test Set Fit: {}".format(round(test_score, 2)))
        print("RMSE: {}".format(round(rmse, 2)))
        print("R^2: {}".format(round(r2, 2)))
        
        #plot the outputs
        plt.scatter(pred, y_test) 
        plt.show()
        
        model_count += 1
        
    fold_count += 1
        


# In[339]:


# load cross validation data into a DataFrame
results = pd.DataFrame(rmse_results)
results.columns = names

# print average results for both classification model
print("Average RMSE Results from {}-fold cross-validation:".format(str(n_folds)))
results = results.mean()
results


# In[340]:


# load cross validation data into a DataFrame
logresults = pd.DataFrame(log_rmse_results)
logresults.columns = names

# print average results for both classification model
print("Average RMSE Results from {}-fold cross-validation:".format(str(n_folds)))
logresults = logresults.mean()
logresults

