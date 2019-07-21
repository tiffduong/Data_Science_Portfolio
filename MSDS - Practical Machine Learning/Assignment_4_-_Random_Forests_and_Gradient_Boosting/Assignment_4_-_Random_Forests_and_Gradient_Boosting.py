
# coding: utf-8

# # Set Up

# In[1]:

import numpy as np
import pandas as pd
import sklearn.linear_model 
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score  
from math import sqrt  # for root mean-squared error calculation
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().magic('matplotlib inline')

# ignore warnings
from sklearn.exceptions import DataConversionWarning
import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings('ignore', category=DataConversionWarning)

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

# In[2]:

print('\nGeneral Description of the Boston DataFrame:')
print(boston_input.info())


# In[3]:

# drop neighborhood from the data being considered
boston = boston_input.drop('neighborhood', 1)

print('\nGeneral description of the Boston DataFrame:')
print(boston.info())


# # Data Exploration

# In[4]:

print('First Five Rows of the Boston DataFrame:')
boston.head()


# In[5]:

print('Descriptive statistics of the Boston DataFrame:')
boston.describe()


# In[6]:

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

# In[29]:

# view the distribution of the mv variable

sns.distplot(boston["mv"])

plt.title("1970's Median Housing Price Distribution", fontsize = 16)
plt.xlabel("Price (in thousands)", fontsize = 14)
plt.ylabel("Distribution (in decimals)", fontsize = 14)
plt.show()

plt.savefig('mvdist.pdf')


# In[8]:

# view the skew of the data
print("Skewness of variable mv: {}".format(boston['mv'].skew()))


# Note: Because mv has a skewness of > 1, it is considered substantially skewed

# In[9]:

# normalize the variable mv by taking the log of its values
logmv = np.log1p(boston['mv'])    # use log1p for correct output when x is a small value

boston['logmv'] = logmv

print("Skewness of log variable mv: {}".format(logmv.skew()))


# Note: normalizing mv by performing a log transformation has reduced the skewness substantially 

# In[30]:

# view distribution of log transformed mv

sns.distplot(logmv)

plt.title("1970's Median Housing Price Distribution (Natural Log)", fontsize = 16)
plt.xlabel("Price (in thousands)", fontsize = 14)
plt.ylabel("Distribution (in decimals)", fontsize = 14)
plt.show()

plt.savefig('logmvdist.pdf')


# In[14]:

sns.pairplot(boston)

plt.show()

plt.savefig('pairplot.pdf')


# # Model Set Up

# <b>Model Set-Up - Defining Model Data</b>

# In[13]:

# set up linear, tree-based models and rmse metric
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

names = ["OLS", "Ridge Regression", "Lasso Regression", "ElasticNet", "Decision Tree", "Random Forest", 'Gradient Boosting']
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
                       selection='cyclic'),
            DecisionTreeRegressor(max_depth=10, random_state = RANDOM_SEED,
                                    max_features='log2'),
            RandomForestRegressor(max_depth=10, random_state = RANDOM_SEED,
                                    max_features='log2', bootstrap=True),
           GradientBoostingRegressor(max_depth=10, random_state = RANDOM_SEED, 
                                     max_features='log2', learning_rate=0.1)]


# In[14]:

# set up preliminary data for data for fitting the models 
# the first column is the median housing value response
# the remaining columns are the explanatory variables
prelim_model_data = np.array([boston.mv,    boston.crim,    boston.zn,    boston.indus,    boston.chas,    boston.nox,    boston.rooms,    boston.age,    boston.dis,    boston.rad,    boston.tax,    boston.ptratio,    boston.lstat]).T

X = np.array([boston.crim,    boston.zn,    boston.indus,    boston.chas,    boston.nox,    boston.rooms,    boston.age,    boston.dis,    boston.rad,    boston.tax,    boston.ptratio,    boston.lstat]).T

y = np.array([boston.mv]).T

# dimensions of the polynomial model X input and y response
# preliminary data before standardization
print('Data dimensions:', prelim_model_data.shape)


# In[15]:

# standard scores for the columns... along axis 0
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ('scaler', StandardScaler())
    ])


# In[16]:

# the model data will be standardized form of preliminary model data
model_data = pipe.fit_transform(prelim_model_data)

# dimensions of the polynomial model X input and y response
# all in standardized units of measure
print('Dimensions for model_data:', model_data.shape)


# <b>K-Fold Cross Validation & Model Set Up</b>

# In[17]:

from sklearn.model_selection import KFold

# define number of folds for cross validation - 5 folds
n_folds = 5

# set up numpy array for storing results
rmse_results = np.zeros((n_folds, len(linmodel)))
test_results = np.zeros((n_folds, len(linmodel)))

kf = KFold(n_splits = n_folds, shuffle=False, random_state = RANDOM_SEED)

#initialize fold count
fold_count = 0

for train_index, test_index in kf.split(model_data):
    # define test and train variables
    # x represents the explanatory variables
    X_train = model_data[train_index, 1:model_data.shape[1]] # 1 to last
    X_test = model_data[test_index, 1:model_data.shape[1]] # 1 to last
    
    # model_data.shape[1]-1 is the response - predictive variable
    y_train = model_data[train_index,0] # 0
    y_test = model_data[test_index,0] # 0
    
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
        mse = mean_squared_error(y_test, pred)
        rmse = sqrt(mean_squared_error(y_test, pred))
        
        rmse_results[fold_count, model_count] = rmse
        test_results[fold_count, model_count] = test_score
        
        print("\n{}".format(name))
        print("Train Set Fit: {}".format(round(train_score, 2)))
        print("Test Set Fit: {}".format(round(test_score, 2)))

        print("RMSE: {}".format(round(rmse, 2)))
        
        model_count += 1
        
    fold_count += 1
        


# # Feature Selection

# In[19]:

# train the random forest regressor with our previous training set
features = ['crim', 'zn', 'indus', 'chas', 'nox', 'rooms', 'age', 'dis', 'rad',
       'tax', 'ptratio', 'lstat']

gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)

# display features with their gini importance scores
for feature in zip(features, gbr.feature_importances_):
    print(feature)


# In[20]:

# displays the important variables
from sklearn.feature_selection import SelectFromModel

sfm = SelectFromModel(gbr)
sfm.fit(X_train, y_train)

for feature_list_index in sfm.get_support(indices=True):
    print(features[feature_list_index])


# In[22]:

# plot feature importances
importances = gbr.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importances', fontsize=16)
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Importance')
plt.show()


# # Regression Decision Tree Visual - Random Forest

# In[34]:

# visualization of the tree
estimator = rfr.estimators_[5]

# export tree visualization as a dot file
from sklearn.tree import export_graphviz

export_graphviz(estimator, out_file='rfr.dot', 
                feature_names = features,
                rounded = True, proportion = False, 
                precision = 2, filled = True)

# Convert to png
from subprocess import call
call(['dot', '-Tpng', 'rfr.dot', '-o', 'rfr.png', '-Gdpi=600'])

# Display in python (View the original png for better quality)
import matplotlib.pyplot as plt
plt.figure(figsize = (14, 18))
plt.imshow(plt.imread('rfr.png'))
plt.axis('off');
plt.show();


# # Model Results

# In[35]:

# load cross validation data into a DataFrame
rmseresults = pd.DataFrame(rmse_results)
rmseresults.columns = names

testresults = pd.DataFrame(test_results)
testresults.columns = names

# print average results for models
print("Average Regressor Performance:")
rmseresults = rmseresults.mean()
rmseresults = rmseresults.reset_index()
rmseresults.columns = ['Regressor', 'RMSE']

testresults = testresults.mean()
testresults = testresults.reset_index()
testresults = pd.DataFrame(testresults)
testresults.columns = ['Regressor', 'Score']

results = pd.merge(testresults, rmseresults, on='Regressor')
results


# In[36]:

sns.barplot(x='Regressor', y='Score', data=results)
plt.title('Regressor Average Test Score', fontsize=16)

plt.savefig('regscore.pdf')


# In[37]:

sns.barplot(x='Regressor', y='RMSE', data=results)
plt.title('Regressor Average RMSE', fontsize=16)

plt.savefig('regrmse.pdf')


# # Employment of Gradient Boosting Regression to Full Dataset

# In[74]:

gbr = GradientBoostingRegressor(max_depth=10, random_state = RANDOM_SEED)

gbr.fit(X,y)
gbrrmse = sqrt(mean_squared_error(y, gbr.predict(X)))
print("Model Score: {}".format(gbr.score(X,y)))
print("RMSE: {}".format(gbrrmse))

