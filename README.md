# Tiffany Duong's Data Science Portfolio
This portfolio is a compliation of notebooks that I have created over the course of my studies for my Masters in Data Science at Northwestern University as well as personal/side projects<br>

## Data Cleansing and Preparation
### BestDeal Transactional Data
[Github](https://github.com/tiffduong/Data_Science_Portfolio/blob/master/MSDS%20-%20Database%20Systems%20%26%20Preparation/Assignment_2_-_Data_Preparation_and_Cleansing.ipynb "Github") | [Nbviewer](https://nbviewer.jupyter.org/github/tiffduong/Data_Science_Portfolio/blob/master/MSDS%20-%20Database%20Systems%20%26%20Preparation/Assignment_2_-_Data_Preparation_and_Cleansing.ipynb "Nbviewer")

An exercise of cleaning transactional data, loading it onto a sqlite engine and executing SQL queries.


## Data Exploration and Analysis
### MSPA Software Survey 
[Github](https://github.com/tiffduong/Data_Science_Portfolio/blob/master/MSDS%20-%20Practical%20Machine%20Learning/Assignment_1_-_MSPA_Software_Survey_Analysis/Assignment_1.ipynb "Github") | [Nbviewer](https://nbviewer.jupyter.org/github/tiffduong/Data_Science_Portfolio/blob/master/MSDS%20-%20Practical%20Machine%20Learning/Assignment_1_-_MSPA_Software_Survey_Analysis/Assignment_1.ipynb "Nbviewer")

I explored data results from a survey given to the MSPA program back in 2016. I analyzed current students' course interests and programming language/software usage to gauge and assess future curriculum planning for the MSDS program.

## NoSQL Databases
### Chicago Food Inspection
[Github](https://github.com/tiffduong/Data_Science_Portfolio/blob/master/MSDS%20-%20Database%20Systems%20%26%20Preparation/Assignment_1_-_Querying_Data_Stored_on_a_NoSQL_Database.ipynb "Github") | [Nbviewer](https://nbviewer.jupyter.org/github/tiffduong/Data_Science_Portfolio/blob/master/MSDS%20-%20Database%20Systems%20%26%20Preparation/Assignment_1_-_Querying_Data_Stored_on_a_NoSQL_Database.ipynb "Nbviewer")

I executed and experimented with queries to pull data of varying degrees of precision/relevance from a NoSQL document-oriented database engine ElasticSearch to assess the current state of failed sanitary inspections in buildings categorized as "children's facilities", such as daycares in the city of Chicago.

## Supervised Learning
### Classification Models
<b> Evaluation of Logistic Regression and Naive Bayes </b> 

[Github](https://github.com/tiffduong/Data_Science_Portfolio/blob/master/MSDS%20-%20Practical%20Machine%20Learning/Assignment_2_-_Evaluating_Classification_Models/Assignment%202%20-%20Evaluating%20Classification%20Models.ipynb "Github") | [Nbviewer](https://nbviewer.jupyter.org/github/tiffduong/Data_Science_Portfolio/blob/master/MSDS%20-%20Practical%20Machine%20Learning/Assignment_2_-_Evaluating_Classification_Models/Assignment%202%20-%20Evaluating%20Classification%20Models.ipynb "Nbviewer")

I used three features (loan, housing, and default) to predict the response of whether or not the bank's client will subscribe to a term deposit. I employed two classification models (Logistic Regression and Naive Bayes) and evaluated them using k-fold cross validation, as well as using the area under the ROC curve as an index of model performance.

### Linear Models
<b> Evaluation of OLS, Ridge Regression, Lasso Regression, ElasticNet</b> 

[Github](https://github.com/tiffduong/Data_Science_Portfolio/blob/master/MSDS%20-%20Practical%20Machine%20Learning/Assignment_3_-_Evaluating_Regression_Models/Assignment%203%20-%20Evaluating%20Regression%20Models.ipynb "Github") | [Nbviewer](https://nbviewer.jupyter.org/github/tiffduong/Data_Science_Portfolio/blob/master/MSDS%20-%20Practical%20Machine%20Learning/Assignment_3_-_Evaluating_Regression_Models/Assignment%203%20-%20Evaluating%20Regression%20Models.ipynb "Nbviewer")

I used twelve features to predict the median value (in thousands of dollars) for housing in the Boston metropolitan area during the 1970's from a dataset of 500+ observations. I employed four linear models (OLS and three regularized linear models - Ridge, Lasso, ElasticNet) and evaluated them using k-fold cross validation, as well as using RMSE as an index of model performance.

### Tree-Based Models and Feature Selection
<b> Evaluation of Decision Trees, Random Forests, Gradient Boosting/b> 

[Github]("Github") | [Nbviewer]("Nbviewer")

I used twelve features to predict the median value (in thousands of dollars) for housing in the Boston metropolitan area during the 1970's from a dataset of 500+ observations. I employed three tree-based models (Decision Trees, Random Forests, Gradient Boosting (learning rate =0.1)), as well as using RMSE as an index of model performance. Then, I used a Gradient Boosting model to determine the feature importance of all of the features when predicting the target variable (median value).

