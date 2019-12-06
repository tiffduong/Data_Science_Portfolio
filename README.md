# Tiffany Duong's Data Science Portfolio
This portfolio is a compilation of notebooks that I have created over the course of my studies for my Masters in Data Science at Northwestern University as well as personal/side projects<br>

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
<b> Evaluation of Decision Trees, Random Forests, Gradient Boosting</b> 

[Github](https://github.com/tiffduong/Data_Science_Portfolio/blob/master/MSDS%20-%20Practical%20Machine%20Learning/Assignment_4_-_Random_Forests_and_Gradient_Boosting/Assignment_4_-_Random_Forests_and_Gradient_Boosting.ipynb "Github") | [Nbviewer](https://nbviewer.jupyter.org/github/tiffduong/Data_Science_Portfolio/blob/master/MSDS%20-%20Practical%20Machine%20Learning/Assignment_4_-_Random_Forests_and_Gradient_Boosting/Assignment_4_-_Random_Forests_and_Gradient_Boosting.ipynb "Nbviewer")

I used twelve features to predict the median value (in thousands of dollars) for housing in the Boston metropolitan area during the 1970's from a dataset of 500+ observations. I employed three tree-based models (Decision Trees, Random Forests, Gradient Boosting (learning rate =0.1)), as well as using RMSE as an index of model performance. Then, I used a Gradient Boosting model to determine the feature importance of all of the features when predicting the target variable (median value).

## Multi-Class Classifiers
### Principal Component Analysis + Random Forest Classifier
[Github](https://github.com/tiffduong/Data_Science_Portfolio/blob/master/MSDS%20-%20Practical%20Machine%20Learning/Assignment_5_-_Principal_Components_Analysis/Assignment_5_-_Principal_Components_Analysis.ipynb "Github") | [Nbviewer](https://nbviewer.jupyter.org/github/tiffduong/Data_Science_Portfolio/blob/master/MSDS%20-%20Practical%20Machine%20Learning/Assignment_5_-_Principal_Components_Analysis/Assignment_5_-_Principal_Components_Analysis.ipynb "Nbviewer")

I used the MNIST dataset and employed two models: a random forest classifier model as a benchmark for model performance and another random forest classifier model that had principal component analysis (PCA) applied to it as a dimensional-reduction method, while preserving 95% variance explained by the feature. Within the second model, I initially applied the fit_transform method to the entire data set (purposeful issue), and then I applied a fit_transform method to the training set and only the transform method to the test set separately and compared performance (using the F1 score, which is the harmonic mean between accuracy and precision, and program runtime) between the wrongly transformed and rightfully transformed data, as well as to the benchmark model.

## Deep Learning
### Multilayer Perceptron (MLP) - Finding Natural Feature Sets in the Hidden Layer
[Github](https://github.com/tiffduong/Data_Science_Portfolio/blob/master/MSDS%20-%20Artificial%20Intelligence%20and%20Deep%20Learning/Assignment_2_-_Finding_Natural_Feature_Sets/Assignment2_Base_Model.ipynb "Github") | [Nbviewer](https://nbviewer.jupyter.org/github/tiffduong/Data_Science_Portfolio/blob/master/MSDS%20-%20Artificial%20Intelligence%20and%20Deep%20Learning/Assignment_2_-_Finding_Natural_Feature_Sets/Assignment2_Base_Model.ipynb "Nbviewer")

I investigated how adjustments made to the number of hidden layers and feature classes of a MLP can contribute to how feature classes are naturally found in the hidden layers based on a 9x9 input grid of alphabet data (81 input nodes, 9 output nodes) and how the pre-determined input classes are then classified. The MLP utilizes backpropagation that was defined in NumPy.

### Multilayer Perceptron (MLP) - MNIST Digits Classification
[Github](https://github.com/tiffduong/Data_Science_Portfolio/blob/master/MSDS%20-%20Practical%20Machine%20Learning/Assignment_6_-_Neural_Networks/Assignment%206%20-%20Neural%20Networks.ipynb "Github") | [Nbviewer](https://nbviewer.jupyter.org/github/tiffduong/Data_Science_Portfolio/blob/master/MSDS%20-%20Practical%20Machine%20Learning/Assignment_6_-_Neural_Networks/Assignment%206%20-%20Neural%20Networks.ipynb "Nbviewer")

I used the MNIST dataset and conducted a 2x2 factorial experiment by adjusting the number of hidden layers and the number of nodes per hidden layer within a multilayer perceptron architecture to classify the MNIST digits images into its proper class. This experiment was done within Tensorflow, which is an open-source ML library, developed by Google. I made use of the Adam Optimization algorithm to improve the model. Performance between the models was compared using the model's process execution time and accuracy.

### Convolutional Neural Networks (CNNs) - Cats and Dogs Binary Classification
[Github](https://github.com/tiffduong/Data_Science_Portfolio/blob/master/MSDS%20-%20Practical%20Machine%20Learning/Assignment_7_-_Convolutional_Neural_Networks/Assignment%207%20-%20Convolutional%20Neural%20Networks.ipynb "Github") | [Nbviewer](https://nbviewer.jupyter.org/github/tiffduong/Data_Science_Portfolio/blob/master/MSDS%20-%20Practical%20Machine%20Learning/Assignment_7_-_Convolutional_Neural_Networks/Assignment%207%20-%20Convolutional%20Neural%20Networks.ipynb "Nbviewer")

I used a subset of the [Kaggle's cats vs. dogs dataset](https://www.kaggle.com/c/dogs-vs-cats "Kaggle's cats vs. dogs dataset") and conducted a 2x2 factorial experiment (MLP vs CNN, grayscale vs. RGB images) to see which type of model design has a higher classification accuracy on a determining if the subject of a given 64x64 image is a cat or dog. This experiment was done within Tensorflow, which is an open-source ML library, developed by Google. I made use of the Adam Optimization algorithm to improve the model. Performance between the models was compared using the model's process execution time and accuracy.

### Convolutional Neural Networks (CNNs) - Distracted Driver Detection (Transfer Learning)
[Github](https://github.com/tiffduong/Data_Science_Portfolio/blob/master/MSDS%20-%20Artificial%20Intelligence%20and%20Deep%20Learning/Final_Project_-_Distracted_Driver_Detection/Distracted_Drivers.ipynb "Github") | [Nbviewer](https://nbviewer.jupyter.org/github/tiffduong/Data_Science_Portfolio/blob/master/MSDS%20-%20Artificial%20Intelligence%20and%20Deep%20Learning/Final_Project_-_Distracted_Driver_Detection/Distracted_Drivers.ipynb "Nbviewer")

I used a image data from the [State Farm's Distracted Driver Detection Competition](https://www.kaggle.com/c/state-farm-distracted-driver-detection/overview) to build a LeNet-5 Model from scratch to classify the images into 10 specified "driving behavior" classes. I also utilized transfer learning and built on top of a pre-trained VGG-16 model (all layers except for the last set of conv/max pooling and fully connected layers are frozen) to classify the images and measure classification accuracy. The models were compiled using Adam Optimization, categorical cross-entropy for loss and trained for 20 epochs with a batch size of 64, and included early stopping. This project was done using Keras, with Tensorflow on the back-end.

### Recurrent Neural Networks (RNNs) - Language Modeling (Movie Review Sentiment)
[Github](https://github.com/tiffduong/Data_Science_Portfolio/blob/master/MSDS%20-%20Practical%20Machine%20Learning/Assignment_8_-_Language_Modeling_with_RNNs/Assignment%208%20-%20Language%20Modeling%20with%20an%20RNN.ipynb "Github") | [Nbviewer](https://nbviewer.jupyter.org/github/tiffduong/Data_Science_Portfolio/blob/master/MSDS%20-%20Practical%20Machine%20Learning/Assignment_8_-_Language_Modeling_with_RNNs/Assignment%208%20-%20Language%20Modeling%20with%20an%20RNN.ipynb "Nbviewer")

I used pretrained word vectors from GloVe embeddings (obtained with Python package chakin) and conducted a 2x2 factorial experiment (different pretrained vectors, small vs. large vocabulary size) to see which type of model design has a higher classification accuracy when determining if the sentiment of a movie review was positive or negative. This experiment was done within Tensorflow, which is an open-source ML library, developed by Google. I made use of the Adam Optimization algorithm to improve the RNN models (50 epochs and batch sizes of 100). Performance between the models was compared using the model's train and test accuracies.
