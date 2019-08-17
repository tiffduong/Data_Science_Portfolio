# Practical Machine Learning (MSDS 422)
>The course introduces machine learning with business applications. It provides a survey of machine learning techniques, including traditional statistical methods, resampling techniques, model selection and regularization, tree-based methods, principal components analysis, cluster analysis, artificial neural networks, and deep learning. Students implement machine learning models with open-source software for data science. They explore data and learn from data, finding underlying patterns useful for data reduction, feature analysis, prediction, and classification.
<br>

## Assignment 1: MSPA Software Survey Analysis
### Management Questions
Imagine that you are an academic administrator responsible for defining the future direction of the graduate program. <br>
The MSPA Survey has been designed with these objectives in mind:<br>
>What are current students' software preferences? <br>
What are students' interests in potential new courses? <br>
Guide software and systems planning for current and future courses. <br>
Guide data science curriculum planning.

## Assignment 2: Evaluating Classification Models
### Management Questions
Imagine that you are advising the bank about machine learning methods to guide telephone marketing campaigns. Which of the two modeling methods would you recommend and why? And, given the results of your research, which group of banking clients appears to be the best target for direct marketing efforts (similar to those used with previous telephone campaigns)? <br>
> Employ two classification models (Logistic Regression and Naive Bayes).<br>
> Evaluate these models with a cross validation design.<br>
> Use the area under the ROC curve as an index of classification performance.<br>

## Assignment 3: Evaluating Regression Models
### Management Questions
Imagine that you are advising a real estate brokerage firm in its attempt to employ machine learning methods. The firm wants to use machine learning to complement conventional methods for assessing the market value of residential real estate. Of the modeling methods examined in your study, which would you recommend to management, and why? <br>
> Employ at least two regression modeling methods (Linear, Ridge, Lasso, ElasticNet). <br>
> Evaluate these models with a cross validation design.<br>
> Use the root mean-squared error (RMSE) as an index of classification performance.<br>

## Assignment 4: Random Forests and Gradient Boosting
### Management Questions
Imagine that you again are advising a real estate brokerage firm in its attempt to employ machine learning methods. The firm wants to use machine learning to complement conventional methods for assessing the market value of residential real estate. Of the modeling methods examined in your study, which would you recommend to management and why? Reviewing the results of the random forests and gradient boosting model you have selected to present to management, which explanatory variables are most important in predicting home prices? <br>
> Employ at least two regression modeling methods (Linear, Ridge, Lasso, ElasticNet). <br>
> Employ Random Forests. <br>
> Evaluate these models with a cross validation design.<br>
> Use the root mean-squared error (RMSE) as an index of classification performance.<br>
> Use Feature Selection to find the importances of the features in the dataset.<br>

## Assignment 5: Principal Components Analysis
### Management Questions
From a management perspective, the predictive accuracy of models must be weighed against the costs of model development and implementation. Suppose you were the manager of a data science team responsible for implementing models for computer vision (classification of images analogous to the MINST problem). Would you recommend using PCA as a preliminary to machine learning classification? Explain your thinking. <br>
> Employ a random forest classification model using the full set of features and use the F1 score and runtime to execute the program as a benchmark for performance.<br>
> Apply Principal Components Analysis to the full set of data to reduce the dimensionality of the dataset, while still retaining 95% of the variability explained by the explanatory variables.<br>
> Employ a random forest classification model using the PCA transformed data set and compare F1 score and runtime.<br>
> Identify the experiemental issue and rerun the experiment in a way that is consistent with a training-and-test regime.<br>

## Assignment 6: Neural Networks
### Management Questions
Suppose you are a financial institution evaluating machine learning technologies for optical character recognition. Initial testing is on the MNIST digits. What can you conclude from your benchmark study?<br>
> Explore tested neural network structures within a benchmark experiment, a factorial design with at least two levels on each of two experimental factors.<br>
> Utilize TensorFlow and ANNs.<br>
> Compare performance based on model accuracy and model runtime.<br>

## Assignment 7: Convolutional Neural Networks
### Management Questions
Assume that we are providing advice to a website provider who is looking for tools to automatically label images provided by end users. As we look across the factors in the study, making recommendations to management about image classification, we are most concerned about achieving the highest possible accuracy in image classification. That is, we should be willing to sacrifice training time for model accuracy. What type of machine learning model works best? If it is a convolutional neural network, what type of network should we use? Part of this recommendation may concern information about the initial images themselves (input data for the classification task). What types of images work best?<br>
> Explore tested neural network structures within a benchmark experiment, a factorial design with at least two levels on each of two experimental factors.<br>
> Utilize TensorFlow and CNNs.<br>
> Compare performance based on model accuracy and model runtime.<br>

## Assignment 8: Language Modeling with RNNs
### Management Questions
Suppose management is thinking about using a language model to classify written customer reviews and call and complaint logs. If the most critical customer messages can be identified, then customer support personnel can be assigned to contact those customers. How would you advise senior management? <br>
> Use pretrained word vectors from GloVe embeddings.<br>
> Utilize TensorFlow and RNNs.<br>
> Compare performance based on model accuracy.<br>
