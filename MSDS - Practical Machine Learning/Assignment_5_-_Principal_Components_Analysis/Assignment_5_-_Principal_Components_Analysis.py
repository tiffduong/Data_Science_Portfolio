#!/usr/bin/env python
# coding: utf-8

# # Set Up

# In[286]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# ignore warnings
from sklearn.exceptions import DataConversionWarning
import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings('ignore', category=DataConversionWarning)
warnings.simplefilter("ignore", category=PendingDeprecationWarning)
warnings.simplefilter('ignore', category=DeprecationWarning)


plt.rcParams.update({'figure.max_open_warning': 0})

# seaborn plot settings
sns.set(style="ticks", rc={'figure.figsize':(12,8)})
sns.set_palette("coolwarm")

# seed value for random number generators to obtain reproducible results
RANDOM_SEED = 1


# # Dataset Specific Set Up

# In[287]:


# original fetch mldata method is not working properly - 
# import the MNIST dataset from a github repo
from six.moves import urllib
from sklearn.datasets import fetch_mldata
from scipy.io import loadmat

mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
mnist_path = "./mnist-original.mat"
response = urllib.request.urlopen(mnist_alternative_url)
with open(mnist_path, "wb") as f:
    content = response.read()
    f.write(content)
mnist_raw = loadmat(mnist_path)
mnist = {
    "data": mnist_raw["data"].T,
    "target": mnist_raw["label"][0],
    "COL_NAMES": ["label", "data"],
    "DESCR": "mldata.org dataset: mnist-original",
}
print("Success! MNIST has been loaded.")


# In[288]:


# set up plotting a digit
def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = matplotlib.cm.binary,
               interpolation="nearest")
    plt.axis("off")


# In[289]:


# set up multi-image plot
def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = matplotlib.cm.binary, **options)
    plt.axis("off")


# # Data Exploration

# In[290]:


X, y = mnist["data"], mnist["target"]
X.shape


# In[291]:


y.shape


# In[292]:


#inspect a random digit
some_digit = X[36000]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap = matplotlib.cm.binary,
           interpolation="nearest")
plt.axis("off")

plt.savefig("some_digit_plot")
print("A Digit Plot:")
plt.show()


# In[293]:


# further inspect random images and display it into a grid to view structure
plt.figure(figsize=(9,9))
example_images = np.r_[X[:12000:600], X[13000:30600:600], X[30600:60000:590]]
plot_digits(example_images, images_per_row=10)
plt.savefig("more_digits_plot")
print("More Digits Plot:")
plt.show()


# # Model Data Preparation

# In[294]:


# utilize the first 60,000 for model development set
# utilize the final 10,000 as a holdout test set
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# normalize the features
# although not necessary for Random Forest, it is important for PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# In[295]:


# view the amount of observations and explanatory variables
X_train.shape


# In[296]:


# view the shape of the predictor holdout test set
y_test.shape


# # 1. Random Forest Classifier on All Features

# In[297]:


get_ipython().run_cell_magic('time', '', '\nimport time\n\n# create an empty list to store time results\nrf_time = []\n\n#fit a Random Forest Classifier onto the training set\nfrom sklearn.ensemble import RandomForestClassifier\n\nstart_time = time.clock()\nrf_clf = RandomForestClassifier(max_features = \'sqrt\', n_estimators=10, bootstrap=True, random_state=RANDOM_SEED)\nrf_clf.fit(X_train,y_train)\nend_time = time.clock()\n\nruntime = end_time - start_time\n\n#append the results to the list\nrf_time.append(runtime)\n\nprint("Success!")')


# In[298]:


# check runtime value
rf_time


# In[299]:


# call the predict method onto the holdout test set
y_pred = rf_clf.predict(X_test)


# In[300]:


from sklearn.metrics import classification_report

# produce a classification report
print("Random Forest Classification Report: \n{}".format(classification_report(y_test, y_pred)))


# In[301]:


# interested in just seeing performance using the F1 score
from sklearn.metrics import f1_score

scores = []

rf_f1score = f1_score(y_test, y_pred, average='weighted')
scores.append(rf_f1score)

print("F1 Score: {}".format(rf_f1score))


# # 2. Principal Components Analysis - Full Set

# In[302]:


get_ipython().run_cell_magic('time', '', '\n# create an empty list to store time results\npca_time = []\n\n# use PCA on all 70,000 observations\nfrom sklearn.decomposition import PCA\n\nstart_time = time.clock()\npca = PCA(n_components=0.95, random_state=RANDOM_SEED)\nX_pca = pca.fit_transform(X)\nend_time = time.clock()\n\nruntime = end_time - start_time\n\n#append the results to the list\npca_time.append(runtime)\n\nprint("Success!")')


# In[303]:


# check runtime value
pca_time


# In[304]:


# show the reduction in features (784 to 154)
X_pca.shape


# In[305]:


# PCA - Explained Variance Ratio Cumulative Distribution
var = pca.explained_variance_ratio_.cumsum()
plt.plot(var)
plt.title("PCA - Explained Variance Ratio Cumulative Distribution", fontsize=16)
plt.savefig("pca_explained_var_ratio.png")


# # 3. Random Forest Classifier - Identified Principal Components

# In[306]:


# set up PCA training and test sets for features
X_train_pca = X_pca[:60000]
X_test_pca = X_pca[60000:]


# In[307]:


get_ipython().run_cell_magic('time', '', '\n# create an empty list to store time results\nrf_pca_time = []\n\n#fit another Random Forest Classifier onto the the training set with the\n# identified principal components\nstart_time = time.clock()\nrf_clf.fit(X_train_pca,y_train)\nend_time = time.clock()\n\nruntime = end_time - start_time\n\n#append the results to the list\nrf_pca_time.append(runtime)\n\nprint("Success!")')


# In[308]:


# check runtime value
rf_pca_time


# In[309]:


# call the predict method onto the holdout test set
y_pred_pca = rf_clf.predict(X_test_pca)


# In[310]:


# produce a classification report
print("Random Forest Classification Report: \n{}".format(classification_report(y_test, y_pred_pca)))


# In[311]:


# interested in just seeing performance using the F1 score
from sklearn.metrics import f1_score
rf_pca_f1score = f1_score(y_test, y_pred_pca, average='weighted')
scores.append(rf_pca_f1score)

print("F1 Score: {}".format(rf_pca_f1score))


# # 4. Model Comparison

# In[312]:


# F1 Score Comparison
columns = ['RF', 'RF+PCA']
column = ['F1_Scores']
scores_comparison = pd.DataFrame(scores).T
scores_comparison.columns = columns
scores_comparison = scores_comparison.T
scores_comparison.columns = column
scores_comparison


# In[313]:


scores_comparison.plot(kind='bar')
plt.title("F1 Score Comparison", fontsize=16)
plt.savefig("f1_score_comparison.png")


# In[314]:


# Runtime Comparison
rt_column = ['RF','PCA','RF+PCA']
rftime = pd.DataFrame(rf_time, columns=['Score'])
pcatime = pd.DataFrame(pca_time, columns=['Score'])
rf_pcatime = pd.DataFrame(rf_pca_time, columns=['Score'])

runtimes = pd.concat([rftime, pcatime, rf_pcatime])
runtimes = runtimes.T
runtimes.columns = rt_column
runtimes = runtimes.T

runtimes


# In[315]:


runtimes.plot(kind='bar')
plt.title("Runtime Comparison (in seconds)", fontsize=16)
plt.savefig("Runtime_Comparison.png")


# # 5. Identify Issue and Re-run the Model

# The issue is that we used PCA on the full set vs. running PCA on the train and test sets individually

# In[316]:


get_ipython().run_cell_magic('time', '', '\n# create an empty list to store time results\npca1_time = []\n\n#use pca on only training\nfrom sklearn.decomposition import PCA\n\nstart_time = time.clock()\npca = PCA(n_components = 154,random_state=RANDOM_SEED)\nX_train_pca1 = pca.fit_transform(X_train)\nX_test_pca1 = pca.transform(X_test)\nend_time = time.clock()\n\nruntime = end_time - start_time\n\n#append the results to the list\npca1_time.append(runtime)\n\nprint("Success!")')


# In[317]:


# show the reduction in features (784 to 154)
X_train_pca1.shape


# In[318]:


# show the reduction in features (784 to 154)
X_test_pca1.shape


# In[319]:


get_ipython().run_cell_magic('time', '', '\n# create an empty list to store time results\nrf_pca1_time = []\n\n#fit another Random Forest Classifier onto the the training set with the\n# identified principal components\nstart_time = time.clock()\nrf_clf.fit(X_train_pca1,y_train)\nend_time = time.clock()\n\nruntime = end_time - start_time\n\n#append the results to the list\nrf_pca1_time.append(runtime)\n\nprint("Success!")')


# In[320]:


# call the predict method onto the holdout test set

y_pred_pca1 = rf_clf.predict(X_test_pca1)


# In[321]:


# produce a classification report
print("Random Forest Classification Report: \n{}".format(classification_report(y_test, y_pred_pca1)))


# In[325]:


# interested in just seeing performance using the F1 score
from sklearn.metrics import f1_score
rf_pca1_f1score = f1_score(y_test, y_pred_pca1, average='weighted')
scores.append(rf_pca1_f1score)

print("F1 Score: {}".format(rf_pca1_f1score))


# # 6. Final Results

# In[323]:


# F1 Score Comparison
columns = ['RF', 'RF+PCA', 'RF+PCA (Fixed)']
column = ['F1_Scores']
scores_comparison = pd.DataFrame(scores).T
scores_comparison.columns = columns
scores_comparison = scores_comparison.T
scores_comparison.columns = column
scores_comparison.plot(kind='bar')
plt.title("F1 Score Comparison (Fixed)", fontsize=16)
plt.savefig("f1_score_comparison_fixed.png")


# In[324]:


# Runtime Comparison
rt_column = ['RF','RF+PCA', 'RF+PCA (Fixed)']
rftime = pd.DataFrame(rf_time, columns=['Score'])
rf_pcatime = pd.DataFrame(rf_pca_time, columns=['Score'])
rf_pcatime1 = pd.DataFrame(rf_pca1_time, columns=['Score'])

runtimes = pd.concat([rftime, rf_pcatime, rf_pcatime1])
runtimes = runtimes.T
runtimes.columns = rt_column
runtimes = runtimes.T
runtimes.plot(kind='bar')
plt.title("Runtime Comparison (in seconds) - Fixed", fontsize=16)
plt.savefig("Runtime_Comparison_fixed.png")


# In[ ]:




