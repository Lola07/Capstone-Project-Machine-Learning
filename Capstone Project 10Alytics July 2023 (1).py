#!/usr/bin/env python
# coding: utf-8

# ## Connecttel
# ### Customer churn Prediction

# #### 1.0 Library Importation

# In[5]:


# Import necessary libraries

# Data analysis libraries
import pandas as pd
import numpy as np

# data visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import missingno as msno # missing data visualization


# Classifiers libraries, preprocessing i.e machine learning algorithms
from sklearn.preprocessing import MinMaxScaler #helps to normalizes data between zero and one
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

#Evaluation Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

import warnings
warnings.filterwarnings("ignore")


# In[6]:


# Environment settings: 
pd.set_option('display.max_column', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_seq_items', None)
pd.set_option('display.max_colwidth', 500)
pd.set_option('expand_frame_repr', True)


# In[7]:


# import dataset 
 
df = pd.read_csv(r"C:\Users\lolab\Downloads\Customer-Churn - Customer-Churn.csv")


# In[8]:


#import first 5 R & c
df.head(5)


# In[9]:


df.tail(5)


# In[10]:


# check dataset information

df.info()


# In[11]:


# Descriptive Statistics of Numerical Columns
df.describe().T


# In[12]:


# Descriptive Statistics of Categorical Columns

df.describe(include=['object', 'bool'])


# In[13]:


df.shape


# In[14]:


df.columns


# In[15]:


# Check for missing values in the dataframe
missing_values_count = df.isnull().sum()
print(missing_values_count)

There are 11 null values found in the TotalCharges Column
# In[16]:


# Visualize missing values using a heatmap
plt.figure(figsize = (10,3))
sns.heatmap(df.isnull(),cbar = True, cmap = "PRGn")


# In[17]:


# display where the missing data is  present/exist in the data.

df[df.isnull().any(axis=1)]


# In[18]:


# visualize missing data.
plt.figure(figsize = (8,5))
sns.heatmap(df.isnull(), cbar=True , cmap="GnBu_r");


# In[ ]:





# In[19]:


# since our mssing values are 11 out of dataset of over 7000, I am dropping them.
#Drop missing data using inplace.
df.dropna(inplace=True)


# In[20]:


df.isnull().sum()


# In[21]:


# Now i  have no missing values, and will porceed to data cleaning


# #### 1.1 Data Cleaning and Pre-Processing
# ##### 1.2 Data Pre-Processing

# In[22]:


# Checking for duplicates
df.duplicated().sum()


# In[23]:


df.sample()


# In[ ]:





# ## Exploratory Data Analysis.
# 
# 
# ##### Univariate Analysis:  
# 
# - Univariate analysis involves examining individual variables in the dataset.I will visualize the distribution of each variable.
# 
# 
# 
# ##### Bivariate Analysis: 
# 
# - Bivariate analysis involves examining the relationship between two variables. I will visualize how the target variable ("Churn") is influenced by other variables.
# 
# 
# 
# ##### Multivariate Analysis: 
# 
# - Multivariate analysis involves exploring relationships between multiple variables simultaneously. I will  use correlation matrices and pair plots.
# 

# In[24]:


# Replace entries under 'StreamingTV', Online security, online backup, device protection, 
#techsupport and 'StreamingMovies' with 'No'

df['StreamingTV'] = df['StreamingTV'].replace('No internet service', 'No')
df['StreamingMovies'] = df['StreamingMovies'].replace('No internet service', 'No')
df['OnlineSecurity'] = df['OnlineSecurity'].replace('No internet service', 'No')
df['OnlineBackup'] = df['OnlineBackup'].replace('No internet service', 'No')
df['TechSupport'] = df['TechSupport'].replace('No internet service', 'No')


# In[ ]:





# In[25]:


df.head(5)


# In[ ]:





# In[26]:


## Visualization relationships using Plots.


# In[27]:


# Create a figure with two subplots
plt.figure(figsize=(16, 6))

# First subplot: Churn by Gender
plt.subplot(1, 2, 1)  # 1 row, 2 columns, subplot 1
sns.countplot(x='gender', hue='Churn', data=df)
plt.title("Churn by Gender")

# Second subplot: Churn by Internet Service
plt.subplot(1, 2, 2)  # 1 row, 2 columns, subplot 2
sns.countplot(x='InternetService', hue='Churn', data=df)
plt.title("Churn by Internet Service")


# Adjust layout and display
plt.tight_layout()
plt.show()


# In[ ]:





# In[28]:


# Box plot for numerical features
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title("Monthly Charges by Churn")
plt.show()

sns.boxplot(x='Churn', y='tenure', data=df)
plt.title("Tenure by Churn")
plt.show()



# In[29]:


# is the tenure in months, weeks or days or years?

There are some outliers in the Tenure by churn plot, this is because some of the values are very large compared to majority of the values. The values occur after Q3 and our churn rate by tenure is mostly in Q1, which happened between customers who have been with the company between 0 - 30 months/weeks.

# In[30]:


#Churn by seniorcitizens and non seniorcitizens
sns.countplot(x='SeniorCitizen', hue='Churn', data=df)
plt.title("Churn by SeniorCitizen")
plt.show()


# In[ ]:





# In[31]:


#churn by seniorcitizens alone

sns.boxplot(x='Churn', y='SeniorCitizen', data=df)
plt.title("SeniorCitizen")
plt.show()

We also have a high rate of churn among senior citizens, could be causeed by many factors,lack of interest as the age, death, sickness, nursing home or disability.
# In[ ]:





# In[32]:


# Scatter plot for numerical features, this has no trend line, separate the x and y against some other features,
# emphasis is on the y axis
sns.scatterplot(x='tenure', y='MonthlyCharges', hue='Churn', data=df)
plt.title("Tenure vs Monthly Charges with Churn")
plt.show()


# In[ ]:




 I am  visualizing relationships between the target variable "Churn" and some key features, such as "gender," "InternetService," "MonthlyCharges," and "tenure." For categorical features, I used bar plots to compare the proportion of churners and non-churners within each category. For numerical features, i used box plots to compare the distribution of values for churners and non-churners. Additionally, I used a scatter plot to visualize the relationship between "tenure" and "MonthlyCharges" with "Churn" as a hue.


# In[33]:


# Bar plot for "MonthlyCharges" by "PaymentMethod".
plt.figure(figsize=(10, 6))
sns.barplot(x='PaymentMethod', y='MonthlyCharges', hue='Churn', data=df)
plt.title("Monthly Charges by Payment Method with Churn")
plt.xticks(rotation=45)
plt.show()



# In[ ]:





# In[34]:


# Count plot for "Churn" by "Contract"
sns.countplot(x='Contract', hue='Churn', data=df)
plt.title("Churn by Contract Type")
plt.show()


From the above plot we can see that , the rate of churn among customers with one month contract or month to month contract is higher ocmpared to other types of contract, this could be because , a trial period was offered for sevral days and the customers opt-out after the trial period, due to some reasons that may be related to price or other factors(Pomotional offers form competiros, better cancellation conditions e.t.c).

going forward the company shold provide more of yearly or bia-annual contracts , which will have certain cancellation conditions to keep customers.
# In[ ]:





# In[35]:


# Violin plot for "tenure" by "Churn"
sns.violinplot(x='Churn', y='tenure', data=df)
plt.title("Distribution of Tenure by Churn")
plt.show()



# In[ ]:





# In[36]:


#checking for outliers


# Selecting only numerical columns for outlier detection
numerical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
numerical_data = df[numerical_columns]

# Plotting box plots to check for outliers
plt.figure(figsize=(6, 4))
sns.boxplot(data=numerical_data)
plt.title("Box Plot of Numerical Features")
plt.xticks(rotation=45)
plt.show()

 No Outliers were detected in our numerical values.
# In[37]:


# Univariate analysis for numerical features
numerical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
fig, axes = plt.subplots(nrows=1, ncols=len(numerical_columns), figsize=(15, 5))

for i, col in enumerate(numerical_columns):
    sns.histplot(df[col], kde=True, ax=axes[i])
    axes[i].set_title(f"Distribution of {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Frequency")

plt.tight_layout()
plt.show()


# In[ ]:





# In[38]:


# Univariate analysis for categorical features
categorical_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                       'InternetService', 'OnlineSecurity', 'DeviceProtection', 'TechSupport',
                       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']
for col in categorical_columns:
    plt.figure(figsize=(6, 5))
    sns.countplot(x=col, data=df, order=df[col].value_counts().index)
    plt.title(f"Count of Customers by {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()


# In[ ]:





# ## Bivariate Analysis

# In[39]:


#Plotting categories against Churn


# In[40]:


for col in categorical_columns[:-1]:
    plt.figure(figsize=(8, 5))
    sns.countplot(x=col, hue='Churn', data=df, order=df[col].value_counts().index)
    plt.title(f"Churn by {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.legend(title="Churn", loc='upper right', labels=["No", "Yes"])
    plt.show()


# In[ ]:





# ## Multivariate Analysis

# In[ ]:





# In[41]:


#Correlation matrix of numerical features , continuous numerical features might be difficult to compare
plt.figure(figsize=(8, 6))
correlation_matrix = df[numerical_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of Numerical Features")
plt.show()


# In[ ]:





# In[42]:


# Pair plot of numerical features against churn
sns.pairplot(df[numerical_columns + ['Churn']], hue='Churn', diag_kind='kde')
plt.suptitle("Pair Plot of Numerical Features with Churn as Hue")
plt.show()


# In[ ]:





# In[43]:


# Pair Plot for numerical features 
numerical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
sns.pairplot(df[numerical_columns], diag_kind='kde')
plt.suptitle('Pair Plot for Numerical Features', y=1.02)
plt.show()


# In[ ]:





# In[44]:


print(df['SeniorCitizen'].unique())


# In[45]:


df.columns


# In[ ]:





# In[46]:


#There is no need for any plots or charts for categorical values

df.head(10)


# In[ ]:





# ## MACHINE LEARNING / MODEL BUILDING
# 

# ### Data preprocessing
# 

# In[47]:


df.head()


# In[ ]:





# In[74]:


# Drop customerID as it is not relevant for the prediction.
df.drop('customerID', axis=1, inplace=True)

# Handle missing values if any
df['TotalCharges'] = df['TotalCharges'].replace(' ', 0).astype(float)




# In[ ]:


# Encode binary categorical variables using LabelEncoder (helps convert 0s and 1s). this is most situable for ordinal
#categorys as you want to preserve the order in the relationsip
binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
label_encoder = LabelEncoder()
for col in binary_cols:
    df[col] = label_encoder.fit_transform(df[col])



# In[ ]:


# One-hot encode multi-class categorical variables. each catergoty becomes a column, and changes them to 0s and 1s. This is 
#situable for nominal variables and preserves the distinction between categories without imposing an ordinal relationship
df = pd.get_dummies(df, columns=['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                                     'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                                     'Contract', 'PaymentMethod'])


# In[75]:


df.head()


# In[76]:


#Split Data into Features and Target
#Separate the data into features (X) and the target variable (y)


X = df.drop('Churn', axis=1)
y = df['Churn']


# In[ ]:





# In[77]:


# splitting the dataset (X and y) into two sets: a training set (X_train and y_train) and a testing set (X_test and y_test). 
# The training set is used to train the machine learning model, 
#while the testing set is used to evaluate the model's performance and generalization on unseen data. 
#The test_size parameter determines the proportion of data to allocate for testing, 
#and random_state ensures reproducibility of the splitting process.


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# In[78]:


# Preprocessing steps for numerical features

# The pipeline first applies the preprocessing steps (scaling) to the numerical features using ColumnTransformer, 
#and then feeds the preprocessed data into the RandomForestClassifier for training and prediction.

# Once the pipeline is created, you can train the model using model.fit(X_train, y_train) 
# and make predictions using model.predict(X_test). The pipeline ensures that all preprocessing steps 
# are consistently applied to both the training and testing datasets, preventing any data leakage and 
# making it easier to maintain the model's integrity.

numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Combine all preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

# Create the pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])


# In[79]:


# Drop samples with missing values from both X_train and y_train
X_train = X_train.dropna()
y_train = y_train[X_train.index]

# Drop samples with missing values from both X_test and y_test
X_test = X_test.dropna()
y_test = y_test[X_test.index]


# In[83]:


model.fit(X_train, y_train)


# In[107]:


#Prediction 

y_pred = model.predict(X_test)


# In[108]:


# Predicted churn value against original churn value for the data that the model has not seen before

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[109]:


# taking the predicted against the ones 

classification_rep = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_rep)



# In[110]:


cm = confusion_matrix(y_test,y_pred)
plt.figure(figsize=(5,5))
sns.heatmap(cm,annot=True,cbar=False,fmt='d',cmap='viridis')


# In[111]:


# Create the pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42))
])
model.fit(X_train, y_train)


# In[112]:


#Prediction 

y_pred = model.predict(X_test)


# In[113]:


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[96]:


# confussion matrix
from sklearn.metrics import confusion_matrix


# In[97]:


cm = confusion_matrix(y_test,y_pred)
plt.figure(figsize=(5,5))
sns.heatmap(cm,annot=True,cbar=False,fmt='d',cmap='viridis')


# In[99]:


classification_rep = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_rep)


# In[115]:


test_df = pd.DataFrame(X_test, columns=X.columns)

# Add the true labels (y_test) and the predicted labels (y_pred) to the DataFrame
test_df['True_Label'] = y_test
test_df['Predicted_Label'] = y_pred




# In[116]:


test_df.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




