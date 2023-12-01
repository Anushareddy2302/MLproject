#!/usr/bin/env python
# coding: utf-8

# # aes228
# 
# Use the "Run" button to execute the code.

# In[ ]:


#loading important library
get_ipython().system('pip install umap-learn')
#importing of the libray which need like pd, np matplotblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier 


# In[ ]:


import pandas as pd
df = pd.read_csv("shopping_trends.csv")
df.head()


# # Exploration of Dataset 

# 

# In[ ]:


#checking the null value.
null_values = df.isnull().sum()
null_values


# In[ ]:


#checking the spread of the data.
plt.figure(figsize=(10, 6))
sns.boxplot(data=df)
plt.title("Box Plot of Data Spread")
plt.show()


# In[ ]:


# exploring statics of numerical data.
print(df.describe())


# In[ ]:


#seeing the corelation between the data.
correlation_matrix = df.corr()
correlation_matrix


# In[ ]:


#plotting the correlation between the data.
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()


# In[ ]:


# Check the independence assumptions between features using pair plots
sns.pairplot(df, diag_kind='kde')
plt.show()


# In[ ]:


# PCA dimension reaction
numerical_features = df.select_dtypes(include=[np.number])
pca = PCA(n_components=2)
pca_result = pca.fit_transform(numerical_features)
df['pca_1'] = pca_result[:, 0]
df['pca_2'] = pca_result[:, 1]
# Plot PCA results
plt.figure(figsize=(10, 8))
sns.scatterplot(x='pca_1', y='pca_2', data=df)
plt.title("PCA")
plt.show()


# In[ ]:


# Dimensionality Reduction: UMAP
umap_model = UMAP(n_components=2)
umap_result = umap_model.fit_transform(numerical_features)
df['umap_1'] = umap_result[:, 0]
df['umap_2'] = umap_result[:, 1]

# Plot UMAP results
plt.figure(figsize=(10, 8))
sns.scatterplot(x='umap_1', y='umap_2', data=df)
plt.title("UMAP")
plt.show()


# In[ ]:


# Dimensionality Reduction: t-SNE
tsne_model = TSNE(n_components=2)
tsne_result = tsne_model.fit_transform(numerical_features)
df['tsne_1'] = tsne_result[:, 0]
df['tsne_2'] = tsne_result[:, 1]

# Plot t-SNE results
plt.figure(figsize=(10, 8))
sns.scatterplot(x='tsne_1', y='tsne_2', data=df)
plt.title("t-SNE Visualization")
plt.show()


# In[ ]:


X = df.drop('Subscription Status', axis=1)  # Features
y = df['Subscription Status']  # Target variable


# In[ ]:


X_encoded = pd.get_dummies(X)
X = X.fillna(0)


# In[ ]:



model = RandomForestClassifier(random_state=42)

k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

cross_val_results = cross_val_score(model, X_encoded, y, cv=k_fold, scoring='accuracy')

# shwoing the cross validation
print("Cross-Validation Results:")
print("Accuracy Mean:", cross_val_results.mean())
print("Accuracy Standard Deviation:", cross_val_results.std())


# Train a simple model first. Use validation set for hyperparameter tuning and/or early stopping. Analyze its performance using cross-validation. Identify potential pitfalls. what to do in this

# In[ ]:


#importing of the library like train_test_sllit
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# In[ ]:



X = df.drop('Subscription Status', axis=1)  # Features
y = df['Subscription Status']  # Target variable

# categorical columns
categorical_columns = ['Gender', 'Item Purchased', 'Category', 'Location', 'Size', 'Color', 'Season',
                        'Payment Method', 'Shipping Type', 'Discount Applied', 'Promo Code Used',
                        'Preferred Payment Method', 'Frequency of Purchases']

# numerical columns in data
numerical_columns = ['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases']


# In[ ]:


# transformation
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_columns),  
        ('cat', OneHotEncoder(), categorical_columns)  # one-hot encode 
    ])

# creaton of model
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])


# In[ ]:


# doing trian-test split.
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# In[ ]:



# on traing 
model.fit(X_train, y_train)


# In[ ]:


#validating the value
y_valid_pred = model.predict(X_valid)


# In[ ]:


#Analysis of the model.
accuracy = accuracy_score(y_valid, y_valid_pred)
conf_matrix = confusion_matrix(y_valid, y_valid_pred)
classification_rep = classification_report(y_valid, y_valid_pred)


# In[ ]:


#showing the preformance
print("Validation Set Performance:")
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_rep)


# In[ ]:


#performing the cross validation
k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
cross_val_results = cross_val_score(model, X, y, cv=k_fold, scoring='accuracy')


# In[ ]:


#Displaying the result.
print("Cross-Validation Results:")
print("Accuracy Mean:", cross_val_results.mean())
print("Accuracy Standard Deviation:", cross_val_results.std())


# # Experiment 1: Feature Selection

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[ ]:


X = df.drop('Subscription Status', axis=1)  # Features
y = df['Subscription Status']  # Target variable


# In[ ]:


categorical_columns = ['Gender', 'Item Purchased', 'Category', 'Location', 'Size', 'Color', 'Season',
                        'Payment Method', 'Shipping Type', 'Discount Applied', 'Promo Code Used',
                        'Preferred Payment Method', 'Frequency of Purchases']

# One-hot encode
X_encoded = pd.get_dummies(X, columns=categorical_columns)

# Create a RandomForestClassifier
rf_classifier = RandomForestClassifier(random_state=42)


# # RFE

# In[ ]:


# Create an RFE selector
rfe_selector = RFE(estimator=rf_classifier, n_features_to_select=5, step=1)


rfe_selector.fit(X_encoded, y)

#getting the selected features.
selected_features = X_encoded.columns[rfe_selector.support_]
selected_features


# In[ ]:



X_encoded = pd.get_dummies(X, columns=['Discount Applied', 'Promo Code Used'])


selected_features = ['Customer ID', 'pca_1', 'umap_1', 'Discount Applied_No', 'Promo Code Used_Yes']

X_selected = X_encoded[selected_features]


# In[ ]:



#Train Model on Selected Features:

# Split the data 
X_train, X_valid, y_train, y_valid = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Create and train a RandomForestClassifier 
model_selected_features = RandomForestClassifier(random_state=42)
model_selected_features.fit(X_train, y_train)

# Validate the model on the validation set
y_valid_pred = model_selected_features.predict(X_valid)

# Evaluate performance
accuracy_selected_features = accuracy_score(y_valid, y_valid_pred)
conf_matrix_selected_features = confusion_matrix(y_valid, y_valid_pred)
classification_rep_selected_features = classification_report(y_valid, y_valid_pred)

print("Validation Set Performance with Selected Features:")
print("Accuracy:", accuracy_selected_features)
print("Confusion Matrix:\n", conf_matrix_selected_features)
print("Classification Report:\n", classification_rep_selected_features)


# # Experiment 2: Hyperparameter Tuning

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:



X = df.drop('Subscription Status', axis=1)  # Features
y = df['Subscription Status']  # Target variable

# List of categorical columns
categorical_columns = ['Gender', 'Item Purchased', 'Category', 'Location', 'Size', 'Color', 'Season',
                        'Payment Method', 'Shipping Type', 'Discount Applied', 'Promo Code Used',
                        'Preferred Payment Method', 'Frequency of Purchases']

# One-hot encode categorical data
X_encoded = pd.get_dummies(X, columns=categorical_columns)


# In[ ]:


# instalizing the grid parameter 
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create a RandomForestClassifier
rf_classifier = RandomForestClassifier(random_state=42)


# In[ ]:


# Create a search grid
grid_search = GridSearchCV(rf_classifier, param_grid, cv=5, scoring='accuracy')

#one hot encoding.
grid_search.fit(X_encoded, y)

# hyperparameters
best_params = grid_search.best_params_

# hyperparameters
print("Best Hyperparameters:", best_params)


# In[ ]:


# Create RandomForestClassifier with the best hyperparameters
model_best_params = RandomForestClassifier(random_state=42, **best_params)
model_best_params.fit(X_train, y_train)

# Validate the model
y_valid_pred_best_params = model_best_params.predict(X_valid)

# Evaluate the performance
accuracy_best_params = accuracy_score(y_valid, y_valid_pred_best_params)
conf_matrix_best_params = confusion_matrix(y_valid, y_valid_pred_best_params)
classification_rep_best_params = classification_report(y_valid, y_valid_pred_best_params)

print("Validation Set Performance with Best Hyperparameters:")
print("Accuracy:", accuracy_best_params)
print("Confusion Matrix:\n", conf_matrix_best_params)
print("Classification Report:\n", classification_rep_best_params)


# In[ ]:




