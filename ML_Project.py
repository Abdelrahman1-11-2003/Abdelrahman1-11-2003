# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 23:39:50 2024

@author: Designer
"""

import pandas as pd
data1 = pd.read_excel('Dataset#1.xlsx')

print(data1.isna().sum(),'\n')
print(data1.shape,'\n')
print(data1.duplicated().sum(),'\n')

data1 = data1.drop_duplicates(keep='first')

print(data1.shape,'\n')

print(data1.duplicated().sum(),'\n')


columns = data1.columns.tolist()
features = columns
features.pop()

# Function to remove outliers for multiple numerical columns
def remove_outliers(df, cols):
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        df = df[~((df[col] < lower_bound) | (df[col] > upper_bound))]
        print(f"Number of outliers removed in '{col}': {len(outliers)}")
    return df

preprocess = remove_outliers(data1,features)
print(preprocess.isna().sum() , '\n',preprocess.shape)



from sklearn.preprocessing import LabelEncoder

y = preprocess['Class']
features = preprocess.drop(columns = ['Class'])
# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Transform the categorical target variable to numerical labels
y = label_encoder.fit_transform(y)


from sklearn.preprocessing import StandardScaler as ss

sc = ss()
features = sc.fit_transform(features)

from sklearn.model_selection import train_test_split

# Split the dataset into training & testing datasets
X_train, X_test, y_train, y_test = train_test_split(features, y, \
test_size = 0.2, random_state = 0)
    
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Naive Bayes': GaussianNB(),
    'Quantile Regression': sm.QuantReg(y_train, sm.add_constant(X_train)),
}

# Train and evaluate each model
for name, model in models.items():
    if name == 'Quantile Regression':
        result = model.fit(q=0.5)  # Fit Quantile Regression at median (q=0.5)
        y_pred = result.predict(sm.add_constant(X_test))
    elif name == 'Bayesian Linear Regression':
        y_pred = model.predict(sm.add_constant(X_test))
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print(f"{name} MSE: {mse}")




import matplotlib.pyplot as plt

# Choose the feature for the x-axis
feature_index = 3
x_feature = features[:, feature_index]

# Plotting
plt.scatter(x_feature, y, alpha=0.5)
plt.title(f'Scatter Plot of Feature {feature_index+1} against Y')
plt.xlabel(f'Feature {feature_index+1}')
plt.ylabel('Y')

# Show the plot
# plt.show()







