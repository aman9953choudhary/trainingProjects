#!/usr/bin/env python
# coding: utf-8

# In[47]:


# Import librarirs
import pandas as pd
import numpy as np
import seaborn as sns


# In[48]:


# load the file
rawDf=pd.read_csv("C:\\Users\\78359\\OneDrive\\Desktop\\PropertyPrice_Data.csv")


# In[49]:


rawDf.head(5) # overveiw of the data with first 5 rows


# In[50]:


rawDf.tail # (5) overveiw of the data with last 5 rows


# In[51]:


# shape of the file
print("Rows in data:",rawDf.shape[0]) # rows -1459 & cols-26
print("Columns in the data:",rawDf.shape[1])


# In[52]:


rawDf.info() # 20 continuous columns & 6 categorical columns


# In[53]:


rawDf.describe().T # here we can see the stastical summary of the data


# In[54]:


# Columns info
rawDf.columns


# In[55]:


# Missing values
rawDf.isna().sum()
# We have missing values in Garage & Garage Built Year column


# In[56]:


# Drop ID column
rawDf.drop('Id',axis=1,inplace=True)


# In[57]:


# Univariate Analysis

# Missing values Imputation
# Garage
rawDf['Garage'].dtype # Categorical column
tempMode=rawDf['Garage'].mode()[0]
rawDf['Garage'].fillna(tempMode,inplace=True)


# In[58]:


# Ensuring the missing value are fill
rawDf["Garage"].isna().sum()


# In[59]:


# Garage_Built_Year
rawDf['Garage_Built_Year'].dtype # Continuous Column
tempMedian=rawDf['Garage_Built_Year'].median()
rawDf['Garage_Built_Year'].fillna(tempMedian,inplace=True)


# In[60]:


# Ensuring the missing values are filled
rawDf["Garage_Built_Year"].isna().sum()


# In[61]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
# Bivaraite Analysis
# Continuous Var 
# correlation
corrDf=rawDf.corr()
corrDf

sns.heatmap(corrDf,cmap='rainbow_r',annot=True,
            xticklabels=rawDf.columns,
            yticklabels=rawDf.columns)
plt.show()


# In[62]:


# Categorical Columns-Using Boxplot
categr_var=rawDf.columns[rawDf.dtypes=='object']
categr_var
sns.boxplot(y=rawDf['Sale_Price'],x=rawDf['Road_Type'])


# In[63]:


road_type=rawDf["Road_Type"].value_counts()
road_type


# In[64]:


plt.figure(figsize=(4,4))
plt.pie(road_type,labels=road_type.index,autopct="%1.1f%%",explode=[0.4,0.4])
plt.show()
# Here can see with the help of pie chart mostly data of Road type is Paved only 0.4% are is Gravel


# In[65]:


#for i in categr_var:
 #   plt()
  #  sns.boxplot(y=rawDf['Sale_Price'],x=rawDf[i])
    


# In[66]:


# Dummy variable creation
fullDf=pd.get_dummies(rawDf,drop_first=True)

fullDf.shape # (1459,37)


# In[67]:


#constant
from statsmodels.api import add_constant     
fullDf1=add_constant(fullDf)


# In[68]:


# Splitting the data sets into train & test
from sklearn.model_selection import train_test_split
trainDf,testDf=train_test_split(fullDf1,train_size=0.7,random_state=2410)

trainDf.shape # (1021, 38)
testDf.shape # (438, 38)


# In[69]:


# Converting into dep. & Indep. var

trainX=trainDf.drop(['Sale_Price'],axis=1).copy()
trainY=trainDf['Sale_Price']

testX=testDf.drop(['Sale_Price'],axis=1).copy()
testY=testDf['Sale_Price']

print(trainX.shape) # (1021, 37)
print(trainY.shape) # (1021)
print(testX.shape) # (438, 37)
print(testY.shape) # (438)


# In[70]:


#########################
# VIF check
#########################

from statsmodels.stats.outliers_influence import variance_inflation_factor

tempMaxVIF = 5 # The VIF that will be calculated at EVERY iteration in while loop
maxVIFCutoff = 5 # 5 is recommended cutoff value for linear regression
trainXCopy = trainX.copy()
counter = 1
highVIFColumnNames = []

while (tempMaxVIF >= maxVIFCutoff):
    
    #print(counter)
    
    # Create an empty temporary df to store VIF values
    tempVIFDf = pd.DataFrame()
    
    # Calculate VIF using list comprehension
    tempVIFDf['VIF'] = [variance_inflation_factor(trainXCopy.values, i) for i in range(trainXCopy.shape[1])]
        
        
   # Create a new column "Column_Name" to store the col names against the VIF values from list comprehension
    tempVIFDf['Column_Name'] = trainXCopy.columns
    
    # Drop NA rows from the df - If there is some calculation error resulting in NAs
    tempVIFDf.dropna(inplace=True)
    
    # Sort the df based on VIF values, then pick the top most column name (which has the highest VIF)
    tempColumnName = tempVIFDf.sort_values(["VIF"], ascending = False).iloc[0,1]
    # tempColumnName = tempVIFDf.sort_values(["VIF"], ascending = True)[-1:]["Column_Name"].values[0]
    
    # Store the max VIF value in tempMaxVIF
    tempMaxVIF = tempVIFDf.sort_values(["VIF"], ascending = False).iloc[0,0]
    # tempMaxVIF = tempVIFDf.sort_values(["VIF"])[-1:]["VIF"].values[0]
    
    print(tempColumnName)
    
    if (tempMaxVIF >= maxVIFCutoff): # This condition will ensure that columns having VIF lower than 5 are NOT dropped
        
        # Remove the highest VIF valued "Column" from trainXCopy. As the loop continues this step will keep removing highest VIF columns one by one 
        trainXCopy = trainXCopy.drop(tempColumnName, axis = 1)    
        highVIFColumnNames.append(tempColumnName)
    
    counter = counter + 1

highVIFColumnNames


# In[ ]:





# In[71]:


highVIFColumnNames.remove('const') # We need to exclude 'const' column from getting dropped/ removed. This is intercept.
highVIFColumnNames


# In[72]:


trainX.drop(highVIFColumnNames,axis=1,inplace=True)
testX.drop(highVIFColumnNames,axis=1,inplace=True)

print(trainX.shape)
print(trainY.shape)


# In[73]:


# Model Building
from statsmodels.api import OLS

MI=OLS(trainY,trainX).fit()

MI.summary() #summary- we can see the model summary


# In[74]:


MI.params


# In[75]:


trainX.shape


# In[76]:


trainY.shape


# In[77]:


# Prediction
# predit()
predictions=MI.predict(testX)
predictions.head()


# In[78]:


# model diagonistics

plt.figure
plt.scatter(testY,predictions)
plt.xlabel("predictions")
plt.ylabel("Test Y Data")
plt.show()


# In[79]:


plt.figure()
sns.distplot(MI.resid)
plt.show()


# In[80]:


import sklearn.metrics as metrics


# In[84]:


print("MAPE: {}".format(metrics.mean_absolute_error(testY,predictions)))
print("MSE:{}".format(metrics.mean_squared_error(testY,predictions)))
print("RMSE:{}".format(np.sqrt(metrics.mean_squared_error(testY,predictions))))


# In[83]:


plt.figure()
sns.distplot((testY,predictions))
plt.show()
# as we can see here its very fluctuating graph


# In[ ]:




