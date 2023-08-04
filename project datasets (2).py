#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing the libararies
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[2]:


# loading the datasets
rawdf=pd.read_csv("C:\\Users\\78359\\Downloads\\census_income_data.csv")


# In[3]:


#  overview of datasets
rawdf.info()


# In[4]:


rawdf.describe().T # Here we can see the Stastical Summary of datasets


# In[5]:


# let us see the the types of values in column
rawdf.head()
# we have ? values in (workclass,occupation,native.country) in these columns


# In[6]:


# let us check the shape of dataset # 
(32561, 15)
print("Number of rows:",rawdf.shape[0])
print("Number of columns:",rawdf.shape[1])


# In[7]:


rawdf.columns # we can see the Columns of the datasets


# In[8]:


rawdf.replace({"?":None},inplace=True) # replacing special charactar with NONE value


# In[9]:


# checking the missing values
rawdf.isna().sum()

# have missing values
#workclass =1836 
#occupation = 1843
#native.country= 583


# In[10]:


print("Dtype of workclass:",rawdf["workclass"].dtype)
print("Dtype of occupation:",rawdf["occupation"].dtype)
print("Dtype of native country:",rawdf["native.country"].dtype)


# In[11]:


# we have method of filling object column with mode values
workmode=rawdf["workclass"].mode()[0]
workmode


# In[12]:


rawdf["workclass"].fillna(workmode,inplace=True)


# In[13]:


rawdf["workclass"].isna().sum()


# In[14]:


# we have missing values in occupation

# checking the dtype of occupation
rawdf["occupation"].dtype


# In[15]:


occumode=rawdf["occupation"].mode()[0]
occumode


# In[16]:


rawdf["occupation"].fillna(occumode,inplace=True)


# In[17]:


# checking the occupation filled
rawdf["occupation"].isna().sum()


# In[18]:


# 3rd missing value in native.country

# checking the datatype of native.county
rawdf["native.country"].dtype
# this one is categorical column


# In[19]:


nativemode=rawdf["native.country"].mode()[0]
nativemode


# In[20]:


# fill the isna values with nativemode
rawdf["native.country"].fillna(nativemode,inplace=True)


# In[21]:


# let us checks the isna values in native.country
rawdf["native.country"].isna().sum()


# In[22]:


# let us checks the values in income column
rawdf["income"].value_counts()
# let us set the messy column (income)


# In[23]:


rawdf["income"]=np.where(rawdf["income"]==">50K",1,0) # we are giving the category of income column in 2 categories


# In[24]:


rawdf["income"].value_counts()
print("greater than 50K:",1)
print("less than or equal to 50K:",0)


# In[25]:


# here we have seen that the same category is divided with their different different names
rawdf["marital.status"].value_counts()


# In[26]:


# so we divided in 2 categories only
# merge the same category columns
rawdf['marital.status']=rawdf['marital.status'].map({"Married-civ-spouse":"married",
                                                     "Married-AF-spouse":"married",
                                                     "Never-married":"Single",
                                                        "Divorced":"Divorced",
                                                        "Separated":"Single",
                                                        "Widowed":"Single",
                                                        "Married-spouse-absent":"Single"})


# In[27]:


rawdf['marital.status'].value_counts()


# In[28]:


# let us see the values in education or education.num
rawdf["education"].value_counts()  


# In[29]:


rawdf["education"].replace({"10th":"high-school",
                           "11th":"high-school",
                           "7th-8th":"high-school",
                          "9th":"high-school",
                           "12th":"high-school",
                           "5th-6th":"Preschool",
                           "1st-4th":"Preschool"},inplace=True)


# In[30]:


rawdf["education"].value_counts()


# In[31]:


# checking the duplicated values
rawdf.duplicated().sum() # we have 24 rows are with duplicated values


# In[32]:


import matplotlib.pyplot as plt


# In[33]:


catogorical=rawdf.columns[rawdf.dtypes=="object"]


# In[34]:


catogorical


# In[35]:


corrdf=rawdf.corr()
corrdf

# we are checking the correlation values here


# In[36]:


sns.heatmap(corrdf, annot=True)


# we can check here the heatmap with correlation values


# In[37]:


x=rawdf["education"].value_counts()
x


# In[38]:


plt.figure(figsize=(14,6))
# create subplot with 1 row and 2 columns
sns.countplot(data=rawdf,x="education",hue="income")
plt.xlabel("education")
plt.legend(title='income',labels=['less than 50k','greater than 50k'])
plt.show()

# as we can see here on the basis of education levels of income
# as usually highest number is HS-grad so the income of lesser than 50K is also present but 
# as compared of earning grater than 50K people mostly present in high school  


# In[39]:


# now we can check the income as the basis of workclass
    
    # firstly we will make pie-chart for checking the in which class are working the most population
y=rawdf['workclass'].value_counts()
y

# as we can see here mostly people are working in the private sector


# In[40]:


plt.figure(figsize=(12,6))
sns.countplot(data=rawdf, x='workclass', hue="income")
plt.xlabel("WORKCLASS")
plt.legend(title='income',labels=['less than 50k','greater than 50k'])
plt.show()

# as we can see here in the private sector mostly employees income is less than 50k
# and most of the employees of greater then 50k is also present in private sector


# In[41]:


#  the next column is marital.status
# we will apply the same process for the analysis
#firstly we will make pie-chart for checking the most number of people in which category
#the we will check income of the people accorfingly to the category


# In[42]:


marital=rawdf['marital.status'].value_counts()
marital


# In[43]:


plt.figure(figsize=(6,12))
plt.pie(marital,labels=marital.index,autopct="%1.1f%%",shadow='Yes')
plt.show()
# as we can see here "Single" category is 53.9% 
# and 46.1% is married


# In[44]:


plt.figure()
sns.countplot(data=rawdf, x="marital.status", hue="income")
plt.xlabel("MARITIAL STATUS")
plt.legend(title="Income",labels=['greater than 50k','less than 50k'])
plt.show()

# as we can here as married here category people counts of earns grater than 50k is more than single category people

# in single category people have more people earnings is less than 50K


# In[45]:


# now we have next column is occupation
print("values of occupation:",rawdf["occupation"].value_counts(normalize=True))



# as we can see here there are most number of people working in Prof-specialty (18.4%)
# and the least number of people is in aremed force 0.9%


# In[46]:


# now we will comapre and analyse our occupation column with income column

plt.figure(figsize=(12,8))
sns.countplot(data=rawdf, x="occupation", hue="income")
plt.xlabel("OCCUPATION")
plt.legend(title="INCOME", labels=["lesser than 50k","greater than 50K"])
plt.show()

# as se can see here mostly people who earns less than 50K is in pro-speicalist category
# # and who eans more than 50K is almost same in with craft-repair


# In[47]:


# our next column is relationship
relationship=rawdf["relationship"].value_counts()
print("values is in relationship:",relationship)

# we will make pie chart to analyse most people in which category comes
plt.figure(figsize=(10,10))
plt.pie(relationship, labels=relationship.index, autopct="%1.1f%%")
plt.show()
# we can see with the help of pie chart
# husband is captured in mostly area 40.5%
# # and other relative are captured least area 3%


# In[48]:


# now we will analyse the our occupation column with income column 
   # will see most earnings people in which category
plt.figure(figsize=(6,8))
sns.countplot(data=rawdf, x="relationship", hue="income")
plt.xlabel("RELATIONSHIP")
plt.legend(title="Income", labels=["Greater than 50K", "lesser than 50K"])
plt.show()
# as we can see here mostly number of lesser than 50K income people present in "Not in family(More than 7,000)" &
# and almost same number of this category people present in husband category also.

# most number of earners more than 50K is present in husband category
# and least number is present in own-child or other relative


# In[49]:


# our next column is race
# firstly we make pie- chart analyse most people in which category
race=rawdf["race"].value_counts()
print(race)
plt.figure(figsize=(8,8))
plt.pie(race, labels=race.index, autopct="%1.1f%%")
plt.show()

# as we can see the most number of peolple is white 85%
# and the least number of people is in other category 0.8% 


# In[50]:


# we will we with the help of countplot most number of earns in which category
plt.figure()
sns.histplot(data=rawdf, x="race", hue="income")
plt.xlabel("RACE")
plt.legend(title="income", labels=["lesser than 50K", "greater than 50K"])
plt.show()

# because of the most people are presented with white category 
# here we can see most people of earns greater than or lesser than both are in white category

# but in the white category most people are earnings grater than 50K 


# In[51]:


# now we have our next column is sex
# and we will apply same category to analyse the things
sex=rawdf["sex"].value_counts()
print(sex)
plt.figure()
explode_val=(0.0,0.2)
plt.pie(sex, labels=sex.index, autopct="%1.1f%%",explode=explode_val)
plt.show()

# as we can see here 66.9% is Male
# and 33.1% is Female


# In[52]:


# now we will analyse our sex column with income
# and will mostly earners are in which category
plt.figure()
sns.histplot(data=rawdf, x="sex", hue="income")
plt.xlabel("SEX")
plt.legend(title="Income", labels=["lesser than 50K", "greater than 50K"])
plt.show()

# as we can in male category almost half people is earning greater than 50K and half people is earning lesser than 50K


# but in female category mostly number is earning greater than 50K as compare of male


# In[53]:


rawdf["hours.per.week"].value_counts()


# In[54]:


sns.distplot(rawdf["hours.per.week"])


# In[55]:


print(rawdf["native.country"].value_counts(normalize=True))
# as we can see here approx 91% data is the united-States 
#so remaining all 9% as collecting in as a value of other countries 

rawdf["native.country"]=np.where(rawdf["native.country"]=="United-States","US","Other_Countries")


# In[56]:


# As we can see here there are so many countries. so its difficukt to understand 
print(rawdf["native.country"].value_counts(normalize=True))
# we can see here approx 91% data is the united-States 
#so remaining all 9% as collecting in as a value of other countries 

rawdf["native.country"]=np.where(rawdf["native.country"]=="United-States","US","Other_Countries")


# so we have changed the values in 2 countires 


# In[57]:


# so we seen that the education number and fnlwgt columns is not much important in this data
# drooping the columns
newrawdf=rawdf.drop(["fnlwgt","education.num"],axis=1)


# In[59]:


print(newrawdf.columns)
print(newrawdf.shape) # (32561,13)
# here we can see that we have droped the 2 columns


# In[60]:


# adding the intercept columns
from statsmodels.api import add_constant
rawdf1=add_constant(newrawdf)


# In[61]:


rawdf1.shape


# In[62]:


# dummy variable creation
fulldf=pd.get_dummies(rawdf1,drop_first=True)
fulldf.shape


# In[63]:


# dividing the datasets
from sklearn.model_selection import train_test_split

traindf,testdf=train_test_split(fulldf, train_size=0.7, random_state=2410)

print("traindf of shape",traindf.shape)
print("testdf of shape",testdf.shape)


# In[64]:


# dividing the data of independent variable or dependent variable
trainX=traindf.drop(["income"],axis=1)
trainY=traindf["income"]


# In[65]:


# dividing the testing data 
testX=testdf.drop(["income"],axis=1)
testY=testdf["income"]


# In[66]:


# checking the datasets shapes after the dividing the variab
print(trainX.shape)
print(trainY.shape)
print(testX.shape)
print(testY.shape)


# In[67]:


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


# In[68]:


highVIFColumnNames.remove("const") # we need to exclude const column
highVIFColumnNames


# In[69]:


trainX=trainX.drop(highVIFColumnNames, axis = 1)
testX=testX.drop(highVIFColumnNames, axis = 1)
print(trainX.shape)
print(testX.shape)


# In[70]:


from statsmodels.api import Logit
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


# In[71]:


# Model Building
M1=Logit(trainY,trainX).fit()


# In[72]:


M1.summary() #summary- we can see the model summary


# In[73]:


tempMaxPValue = 0.1
maxPValueCutoff = 0.1
trainXCopy = trainX.copy()
counter = 1
highPValueColumnNames = []


while (tempMaxPValue >= maxPValueCutoff):
    
    print(counter)    
    


    tempModelDf = pd.DataFrame()    
    Model = Logit(trainY, trainXCopy).fit()
    tempModelDf['PValue'] = Model.pvalues
    tempModelDf['Column_Name'] = trainXCopy.columns
    tempModelDf.dropna(inplace=True) # If there is some calculation error resulting in NAs
    tempColumnName = tempModelDf.sort_values(["PValue"], ascending = False).iloc[0,1]
    tempMaxPValue = tempModelDf.sort_values(["PValue"], ascending = False).iloc[0,0]
    
    if (tempMaxPValue >= maxPValueCutoff): # This condition will ensure that ONLY columns having p-value lower than 0.1 are NOT dropped
        print(tempColumnName, tempMaxPValue)    
        trainXCopy = trainXCopy.drop(tempColumnName, axis = 1)    
        highPValueColumnNames.append(tempColumnName)
    
    counter = counter + 1

highPValueColumnNames

# drop the insignificant columns

trainX=trainX.drop(highPValueColumnNames,axis=1)
testX=testX.drop(highPValueColumnNames,axis=1)


# In[74]:


print(trainX.shape)
print(testX.shape)


# In[75]:


# final model
FinalModel=Logit(trainY,trainX).fit()
DT=DecisionTreeClassifier().fit(trainX,trainY)


# In[76]:


# prediction
testX["test_pred"]=FinalModel.predict(testX)
testX


# In[77]:


# converting prediction into 0 class or 1 class
testX["class"]=np.where(testX["test_pred"]>=0.5,1,0)


# In[78]:


# model evaluation
from sklearn.metrics import confusion_matrix, classification_report
confusion_matrix(testX["class"],testY)


# In[79]:


print(classification_report(testX["class"],testY))

# we have received the accuracy is 85% by logistic regression


# In[ ]:





# In[ ]:




