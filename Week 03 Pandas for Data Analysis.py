#!/usr/bin/env python
# coding: utf-8

# # 1. Pandas Data Structures: Series and Dataframes

# Series is a one-dimensional labeled array.
# A Series is essentially a column of data with a label or name and an associated index that is used to access the data.
# 
# Dataframe is a two-dimensional labeled data structure.
# A Dataframe is essentially a tabel where each column is a series and each row has a label or index.

# In[35]:


import pandas as pd


# In[2]:


#creating series in pandas
data = [1,2,3,4,5]
series = pd.Series(data, index = ['A','B','C','D', 'E'])
print(series)


# In[3]:


# creating DataFrame in pandas
data = {'Name':['vidhya','soni','dev','riya','rohan','karthik'],
       'Age': [23,5,66,74,22,42],
       'Color': ['Blue','Red','Green','Blue','Pink','Red']}
df = pd.DataFrame(data)
print(df)


# # 2. Data Transformation with Pandas - Grouping, Merging, and Concatenating

# 1. Group-by: It is used to group your data bases on one or more columns. This allows you to apply aggregate functions such as mean, sum, or count to your data,based on the groups.
# 
# 2. Merging and joining dataframes: It is used to combine data from multiple dataframes into a single dataframe.
# 
# 3. Concatenating: It is mainly used to reorganise data. It allows you to combine dataframes vertically or horizontally, while reshaping allows you to rearrange the data in your dataframes into different shapes.

# In[4]:


import pandas as pd


# In[5]:


# Group by
data = {'Name':['vidhya','soni','dev','riya','rohan','karthik'],
       'Age': [23,15,26,44,22,62],
       'Country': ['India','Africa','Brazil','India','Nepal','Brazil']}
df = pd.DataFrame(data)
groupeddata = df.groupby(['Country'])
print(groupeddata['Age'].mean()) #shows the avg age of the ppl grouping by country they live in 


# In[6]:


# merging and joining
data1 = {'Name':['vidhya','ganesh','dev','riya','rohan','karthik'],
        'Age': [23,65,73,24,66,45]}
data2 = {'Name':['vidhya','ganesh','dev','riya','rohan','karthik'],
        'Color': ['Blue','Red','Green','Blue','Pink','Red']}
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)
mergeddata = pd.merge(df1, df2, on = 'Name') #merging the 2 dataframes/tables based on the name
print(mergeddata)


# In[7]:


# concatenating two dataframes - feilds should be same
data1 = {'Name':['vidhya','ganesh'],
        'Age': [23,65]}
data2 = {'Name':['dev','riya','rohan','karthik'],
        'Age': [23,65,73,43]}
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)
concatenateddata = pd.concat([df1,df2], axis = 0) #passing dataframes as a list, Axis 0 refers to the rows, so the DataFrames will be concatenated row-wise, meaning one DataFrame will be stacked on top of the other.
print(concatenateddata)
# here the index is not proper so you can customise it


# # 3. Indexing  & Slicing 

# The indexing and slicing techniques are useful when you need to extract specific pieces of information from your data. 
# 
# Indexing in Pandas allows you to extract specific values or groups of values from a Series or a Dataframe.
# 
# Slicing in Pandas allows you to extract a portion of the data in your series or Dataframe. helps to extract rows by giving start and stop points.

# In[8]:


import pandas as pd


# In[9]:


# Indexing
data = {'Name':['vidhya','soni','dev','riya','rohan','karthik'],
       'Age': [23,15,26,44,22,62],
       'Country': ['India','Africa','Brazil','India','Nepal','Brazil']}
df = pd.DataFrame(data)
print(df)


# In[10]:


# 1. indexing by column name

print(df['Age'])


# 2. df.loc and df.iloc
# 
# df.loc[]: This method is primarily label-based. It allows you to access a group of rows and columns by labels or boolean array. You can use this method by passing row and column labels.
# 
# 
# Example:
# #### Accessing a specific row and column by label
# df.loc[3, 'column_name']
# 
# #### Accessing a slice of rows and columns by labels
# df.loc[2:5, 'column_name']
# 
# #### Accessing rows based on a boolean condition
# df.loc[df['column_name'] > 10]
# 
# df.iloc[]: This method is primarily integer-based. It allows you to access a group of rows and columns by integer position. You can use this method by passing integer indexes.
# 
# Example:
# #### Accessing a specific row and column by integer position
# df.iloc[3, 4]
# 
# #### Accessing a slice of rows and columns by integer positions
# df.iloc[2:5, 1:4]
# 
# #### Accessing rows and columns based on integer positions
# df.iloc[[1, 3, 5], [0, 2]]
# 
# df.loc is used when you want to access DataFrame elements using labels, while df.iloc is used when you want to access DataFrame elements using integer positions.

# In[11]:


print(df)


# In[12]:


#df.loc and df.iloc

print(df.loc[2,'Name']) # row indexer and col indexer in the list as 2, 'Name' respectively

print(df.iloc[3,1]) # row indexer and col indexer in the list as 3, 1 is the index of 'Age' column respectively


# In[13]:


# Slicing - here the start point is 1 and stop point is 5
#1. [start:stop]
print(df[1:5]) #prints the rows from 1st index to the 5th index where 5 is not included.


# In[14]:


# 2. Slicing using the loc function by providing the [start:stop, col_indexer]
print(df.loc[1:3, ['Name','Country']])


# # 4. Sorting, Filtering, Mapping of Data

# Pandas provides a wide range of functions for manipulating and analyzing data, including sorting, filtering, reshaping , and mapping.
# 
# Sorting is the process of arranging data in a particular order.
# 
# 
# Filtering data is the process of selecting a subset of data based on certain conditions.
# 
# 
# Reshaping data is the process of transforming data from one form to another.
# 
# 
# Mapping data is the process of replacing one value with another value.

# In[15]:


import pandas as pd


# In[16]:


data = {'Name': ['Kavya', 'Ganesh', 'Manu', 'Kittu'],
       'English': [52,86,45,94],
       'Maths': [45,74,98,99],
       'Hindi':[26,78,59,99]}
df = pd.DataFrame(data, index = [1,2,3,4])
df


# In[17]:


# Sorting the data using 'sort_values' function based on one or more cols
engdatasort = df.sort_values ( by = ['English'], ascending = False)
print(engdatasort)
print('\n') 

# Sorting multiple cols
data_s = df.sort_values(by = ['English','Maths'], ascending = [True, False]) #sorts english col with asc and maths with desc
print(data_s)


# In[18]:


# filtering data - we can filter the data using boolean indexing
filtereddata = df[df['Maths']>45]
print(filtereddata)
print('\n')

#passing multiple conditions
filterdata = df[(df['Maths']>60) & (df['Hindi']<60)]
print(filterdata)


# In[19]:


# Mapping data - mapping the proficiency with the each name.
mapped = {'Kavya':2, 'Ganesh':3, 'Manu':4, 'Kittu':5}
df['Proficiency'] = df['Name'].map(mapped)
print(df)


# # 5. Data Cleaning

# Data Cleaning and preparation are essential steps in the data analysis pipeline.
# The process of data cleaning involves identifying and handling issues within a dataset that could cause inaccurate or biased analysis results. This involves handling missing values, removing duplicates, scaling data and encoding categorical data.

# In[20]:


import pandas as pd


# In[21]:


#handling duplicate values and outliers.
data = {'A': [1,3,2,3,4,3,3,9],
       'B': [1,3,4,3,2,5,9,6]}
df = pd.DataFrame(data)
df


# In[22]:


df.duplicated() 


# In[23]:


df1 = df.drop_duplicates()
df1


# In[24]:


q = df['A'].quantile(0.99)  
#calculates the 99th percentile (quantile) of the values in the 'A' column of the DataFrame df.
#calculate the value below which 99% of the data in column 'A' lies. In other words, 99% of the values in column 'A' are less than or equal to the value stored in variable q.
#0.99 is the quantile range
q 


# In[25]:


# clipping
#The line df['A'] = df['A'].clip(lower=None, upper=q) modifies the values in column 'A' of the DataFrame df. It clips the values to be within a specified range, where the lower limit is set to None and the upper limit is set to the value stored in the variable q, which represents the 99th percentile of the values in column 'A'.
#After executing this line, the values in column 'A' that are greater than the 99th percentile (q) will be replaced with q
#If a value is less than the 99th percentile, it remains unchanged. If you want to clip the values only from below (setting an upper limit), you can specify upper=None instead.
df['A'] = df['A'].clip(lower = None , upper = q)
df


# #Handling missing values is an important task in data preprocessing to ensure accurate analysis and modeling. Python provides several ways to handle missing values. Here are some common methods:
# 
# * Dropping Missing Values: Remove rows or columns with missing values using methods like dropna().
# 
# df.dropna()  # Drop rows with any missing value
# 
# 
# df.dropna(axis=1)  # Drop columns with any missing value
# * Imputation: Replace missing values with a specific value, such as the mean, median, or mode of the column. This can be done using methods like fillna().
# 
# df.fillna(df.mean())  # Replace missing values with the mean of the column
# 
# * Forward Fill (ffill) or Backward Fill (bfill): Propagate non-null values forward or backward along a Series or DataFrame.
# 
# df.fillna(method='ffill')  # Forward fill missing values
# df.fillna(method='bfill')  # Backward fill missing values
# 
# * Interpolation: Interpolate missing values based on other values in the dataset.
# 
# df.interpolate()  # Interpolate missing values using linear interpolation
# 
# * Using Machine Learning Models: Train a machine learning model to predict missing values based on other features in the dataset.
# 
# * Indicator Variables: Create an indicator variable to denote missingness.
# 
# df['A_missing'] = df['A'].isnull().astype(int)  # Create a new column indicating missing values in column 'A'
# 
# * Custom Function: Define a custom function to handle missing values based on domain knowledge or specific requirements.
# 
# def custom_imputation(df):
#     # Custom logic to impute missing values
#     
#     
#     return df
# 
# df = custom_imputation(df)
# 

# * Mean: Often used when missing values are missing completely at random (MCAR) or missing at random (MAR), especially in numerical data.used if the data doesnt contain outliers.
# * Median: Preferred when the data contains outliers or is skewed, as it's less sensitive to extreme values.
# * Mode: Suitable for categorical data or when dealing with variables with few unique values.

# In[36]:


# Handling missing values
import numpy as np
data = {'A':[2,1,4,np.nan,np.nan], 'B': [3,5,np.nan,np.nan,1]}
df = pd.DataFrame(data)
df


# In[37]:


# isna() method - returns true for missing values
print(df.isna())


# In[38]:


#fillna() method - replacing / filing the missing values with a particular value
print(df.fillna(value = 0))


# In[39]:


# Mean() method
df['A'] = df['A'].fillna(df['A'].mean())
df['B'] = df['B'].fillna(df['B'].mean())
df


# In[40]:


# data normalisation and scaling of data technique - used to transform the numerical data to a common scale.
#it is done to bias the higher values ie to scale down the values to a particular value.

max_value = df.max()
min_value = df.min()
print(max_value)
print(min_value)


# In[41]:


# Encoding of categorical data
data = {'A': ['Red','Blue','Yellow','Pink','White','Black'],
       'B': [1,3,4,3,2,0]}
df = pd.DataFrame(data)
df


# In[42]:


# one-hot encoding usign 'get-dummies' method
df = pd.get_dummies(df, columns = ['A'])
print(df)


# In[45]:


# 'replace' method
df['A'] = df['A'].replace({'Red': 15, 'Blue': 25, 'Yellow': 14, 'Pink': 21, 'White': 12, 'Black': 23})
print(df)


# # 6. Data Exploration

# Data Exploration and visualization are two critical components of data analysis. They help to understand the underlying structure and patterns of the data, which inturn can lead to meaningful insights and decisions.

# In[46]:


import pandas as pd


# In[48]:


# loading a standard dataset
df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
df


# In[49]:


# Understanding descriptive statistics
df.describe()


# In[50]:


df.head()


# In[51]:


# Integrating with other libraries
import numpy as np
np_array = df.values #converting pandas dataframe to numpy arrays


# In[52]:


np_array


# In[55]:


import matplotlib.pyplot as plt
df.plot()
plt.show()


# In[56]:


import seaborn as sb
sb.scatterplot(data = df, x = 'petal_length', y= 'petal_width')
plt.show()


# In[57]:


df.hist(column = 'petal_length')
plt.show()


# In[58]:


df.hist(column = 'sepal_length')
plt.show()


# In[59]:


df.boxplot(column = ['petal_length','sepal_length','sepal_width','petal_width'])
plt.show() #box plot of each col


# #Understanding correlation and covariance - Correlation is a statistical measure that describes the relationship between two variables. It indicates both the strength and direction of the relationship between the variables.
# * If the coefficient is close to 1, it indicates a strong positive correlation, meaning that as one variable increases, the other variable tends to increase as well.
# * If the coefficient is close to -1, it indicates a strong negative correlation, meaning that as one variable increases, the other variable tends to decrease.
# * If the coefficient is close to 0, it indicates little to no linear relationship between the variables.

# In[60]:


df.corr() # correlation


# #Understanding covariance - Covariance is a statistical measure that describes the extent to which two variables change together. It indicates the direction of the linear relationship between two variables.
# * Positive covariance indicates that the two variables tend to increase or decrease together, while negative covariance indicates that one variable tends to increase when the other decreases.

# In[62]:


df.cov() 


# In[66]:


#plotly - python library used to create interactive plots when integrated with pandas
get_ipython().system('pip install plotly')


# In[67]:


import plotly.express as px
figscatter = px.scatter(df, x = 'sepal_width', y = 'sepal_length')
figscatter.show()


# # 7. Time Series Analysis with Pandas
# Time series analysis is a statistical technique that can be used to extract meaningful insights from time series data ie the data that changes overtimes continuously. It is a commonly used in various fields, such as finance, economics, weather forecasting , and many others.

# In[68]:


import pandas as pd


# In[69]:


#DateTimeIndex and DateTime are the 2 objects used to represent the date and time datatypes in pandas
import datetime as dt
date = pd.datetime(2022,8,6,1,34,23) #(year,month,date,hour,min,sec)
print(date.month)


# In[70]:


print(date.year)


# In[73]:


df = pd.DataFrame({'A':[1,2,3]}, index = pd.DatetimeIndex(['2022-08-16','2002-12-07','2032-03-30']))
print(df)


# #Handling and manipulation of timeseries data
# 
# * Shifting - It is used to move the timeseries data forwards or backwards by a specified no of time periods.
# 
# * Lagging - It is used to create a new colmn with the timeseries data shifted by a specified no of timeperiods.
# * Resampling - It is used to change the frequency of the timeseries data ie getting data from daily to monthly
# 

# In[75]:


shifted_data = df.shift(periods = 2)
print(shifted_data)


# In[77]:


# lagging
df['lagged'] = df['A'].shift(periods=1)
df


# In[79]:


# Resampling
resampleddata = df.resample('M').sum() # M - monthly
print(resampleddata)


# # 8. Time Series Analysis with Pandas - Part - 02

# Time series data can ofter be decomposed into its trend, seasonal, and residual components.
# The trend component represents the longterm behaviour of a timeseries data while the seasonal component represents the periodic fluctuations in it. The residual component represents the random noise which is present in it.

# In[88]:


#Time decompostion using pandas 
import pandas as pd
import numpy as np
dates = pd.date_range(start = '2022-01-01',end = '2022-04-30')
values = np.random.randint(low = 1, high = 100, size = len(dates))
df = pd.DataFrame({'date':dates,'value':values})
df = df.set_index('date')
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(df['value'],period = 8)

df['trend'] = decomposition.trend
df['seasonal'] = decomposition.seasonal
df['residual'] = decomposition.resid 
df


# ## Time series forcastin with ARIMA and SARIMA models
# ARIMA models are used for time series data that does not have a seasonal component, while SARIMA models are used for time series data that does have a seasonal component.
# 
# ARIMA - Auto Regressive Integrated Moving Average model
# 
# 
# SARIMA - Seasonal Auto Regressive Integrated Moving Average model

# In[96]:


#Generate Time Series Data:
dates = pd.date_range(start = '2022-01-01',end = '2022-04-30')
values = np.random.randint(low = 1, high = 100, size = len(dates))
df = pd.DataFrame({'date':dates,'value':values})
df = df.set_index('date')
#This part generates a time series dataset spanning from January 1, 2022, to April 30, 2022, with random values for each date. The pd.date_range() function creates a range of dates, and np.random.randint() generates random integer values between 1 and 100. Finally, a DataFrame df is created with the dates as the index and the random values in the 'value' 


# In[105]:


from statsmodels.tsa.arima.model import ARIMA #This line imports the ARIMA (AutoRegressive Integrated Moving Average) model from the statsmodels library. ARIMA is a popular time series forecasting method.

#Fitting an ARIMA(1,1,1) model to the time series data
model = ARIMA(df['value'], order = (1,1,1)) 
#Here, an ARIMA(1,1,1) model is created and fitted to the time series data. The (1, 1, 1) in the order parameter represents the autoregressive order, differencing order, and moving average order, respectively. This specifies that the model includes one lagged observation, one differencing, and one lagged moving average term.
# order has 3 components ie autoregression, integration/differenciation , moving average
model_fit = model.fit() 

# Making a 7-day forecast
forecast = model_fit.forecast(steps=7)
#Finally, a 7-day forecast is generated using the forecast() method of the fitted ARIMA model. The steps parameter specifies the number of steps (or periods) ahead to forecast. In this case, it's set to 7 days.
forecast


# In[106]:


# SARIMA MOdel
from statsmodels.tsa.statespace.sarimax import SARIMAX #This line imports the SARIMAX (Seasonal Autoregressive Integrated Moving Average with Exogenous Regressors) model from the statsmodels library. SARIMAX is an extension of the ARIMA model that includes seasonal components.

model = SARIMAX(df['value'], order = (1,1,1), seasonal_order = (1,1,1,7)) #Here, a SARIMA(1,1,1)(1,1,1,7) model is created and fitted to the time series data. The (1, 1, 1) in the order parameter represents the autoregressive order, differencing order, and moving average order, respectively. The (1, 1, 1, 7) in the seasonal_order parameter represents the seasonal autoregressive order, seasonal differencing order, seasonal moving average order, and the length of the seasonal cycle (which is 7 in this case, indicating weekly seasonality).
model_fit = model.fit()

# Make a 7-day forecast
forecast = model_fit.forecast(steps=7) #Finally, a 7-day forecast is generated using the forecast() method of the fitted SARIMA model. The steps parameter specifies the number of steps (or periods) ahead to forecast, which is set to 7 days in this case.
print(forecast)


# # Assignment - Datawars Python - 
# 
# #### INTRODUCTION

# 1. Loading the Data

# In[36]:


import pandas as pd


# In[38]:


df = pd.read_csv("S&P500.csv", index_col='Date', parse_dates=True)
df.head()


# In[41]:


df.tail()


# In[42]:


#2. Analyzing the Data
#Quick summary statistics of our DataFrame:
df.describe()


# Single column statistics:

# In[43]:


df['Close'].min()


# In[44]:


df['Close'].max()


# Single column selection:

# In[45]:


df['Close'].head()


# Visualizations:
# 
# A simple line chart showing Close price:

# In[47]:


df['Close'].plot(figsize=(14, 7), title='S&P Closing Price | 2017 - 2022')


# In[48]:


#A more advanced chart combining Close Price and Volume:
ax1 = df['Close'].plot(figsize=(14, 7), title='S&P Closing Price | 2017 - 2022')

ax2 = ax1.twinx()
df['Volume'].plot(ax=ax2, color='red', ylim=[df['Volume'].min(), df['Volume'].max() * 5])

ax1.figure.legend(["Close", "Volume"])


# In[49]:


#A few statistical visualizations.

#A histogram:
df['Volume'].plot(kind='hist')


# In[50]:


#A box plot:

df['Volume'].plot(kind='box', vert=False)


# 3. Data Wrangling

# In[51]:


#Close price and it's SMA (Simple Moving Average):
df['Close SMA'] = df['Close'].rolling(60).mean()


# In[52]:


df[['Close', 'Close SMA']].tail(10)


# In[53]:


ax = df[['Close', 'Close SMA']].plot(figsize=(14, 7), title='Close Price & its SMA')


# Calculate the bands as:
# 
# Lower Band = Close SMA - 2 * rolling std
# 
# 
# Upper Band = Close SMA + 2 * rolling std

# In[54]:


df['Lower Band'] = df['Close SMA'] - (2 * df['Close'].rolling(60).std())
df['Upper Band'] = df['Close SMA'] + (2 * df['Close'].rolling(60).std())


# In[55]:


df[['Close', 'Close SMA', 'Lower Band', 'Upper Band']].tail()


# In[56]:


df[['Close', 'Lower Band', 'Upper Band']].plot(figsize=(14, 7), title='Close Price & its SMA')


# In[57]:


#Let's find that lower point that crosses the lower band:
ax = df[['Close', 'Lower Band', 'Upper Band']].plot(figsize=(14, 7), title='Close Price & its SMA')
ax.annotate(
    "Let's find this point", xy=(pd.Timestamp("2020-03-23"), 2237), 
    xytext=(0.9, 0.1), textcoords='axes fraction',
    arrowprops=dict(facecolor='red', shrink=0.05),
    horizontalalignment='right', verticalalignment='bottom');


# In[58]:


#We can quickly query all the dates that crossed the lower band (in the period 2020-03-01 to 2020-06-01)
df.loc['2020-03-01': '2020-06-01'].query("Close < `Lower Band`").head()


# In[59]:


#And finally, we can zoom in in that period:
df.loc['2020-01-01': '2020-06-01', ['Close', 'Lower Band', 'Upper Band']].plot(figsize=(14, 7), title='Close Price & its SMA | 2020-01-01 to 2020-06-01')


# Assignment 
# ### Intro to Pandas Series

# Check your knowledge: create a series
# 
# Create a series under the variable my_series that contains three elements 9, 11 and -5. The index of the series should be ['a', 'b', 'c'] and the name should be "My First Series".
# 
# 

# In[72]:


# 1. Check your knowledge: build a series
#Create a series called my_series
import pandas as pd
my_series = pd.Series([9,11,-5], index = ['a','b','c'], name = 'My First Series')
my_series


# Intro to Series
# 
# 
# Take a look the following list of companies:

# In[73]:


<img src="companies-revenue-seires.png" width="450px">


# ![image.png](attachment:image.png)

# In[74]:


#We'll represent them using a Series in the following way:
companies = [
    'Apple', 'Samsung', 'Alphabet', 'Foxconn',
    'Microsoft', 'Huawei', 'Dell Technologies',
    'Meta', 'Sony', 'Hitachi', 'Intel',
    'IBM', 'Tencent', 'Panasonic'
]


# In[75]:


s = pd.Series([
    274515, 200734, 182527, 181945, 143015,
    129184, 92224, 85965, 84893, 82345,
    77867, 73620, 69864, 63191],
    index=companies,
    name="Top Technology Companies by Revenue")


# In[76]:


s


# In[77]:


#Basic selection and location
#Selecting by index:
s['Apple']


# In[83]:


#.loc is the preferred way:
s.loc['Apple']


# In[84]:


#Selection by position:
s.iloc[0]


# In[85]:


s.iloc[-1]


# In[86]:


#Errors in selection:
# this code will fail
s.loc["Non existent company"]


# In[87]:


# This code also fails, 132 it's out of boundaries
# (there are not so many elements in the Series)
s.iloc[132]


# In[88]:


#We could prevent these errors using the membership check in:
"Apple" in s


# In[89]:


"Snapchat" in s


# Multiple selection:

# In[90]:


#By index:
s[['Apple', 'Intel', 'Sony']]


# In[91]:


#By position:
s.iloc[[0, 5, -1]]


# Activities:
#     
# 2. Check your knowledge: location by index
#     
#     
# Select the revenue of Intel and store it in a variable named intel_revenue:

# In[93]:


intel_revenue = s.iloc[-4]  #or intel_revenue = s.loc['Intel']
intel_revenue


# 3. Check your knowledge: location by position
# 
# 
# Select the revenue of the "second to last" element in our series s and store it in a variable named second_to_last:

# In[94]:


second_to_last = s.iloc[-2]
second_to_last
#As we're referring to an element from the end of the list ("second to last"), it makes sense to do negative indexing:
#You could have also done positive index; there are 14 elements in the series, so s.iloc[12] is also the second to last element (13 is the last one, Series are 0-indexed as Python lists).


# 4. Check your knowledge: multiple selection
# 
# 
# Use multiple label selection to retrieve the revenues of the companies:
# 
# 
# * Samsung
# * Dell Technologies
# * Panasonic
# * Microsoft
# 
# Store the result in the variable sub_series. Important! The values must be in that order.

# In[97]:


sub_series =  s[['Samsung', 'Dell Technologies', 'Panasonic', 'Microsoft']] 
# or sub_series = s.loc[["Samsung", "Dell Technologies", "Panasonic", "Microsoft"]]
sub_series


# #### Series Attributes and Methods

# In[98]:


s.head()


# In[99]:


s.tail()


# In[100]:


#Main Attributes
#The underlying data:
s.values


# In[101]:


#The index:
s.index


# In[102]:


#The name (if any):
s.name


# In[103]:


s.dtype


# In[104]:


s.size


# In[105]:


#len also works:

len(s)


# Statistical methods

# In[108]:


s.describe()


# In[109]:


s.mean()


# In[110]:


s.median()


# In[111]:


s.std()


# In[112]:


s.min(), s.max()


# In[113]:


s.quantile(.75)


# In[114]:


s.quantile(.99)


# Activities

# In[116]:


# Run this cell to complete the activity
american_companies = s[[
    'Meta', 'IBM', 'Microsoft',
    'Dell Technologies', 'Apple', 'Intel', 'Alphabet'
]]
american_companies


# 5. What's the average revenue of American Companies?
# 
# What's the average revenue of the companies contained in the variable american_companies? Enter the whole number (that is, without decimals).

# In[118]:


american_companies.mean()


# 6. What's the median revenue of American Companies?

# In[119]:


american_companies.median()


# #### Sorting Series¶

# Sorting by values or Index
# 
# Sorting by values, notice it's in "ascending mode":

# In[121]:


s.sort_values()


# Sorting by index (lexicographically by company's name), notice it's in ascending mode:

# In[123]:


s.sort_index()


# To sort in descending mode:

# In[124]:


s.sort_values(ascending=False).head()


# In[125]:


s.sort_index(ascending=False).head()


# ### Activities

# ##### 7. What company has the largest revenue?
# 
# Using all the companies (stored in the Series in s), which company has the largest revenue?
# 
# 
# Alphabet
# 
# 
# Apple
# 
# 
# Samsung
# 
# 
# Tencent

# In[127]:


s.sort_values(ascending=False)


# ##### 8. Sort company names lexicographically. Which one comes first?
# 
# Using all the companies (stored in the Series in s), which name is the "first" one in lexicographic (or alphabetical) order. That is, aa comes before than ab.
# 
# 
# Dell
# 
# 
# Alphabet
# 
# 
# IBM
# 
# 
# Apple

# In[128]:


s.sort_index()


# ### Immutability

# Run the sort methods above and check the series again, you'll see that `s` has NOT changed:

# In[129]:


s.head()


# We will sort the series by revenue, ascending, and we'll mutate the original one. Notice how the method doesn't return anything:

# In[130]:


s.sort_values(inplace=True)


# But now the series is sorted by revenue in ascending order:

# In[131]:


s.head()


# We'll now sort the series by index, mutating it again:

# In[133]:


s.sort_index(inplace=True)


# In[134]:


s.head()


# ### Activities

# ##### 9. Sort American Companies by Revenue
# Create a new variable american_companies_desc that contains the results of sorting american_companies by revenue (this is, by value) in descending order.
# 

# In[136]:


american_companies_desc = american_companies.sort_values(ascending= False)
american_companies_desc


# ##### 10. Sort (and mutate) international companies
# 
# Now it's time to do what we told you NOT to do, but we need practice it. There's a new series defined named international_companies. Your task is to sort them by Revenue in descending order (larger to smaller) but doing it in place, that is, modifying the series.
# 
# 
# If you make a mistake, you can always re-run the cell that generates the Series.

# In[138]:


# Run this cell to complete the activity
international_companies = s[[
    "Sony", "Tencent", "Panasonic",
    "Samsung", "Hitachi", "Foxconn", "Huawei"
]]
international_companies


# In[146]:


international_companies.sort_values(ascending=False, inplace=True)
international_companies


# #### Modifying series

# Modifying values:

# In[147]:


s


# In[148]:


s['IBM']  = 0


# In[149]:


s


# In[150]:


s.sort_values().head()


# In[159]:


#Adding elements:
s['Tesla'] = 21450


# In[160]:


s.sort_values().head()


# In[161]:


#Removing elements:
del s['Tesla']
s.sort_values().head()


# ### Activities

# ##### 11. Insert Amazon's Revenue
# Insert a new element in our series s, Amazon with a total revenue of: $469,822 (million dollars).

# In[163]:


s['Amazon'] = 469,822


# ##### 12. Delete the revenue of Meta
# 
# Remove the entry for Meta from the series s.

# In[164]:


del s['Meta']


# ### Concatenating Series

# We can append series to other series using the `.concat()` method:

# In[166]:


another_s = pd.Series([21_450, 4_120], index=['Tesla', 'Snapchat'])


# In[167]:


another_s


# In[168]:


s_new = pd.concat([s, another_s])


# In[170]:


# The original series s is not modified:
s


# In[172]:


#s_new is the concatenation of s and another_s:
s_new


# ### Series Practice with World Bank's data

# Take a look at raw data
# 
# 
# !head world_data.csv
# 
# 
# Country Name,Region Code,Country Code,"GDP, PPP (current international $)"," Population, total ",Population CGR 1960-2015,Internet users (per 100 people),Popltn Largest City % of Urban Pop,"2014 Life expectancy at birth, total (years)","Literacy rate, adult female (% of females ages 15 and above)",Exports of goods and services (% of GDP)
# Aruba,MA,ABW,," 103,889 ",1.19%,88.7,,75.5,97.5139617919922,
# Andorra,EU,AND,," 70,473 ",3.06%,96.9,,,,
# Afghanistan,ME,AFG," 62,912,669,167 "," 32,526,562 ",2.36%,8.3,53.4%,60.4,23.8738498687744,0.073278411818003
# Angola,AF,AGO," 184,437,662,368 "," 25,021,974 ",2.87%,12.4,50.0%,52.3,60.744800567627,0.373074223085945
# Albania,EU,ALB," 32,663,238,936 "," 2,889,167 ",1.07%,63.3,27.3%,77.8,96.7696914672852,0.271049844901716
# Arab World,,ARB," 6,435,291,560,152 "," 392,022,276 ",2.66%,39.5,29.8%,70.6,,
# United Arab Emirates,ME,ARE," 643,166,288,737 "," 9,156,963 ",8.71%,91.2,30.8%,77.4,95.0763397216797,
# Argentina,SA,ARG," 882,358,844,160 "," 43,416,755 ",1.36%,69.4,38.1%,76.2,98.1347808837891,0.110578189784346
# Armenia,RU,ARM," 25,329,201,238 "," 3,017,712 ",0.88%,58.2,55.2%,74.7,99.73046875,0.297333847463774
# 
# 
# import pandas as pd
# df = pd.read_csv('world_data.csv')
# df
# 
# 
# Country Name	Region Code	Country Code	GDP, PPP (current international $)	Population, total	Population CGR 1960-2015	Internet users (per 100 people)	Popltn Largest City % of Urban Pop	2014 Life expectancy at birth, total (years)	Literacy rate, adult female (% of females ages 15 and above)	Exports of goods and services (% of GDP)
# 0	Aruba	MA	ABW	NaN	103,889	1.19%	88.7	NaN	75.5	97.513962	NaN
# 1	Andorra	EU	AND	NaN	70,473	3.06%	96.9	NaN	NaN	NaN	NaN
# 2	Afghanistan	ME	AFG	62,912,669,167	32,526,562	2.36%	8.3	53.4%	60.4	23.873850	0.073278
# 3	Angola	AF	AGO	184,437,662,368	25,021,974	2.87%	12.4	50.0%	52.3	60.744801	0.373074
# 4	Albania	EU	ALB	32,663,238,936	2,889,167	1.07%	63.3	27.3%	77.8	96.769691	0.271050
# ...	...	...	...	...	...	...	...	...	...	...	...
# 259	Yemen, Rep.	ME	YEM	NaN	26,832,215	3.04%	25.1	31.9%	63.8	54.850632	NaN
# 260	South Africa	AF	ZAF	723,515,991,686	54,956,920	2.11%	51.9	26.4%	57.2	93.428932	0.308972
# 261	Congo, Dem. Rep.	AF	COD	60,482,256,092	77,266,814	2.99%	3.8	35.3%	58.7	65.897346	0.294904
# 262	Zambia	AF	ZMB	62,458,409,612	16,211,767	3.08%	21.0	32.9%	60.0	80.566971	NaN
# 263	Zimbabwe	AF	ZWE	27,984,877,195	15,602,751	2.62%	16.4	29.7%	57.5	85.285133	0.262450
# 264 rows × 11 columns
# 
# 
# df.columns
# 
# 
# Index(['Country Name', 'Region Code', 'Country Code',
#        'GDP, PPP (current international $)', ' Population, total ',
#        'Population CGR 1960-2015', 'Internet users (per 100 people)',
#        'Popltn Largest City % of Urban Pop',
#        '2014 Life expectancy at birth, total (years)',
#        'Literacy rate, adult female (% of females ages 15 and above)',
#        'Exports of goods and services (% of GDP)'],
#       dtype='object')
#       
#       
# Creating a pandas series from a dataframe df
# 
# 
# # Converting columns to pandas series
# 
# 
# country_name = pd.Series(df['Country Name'])
# country_code = pd.Series(df['Country Code'])
# population = pd.Series(df[' Population, total '])
# gdp = pd.Series(df['GDP, PPP (current international $)'])
# internet_users = pd.Series(df['Internet users (per 100 people)'])
# life_expectancy = pd.Series(df['2014 Life expectancy at birth, total (years)'])
# literacy_rate = pd.Series(df['Literacy rate, adult female (% of females ages 15 and above)'])
# exports = pd.Series(df['Exports of goods and services (% of GDP)'])
# 
# 
# country_name.head()
# 
# 
# 0          Aruba
# 1        Andorra
# 2    Afghanistan
# 3         Angola
# 4        Albania
# Name: Country Name, dtype: object
# 
# 
# country_code.head()
# 
# 
# 0    ABW
# 1    AND
# 2    AFG
# 3    AGO
# 4    ALB
# Name: Country Code, dtype: object
# 
# 
# population.head()
# 
# 
# 0        103,889 
# 1         70,473 
# 2     32,526,562 
# 3     25,021,974 
# 4      2,889,167 
# Name:  Population, total , dtype: object
# 
# 
# gdp.head()
# 
# 
# 0                  NaN
# 1                  NaN
# 2      62,912,669,167 
# 3     184,437,662,368 
# 4      32,663,238,936 
# Name: GDP, PPP (current international $), dtype: object
# 
# 
# internet_users.head()
# 
# 
# 0    88.7
# 1    96.9
# 2     8.3
# 3    12.4
# 4    63.3
# Name: Internet users (per 100 people), dtype: float64
# 
# 
# life_expectancy.head()
# 
# 
# 0    75.5
# 1     NaN
# 2    60.4
# 3    52.3
# 4    77.8
# Name: 2014 Life expectancy at birth, total (years), dtype: float64
# 
# 
# literacy_rate.head()
# 
# 
# 0    97.513962
# 1          NaN
# 2    23.873850
# 3    60.744801
# 4    96.769691
# Name: Literacy rate, adult female (% of females ages 15 and above), dtype: float64
# 
# 
# exports.head()
# 
# 
# 0         NaN
# 1         NaN
# 2    0.073278
# 3    0.373074
# 4    0.271050
# Name: Exports of goods and services (% of GDP), dtype: float64
# Activities
# 
# 
# 1: What is the data type?
# 
# 
# # try to find the dtype of `country_name`
# 
# 
# country_name.dtype
# 
# 
# dtype('O')
# 
# 
# 2: What is the size of the series?
# 
# 
# # try to get the shape of `gdp`
# 
# 
# gdp.size
# 
# 
# 264
# 
# 
# 3: What is the data type?
# 
# 
# # try to get the dtype of `internet_users`
# 
# 
# internet_users.dtype
# 
# 
# dtype('float64')
# 
# 
# 4: What is the value of the first element?
# 
# 
# # try to get the value of the first element in the `population` series
# 
# 
# population.iloc[0]
# 
# 
# ' 103,889 '
# 
# 
# 5: What is the value of the last element?
# 
# 
# life_expectancy.iloc[-1]
# 57.5
# 
# 
# 6: What is the value of the element with index 29?
# literacy_rate.iloc[29]
# 95.4420318603516
# 
# 
# 7: What is the value of the last element in the series?
# gdp.iloc[-1]
# ' 27,984,877,195 '
# 
# 
# 8: What is the mean of the series?
# 
# internet_users.mean()
# 47.557258064516134
# 
# 
# 9: What is the standard deviation?
# 
# internet_users.std()
# 27.690496399160462
# 
# 
# 10: What is the median of the series?
# 
# exports.median()
# 0.30183071080490154
# 
# 
# 11: What is the minimum value of the series?
# 
# life_expectancy.min()
# 48.9
# 
# 
# 12: What is the average literacy rate?
# 
# 
# literacy_rate.mean()
# 
# 
# 80.91936549162253
# 
# 
# ### Sorting
# 
# 
# 13: Sort the series in ascending order
# 
# 
# country_name_sorted = country_name.sort_values()
# 
# 
# 14: Sort multiple series at once
# 
# 
# literacy_rate_sorted = literacy_rate.sort_values()
# 
# 
# country_name_sorted_by_literacy_rate = country_name[literacy_rate.sort_value]

# ### Series Practice with S&P Companies' Market Cap

# In[ ]:


Take a look at the raw data:
Company stock symbols

symbols
get_ipython().system('head sp500-symbols.csv')
Name,Symbol
3M Company,MMM
A.O. Smith Corp,AOS
Abbott Laboratories,ABT
AbbVie Inc.,ABBV
Accenture plc,ACN
Activision Blizzard,ATVI
Acuity Brands Inc,AYI
Adobe Systems Inc,ADBE
Advance Auto Parts,AAP
Market cap raw data:

get_ipython().system('head sp500-marketcap.csv')
Symbol,Market Cap
MMM,138721055226
AOS,10783419933
ABT,102121042306
ABBV,181386347059
ACN,98765855553
ATVI,52518668144
AYI,6242377704
ADBE,94550214268
AAP,8123611867
import pandas as pd
market_cap = pd.read_csv("sp500-marketcap.csv", index_col="Symbol")['Market Cap']
market_cap.head()
Symbol
MMM     138721055226
AOS      10783419933
ABT     102121042306
ABBV    181386347059
ACN      98765855553
Name: Market Cap, dtype: int64
symbols = pd.read_csv("sp500-symbols.csv", index_col="Name")['Symbol']
symbols.head()
Name
3M Company              MMM
A.O. Smith Corp         AOS
Abbott Laboratories     ABT
AbbVie Inc.            ABBV
Accenture plc           ACN
Name: Symbol, dtype: object     

Basic series attributes
We'll start by doing a simple reconnaissance of the series we're working with.

1. Name of the market_cap Series

What's the name of the series contained in the market_cap variable?
market_cap.name
'Market Cap'

2. Name of the symbols Series

What's the name of the series contained in the symbols variable?
symbols.name
'Symbol'

3. What's the dtype of market_cap

What's the dtype of the series contained in the market_cap variable?
market_cap.dtype
dtype('int64')

4. What's the dtype of symbols

What's the dtype of the series contained in the symbols variable?

symbols.dtype
dtype('O')

5. How many elements do the series have?

How many elements market_cap series contains?
market_cap.count()
505

6. What's the minimum value for Market Cap?
 market_cap.min()

7. What's the maximum value for Market Cap?
market_cap.max()

## Selection and Indexing
market_cap.head()
1. What's the symbol of Oracle Corp.?
symbols['Oracle Corp.']
'ORCL'
2. What's the Market Cap of Oracle Corp.?
market_cap['ORCL']
symbols.loc['Oracle Corp.'] # ORCL
market_cap.loc['ORCL']
202302349740
3. What's the Market Cap of Wal-Mart Stores?
symbols.loc['Wal-Mart Stores']
market_cap['WMT']
304680931618
4. What's the symbol of the 129th company?
symbols.iloc[128]
'STZ'
5. What's the Market Cap of the 88th company in symbols?
symbols.iloc[87]
market_cap['CPB']
13467193376


6.Create a new series only with FAANG Stocks

There's a common term in investing (and in tech) which is FAANG companies. This refers to "big tech" companies by their acronyms. For example, FAANG means the following companies: Facebook Apple Amazon Netflix and Google (read more about FAANG and Big Tech in Wikipedia).

Here FAANG refers to acronym of few companies but there are other big tech companies like Microsoft. So, the term FAANG is not a strict definition of big tech companies.

Your task is to create a new series, under the variable faang_market_cap, containing the market cap of the following companies:

Amazon.com Inc
Apple Inc.
Microsoft Corp.
Alphabet Inc Class A (this is Google's main stock)
Facebook, Inc.
Netflix Inc.
Important! The stocks must be in THIS order. You will need to find the Symbols of the companies first.

Also important, as stated above, you MUST create a variable containing your new series. Your code should look something like:

faang_market_cap = ... # your code
There's a way to combine everything in a one-liner. Try to solve this task without looking at the solution; but after you've finished it, take a peak at it because there's a neat trick explained at the end of the solution.

#Ans# We start by first looking up the symbols of the companies:

>>> symbols[["Amazon.com Inc", "Apple Inc.", "Microsoft Corp.", "Alphabet Inc Class A", "Facebook, Inc.", "Netflix Inc.", ]]

Name
Amazon.com Inc           AMZN
Apple Inc.               AAPL
Microsoft Corp.          MSFT
Alphabet Inc Class A    GOOGL
Facebook, Inc.             FB
Netflix Inc.             NFLX
Name: Symbol, dtype: object
The symbols are then: "AMZN", "AAPL", "MSFT", "GOOGL", "FB", "NFLX". Now we can define the variable as the result of the multi selection:

faang_market_cap = market_cap[["AMZN", "AAPL", "MSFT", "GOOGL", "FB", "NFLX"]]
One neat trick with Pandas is that we can use the values of one series to select elements from another series. So we could have just done:

faang_market_cap = market_cap[symbols[["Amazon.com Inc", "Apple Inc.", "Microsoft Corp.", "Alphabet Inc Class A", "Facebook, Inc.", "Netflix Inc.", ]]]

7. Select the market cap of companies in position 1st, 100th, 200th, etc.
position_companies = market_cap.iloc[[0,99,199,299,399,499]]
position_companies
Symbol
MMM    138721055226
CTL     18237196861
FL       5819080328
MAT      5843402350
ROP     27247789759
XL      10753423590
Name: Market Cap, dtype: int64

                      
Sorting Series

                      
1. What's the 4th company sorted lexicographically by their symbol?

symbols.sort_values().head()
Name
Agilent Technologies Inc       A
American Airlines Group      AAL
Advance Auto Parts           AAP
Apple Inc.                  AAPL
AbbVie Inc.                 ABBV
Name: Symbol, dtype: object

2. What's the Market Cap of the 7th company (in descending order)?
market_cap.sort_index(ascending = False).iloc[6]
13390513478

# Assignment 
# ### Filtering and Conditional Selection with Series

# In[176]:


import pandas as pd
companies = [
    'Apple', 'Samsung', 'Alphabet', 'Foxconn',
    'Microsoft', 'Huawei', 'Dell Technologies',
    'Meta', 'Sony', 'Hitachi', 'Intel',
    'IBM', 'Tencent', 'Panasonic'
]
s = pd.Series([
    274515, 200734, 182527, 181945, 143015,
    129184, 92224, 85965, 84893, 82345,
    77867, 73620, 69864, 63191],
    index=companies,
    name="Top Technology Companies by Revenue")
s


# ### Boolean Arrays

# In[177]:


s.loc[[
    True,      # Apple
    False,     # Samsung
    True,      # Alphabet
    False,     # Foxconn
    True,      # Microsoft
    False,     # Huawei
    True,      # Dell
    True,      # Meta
    False,     # Sony
    False,     # Hitachi
    True,      # Intel
    True,      # IBM
    False,     # Tencent
    False,     # Panasonic
]]


# Activities
# 1. Select only the Japanese companies

# In[178]:


#japanese_boolean_array = [False, False, True, False, etc...]
japanese_boolean_array = [
    False,     # Apple
    False,     # Samsung
    False,     # Alphabet
    False,     # Foxconn
    False,     # Microsoft
    False,     # Huawei
    False,     # Dell
    False,     # Meta
    True,      # Sony
    True,      # Hitachi
    False,     # Intel
    False,     # IBM
    False,     # Tencent
    True,      # Panasonic
]


# In[179]:


japanese_companies = s.loc[japanese_boolean_array]
japanese_companies


# ### Conditional Selection

# In[180]:


s


# In[181]:


s > 100_000


# ##### What are the companies which revenues exceed the $100 billion dollars?

# In[182]:


s.loc[s > 100_000]


# #### Activities
# 1. Select companies with less than $90,000M in Revenue

# In[183]:


less_90_rev = s.loc[s<90_000]
less_90_rev


# In[184]:


#2. Select companies with revenue of more than $150,000M

more_150_rev = s.loc[s>150_000]
more_150_rev


# ### Combining Series methods with comparison operators

# ##### Company with the most revenue

# In[185]:


s.max()


# In[186]:


s.loc[s == s.max()]


# In[187]:


##### Company with the revenue above average:


# In[188]:


s.mean()


# In[189]:


s.loc[s >= s.mean()]


# ##### Companies who's revenue is greater than the average + 1 standard deviation:

# In[190]:


s.loc[s > (s.mean() + s.std())]


# In[191]:


### Boolean Operators


# In[192]:


s


# In[193]:


#### Companies with revenue greater than `$150,000M` or less than `$80,000M`


# In[194]:


##### Revenue greater than `$150,000M`


# In[195]:


s > 150_000


# In[196]:


##### Revenue less than `$80,000M`


# In[197]:


s < 80_000


# In[198]:


##### Putting it altogether:


# In[199]:


(s > 150_000) | (s < 80_000)


# In[200]:


#### Selecting the companies matching the expression:


# In[201]:


s.loc[(s > 150_000) | (s < 80_000)]


# In[202]:


#### The NOT (`~`) operator


# In[203]:


s.loc[s >= 150_000]


# In[204]:


s.loc[~(s < 150_000)]


# In[205]:


#### Activities


# In[206]:


##### 1. Select companies the companies with the MOST and LESS revenue


# In[207]:


most_and_less_rev = s.loc[(s == s.max()) | (s == s.min())]
most_and_less_rev


# In[208]:


##### 2. Select companies with revenue between `$80,000M` and `$150,000M`


# In[209]:


between_80_and_150 = s.loc[(s>80_000) & (s<150_000)]
between_80_and_150


# ### Practicing Series Filtering with S&P500 and Census Data

# In[210]:


import pandas as pd
# for visualizations, don't worry about these for now
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ### Datasets
# In this project you'll practice your Series filtering skills.
# 
# 
# Before we get started, let's introduce the datasets used. Make sure your lab is running!
# 
# 
# Datasets
# 
# 
# Both datasets used for this project were taken from the publicly available and Open Source RDatasets repository.
# 
# 
# #### Age of First Marriage
# The first one is titled Age at first marriage of 5,534 US women (source). It reads:
# 
# 
# Age at first marriage of 5,534 US women who responded to the National Survey of Family Growth (NSFG) conducted by the CDC in the 2006 and 2010 cycle.
# 
# 
# There are a total of 5,534 observations.
# 
# 
# S&P500 Returns (1990's)
# 
# 
# The second one is titled Returns of the Standard and Poors 500 (source) contains daily returns for S&P500 in the 1990's (1991-1999). It contains 2,780 values.
# 
# 
# #### Reading the data
# We can use the pandas built-in read_csv method to read the data that is stored in CSV format. Most commonly, read_csv is used to read data into DataFrames, but as this project deals with Series, we pass the parameter squeeze=True to make it a series. Bottom line is: don't worry about it for now, both datasets should be available for you in the variables age_marriage and sp500.
# 
# 
# We can also display a quick histogram about our data to understand how it is distributed. This is completely optional.

# ##### Age of First Marriage

# In[211]:


age_marriage = pd.read_csv("age_at_mar.csv", index_col=0).squeeze("columns") 

age_marriage.head()


# In[212]:


age_marriage.shape


# In[213]:


fig, ax = plt.subplots(figsize=(14, 7))
sns.histplot(age_marriage, ax=ax)


# ##### S&P Returns 1990's

# In[220]:


sp500 = pd.read_csv('SP_500.csv', index_col=0).squeeze("columns") 
sp500.head()


# In[221]:


sp500.shape


# In[222]:


fig, ax = plt.subplots(figsize=(14, 7))
sns.histplot(sp500, ax=ax)
#Create Figure and Axes: fig, ax = plt.subplots(figsize=(14, 7))
#This line creates a new figure and axes using plt.subplots(). The figsize parameter specifies the size of the figure (width, height) in inches. In this case, the width is set to 14 inches, and the height is set to 7 inches. The fig variable stores the figure object, and the ax variable stores the axes object.
#Plot Histogram: sns.histplot(sp500, ax=ax)
#This line creates a histogram plot of the data stored in the variable sp500 using seaborn's histplot() function. The ax=ax parameter specifies that the plot should be drawn on the axes created earlier (ax). The histogram visualizes the distribution of the data, showing the frequency of occurrence of different values.
#Explanation:The histogram plot provides a visual representation of the distribution of the data in sp500. It can help to understand the central tendency, spread, and shape of the data distribution. 
#In this case, sp500 likely contains data related to the S&P 500 stock index, and the histogram shows how frequently different values occur within that dataset.


# #### Activities

# 1. Rename both series with the names specified below, given their variables:
# 
# * age_marriage: should be named "Age of First Marriage"
# * sp500: should be named "S&P500 Returns 90s"

# In[227]:


age_marriage.name = "Age of First Marriage"
sp500.name = "S&P500 Returns 90s"
print(age_marriage)
print(sp500)


# ##### Basic Analysis
# ##### 2. What's the maximum value in age_marriage?

# In[228]:


age_marriage.max()


# ##### 3. What's the median Age of Marriage?

# In[229]:


age_marriage.median()


# ##### 4. What's the minimum return from S&P500?

# In[230]:


sp500.min()


# ### Simple Selection and Filtering
# ##### 5. How many Women marry at age 21?

# In[232]:


age_marriage.loc[age_marriage == 21].shape # or age_marriage.value_counts()


# ##### 6. How many Women marry at 39y/o or older?

# In[233]:


len(age_marriage[age_marriage >= 39])


# ##### 7. How many positive S&P500 returns are there?
# 
# The following visualization shows a red vertical line at the point `0`, we're looking for everything at the right of that line:

# In[236]:


ax = sns.histplot(sp500)
ax.axvline(0, color='red')
sp500.loc[sp500 > 0].shape #or use == >  #len(sp500[sp500 > 0])


# ##### 8. How many returns are less or equals than -2?
# 
# (Left to the red line)

# In[237]:


ax = sns.histplot(sp500)
ax.axvline(-2, color='red')


# In[238]:


len(sp500[sp500 <= -2]) # or use sp500.loc[sp500 <= -2].shape


# ### Advanced Selection with Boolean Operators

# ##### 9. Select all women below 20 or above 39
# 
# The segments depicted below:

# In[240]:


fig, ax = plt.subplots(figsize=(14, 7))
sns.histplot(age_marriage, ax=ax)
ax.add_patch(Rectangle((10, 0), 9, 450, alpha=.3, color='red'))
ax.add_patch(Rectangle((39, 0), 5, 450, alpha=.3, color='red'))


# Let's break this one step by step.
# 
# First, we have the condition "below 20", that we can write as: age_marriage < 20.
# 
# Then, we have "above 39", that we can write: age_marriage > 39.
# 
# Now, the task asks us for either condition. Women below 20 OR above 39, so we must use the or operator | to combine both expressions: (age_marriage < 20) | (age_marriage > 39). We must surround each individual expression within parentheses to avoid syntax issues.
# 
# Putting everything together we have:

# In[242]:


age_20_39 = age_marriage.loc[(age_marriage < 20) | (age_marriage > 39)]
# or age_20_39 = age_marriage[(age_marriage < 20) | (age_marriage >39)]
age_20_39


# ##### 10. Select all women whose ages are **even**, and are older than 30 y/o
# This is pretty similar to the previous activity, we need the and & boolean operator. The only "new" concept is that we can use the modulo operator % to check if a number is even or not, as it computes the "remainder" of the division. In summary:
# 
# age_30_even = age_marriage.loc[(age_marriage > 30) & (age_marriage % 2 == 0)]

# In[243]:


age_30_even = age_marriage.loc[(age_marriage % 2 == 0) & (age_marriage > 30)]
age_30_even


# ##### 11. Select the S&P500 returns between 1.5 and 3
# 
# The ones depicted below:

# In[244]:


fig, ax = plt.subplots(figsize=(14, 7))
sns.histplot(sp500, ax=ax)
ax.add_patch(Rectangle((1, 0), 1.5, 250, alpha=.3, color='red'))


# In[245]:


sp_15_to_3 = sp500.loc[ (sp500 >1.5) & (sp500<3) ]


# In[246]:


sp_15_to_3


# ### Vectorized Operations with Series

# In this brief project, we'll learn about "Vectorized Operations". In particular, we'll learn about Vectorized Operations applied to Pandas Series; but in reality, they're a concept original from NumPy, and we'll use it A LOT with DataFrames.
# 
# 
# So, the examples we'll see here might look trivial, but trust us that they'll be very useful throughout all your Pandas journey
# 
# ##### Understanding Vectorized Operations
# Vectorized Operations means just applying a "global" function to an entire Series. Let's derive an example from a Spreadsheet, in which we create a new column by applying an operation to ANOTHER column:
# ##### Preview
# With Series, it's going to be pretty much the same, it might look even simpler. Start the lab if you haven't already and take a look at the first operations.
# 
# First, we initialize the Series we've been using, this time we name it revenue_in_millions. That's the same series we've used so far, and it captures the revenue of the companies (listed in the Index) in Millions of dollars.
# 
# 
# That's it! That's a vectorized operation. We say it's "vectorized" because it doesn't act on just 1 value, but in the whole vector of values contained in the Series.
# 
# 
# Available Operators
# For now, we'll mostly focus on the regular arithmetic operators: +, -, *, /, **, etc. But you'll see in further labs that we can create vectorized operations with String operations or even our own custom functions.

# In[253]:


import pandas as pd
companies = [
    'Apple', 'Samsung', 'Alphabet', 'Foxconn',
    'Microsoft', 'Huawei', 'Dell Technologies',
    'Meta', 'Sony', 'Hitachi', 'Intel',
    'IBM', 'Tencent', 'Panasonic'
]

revenue_in_millions = pd.Series([
    274515, 200734, 182527, 181945, 143015,
    129184, 92224, 85965, 84893, 82345,
    77867, 73620, 69864, 63191],
    index=companies,
    name="Top Technology Companies by Revenue")


# #### Understanding Vectorized Operations
# 
# We'll now compute the revenue in Billions:

# In[254]:


revenue_in_billions = revenue_in_millions / 1000
revenue_in_billions


# #### Activities
# ##### 1. Subtract $50B from all companies in `revenue_in_billions`
# 
# 
# The recession just hit! Let's say you need to subtract $50B from all the companies in revenue_in_billions. Store the new series in the variable revenue_recession

# In[256]:


revenue_recession = revenue_in_billions - 50
revenue_recession


# ##### 2. Create a new series expressing revenue in dollars (units)

# In[258]:


revenue_in_dollars = revenue_in_millions * 1_000_000
##  revenue_in_dollars = revenue_in_billions * 1_000_000_000
revenue_in_dollars


# ### Operations between Series

# In[259]:


recession_impact = pd.Series([
    0.91, 0.93, 0.98, 0.97, 0.99, 0.89, 0.87,
    0.82, 0.93, 0.93, 0.89, 0.97, 0.97, 0.94], index=companies)
recession_impact


# The result of applying the recession impact:

# In[261]:


revenue_in_millions * recession_impact


# We can calculate the dollar amount of the impact by combining multiple operations:

# In[262]:


# Absolute impact in Millions
revenue_in_millions - (revenue_in_millions * recession_impact)


# In[263]:


# Absolute impact in Billions
(revenue_in_millions - (revenue_in_millions * recession_impact)) / 1_000


# In[264]:


#### Activities
##### Calculate revenue per employee, in dollars
number_of_employees = pd.Series([
    164000, 266673, 150028, 1290000, 221000, 195000,
    165000, 71970, 109700, 368250, 121100, 282100, 112771, 240198
], index=companies)


# In[265]:


revenue_per_employee = (revenue_in_millions / number_of_employees) * 1_000_000
revenue_per_employee

### Practicing Series Vectorized Operations with Penguins Data

Look at the dataset
import pandas as pd
# Read the dataset into a DataFrame
df = pd.read_csv('penguins_cleaned.csv')
df
species	island	culmen_length_mm	culmen_depth_mm	flipper_length_mm	body_mass_g	sex
0	Adelie	Torgersen	39.1	18.7	181.0	3750.0	MALE
1	Adelie	Torgersen	39.5	17.4	186.0	3800.0	FEMALE
2	Adelie	Torgersen	40.3	18.0	195.0	3250.0	FEMALE
3	Adelie	Torgersen	36.7	19.3	193.0	3450.0	FEMALE
4	Adelie	Torgersen	39.3	20.6	190.0	3650.0	MALE
...	...	...	...	...	...	...	...
328	Gentoo	Biscoe	47.2	13.7	214.0	4925.0	FEMALE
329	Gentoo	Biscoe	46.8	14.3	215.0	4850.0	FEMALE
330	Gentoo	Biscoe	50.4	15.7	222.0	5750.0	MALE
331	Gentoo	Biscoe	45.2	14.8	212.0	5200.0	FEMALE
332	Gentoo	Biscoe	49.9	16.1	213.0	5400.0	MALE
333 rows × 7 columns

# Convert all columns to pandas Series
species = df['species']
island = df['island']
culmen_length_mm = df['culmen_length_mm']
culmen_depth_mm = df['culmen_depth_mm']
flipper_length_mm = df['flipper_length_mm']
body_mass_g = df['body_mass_g']
gender = df['sex']
print("Species: ", species)
Species:  0      Adelie
1      Adelie
2      Adelie
3      Adelie
4      Adelie
        ...  
328    Gentoo
329    Gentoo
330    Gentoo
331    Gentoo
332    Gentoo
Name: species, Length: 333, dtype: object
print("Island: ", island)
Island:  0      Torgersen
1      Torgersen
2      Torgersen
3      Torgersen
4      Torgersen
         ...    
328       Biscoe
329       Biscoe
330       Biscoe
331       Biscoe
332       Biscoe
Name: island, Length: 333, dtype: object
print("Culmen Length (mm): ", culmen_length_mm)
Culmen Length (mm):  0      39.1
1      39.5
2      40.3
3      36.7
4      39.3
       ... 
328    47.2
329    46.8
330    50.4
331    45.2
332    49.9
Name: culmen_length_mm, Length: 333, dtype: float64
print("Culmen Depth (mm): ", culmen_depth_mm)
Culmen Depth (mm):  0      18.7
1      17.4
2      18.0
3      19.3
4      20.6
       ... 
328    13.7
329    14.3
330    15.7
331    14.8
332    16.1
Name: culmen_depth_mm, Length: 333, dtype: float64
print("Flipper Length (mm): ", flipper_length_mm)
Flipper Length (mm):  0      181.0
1      186.0
2      195.0
3      193.0
4      190.0
       ...  
328    214.0
329    215.0
330    222.0
331    212.0
332    213.0
Name: flipper_length_mm, Length: 333, dtype: float64
print("Body Mass (g): ", body_mass_g)
Body Mass (g):  0      3750.0
1      3800.0
2      3250.0
3      3450.0
4      3650.0
        ...  
328    4925.0
329    4850.0
330    5750.0
331    5200.0
332    5400.0
Name: body_mass_g, Length: 333, dtype: float64
print("Gender: ", gender)
Gender:  0        MALE
1      FEMALE
2      FEMALE
3      FEMALE
4        MALE
        ...  
328    FEMALE
329    FEMALE
330      MALE
331    FEMALE
332      MALE
Name: sex, Length: 333, dtype: object
Activities
1. Add a constant value.

body_mass_g_plus_100 = body_mass_g +100

2. Subtract the 'culmen_length_mm' series from the 'flipper_length_mm' series

length_difference = flipper_length_mm - culmen_length_mm

3. Multiply to series

double_culmen_depth_mm = culmen_depth_mm *2

4. Raise the 'flipper_length_mm' series to the power

flipper_length_mm_squared = flipper_length_mm **2

5. Calculate the mean of the 'culmen_length_mm' series and subtract it from each value in the series

culmen_length_mm_mean_centered = culmen_length_mm - culmen_length_mm.mean()

6. Concatenate the 'species' and 'gender' series

species_and_gender = species + '-' + gender 

7. Perform element-wise addition

culmen_length_plus_depth_mm = culmen_length_mm + culmen_depth_mm

8. Sort culmen_length_mm in descending order

culmen_length_mm_sorted = culmen_length_mm.sort_values(ascending = False)

9. Divide flipper_length_mm by culmen_length_mm

length_ratio = flipper_length_mm /culmen_length_mm### Series Practice: Vectorized Operations using NBA data
import pandas as pd
df = pd.read_csv('nba_player_stats_1985.csv', index_col='Player')
df.head()
height	weight	collage	born	G	MP	FG	FGA	FT	FTA	...	2PA	ORB	DRB	TRB	AST	STL	BLK	TOV	PF	PTS
Player																					
A.C. Green	203.0	106.0	NaN	1960.0	1361.0	39044.0	4778.0	9686.0	3247.0	4447.0	...	9177.0	3576.0	6553.0	10129.0	1469.0	1103.0	562.0	1508.0	2581.0	12928.0
A.J. Bramlett	196.0	88.0	NaN	1973.0	8.0	61.0	4.0	21.0	0.0	0.0	...	21.0	12.0	10.0	22.0	0.0	1.0	0.0	3.0	13.0	8.0
A.J. English	196.0	95.0	NaN	1963.0	151.0	3108.0	617.0	1418.0	259.0	333.0	...	1353.0	140.0	175.0	315.0	320.0	57.0	24.0	203.0	287.0	1502.0
A.J. Guyton	208.0	99.0	NaN	1976.0	80.0	1246.0	166.0	440.0	37.0	45.0	...	247.0	22.0	58.0	80.0	147.0	20.0	12.0	62.0	58.0	442.0
A.J. Hammons	198.0	99.0	NaN	1993.0	22.0	163.0	17.0	42.0	9.0	20.0	...	32.0	8.0	28.0	36.0	4.0	1.0	13.0	10.0	21.0	48.0
5 rows × 23 columns

free_throws_attempts
# Game info
games_played = df['G']
minutes_played = df['MP']
​
# Field Goals info
field_goals = df['FG']
field_goals_attempts = df['FGA']
​
# Free Throws info
free_throws = df['FT']
free_throws_attempts = df['FTA']
games_played.head()
Player
A.C. Green       1361.0
A.J. Bramlett       8.0
A.J. English      151.0
A.J. Guyton        80.0
A.J. Hammons       22.0
Name: G, dtype: float64
field_goals.head()
Player
A.C. Green       4778.0
A.J. Bramlett       4.0
A.J. English      617.0
A.J. Guyton       166.0
A.J. Hammons       17.0
Name: FG, dtype: float64
field_goals_attempts
field_goals_attempts.head()
Player
A.C. Green       9686.0
A.J. Bramlett      21.0
A.J. English     1418.0
A.J. Guyton       440.0
A.J. Hammons       42.0
Name: FGA, dtype: float64
MJ's Field Goals:

field_goals.loc['Michael Jordan*']
12192.0
Arithmetic Operations
1. Calculate field goal accuracy
field_goal_perc = (field_goals /field_goals_attempts) *100
field_goal_perc
Player
A.C. Green            49.328928
A.J. Bramlett         19.047619
A.J. English          43.511989
A.J. Guyton           37.727273
A.J. Hammons          40.476190
                        ...    
Zeljko Rebraca        52.043868
Zendon Hamilton       44.202899
Zoran Dragic          36.666667
Zoran Planinic        40.534979
Zydrunas Ilgauskas    47.579733
Length: 2553, dtype: float64
2. What's the FG% of Michael Jordan
field_goal_perc.loc["Michael Jordan*"]
49.68822594449199
3. Field goals per Game
field_goals_per_game
field_goals_per_game = field_goals / games_played
4. Which player has the highest 'Field Goal per Game' value?
field_goals_per_game.sort_values(ascending = False).head()
Player
Michael Jordan*       11.373134
Larry Bird*            9.965863
Alex English*          9.874336
LeBron James           9.823751
Dominique Wilkins*     9.518782
dtype: float64
5. Calculate "Total Points"
#In the NBA lingo, field goals account for all the "goals" scored by a player, EXCEPT free throws. So, if we want to calculate the total number of points scored by a player, we must add field goals and free throws. Field goals are a combination of 2-point and 3-point goals. For this exercise, you can safely assume that all "field goals" have a value of 2.
​
#Calculate Total Points scored by a player, by adding the series containing field goals and free throws. Store your results in the variable total_points.
​
total_points = (field_goals * 2) + free_throws
6. Who's the player with the most Total Points?
total_points
total_points.sort_values(ascending = False).head(1)
Player
Karl Malone*    36843.0
dtype: float64
7. Total Points per Minute
points_per_minute
points_per_minute = total_points/minutes_played
8. Who has a better Points per Minute score; MJ or Kevin Durant?
points_per_minute.loc[points_per_minute.index == 'Michael Jordan*']
Player
Michael Jordan*    0.773232
dtype: float64
points_per_minute.loc[points_per_minute.index == 'Kevin Durant']
Player
Kevin Durant    0.679694
dtype: float64
9. Calculate FT
total_attempts
ft_perc = free_throws / free_throws_attempts
10. Who's the player with best FT% record: MJ or Larry Bird?
ft_perc.loc[ft_perc.index == 'Larry Bird*']
Player
Larry Bird*    0.906006
dtype: float64
l Jordan*']
ft_perc.loc[ft_perc.index == 'Michael Jordan*']
Player
Michael Jordan*    0.835271
dtype: float64
Boolean Operations
11. Find the top 25% players by "free throw accuracy"
ft_top_25 = ft_perc >= ft_perc.quantile(.75)
ft_top_25
Player
A.C. Green            False
A.J. Bramlett         False
A.J. English          False
A.J. Guyton            True
A.J. Hammons          False
                      ...  
Zeljko Rebraca        False
Zendon Hamilton       False
Zoran Dragic          False
Zoran Planinic        False
Zydrunas Ilgauskas    False
Length: 2553, dtype: bool
12. How many players are in the top 25% by free throw accuracy?
ft_top_25.sum() # or ft_perc.loc[ft_top_25]
613
13. Find those players that scored 0 points in their history
players_0_points
players_0_points = total_points == 0
14. How many players have scored 0 points?
sum
players_0_points.sum()
41
# ### Introduction to DataFrames
What is a DataFrame?
Simply put, DataFrames are Python's way of displaying data in tablular form.

By using Python's powerful library for Data Analysis - pandas with DataFrames it offers us as users an efficient way to work with large amounts of structured data.

Is it similar to Excel?

Just as Excel use spreadsheets, Python uses pandas dataframes.

Python is often preferred over Excel due to its scalability and speed.

How is a DataFrame composed?
So what does a DataFrame look like? It is made up of rows and columns making it two dimensional. Let's take a look below:


We see that our Index is made up of the companies, however an index can also be made up of numbers.

An Index is like an address, it can be used to locate both rows and columns.

More on that later.

So, let's jump into our Jupyter notebook and create the DataFrame from this example.

The dataset for today's lab contains information on the Top Tech Companies in the World as shown below:


Preview
Our DataFrame contains five columns: - Revenue - Employees - Sector - Founding Date - Country

The syntax to create a dataframe is:

import pandas as pd
pd.DataFrame(data, index)
data: These are the values from our dataset.
index: This is like the address of the data we are storing.
Basic Navigation & Browsing Techniques
Basic Navigation & Browsing Techniques
So, how do we truely harness the power of pandas DataFrames?

Let's explore some functions:

a) head
The head() method displays the first few rows of your DataFrame, making it easier to get a sense of the overall structure and content. It will display the first five rows by default.

df.head()
We can expand our dataset further and specify the number of rows .head(n) within the brackets, as shown below:

df.head(10)
This function is especially useful as you can quickly inspect your DataFrame using the head() method to ensure that all of the data is stored correctly and as expected.

b) tail
The tail() method is similar to head() except it displays the last few rows of your DataFrame. .

df.tail()
Also similar to the head() we can specify the number of rows we want to display with .tail(n).

df.tail(10)
This is useful for quickly identifying any problems with your dataset, as any errors will most likely be found at the end rather than the beginning.

c) info
The info() method returns a list of all the columns in your DataFrame, along with their names, data types, number of values, and memory usage.

df.info()
This makes it easy to gain insight into how much space is being taken up by each column and can help identify potential problems such as missing values or incorrect data types.

d) shape
The shape method returns a tuple with the number of rows and columns (rows, columns) in our DataFrame.

df.shape
This gives us a quick insight into the dimensionality of our DataFrame.

e) describe
The describe() method displays descriptive statistics for numerical columns in your DataFrame, including the mean, median, standard deviation, minimum and maximum values.

df.describe()
This can be very useful for understanding the distribution of values across a specific dataset or column without having to manually calculate each statistic.

f) nunique
The nunique() method counts the number of distinct elements.

df.nunique()
This can be very useful for understanding the number of categories we have in a column for example.

g) isnull
The isnull() method detected missing values by creating a DataFrame object with a boolean value of True for NULL values and otherwise False.

df.isnull()
We can take this one step further by applying the sum() function to get a total number of NULL values in our DataFrame.

df.isnull().sum()

1 Output the first four rows of the df using the head function

Now it's your turn! Try outputting the first four rows of the dataframe using the head function. Store the result in the variable head_first_4.

head_first_4 = df.head(4)
2 Output the last six rows of the df using the tail function.

Now try output the last six rows of the dataframe using the tail function. Store the result in the variable tail_last_6.

tail_last_6 = df.tail(6)
# ![download.png](attachment:download.png)
Column selection:
Before we begin this section it is important we can differenciate between a Series and a DataFrame.

A Series is a single column in a DataFrame
A DataFrame is an entire table of data.
Let's take a look at how we can choose these items:

a) Select One Column
The [] operator can be used to select a specific column within a DataFrame. The output is a Series.

df['column_name']
b) Select One Column and Apply Methods
DataFrame's also allow the user to apply methods on columns, these functions include sum(), mean(), min(), max(), median() and more.

df['column_name'].sum()
c) Select Multiple Columns
To select multiple columns, use the [] operator with a list of column names as the argument. This creates another DataFrame.

df[['column_name_1', 'column_name_2','column_name_3']]
We can save this new DataFrame under a new variable name so that we can come back to it later.

new_df = df[['column_name_1', 'column_name_2','column_name_3']]
d) Select Multiple Columns and Apply Methods
We can take a shortcut and apply methods on more than one column at the same time.

df[['column_name_1', 'column_name_2','column_name_3']].mean()
3
Select the column Employees

Let's practice these skills, select the column Employees into the variable employees_s. You'll notice that the result of this selection is a Series.

Correct!
As we previously saw, selection of columns uses just square brackets []:

employees_s = df['Employees']
Try outputting the values of employee_s, you'll notice it's a Series.

4
Output the median Employees to the nearest whole number

Now, take it one step further and find the median of each row for the column Employees. Store the result in the variable employees_median

Correct!
We can do all at once and chain the operations of first, selection (df['Employees']) and then immediately calculate the median:

employees_median = df["Employees"].median()
5
Calculate the mean for columns Revenue and Employees

Lastly, let's calculate the mean for the columns Revenue and Employees. Store the result in the variable r_e_mean.

Your result should be a Series, and it should look something like:

Revenue      XXX
Employees    YYY
dtype: float64
Round off the mean to the nearest whole number.

Correct!
r_e_mean = round(df[["Revenue", "Employees"]].mean())Selection by Index
Selection by Index - loc
Index selection .loc is a Python DataFrames method that allows users to select DataFrame rows and columns by their labels or integer positions.

It is most commonly used when a user needs to access specific elements within a DataFrame, such as selecting all rows with a specific label or values in a specific column.

df.loc[row_label, column_label]
We can use : in place of row_label or column_label to call all the data.

df.loc[:, column_label]

df.loc[row_label,:]
We can also pass multiple columns in place of column_label or multiple rows in place of row_label.

df.loc[['row_name_1', 'row_name_2','row_name_3'], column_label]

df.loc[row_label,['column_name_1', 'column_name_2','column_name_3']]
Slicing is a powerful feature of pandas that enables us to access specific parts of our DataFrame.

start:stop:step

If we don't specify the step, the default value is 1.

a) With column's
df.loc[`row_label`, `column_name_start`:`column_name_stop`]
b) With rows's
df.loc[`row_name_start`:`row_name_stop` , `column_label`]
c) With step
df.loc[`row_label`, `column_name_start`:`column_name_stop`:n]

df.loc[`row_name_start`:`row_name_stop`:n , `column_label`]
d) With step and :
df.loc[:`, `column_name_start`:`column_name_stop`:n]

df.loc[`row_name_start`:`row_name_stop`:n , :]
6
Select the Revenue, Employees & Sector for the companies Apple, Alphabet and Microsoft

Now let's leverage your .loc selection skills. Your task is to select the columns Revenue, Employees & Sector for the companies Apple, Alphabet and Microsoft. Your result should be stored in a variable index_selection and it should be a DataFrame looking something like:

Revenue  Employees                Sector
Apple       274515     147000  Consumer Electronics
Alphabet    182527     135301     Software Services
Microsoft   143015     163000     Software Services
Correct!
Selection by Position - iloc
Selection by position .iloc is a useful Python DataFrames method that allows users to select rows and columns of a DataFrame based on their integer positions.

This is especially useful when users need to access elements within a DataFrame that do not have labels or specific column names.

df.iloc[row_position, column_position]
We can use : in place of row_position or column_position to call all the data.

df.iloc[:, column_position]

df.iloc[row_position,:]
We can also pass multiple columns in place of column_position or multiple rows in place of row_position.

df.iloc[['row_position_1', 'row_position_2','row_position_3'], column_position]

df.iloc[row_position,['column_position_1', 'column_position_2','column_position_3']]
Slicing is a powerful feature of pandas that enables us to access specific parts of our DataFrame.

start:stop:step

a) With column's
df.iloc[`row_position`, `column_position_start`:`column_position_stop`]
b) With rows's
df.iloc[`row_position_start`:`row_position_stop` , `column_position`]
c) With step
df.iloc[`row_position`, `column_position_start`:`column_position_stop`:n]

df.iloc[`row_position_start`:`row_position_stop`:n , `column_position`]
d) With step and :
df.iloc[:`, `column_position_start`:`column_position_stop`:n]

df.iloc[`row_position_start`:`row_position_stop`:n , :]
7
Perform a selection using .iloc and positional selection

Now it's time to put your iloc skills to the practice. Your task is to select the companies in positions: 2nd, 4th and 6th. And the columns in positions 1st, 2nd and the last one. Store your result in the variable position_selection.

Your result should look something like:

    Revenue Employees   Country
Samsung 200734  267937  South Korea
Foxconn 181945  878429  Taiwan
Huawei  129184  197000  China
Correct!


DataFrame's and Immutability

What is Immutability?
Immutability is a key concept in Python DataFrames, which means that once a DataFrame is created, it cannot be changed.

This means that any changes made within the DataFrame will result in the creation of a new DataFrame rather than modifying the original. We seen an example of this earlier when we created the DataFramedf_new based on our results from df.

Conclusion
By understanding these Python DataFrame's concepts and methods, you can confidently use Python to explore and analyze datasets.

Python DataFrame's are powerful tools for working with large amounts of data quickly and easily. It can help you explore, understand, and gain insights into your datasets with just a few lines of code.Introduction to DataFrames
import pandas as pd
Firstly, take a look at the dataset:

tech_table.png

# Creating an empty DataFrame
df = pd.DataFrame()
print(df)
Empty DataFrame
Columns: []
Index: []
Now, let's add our data:

# Lists of data
data = {'Revenue': [274515,200734,182527,181945,143015,129184,92224,85965,84893,
                    82345,77867,73620,69864,63191],
        'Employees': [147000,267937,135301,878429,163000,197000,158000,58604,
                      109700,350864,110600,364800,85858,243540],
        'Sector': ['Consumer Electronics','Consumer Electronics','Software Services',
                   'Chip Manufacturing','Software Services','Consumer Electronics',
                   'Consumer Electronics','Software Services','Consumer Electronics',
                   'Consumer Electronics','Chip Manufacturing','Software Services',
                   'Software Services','Consumer Electronics'],
        'Founding Date':['01-04-1976','13-01-1969','04-09-1998','20-02-1974',
                         '04-04-1975','15-09-1987','01-02-1984','04-02-2004',
                         '07-04-1946','01-01-1910','18-07-1968','16-06-1911',
                         '11-11-1998','07-03-1918'],
        'Country':['USA','South Korea','USA','Taiwan','USA','China','USA','USA',
                   'Japan','Japan','USA','USA','China','Japan']} 
index = ['Apple','Samsung','Alphabet','Foxconn','Microsoft','Huawei',
         'Dell Technologies','Meta','Sony','Hitachi','Intel','IBM',
         'Tencent','Panasonic']
# Creating a dataframe with our data 
df = pd.DataFrame(data, index)
# Let's see our dataframe
df
Revenue	Employees	Sector	Founding Date	Country
Apple	274515	147000	Consumer Electronics	01-04-1976	USA
Samsung	200734	267937	Consumer Electronics	13-01-1969	South Korea
Alphabet	182527	135301	Software Services	04-09-1998	USA
Foxconn	181945	878429	Chip Manufacturing	20-02-1974	Taiwan
Microsoft	143015	163000	Software Services	04-04-1975	USA
Huawei	129184	197000	Consumer Electronics	15-09-1987	China
Dell Technologies	92224	158000	Consumer Electronics	01-02-1984	USA
Meta	85965	58604	Software Services	04-02-2004	USA
Sony	84893	109700	Consumer Electronics	07-04-1946	Japan
Hitachi	82345	350864	Consumer Electronics	01-01-1910	Japan
Intel	77867	110600	Chip Manufacturing	18-07-1968	USA
IBM	73620	364800	Software Services	16-06-1911	USA
Tencent	69864	85858	Software Services	11-11-1998	China
Panasonic	63191	243540	Consumer Electronics	07-03-1918	Japan
Basic Navigation & Browsing Techniques
a) head()

# First 5 rows by default
df.head()
Revenue	Employees	Sector	Founding Date	Country
Apple	274515	147000	Consumer Electronics	01-04-1976	USA
Samsung	200734	267937	Consumer Electronics	13-01-1969	South Korea
Alphabet	182527	135301	Software Services	04-09-1998	USA
Foxconn	181945	878429	Chip Manufacturing	20-02-1974	Taiwan
Microsoft	143015	163000	Software Services	04-04-1975	USA
# df.head(n)
df.head(2)
Revenue	Employees	Sector	Founding Date	Country
Apple	274515	147000	Consumer Electronics	01-04-1976	USA
Samsung	200734	267937	Consumer Electronics	13-01-1969	South Korea
b) tail()

# Last 5 rows by default
df.tail()
Revenue	Employees	Sector	Founding Date	Country
Hitachi	82345	350864	Consumer Electronics	01-01-1910	Japan
Intel	77867	110600	Chip Manufacturing	18-07-1968	USA
IBM	73620	364800	Software Services	16-06-1911	USA
Tencent	69864	85858	Software Services	11-11-1998	China
Panasonic	63191	243540	Consumer Electronics	07-03-1918	Japan
# df.tail(n)
df.tail(3)
Revenue	Employees	Sector	Founding Date	Country
IBM	73620	364800	Software Services	16-06-1911	USA
Tencent	69864	85858	Software Services	11-11-1998	China
Panasonic	63191	243540	Consumer Electronics	07-03-1918	Japan
c) info()

 df.info()
<class 'pandas.core.frame.DataFrame'>
Index: 14 entries, Apple to Panasonic
Data columns (total 5 columns):
 #   Column         Non-Null Count  Dtype 
---  ------         --------------  ----- 
 0   Revenue        14 non-null     int64 
 1   Employees      14 non-null     int64 
 2   Sector         14 non-null     object
 3   Founding Date  14 non-null     object
 4   Country        14 non-null     object
dtypes: int64(2), object(3)
memory usage: 672.0+ bytes
d) shape

 df.shape
(14, 5)
e) describe()

 df.describe()
Revenue	Employees
count	14.000000	14.000000
mean	124420.642857	233616.642857
std	63686.481231	207583.087389
min	63191.000000	58604.000000
25%	78986.500000	116775.250000
50%	89094.500000	160500.000000
75%	172212.500000	261837.750000
max	274515.000000	878429.000000
f) nunique()

 df.nunique()
Revenue          14
Employees        14
Sector            3
Founding Date    14
Country           5
dtype: int64
g) isnull()

 df.isnull()
Revenue	Employees	Sector	Founding Date	Country
Apple	False	False	False	False	False
Samsung	False	False	False	False	False
Alphabet	False	False	False	False	False
Foxconn	False	False	False	False	False
Microsoft	False	False	False	False	False
Huawei	False	False	False	False	False
Dell Technologies	False	False	False	False	False
Meta	False	False	False	False	False
Sony	False	False	False	False	False
Hitachi	False	False	False	False	False
Intel	False	False	False	False	False
IBM	False	False	False	False	False
Tencent	False	False	False	False	False
Panasonic	False	False	False	False	False
# Let's see what happens when we apply sum()
df.isnull().sum()
Revenue          0
Employees        0
Sector           0
Founding Date    0
Country          0
dtype: int64
Check you knowledge:

1. Output the first four rows of the df
4
head_first_4 = df.head(4)
2. Output the last six rows of the df
6
tail_last_6 = df.tail(6)
Column Selection
Let's now look into how we can select specific data.

a) Select One Column

#name_of_df["name_of_column"]
​
df['Revenue']
Apple                274515
Samsung              200734
Alphabet             182527
Foxconn              181945
Microsoft            143015
Huawei               129184
Dell Technologies     92224
Meta                  85965
Sony                  84893
Hitachi               82345
Intel                 77867
IBM                   73620
Tencent               69864
Panasonic             63191
Name: Revenue, dtype: int64
b) Select One Column and Apply Methods

With numeric datatypes:

# Find the lowest revenue 
df['Revenue'].min()
63191
# Find the highest revenue
df['Revenue'].max()
274515
# Find the average revenue
df['Revenue'].mean()
124420.64285714286
# Find the average revenue rounded to the nearest whole number 
round(df['Revenue'].mean())
124421
# Find the median revenue rounded to the nearest whole number   
round(df['Revenue'].median())
89094
With string datatypes:

df['Sector'].min()
'Chip Manufacturing'
df['Sector'].max()
'Software Services'
df['Sector'].count()
14
df['Sector'].nunique()
3
c) Select Multiple Columns

In order to select multiple columns in our dataframe we have to create a list.

# ERROR?!
df['Revenue', 'Employees','Country']
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
File /usr/local/lib/python3.11/site-packages/pandas/core/indexes/base.py:3652, in Index.get_loc(self, key)
   3651 try:
-> 3652     return self._engine.get_loc(casted_key)
   3653 except KeyError as err:

File /usr/local/lib/python3.11/site-packages/pandas/_libs/index.pyx:147, in pandas._libs.index.IndexEngine.get_loc()

File /usr/local/lib/python3.11/site-packages/pandas/_libs/index.pyx:176, in pandas._libs.index.IndexEngine.get_loc()

File pandas/_libs/hashtable_class_helper.pxi:7080, in pandas._libs.hashtable.PyObjectHashTable.get_item()

File pandas/_libs/hashtable_class_helper.pxi:7088, in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: ('Revenue', 'Employees', 'Country')

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
Cell In[31], line 2
      1 # ERROR?!
----> 2 df['Revenue', 'Employees','Country']

File /usr/local/lib/python3.11/site-packages/pandas/core/frame.py:3760, in DataFrame.__getitem__(self, key)
   3758 if self.columns.nlevels > 1:
   3759     return self._getitem_multilevel(key)
-> 3760 indexer = self.columns.get_loc(key)
   3761 if is_integer(indexer):
   3762     indexer = [indexer]

File /usr/local/lib/python3.11/site-packages/pandas/core/indexes/base.py:3654, in Index.get_loc(self, key)
   3652     return self._engine.get_loc(casted_key)
   3653 except KeyError as err:
-> 3654     raise KeyError(key) from err
   3655 except TypeError:
   3656     # If we have a listlike key, _check_indexing_error will raise
   3657     #  InvalidIndexError. Otherwise we fall through and re-raise
   3658     #  the TypeError.
   3659     self._check_indexing_error(key)

KeyError: ('Revenue', 'Employees', 'Country')

# Ensure you use double square brackets [[]]
df[['Revenue', 'Employees','Country']]
# We can save this dataframe to another dataframe
df_new = df[['Revenue', 'Employees','Country']]
# Now when we call our new dataframe 
df_new
Revenue	Employees	Country
Apple	274515	147000	USA
Samsung	200734	267937	South Korea
Alphabet	182527	135301	USA
Foxconn	181945	878429	Taiwan
Microsoft	143015	163000	USA
Huawei	129184	197000	China
Dell Technologies	92224	158000	USA
Meta	85965	58604	USA
Sony	84893	109700	Japan
Hitachi	82345	350864	Japan
Intel	77867	110600	USA
IBM	73620	364800	USA
Tencent	69864	85858	China
Panasonic	63191	243540	Japan
# While our original dataframe remains the same
df
Revenue	Employees	Sector	Founding Date	Country
Apple	274515	147000	Consumer Electronics	01-04-1976	USA
Samsung	200734	267937	Consumer Electronics	13-01-1969	South Korea
Alphabet	182527	135301	Software Services	04-09-1998	USA
Foxconn	181945	878429	Chip Manufacturing	20-02-1974	Taiwan
Microsoft	143015	163000	Software Services	04-04-1975	USA
Huawei	129184	197000	Consumer Electronics	15-09-1987	China
Dell Technologies	92224	158000	Consumer Electronics	01-02-1984	USA
Meta	85965	58604	Software Services	04-02-2004	USA
Sony	84893	109700	Consumer Electronics	07-04-1946	Japan
Hitachi	82345	350864	Consumer Electronics	01-01-1910	Japan
Intel	77867	110600	Chip Manufacturing	18-07-1968	USA
IBM	73620	364800	Software Services	16-06-1911	USA
Tencent	69864	85858	Software Services	11-11-1998	China
Panasonic	63191	243540	Consumer Electronics	07-03-1918	Japan
d) Select Multiple Columns

Similar to the series, we can apply methods on our new dataframe.

new_df = df[['Revenue', 'Employees']] # we removed the Country column as it is not a numerical column
new_df.mean()
Revenue      124420.642857
Employees    233616.642857
dtype: float64
Check you knowledge:

3. Output a Series with the column Employees
employees_s = employees_s = df['Employees']
4. Output the median Employees to the nearest whole number
employees_median = df["Employees"].median()
5. Output the mean for columns Revenue andEmployees to the nearest whole number
...
r_e_mean = round(df[["Revenue", "Employees"]].mean())
Selection by Index - loc
loc[row_label, column_label]

# Find the revenue for Samsung 
​
# loc[row_label, column_label]
​
df.loc['Samsung','Revenue']
200734
Notice if we use : in place of row_label, it will return all the data from the specified column.

Thus, we have a Series

# loc[row_label, column_label]
​
df.loc[:,'Revenue']
Apple                274515
Samsung              200734
Alphabet             182527
Foxconn              181945
Microsoft            143015
Huawei               129184
Dell Technologies     92224
Meta                  85965
Sony                  84893
Hitachi               82345
Intel                 77867
IBM                   73620
Tencent               69864
Panasonic             63191
Name: Revenue, dtype: int64
Let's now use : in place of row_label or column_label

# row_label
df.loc['Samsung',:]
Revenue                        200734
Employees                      267937
Sector           Consumer Electronics
Founding Date              13-01-1969
Country                   South Korea
Name: Samsung, dtype: object
# column_label
df.loc[:, 'Revenue']
Apple                274515
Samsung              200734
Alphabet             182527
Foxconn              181945
Microsoft            143015
Huawei               129184
Dell Technologies     92224
Meta                  85965
Sony                  84893
Hitachi               82345
Intel                 77867
IBM                   73620
Tencent               69864
Panasonic             63191
Name: Revenue, dtype: int64
Let's select a list of values this time:

# Multiple columns
df.loc[['Apple','Samsung','Sony'], 'Revenue']
Apple      274515
Samsung    200734
Sony        84893
Name: Revenue, dtype: int64
# Multiple rows
df.loc['Apple', ['Employees','Country']]
Employees    147000
Country         USA
Name: Apple, dtype: object
rows = ['Apple','Samsung','Sony']
columns = ['Employees','Sector','Country']
# loc[row_label, column_label]
​
df.loc[rows,columns]
Employees	Sector	Country
Apple	147000	Consumer Electronics	USA
Samsung	267937	Consumer Electronics	South Korea
Sony	109700	Consumer Electronics	Japan
Slicing start:stop:step

a) With columns

df.loc['Apple', 'Employees':'Founding Date']
Employees                      147000
Sector           Consumer Electronics
Founding Date              01-04-1976
Name: Apple, dtype: object
b) With rows

df.loc['Apple':'Sony', 'Employees']
Apple                147000
Samsung              267937
Alphabet             135301
Foxconn              878429
Microsoft            163000
Huawei               197000
Dell Technologies    158000
Meta                  58604
Sony                 109700
Name: Employees, dtype: int64
c) With step

df.loc['Apple':'Sony':2, columns]
Employees	Sector	Country
Apple	147000	Consumer Electronics	USA
Alphabet	135301	Software Services	USA
Microsoft	163000	Software Services	USA
Dell Technologies	158000	Consumer Electronics	USA
Sony	109700	Consumer Electronics	Japan
d) With step and :

df.loc['Apple':'Sony':2, :]
Revenue	Employees	Sector	Founding Date	Country
Apple	274515	147000	Consumer Electronics	01-04-1976	USA
Alphabet	182527	135301	Software Services	04-09-1998	USA
Microsoft	143015	163000	Software Services	04-04-1975	USA
Dell Technologies	92224	158000	Consumer Electronics	01-02-1984	USA
Sony	84893	109700	Consumer Electronics	07-04-1946	Japan
Check you knowledge:

6. Using Index Selection, select the Revenue, Employees & Sector for the companies Apple, Alphabet and Microsoft
Include a step value in your output.

index_selection = df.loc['Apple':'Microsoft':2, 'Revenue':'Sector']
index_selection
Revenue	Employees	Sector
Apple	274515	147000	Consumer Electronics
Alphabet	182527	135301	Software Services
Microsoft	143015	163000	Software Services
Selection by Position - iloc
iloc[row_position, column_position]

The following output's for the code below will be the same as the loc examples.

# Find the revenue for Samsung 
df.iloc[1, 0]
200734
Notice if we use : in place of row_position, it will again return all the data from the specified column.

Thus, we have a Series

df.iloc[:,0]
Apple                274515
Samsung              200734
Alphabet             182527
Foxconn              181945
Microsoft            143015
Huawei               129184
Dell Technologies     92224
Meta                  85965
Sony                  84893
Hitachi               82345
Intel                 77867
IBM                   73620
Tencent               69864
Panasonic             63191
Name: Revenue, dtype: int64
Let's now use : in place of row_position or column_position

# row_position
df.iloc[1,:]
Revenue                        200734
Employees                      267937
Sector           Consumer Electronics
Founding Date              13-01-1969
Country                   South Korea
Name: Samsung, dtype: object
# column_position
df.iloc[:,0]
Apple                274515
Samsung              200734
Alphabet             182527
Foxconn              181945
Microsoft            143015
Huawei               129184
Dell Technologies     92224
Meta                  85965
Sony                  84893
Hitachi               82345
Intel                 77867
IBM                   73620
Tencent               69864
Panasonic             63191
Name: Revenue, dtype: int64
Let's select a list of values this time:

# Multiple columns
df.iloc[[0,1,8], 0]
Apple      274515
Samsung    200734
Sony        84893
Name: Revenue, dtype: int64
# Multiple rows
df.iloc[0, [1,4]]
Employees    147000
Country         USA
Name: Apple, dtype: object
rows_i = [0,1,8]
columns_i = [1,2,4]
df.iloc[rows_i,columns_i]
Employees	Sector	Country
Apple	147000	Consumer Electronics	USA
Samsung	267937	Consumer Electronics	South Korea
Sony	109700	Consumer Electronics	Japan
Slicing start:stop:step:

a) With columns

df.iloc[0, 1:4]
Employees                      147000
Sector           Consumer Electronics
Founding Date              01-04-1976
Name: Apple, dtype: object
b) With rows

df.iloc[0:8, 1]
Apple                147000
Samsung              267937
Alphabet             135301
Foxconn              878429
Microsoft            163000
Huawei               197000
Dell Technologies    158000
Meta                  58604
Name: Employees, dtype: int64
c) With step

df.iloc[0:9:2, columns_i]
Employees	Sector	Country
Apple	147000	Consumer Electronics	USA
Alphabet	135301	Software Services	USA
Microsoft	163000	Software Services	USA
Dell Technologies	158000	Consumer Electronics	USA
Sony	109700	Consumer Electronics	Japan
d) With step & :

df.iloc[0:9:2, :]
Revenue	Employees	Sector	Founding Date	Country
Apple	274515	147000	Consumer Electronics	01-04-1976	USA
Alphabet	182527	135301	Software Services	04-09-1998	USA
Microsoft	143015	163000	Software Services	04-04-1975	USA
Dell Technologies	92224	158000	Consumer Electronics	01-02-1984	USA
Sony	84893	109700	Consumer Electronics	07-04-1946	Japan
Check you knowledge:

7. Using Position Selection, select the Revenue, Employees & Country for the companies Samsung, Foxconn and Huawei.
position_selection = df.iloc[1:7:2, [0,1,4]]
position_selection
Revenue	Employees	Country
Samsung	200734	267937	South Korea
Foxconn	181945	878429	Taiwan
Huawei	129184	197000	China
# ### Exploring DataFrames with Currency Data
import pandas as pd
df = pd.read_csv("currencies.csv")
df
	Name	Symbol	Code	Countries	Digits	Number
0	UK Pound	£	GBP	Guernsey,Isle Of Man,Jersey,United Kingdom Of ...	2.0	826.0
1	Czech Koruna	Kč	CZK	Czechia	2.0	203.0
2	Latvian Lat	Ls	LVL	NaN	NaN	NaN
3	Swiss Franc	CHF	CHF	Liechtenstein,Switzerland	2.0	756.0
4	Croatian Kruna	kn	HRK	Croatia	2.0	191.0
5	Danish Krone	kr	DKK	Denmark,Faroe Islands (The),Greenland	2.0	208.0
6	Korean Won	₩	KRW	Korea (The Republic Of)	0.0	410.0
7	Swedish Krona	kr	SEK	Sweden	2.0	752.0
8	Turkish Lira	₤	TRY	Turkey	2.0	949.0
9	Hungarian Forint	Ft	HUF	Hungary	2.0	348.0
10	Brazilian Real	R$	BRL	Brazil	2.0	986.0
11	Lithuanian Litas	Lt	LTL	NaN	NaN	NaN
12	Bulgarian Lev	лB	BGN	Bulgaria	2.0	975.0
13	Polish Zloty	zł	PLN	Poland	2.0	985.0
14	US Dollar	$	USD	American Samoa,Bonaire, Sint Eustatius And Sab...	2.0	840.0
15	Russian Ruble	руб	RUB	Russian Federation (The)	2.0	643.0
16	Japanese Yen	¥	JPY	Japan	0.0	392.0
17	Romanian New Leu	lei	RON	Romania	2.0	946.0
18	Norwegian Krone	kr	NOK	Bouvet Island,Norway,Svalbard And Jan Mayen	2.0	578.0
19	Australian Dollar	$	AUD	Australia,Christmas Island,Cocos (Keeling) Isl...	2.0	36.0
20	Israeli Shekel	₪	ILS	Israel	2.0	376.0
21	New Zealand Dollar	$	NZD	Cook Islands (The),New Zealand,Niue,Pitcairn,T...	2.0	554.0
22	Indonesian Rupiah	Rp	IDR	Indonesia	2.0	360.0
23	Philippine Peso	₱	PHP	Philippines (The)	2.0	608.0
24	Euro	€	EUR	Andorra,Austria,Belgium,Cyprus,Estonia,Europea...	2.0	978.0
25	Canadian Dollar	$	CAD	Canada	2.0	124.0
26	Chinese Yuan	¥	CNY	China	2.0	156.0
27	Hong Kong Dollar	$	HKD	Hong Kong	2.0	344.0
28	Indian Rupee	₨	INR	Bhutan,India	2.0	356.0
29	Mexican Peso	$	MXN	Mexico	2.0	484.0
30	Malaysian Ringgit	RM	MYR	Malaysia	2.0	458.0
31	South African Rand	R	ZAR	Lesotho,Namibia,South Africa	2.0	710.0
32	Thai Baht	฿	THB	Thailand	2.0	764.0
33	Singaporean Dollar	$	SGD	Singapore	2.0	702.0

4. Count the number of unique currencies in the dataset

df.nunique()
nunique() method returns the number of unique values in each column. The Name column has 34 unique values, which means there are 34 unique currencies in the dataset.

Name         34
Symbol       25
Code         34
Countries    32
Digits        2
Number       32
dtype: int64

5.Identify the number of missing values in each column

Write the count of all missing values in the dataset.

df.isnull().sum()
isnull() method returns a boolean DataFrame indicating whether each value is missing or not. The sum() method sums the number of missing values in each column.

Name         0
Symbol       0
Code         0
Countries    2
Digits       2
Number       2
dtype: int64
Here we can see that the Countries, Digits, and Number columns have 2 missing values each. So the total number of missing values in the dataset is 6.

6. Determine the highest currency number in the dataset

Write answer in the form of a float. For example, if the answer is 98, write 98.0.

986.0

df['Number'].max()

7. Select Currency Names

Extract the 'Name' column from the DataFrame df and assign it to a new variable names.

names = df['Name']

8. Get the details of the 3rd row

Extract the 3rd row from the DataFrame df and assign it to a new variable row_3.

row_3 = df.loc[2]

9. Select rows 10 to 15 (inclusive) from the DataFrame

Select rows 10 to 15 (inclusive) from the DataFrame df and assign it to a new variable rows.

You were asked for rows, not index numbers.

rows = df.loc[9:14]
Output:

Name    Symbol  Code    Countries   Digits  Number
9   Hungarian Forint    Ft  HUF Hungary 2.0 348.0
10  Brazilian Real  R$  BRL Brazil  2.0 986.0
11  Lithuanian Litas    Lt  LTL NaN NaN NaN
12  Bulgarian Lev   лB  BGN Bulgaria    2.0 975.0
13  Polish Zloty    zł  PLN Poland  2.0 985.0
14  US Dollar   $   USD American Samoa,Bonaire, Sint Eustatius And Sab...   2.0 840.0
The argument [9:14] inside the .loc indexer specifies the range of rows to be selected. Note that the index labels start from 0, so rows 10 to 15 correspond to index labels 9 to 14.

10. Extract Alternating Rows from DataFrame

Your task is to extract the alternating rows from the DataFrame df, where "alternating" refers to the 1st, 3rd, 5th, and so on (not based on index order). Store these selected rows in a new variable named rows_every_other.

Correct!
rows_every_other = df.iloc[::2]

11. Select columns with indices 2, 4, and 5 from the DataFrame

Select columns with indices 2, 4, and 5 from the DataFrame df and assign it to a new variable cols.

cols = df.iloc[:, [2, 4, 5]]

12. Select the first three columns of the dataframe

Select the first three columns of the dataframe df and assign it to a new variable cols_first_three.

Dataframe columns also include index column.

The expected output looks like this (it contains more rows than below):

|    | Name               | Symbol   |
|---:|:-------------------|:---------|
|  0 | UK Pound           | £        |
|  1 | Czech Koruna       | Kč       |
|  2 | Latvian Lat        | Ls       |
|  3 | Swiss Franc        | CHF      |
|  4 | Croatian Kruna     | kn       |
|  5 | Danish Krone       | kr       |
|  6 | Korean Won         | ₩        |
Correct!
cols_first_three = df.iloc[:, :2]  #or cols_first_three = df.iloc[:,0:2]
# ### Exploring DataFrames with Pokemon data
What's the type of the variable defense_col?

type(df['Defense'])
pandas.core.series.Series
# In[2]:


## Project: Querying and Filtering Pokemon data

This project will help you practice your pandas querying and filtering skills. Let's begin!

<center>
<img src="./mikel-DypO_XgAE4Y-unsplash.jpg" >
    <p align="center">
        Photo by <a href="https://unsplash.com/@mykelgran?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Mikel</a> on <a href="https://unsplash.com/s/photos/pokemon?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>.
    </p>
</center>  


# ### Modifying DataFrames: creating columns and more
# 
# In this lab we'll cover the most common types of operations to "modify" dataframes. This includes:
# 
# Creating new columns
# Deletion: deleting rows or columns
# Modifications: renaming columns, changing column types, modifying values
# Adding new rows

# ##### Creating new columns
# We'll start with one of the most straightforward operations: creating new columns. We can create new columns in multiple ways, but let's start with the most common one:
# 
# **Expressions (and vectorized operations)**
The syntax is extremely intuitive, it's just assigning "the new column" to a given expression:

df["New Column Name"] = [EXPRESSION]
In this case, the expression can be anything. Examples:

# A  simple arithmetic expression between two columns
df["New Column Name"] = df["Column 1"] + df["Column 2"]

# A boolean expression
df["New Column Name"] = df["Column 1"] > 1000

# A more advanced expression multiple columns
df["New Column Name"] = df["Column 1"] * (df["Column 2"] / df["Column 3"]) / df["Column 4"].std
# In[3]:


import pandas as pd


# ![](tech_table.png)

# In[5]:


# Lists of data
data = {'Revenue': [274515,200734,182527,181945,143015,129184,92224,85965,84893,
                    82345,77867,73620,69864,63191],
        'Employees': [147000,267937,135301,878429,163000,197000,158000,58604,
                      109700,350864,110600,364800,85858,243540],
        'Sector': ['Consumer Electronics','Consumer Electronics','Software Services',
                   'Chip Manufacturing','Software Services','Consumer Electronics',
                   'Consumer Electronics','Software Services','Consumer Electronics',
                   'Consumer Electronics','Chip Manufacturing','Software Services',
                   'Software Services','Consumer Electronics'],
        'Founding Date':['01-04-1976','13-01-1969','04-09-1998','20-02-1974',
                         '04-04-1975','15-09-1987','01-02-1984','04-02-2004',
                         '07-04-1946','01-01-1910','18-07-1968','16-06-1911',
                         '11-11-1998','07-03-1918'],
        'Country':['USA','South Korea','USA','Taiwan','USA','China','USA','USA',
                   'Japan','Japan','USA','USA','China','Japan']} 
index = ['Apple','Samsung','Alphabet','Foxconn','Microsoft','Huawei',
         'Dell Technologies','Meta','Sony','Hitachi','Intel','IBM',
         'Tencent','Panasonic']


# In[7]:


df = pd.DataFrame(data , index= index)


# In[8]:


df


# ##### Creating new columns
# Expressions (and vectorized operations)

# In[11]:


#Let's use our sample DataFrame to calculate "Revenue per Employee" (as in the GIF above). The expression is just:

df["Revenue per Employee"] = df["Revenue"] / df["Employees"]
#We call these expressions "vectorized operations", as they act upon the whole dataframe, regardless if it has 100 rows, or 1 billion. Vectorized Operations are extremely fast, even with large number of data.
df["Revenue per Employee"]


# In[12]:


df.head()


# #### Activities
# ##### 1. Create a new column: `Revenue in $`
# The column Revenue is expressed in millions of dollars. Create a new one, Revenue in $ with the values for revenue expressed in $US Dollars (single units).

# In[13]:


df['Revenue in $'] = df["Revenue"] * 1_000_000
df.head()


# ##### 2. Create a new column: `Is American?`
# 

# In[15]:


df['Is American?'] = df['Country'] == 'USA'
df.head()


# **Columns out of Fixed Values**
# 
# 
# ***Single (hardcoded) value***
# We can create columns by also providing values directly. In its simplest form, we just assign the new column to a hardcoded value:
# 
# 
# `df["New Column"] = VALUE`
# This will set EVERY rows in the dataframe with that given value. In our notebook, we're setting the value `Is Tech?` to "Yes".
# 
# 
# ***Collection of values***
# Instead of providing just one value for the entire dataframe (and for every single row), we can provide a more "granular" collection containing the value for each row we want to assign.
# 
# 
# Let's look at the example in the associated notebook. In the variable stock_prices we're storing the stock prices of the given companies. We'll then assign the values to the column "Stock Price" directly:
# 
# 
# `stock_prices = [143.28, 49.87, 88.26, 1.83, 253.75, 0,
#                 43.4, 167.32, 89.1, 52.6, 25.58, 137.35, 48.23, 8.81]
# 
# df['Stock Price'] = stock_prices`
# 
# 
# This works because the list stock_prices contains the same number of elements as in the DataFrame.
# 
# 
# Note: The stock prices here are estimate. Not all companies are listed in the same exchange, so we just estimated the value in dollars. Also, Huawei is not publicly listed, so we assigned a value of $0.

# #### Creating Columns out of Fixed Values
# #### A single value

# In[29]:


df['Is Tech?'] = "Yes"
df.head()


# #### Collection of values

# In[17]:


stock_prices = [143.28, 49.87, 88.26, 1.83, 253.75, 0,
                43.4, 167.32, 89.1, 52.6, 25.58, 137.35, 48.23, 8.81]
df['Stock Price'] = stock_prices
df.head()


# ##### 3. Create a new column: `CEO`

# In[26]:


ceo_list = [
    "Tim Cook", "Kim Ki Nam", "Sundar Pichai",
    "Young Liu", "Satya Nadella", "Ren Zhengfei",
    "Michael Dell", "Mark Zuckerberg",
    "Kenichiro Yoshida", "Toshiaki Higashihara", "Patrick Gelsinger",
    "Arvind Krishna", "Ma Huateng", "Yuki Kusumi"]
#Create a new column with the CEOs of each company
df['CEO'] = ceo_list
df.head()


# ### Deleting Columns with `del`
# 
# Deleting Columns
# There are mainly two ways of deleting columns, using the del keyword and with the drop method. For now we'll focus only on the del keyword as the drop method introduces a few more complexities that we'll need to address later.
# 
# The del keyword is the simplest and most intuitive expression, just: del df["Column Name"]. It will modify the underlying dataframe, so use it carefully!
# 
# For example, let's delete the column Is Tech? that we created before.

# In[30]:


del df["Is Tech?"]
df.head()


# ##### 4. Delete the column `CEO`

# In[28]:


del df['CEO']
df.head()

Mutability and Immutability

This is a VERY important concept in Data Science (and programming in general). When solving problems, we usually have the option to resolve them with a "mutable" solution, that is, modifying (or mutating) the underlying dataframe, or with an immutable solution, which performs the changes but without modifying the underlying data.

For example, most of the String methods in Python are immutable. You can perform a wide variety of operations (replace, title, upper, lower, etc) but the original string is NOT modified, these operations return NEW strings (new copies) with the desired changes applied. Take a look at the notebook for a few of these strings examples and pay attention at how the string s is not modified after any of the operations.

Favor Immutability
Python's decision for strings (and other, non mentioned modules) is not a coincidence. Most of the time (and only under rare exceptions), we should prefer immutable solutions. Specially in Pandas, operations that don't modify the underlying DataFrames or Series. That way, you can always safely try things without the risk of losing important data.

Here's an example of the flow you should expect when performing immutable operations (don't worry about the methods below, we'll learn about them in this and other projects):

df = df.read_csv()
df_renamed = df.rename(...) # rename columns
df_notna = df_renamed.dropna(...)  # dropping null values
df_cleaned = df_notna.drop(...)  # dropping some values
As you can see, the result of each operation is the "entry point" of the following operation, creating a chain. This is intentional, because, as you'll see, we'll use this "chaining" to our advantage. It's pretty common to see expressions that are a combination (chaining) of multiple methods one after the other:

df.dropna().drop([...]).rename([...]).sort_values().head()
The inplace parameter
Before moving forward, we need to make a special mention about the inplace parameter.

The inplace parameter is EVERYWHERE in pandas methods, both for DataFrames and Series. For example, df.dropna(inplace=True), df.drop([...], inplace=True), df.drop_duplicates(inplace=True), etc.

The inplace parameter changes the behavior of a given method from immutable (default) to mutable, modifying the underlying DataFrame. Again, by default, inplace is always False, as Pandas is always favoring immutability. You can alter that behavior by setting inplace=True, although, as we just mentioned, it's NOT recommended, except in some special cases.

Now, let's move to the next section to put these concepts to use!

Deleting rows
The method to delete "arbitrary" rows is: .drop. It has some variations, as it can also be used to delete columns, but let's start with the basics.

The .drop() method accepts the indices of the values we want to remove, and as we previously mentioned, by default, is immutable.

In the notebook you can see an example of deleting multiple rows:

df.drop(["Microsoft", "Tencent", "Samsung", "Alphabet", "Meta", "Hitachi", "Apple"])
Again, this method is IMMUTABLE. It doesn't modify the underlying dataframe: it immediately returns a new DataFrame with the modifications done. The common pattern is to assign the results of .drop to a variable: df_new = df.drop(...). This allows us to re-play any operation if we find a mistake in the process.
# ### Mutability vs Immutability

# In[37]:


s = "Hello World"


# In[38]:


s.replace("World", "Datawars")
s # not modified


# In[39]:


s.lower()


# In[40]:


# not modified
s


# ### Deleting rows

# In[44]:


df.drop(["Microsoft", "Tencent", "Samsung", "Alphabet", "Meta", "Hitachi", "Apple"])


# In[42]:


# the underlying `df` has not changed
df.head()


# ##### 5. Drop Microsoft from the `df`

# In[47]:


df_no_windows = df.drop(['Microsoft'])
df_no_windows


# #### Mutable modification with `inplace`

# In[48]:


# This method produces no result
df.drop("Huawei", inplace=True)


# In[49]:


df #modifies


# ##### 6. Delete *inplace* the values for IBM and Dell

# In[50]:


df.drop(['IBM','Dell Technologies'], inplace = True)


# ### Deleting rows based on a condition

# In[51]:


df.sort_values(by='Revenue').head()


# We need to replicate the following expression using conditions:

# In[52]:


df.drop(["Intel", "Tencent", "Panasonic"])


# The condition:
# > Companies with less than M$80,000

# In[53]:


df.loc[df["Revenue"] < 80_000]


# The companies that match that query:

# In[54]:


df.loc[df["Revenue"] < 80_000].index


# The resulting `.drop()` expression:

# In[56]:


# .drop() and .sort_values() chaining in action
df.drop(df.loc[df["Revenue"] < 80_000].index).sort_values(by='Revenue')


# ##### 7. Delete companies with revenue lower than the mean

# In[58]:


df_high_revenue = df.drop(df.loc[df['Revenue'] < df['Revenue'].mean()].index)


# ##### 8. Drop the companies that are NOT from the USA

# In[ ]:


df_usa_only = df.drop(df.loc[df['Country'] != 'USA'].index)


# ##### 9. Japanese companies sorted by Revenue (desc)
# Using chaining methods, perform the following two operations in the same expression: * drop all the companies that are NOT Japanese * sort them by Revenue in descending order
# 
# Store your results in the variable df_jp_desc

# In[59]:


df_jp_desc = df.drop(df.loc[df['Country'] != 'Japan'].index).sort_values(by = 'Revenue', ascending = False)


# In[60]:


df_jp_desc


# In[63]:


df


# **Removing columns with .drop()**
# 
# Finally, it's worth mentioning that the .drop() method can be used to delete columns as well, as an immutable alternative to del. The syntax is the same as removing rows, but to indicate that we want to delete columns, we must pass axis=1 as a parameter. By default, the axis parameter is 0, which means "delete at row level"; by setting it to 1 we're indicating we're deleting columns.

# In[65]:


#### Removing columns with Drop
df.drop(['Revenue', 'Employees'], axis=1)


# In[66]:


df


# In[67]:


df.columns


# ### Mastering DataFrame Mutations with Hollywood data
In this lab, you'll engage with an actual dataset, applying various data manipulation techniques. Our focus will be a movie dataset, including information like the film title, release year, budget, gross earnings, and more.

This lab will cover the following:

Creating new columns in a data frame by doing basic arithmetic operations (addition, subtraction, division) on existing columns.
Creating new columns in a data frame by applying boolean operations (less-than <, greater-than >, equals ==, etc.) on existing columns.
Deleting rows based on specific conditions.
Removing single or multiple columns based on conditions.
Let's start by loading our dataset.

You can do this by using the pandas.read_csv() function to load the dataset into a pandas dataframe. Afterwards, store the dataframe in a variable nameddf.

Here is a sample code:

import pandas as pd

df = pd.read_csv("movies.csv")
df
IMPORTANT NOTE: Please ensure you complete all activities in the lab in sequence. Each activity builds on the one before it, so skipping an activity will prevent further progress. Complete each task fully before moving on to the next for a successful learning experience.

Activities
1
Create a new column revenue

Add a new column named revenue to the dataframe df. This new column should reflect the difference between the values in the gross and budget columns.

Correct!
df["revenue"] = df["gross"] - df["budget"]
In the above solution, new column revenue is created using basic arithmetic operation from gross and budget.

2
Create a new column percentage_profit

Create a new column called percentage_profit. You will calculate its values as the proportion of the gross earnings out of the total revenue for each row. For example, if the gross earning is 100 million out of a total revenue of 200 million, the percentage_profit will be 50%.

Express profit percentage as a percentage.
Correct!
df["percentage_profit"] = (df["revenue"] / df["gross"]) * 100
In the above solution, new column percentage_profit is created using basic arithmetic operation from revenue and gross.

3
Create a new column high_budget_movie

Add a new column named high_budget_movie to dataframe df. This column should label each movie with True if it has a budget over 100 million, or False if it does not.

Correct!
The activity requires to create a column that will be of "boolean" type; that is: it'll have values that are True or False.

We can use the result of boolean expressions to create new columns. But first, take a look at the boolean expression alone:

>>> df["budget"] > 100000000
0    False
1    False
2    False
3    False
4    False
...
This expression results in a boolean Series with the same index as the movies, so we can just use it to create the new column by assigning the expression to the column name:

df["high_budget_movie"] = df["budget"] > 100000000
4
Create a new column successful_movie

Add a new column named successful_movie. Assign it the value True if a movie's percentage_profit exceeds 50. If it doesn't, assign False.

Correct!
df["successful_movie"] = df["percentage_profit"] > 50
5
High-Rated Movies

Create a new column called high_rated_movie. If the movie's score is more than 8, label it as True. If not, label it as False.

Correct!
df["high_rated_movie"] = df["score"] > 8
6
Create a new column is_new_release

Create a new column named is_new_release. This column should indicate True if the year column's value is beyond 2020, and False if it's not.

Correct!
df["is_new_release"] = df["year"] > 2020
7
Create a new column is_long_movie

Create a new column is_long_movie which is True if the value of runtime column is greater than 150 minutes and False otherwise.

Correct!
df["is_long_movie"] = df["runtime"] > 150
8
Drop unsuccessful movie.

Delete all rows in the dataframe df where the column successful_movie is labeled as False. Use the inplace attribute to make sure these modifications are permanent.

Correct!
index_to_drop = df[df["successful_movie"] == False].index
df.drop(index_to_drop, inplace=True)
The code above is written in Python and performs the following operations on a DataFrame object (df):

It creates a new variable named "index_to_drop", which contains the indices (row labels) of all the rows in the DataFrame where the value in the "successful_movie" column is False. This is done by using boolean indexing on the "successful_movie" column and selecting all the rows where the value is False.
It then uses the drop method to remove the rows with indices specified in the "index_to_drop" variable from the DataFrame. The inplace parameter is set to True, which means that the changes to the DataFrame will be made in place, without creating a copy of the DataFrame.
So, this code will remove all the movies from the DataFrame that have a "successful_movie" value of False, which means that the movie did not generate positive revenue. The result will be a DataFrame that only contains the successful movies, based on the definition of success used in the "successful_movie" column.

Another way to do this is to use boolean indexing directly on the DataFrame, without creating a separate variable for the indices:

df.drop(df[df["successful_movie"] == False].index, inplace=True)
You can also use the following code to achieve the same result:

df = df[df["successful_movie"] == True]
In this case, the boolean indexing is used to select all the rows where the "successful_movie" value is True, and the result is assigned back to the df variable, which overwrites the original DataFrame with the filtered DataFrame.

All of these approaches will result in the same DataFrame, which contains only the successful movies.

9
Drop high budget movie

Create a new dataframe named low_budget_df by removing all rows from the original dataframe where the budget value exceeds 100 million. Remember, changes shouldn't affect the original dataframe.

Correct!
Solution 1: Using boolean indexing to select rows with budget <= 100 million

low_budget_df = df[df["budget"] <= 100000000]
Solution 2: Using df.drop() to drop rows with budget > 100 million and store the result in a new dataframe

low_budget_df = df.drop(df[df["budget"] > 100000000].index)
In the first solution, the code df[df["budget"] <= 100000000] selects all the rows where the "budget" value is less than or equal to 100 million, and the result is stored in the low_budget_df variable.

The second solution uses the df.drop() method to drop rows where the "budget" value is greater than 100 million. The df.drop() method takes the index of the rows to be dropped as an argument, which is obtained by using boolean indexing on the "budget" column. The resulting dataframe with the dropped rows is then stored in the low_budget_df variable.

Regardless of the solution used, after dropping the rows, the new dataframe should have 2305 rows and 22 columns. It's important to note that in both solutions, the original dataframe df remains unchanged, and the filtered dataframe is stored in the low_budget_df variable as requested.

10
Removing Low-Voted Movies

Remove all the rows from the dataframe where the votes count is below 1000. Assign this updated dataframe to a new variable named high_voted_df. Ensure you do not make these changes to the original dataframe.

Correct!
high_voted_df = df.drop(df[df["votes"] < 1000].index)
11
Drop the column budget

To delete the budget column from the movie dataframe, apply the drop method and include the column's name, budget. Make sure to specify the axis to show you're referring to a column, not a row. Also, set the inplace parameter to True so the change isn't temporary but permanent.

Correct!
df.drop("budget", axis=1, inplace=True)
12
Drop the director and writer columns from the dataframe.

Remove the director and writer columns from the dataframe df. To do this, employ the drop method, designating director and writer as the column names. Set the axis to confirm that these are columns not rows. Make sure to adjust the inplace parameter to False, this way you're forming a new dataframe named new_df without altering the original one.

Please remember, in this activity, your task is to build a new dataframe named new_df.

Correct!
new_df = df.drop(['director', 'writer'], axis=1, inplace=False)
Here's what's happening in the above code in step by step:

The drop method is called on the df DataFrame object, with the following parameters:
['director', 'writer']: This is a list of the column labels (names) of the columns to be removed.
axis=1: This specifies that the columns, not rows, are to be removed.
inplace=False: This specifies that a copy of the DataFrame, rather than modifying the original DataFrame in place, should be returned.
The result of the drop method is assigned to the new DataFrame object named "new_df".
So, this code will create a new DataFrame that is identical to the original DataFrame, except that it does not contain the "director" and "writer" columns. The original DataFrame is not modified by this operation.

13
Drop Out Low-Rated and Low-Voted Movies

Drop all the rows where the value of score is less than 5 and the value of votes is less than 1000. Drop the rows from the original dataframe df.

Correct!
df.drop(df[(df["score"] < 5) & (df["votes"] < 1000)].index, inplace=True)
14
Top High-Rated Movies

Create a new DataFrame named top_rated_movies, which should include the top five highly-rated movies. Sort this DataFrame based on the score column in descending order.

Correct!
top_rated_movies = df.sort_values(by="score", ascending=False).head(5)
15
Removing Specific Rows

Remove rows with index 2 and 10 from the DataFrame df.

Correct!
rows_to_drop = [2, 10]
df = df.drop(index=rows_to_drop)
16
Sci-Fi Blockbusters

Create a new DataFrame named sci_fi_blockbusters containing movies that are 'Sci-Fi' genre and have a gross greater than $150 million.

Correct!
sci_fi_blockbusters = df[(df['genre'] == 'Sci-Fi') & (df['gross'] > 150000000)]
17
Age of Movies

Create a new column named age to calculate the age of the movie in years. Find it by subtracting the year column from the current year.

Consider 2023 as the current year.

Correct!
df["age"] = 2023 - df["year"]
18
Movies Released in Summer

Create a new DataFrame containing movies released in June, July, or August. Store the result in dataframe summer_movies.

Correct!
summer_movies = df[df['released'].str.contains('June|July|August')]
# ### Mastering DataFrame Mutations with Wine Quality data

# In[1]:


import pandas as pd
import numpy as np


# In[3]:


wine_quality_df = pd.read_csv('winequality-red.csv', sep= ';')
wine_quality_df.head()


# In[4]:


wine_quality_df.info()


# In[5]:


wine_quality_df.describe()


# To ensure the integrity of our original data set, it's a best practice to work with a copy of the data frame when performing data manipulation. By creating a copy, we can freely experiment with various techniques and make modifications without affecting the original data. This way, we can have peace of mind knowing that the original data set remains untouched.

# In[6]:


df = wine_quality_df.copy()


# **1. What is maximum amount of citric acid in the wine? Enter the answer to 1 decimal point.**

# In[7]:


df['citric acid'].max()


# **2. How many missing values are in the dataset?**

# In[8]:


df.isna().sum()


# **3. What is the median wine quality?**

# In[9]:


df['quality'].median()


# #### Row and Column modification
# **4. Rename the columns to have underscore instead of space. For example old name: fixed acidity to new name: fixed_acidity**

# In[10]:


df.rename(columns = {"fixed acidity": "fixed_acidity",
                     "volatile acidity": "volatile_acidity",
                     "citric acid": "citric_acid",
                     "residual sugar": "residual_sugar",
                     "fixed acidity": "fixed_acidity",
                     "free sulfur dioxide": "free_sulfur_dioxide",
                     "total sulfur dioxide": "total_sulfur_dioxide"}, inplace = True)


# In[11]:


df.columns = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
       'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
       'pH', 'sulphates', 'alcohol', 'quality']


# **5. Drop the first and last row**

# In[12]:


df_first_last = df.drop([df.index[0], df.index[-1]])


# **6. The dataset contains an outlier. Remove the row where that contains the maximum total sulfur dioxide.**

# In[13]:


df_drop = df.drop(df[df['total_sulfur_dioxide'] == df['total_sulfur_dioxide'].max()].index)


# **7. We notice that all the datatypes are float besides the quality column. Convert the column to float datatype**

# In[14]:


df["quality_float"] = df['quality'].astype(float)


# **8. Remove these columns from the dataset**

# In[15]:


df_drop_three = df.drop(['density','residual_sugar','chlorides'], axis = 1)


# ##### Basic Column operations
# **9.  Create a new column that calculates the alcohol content in terms of percentage (%)**

# In[16]:


df['alcohol_perc'] = (df['alcohol'] / df['alcohol'].max())*100


# **10. Create a new column in the data frame that contains the sum of sulfates and citric_acid for the red wine.**

# In[17]:


df['sulphate_citric_acid']= df['sulphates'] + df['citric_acid']


# **11. Create a new column that where the alcohol content is less than its mean.**

# In[18]:


df['deviation_alcohol'] = df['alcohol']< df['alcohol'].mean()
df['deviation_alcohol']


# **12. Convert the wine quality scores into categorical labels: "low", "medium", "high".**

# In[19]:


df['quality_label'] = ['low' if x <= 5 else 'medium' if x <= 7 else 'high' for x in df['quality']]


# **13. Create a new column that calculates the ratio of free sulfur dioxide to total sulfur dioxide.**

# In[20]:


df['free_total_ratio'] = df['free_sulfur_dioxide'] / df['total_sulfur_dioxide']


# #### Practice DataFrame Mutations using Airbnb Data
### Look at the dataset

# importing pandas library
import pandas as pd

df = pd.read_csv('AB_US_2023.csv', low_memory=False, parse_dates=['last_review'])
df.head()

df.columns

df.shape

df.info()

df.describe()

df.isnull().sum()

### Activities

##### 1. Create a New Column `price_per_minimum_stay`

df['price_per_minimum_stay'] = df['price'] * df['minimum_nights']

##### 2. Delete all rows where the price is greater than $500

df = df[df['price'] <= 500]
#df.drop(df[df['Price'] > 500].index, inplace=True)


##### 3. Delete the `host_name` and `neighbourhood_group` columns from the DataFrame `df`

df = df.drop(['host_name', 'neighbourhood_group'], axis=1)

##### 4. Rename the column `number_of_reviews` to `reviews_count`

df = df.rename(columns={'number_of_reviews': 'reviews_count'})

##### 5. Convert the `price` column from integer to float data type

df['price'] = df['price'].astype(float)

##### 6. Replace all occurrences of `Private room` in the `room_type` column with `Private`

df.loc[df['room_type'] == 'Private room', 'room_type'] = 'Private'
#df['room_type'] = df['room_type'].replace('Private room', 'Private')
#In this solution, df['room_type'] == 'Private room' creates a boolean mask that identifies the rows where the 'room_type' column has the value 'Private room'. Then, df.loc[boolean_mask, 'room_type'] selects the subset of the 'room_type' column where the boolean mask is True. Finally, we assign the value 'Private' to this subset, effectively replacing 'Private room' with 'Private'.


##### 7. Add new row with the given details

new_row_data = {'id': 851792795339743534, 'name': 'Tony Stark Apartment', 'host_id': 67890, 'room_type': 'Entire home/apt',
                'price': 150, 'minimum_nights': 3, 'reviews_count': 10}

new_row = pd.DataFrame(new_row_data, index=[len(df)])
df = pd.concat([df, new_row])
#In this solution, we create a new DataFrame new_row with the new row data and index. Then, we use pd.concat() to concatenate the two DataFrames df and new_row along the row axis.

##### 8. Remove the `availability_365` column from the DataFrame without creating a new DataFrame

df.drop('availability_365',axis = 1, inplace = True)

##### 9. Sort the DataFrame by the `price` column in descending order

sorted_df = df.sort_values('price', ascending = False)

##### 10. Convert all prices from US dollars to euros

df['price_eur'] = df['price']*0.85

##### 11. Modify the `price_per_minimum_stay` by doubling the rates.

df['price_per_minimum_stay'] = df['price_per_minimum_stay'] * 2

##### 12. Create a new column named `year` that contains the year information from the `last_review` column

df['year'] = df['last_review'].dt.year
# In[ ]:




