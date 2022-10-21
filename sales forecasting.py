#!/usr/bin/env python
# coding: utf-8

# In[2]:


#lets import the libraries 
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[8]:


#lets import the dataset
df=pd.read_excel(r"file:///C:\Users\namtala\AppData\Local\Temp\Temp1_Sales-Analysis-master.zip\Sales-Analysis-master\superstore_sales.xlsx")


# In[10]:


#shows you the first five rows of the dataset
df.head()


# In[17]:


#gives you a summary of the dataset
df.info()


# In[14]:


df.shape


# In[12]:


#shows you the last five rows of the dataset
df.tail()


# In[16]:


#checks the column names
df.columns


# In[18]:


#check for missing values in the dataset
df.isnull()


# In[19]:


#shows exact number of missing values in each column
df.isnull().sum()


# In[20]:


#lets get the descriptive statistics of the dataset
df.describe()


# In[21]:


#lets perform some exploratory data analysis
df['order_date'].min()


# In[22]:


df['segment'].count()


# In[25]:


df['segment'].max()


# In[27]:


#getting month year from the dataset
df['month_year']=df['order_date'].apply(lambda x: x.strftime('%Y-%m'))


# In[44]:


#Grouping month year and getting total sales for each
df_trend=df.groupby('month_year').sum()['sales'].reset_index()


# In[49]:


#setting the figure size and then rotating it in order to see the values
plt.figure(figsize=[20,8])
plt.plot(df_trend['month_year'], df_trend['sales'],color='r')
plt.xticks(rotation='vertical',size=8)
plt.show()


# In[60]:


#which are the top ten products by sale
#first group the product name column by sales and then make it a dataframe, and then put it in an object called product sales
prod_sales=pd.DataFrame(df.groupby(['product_name']).sum()['sales'])


# In[73]:


#sort dataframe in descending order using sales
prod_sales=prod_sales.sort_values('sales',ascending=False)


# In[74]:


#fetch the top ten most selling products
prod_sales[:10]


# In[9]:


#the most selling products
#group the product name column by quantity and then convert it into a dataframe
most_sell_prod=pd.DataFrame(df.groupby(['product_name']).sum()['quantity'])


# In[12]:


#sort the most sold product
most_sell_prod=most_sell_prod.sort_values('quantity',ascending=False)


# In[13]:


#top 10 most selling products
most_sell_prod[:10]


# In[16]:


#plot the most preferred ship mode
plt.figure(figsize=(15,9))
sns.countplot(df['ship_mode'])
plt.show()


# In[20]:


# group by most profitable category and subcategory and then put the dataframe in an object
cat_subcat_profit=pd.DataFrame(df.groupby(['category','sub_category']).sum()['profit'])


# In[21]:


#sort the result
cat_subcat_profit.sort_values(['category','profit'],ascending=False)


# In[ ]:




