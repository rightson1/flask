
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer

import warnings
warnings.filterwarnings('ignore')
housing =pd.DataFrame(pd.read_csv("house_listingskenya.csv"))

housing.head()
housing.shape
housing.info()
housing.describe()
housing.isnull().sum()
housing['price'].fillna(housing['price'].mean(), inplace=True)
housing.info()

condition = housing['bedrooms'] < 30
df = housing[condition]

condition2 = df['bathrooms'] < 15
condition3 = df['parking'] < 20
condition4 = df['toilets'] < 10

combined_condition = condition2 & condition3 & condition4
df = df[combined_condition]


# In[14]:


new_data = pd.DataFrame({'type': ['House'], 'bedrooms': [5],
                         'category': ['For_Rent'], 'state': ['Kajiado'], 'locality': ['Kitengela'],
                         'bathrooms': [5], 'toilets': [0], 'furnished': [0], 'serviced': [0], 'shared': [0],
                         'parking': [0], 'sub_type': ['Unknown'], 'listmonth': [7.0], 'listyear': [2020.0]
})

if 'For_Rent' in new_data['category'].values:
    df = df.loc[df['price'] < 1000000]
    print("final1")
else:
    df = df.loc[df['price'] < 200000000]
    print("final2")

def preprocess_new_data(type_, bedrooms, category, state, locality, bathrooms, toilets, furnished, serviced, shared, parking, sub_type, listmonth, listyear):
    new_data = pd.DataFrame({
        'type': [type_],
        'bedrooms': [bedrooms],
        'category': [category],
        'state': [state],
        'locality': [locality],
        'bathrooms': [bathrooms],
        'toilets': [toilets],
        'furnished': [furnished],
        'serviced': [serviced],
        'shared': [shared],
        'parking': [parking],
        'sub_type': [sub_type],
        'listmonth': [listmonth],
        'listyear': [listyear]
    })
    
    return new_data
df.shape


df.isnull().sum()


# In[18]:


del df["id"]
del df["price_qualifier"]
del df["sub_locality"]


# In[19]:


df.shape


# In[20]:


df['toilets'].fillna(df['toilets'].mean(), inplace=True)


# In[21]:


df.isnull().sum()


# In[22]:


df.shape


# In[23]:


df['sub_type'].fillna('Unknown', inplace=True)
df['locality'].fillna('Unknown', inplace=True)


# In[24]:


df.isnull().sum()


# In[25]:


from datetime import datetime
df['listdate'] = pd.to_datetime(df['listdate'])
df['listyear'] = df['listdate'].dt.year.astype(float)
df['listmonth'] = df['listdate'].dt.month.astype(float)


# In[26]:


del df["listdate"]


# In[27]:


df['listyear'].fillna(df['listyear'].mean(), inplace=True)
df['listmonth'].fillna(df['listmonth'].mean(), inplace=True)


# In[28]:


label_encoder = LabelEncoder()
categorical_columns = ['category','sub_type','locality','type','state',]

for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])


# In[29]:


condition2= df['category']<2
condition3= df['type']<3
condition4= df['shared']<1



combined_condition= condition2&condition3&condition4
df= df[combined_condition]


import pandas as pd
y = df['price']
x = df.drop('price', axis=1)


# In[32]:


from sklearn.model_selection import train_test_split

x_train,x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=45)


# In[33]:


from sklearn.preprocessing import Normalizer

scaler = Normalizer()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


label_encoder = LabelEncoder()
categorical_columns = ['category','sub_type','locality','type','state']

for column in categorical_columns:
    new_data[column] = label_encoder.fit_transform(new_data[column])


# In[36]:




scaler = Normalizer()
new_data = scaler.fit_transform(new_data)


# In[37]:


import xgboost as xgb
from sklearn.metrics import mean_squared_error


# In[38]:


model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(x_train, y_train)


# In[39]:


y_pred = model.predict(x_test)


# In[40]:


mse = mean_squared_error(y_test, y_pred)
# print("Mean Squared Error:", mse)


# In[41]:


rmse = np.sqrt(mse)


# In[42]:


# print(rmse)


# In[43]:


prediction= model.predict(new_data)


# In[44]:


print(prediction)


# In[ ]:





# In[ ]:




