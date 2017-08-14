
# coding: utf-8

# # Online Retail
# 
# - http://archive.ics.uci.edu/ml/datasets/online+retail#
# 
# 
# ## Data Set Information:
# 
# This is a transnational data set which contains all the transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail.The company mainly sells unique all-occasion gifts. Many customers of the company are wholesalers.
# 
# 
# ## Attribute Information:
# 
# - InvoiceNo: Invoice number. Nominal, a 6-digit integral number uniquely assigned to each transaction. If this code starts with letter 'c', it indicates a cancellation. 
# - StockCode: Product (item) code. Nominal, a 5-digit integral number uniquely assigned to each distinct product. 
# - Description: Product (item) name. Nominal. 
# - Quantity: The quantities of each product (item) per transaction. Numeric.	
# - InvoiceDate: Invice Date and time. Numeric, the day and time when each transaction was generated. 
# - UnitPrice: Unit price. Numeric, Product price per unit in sterling. 
# - CustomerID: Customer number. Nominal, a 5-digit integral number uniquely assigned to each customer. 
# - Country: Country name. Nominal, the name of the country where each customer resides.
# 
# 

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
import itertools
import operator
import statsmodels.api as sm


# In[2]:

online_retail = pd.read_excel('data/Online Retail.xlsx')


# In[3]:

online_retail.describe()


# In[4]:

online_retail.head()


# In[5]:

online_retail['InvoiceDate'] = online_retail['InvoiceDate'].astype('datetime64[ns]')


# In[6]:

online_retail.head()


# In[7]:

online_retail.info()


# In[8]:

(online_retail['CustomerID'].isnull()).any()


# In[9]:

online_retail[online_retail['CustomerID'].isnull()]


# In[10]:

#calculate revenue? total sum of the price
online_retail.set_index('InvoiceDate', inplace=True)


# In[11]:

# http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
y = online_retail['Quantity'].resample('MS').sum()


# In[12]:

y = y.fillna(y.bfill())


# In[13]:

y.head()


# In[15]:

y.isnull().any()


# In[16]:

y.plot(figsize=(15,6))
plt.show()


# In[17]:

p = d = q = range(0, 2)

pdq = list(itertools.product(p, d, q))

# try adjust the `s` parameter
s = 1
seasonal_pdq = [(x[0], x[1], x[2], s) for x in list(itertools.product(p, d, q))]


# In[18]:

print('Example of parameter conbination for Seasonal ARIMA')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[2]))


# In[19]:

warnings.filterwarnings('ignore')
history = {}
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                           order=param,
                                           seasonal_order=param_seasonal,
                                           enforce_stationarity=False,
                                           enforce_invertibility=False)
            
            results = mod.fit()
            history[(param, param_seasonal)] = results.aic
            print('ARIMA{}x{} - AIC: {}'.format(param, param_seasonal, results.aic))
        except:
            continue


# Get the combination that results the minimum AIC

# In[20]:

sorted_x = sorted(history.items(), key=operator.itemgetter(1))


# In[21]:

param, param_seasonal =  sorted_x[0][0][0], sorted_x[0][0][1]


# In[22]:

print(param)
print(param_seasonal)


# In[23]:

model = sm.tsa.statespace.SARIMAX(y,
                         order = param,
                         seasonal_order=param_seasonal,
                         enforce_stationarity=False,
                         enforce_invertibility=False)

results = model.fit()


# In[24]:

print(results.summary())


# In[25]:

# http://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAXResults.plot_diagnostics.html
results.plot_diagnostics(lags=1, figsize=(15,6))
plt.show()


# ## Validating Forecasts
# 
# - one-step ahead forecast
# - dynamic forecast

# ### One-step ahead forecast

# In[26]:

start_date = '2011-10-01'
pred = results.get_prediction(start=pd.to_datetime(start_date), dynamic=False)


# In[27]:

pred_ci = pred.conf_int()


# In[28]:

ax = y['2011':].plot(label='observed')

pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)

ax.fill_between(pred_ci.index,
               pred_ci.iloc[:, 0],
               pred_ci.iloc[:, 1], color='k',
               alpha=.2)

ax.set_xlabel('Date')
ax.set_ylabel('Total Unit Sales')

plt.legend()

plt.show()


# In[29]:

y_forecasted = pred.predicted_mean
y_truth = y[start_date:]

mse = ((y_forecasted - y_truth) ** 2).mean()

print('The Mean Squared Error of our forecasts is {}'.format(round(mse,2)))


# ### Dynamic forecast

# In[30]:

pred_dynamic = results.get_prediction(start=pd.to_datetime(start_date), dynamic=True)

pred_dynamic_ci = pred_dynamic.conf_int()


# In[31]:

ax = y['2011':].plot(label='observed')

pred_dynamic.predicted_mean.plot(ax=ax, label='Dynamic Forecast', alpha=.7)

ax.fill_between(pred_dynamic_ci.index,
               pred_dynamic_ci.iloc[:, 0],
               pred_dynamic_ci.iloc[:, 1], color='k',
               alpha=.2)

ax.fill_betweenx(ax.get_ylim(), pd.to_datetime(start_date), y.index[-1], alpha=.1, zorder=-1)

ax.set_xlabel('Date')
ax.set_ylabel('Total Unit Sales')

plt.legend()

plt.show()


# In[32]:

y_forecasted = pred_dynamic.predicted_mean
y_truth = y[start_date:]
mse = ((y_forecasted - y_truth) ** 2).mean()

print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))


# The dynamic forecast results lower MSE than the one-step ahead

# ## Visualizing Forecasts

# In[35]:

pred_uc = results.get_forecast(steps=2)

pred_ci = pred_uc.conf_int()


# In[36]:

ax = y.plot(label='observed', figsize=(20,15))

pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
               pred_ci.iloc[:, 0],
               pred_ci.iloc[:, 1],
               color='k',
               alpha=.25)

ax.set_xlabel('Date')
ax.set_ylabel('Total Unit Sales')

plt.legend()
plt.show()


# In[ ]:



