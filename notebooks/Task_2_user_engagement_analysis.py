#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


import os 
import sys
import pandas as pd
import psycopg2 
from dotenv import load_dotenv
from sqlalchemy import create_engine


# In[2]:


current_dir = os.getcwd()
print(current_dir)

parent_dir = os.path.dirname(current_dir)
print(parent_dir)

sys.path.insert(0,parent_dir)


# In[3]:


from importlib import reload
import scripts.data_loader, scripts.clean_and_transform, scripts.analysis_1, scripts.visualization, scripts.analysis_2
reload(scripts.data_loader)
reload(scripts.clean_and_transform)
reload(scripts.analysis_1)
reload(scripts.analysis_2)
reload(scripts.visualization)


# In[4]:


from scripts.data_loader import DataLoader, LoadSqlData
from scripts.clean_and_transform import DropNullRows, NullValueFiller, DropUndefined
from scripts.analysis_1 import UserSessionAggregator, DataDescriber, VariableTransformer, MetricsAnalyzer, DispersionAnalyzer, PCAAnalyzer, HandsetAnalysis
from scripts.analysis_2 import TelecomEngagementAnalysis
from scripts.visualization import UnivariateAnalyzer, BivariateAnalyzer, CorrelationAnalyzer


# # Load data from postgreSQL

# In[5]:


# Define your SQL query
query = "SELECT * FROM xdr_data"

# Create an instance of the LoadSqlData class
data_loader = LoadSqlData(query)
# Load data using psycopg2
data = data_loader.load_data_from_postgres()
data.head()


# In[6]:


# Load data using SQLAlchemy
df_sqlalchemy = data_loader.load_data_using_sqlalchemy()
df_sqlalchemy.head()


# In[7]:


data.shape


# # Clean the data

# In[8]:


# Drop  undefined values from Handset Type
drop_undefined = DropUndefined(data)
data = drop_undefined.DeleteUndefined(column='Handset Type', value='undefined')


# In[9]:


# drop null rows for the follwing columns
# Bearer Id, Start, End, IMSI, MSISDN/Number, IMEI,Last Location Name, Handset Manufacturer, Handset Type
col_1 = ['Bearer Id', 'Start', 'End', 'IMSI', 'MSISDN/Number', 'IMEI', 'Last Location Name', 'Handset Manufacturer', 'Handset Type']
dropper = DropNullRows(columns_to_check=col_1)

# Drop rows where the specified column has null values
data = dropper.drop_if_null(data)


# In[10]:


data.shape


# In[11]:


null_columns = ['Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)', 
           'DL TP < 50 Kbps (%)','50 Kbps < DL TP < 250 Kbps (%)', '250 Kbps < DL TP < 1 Mbps (%)', 'DL TP > 1 Mbps (%)', 
           'UL TP < 10 Kbps (%)','10 Kbps < UL TP < 50 Kbps (%)', '50 Kbps < UL TP < 300 Kbps (%)', 'UL TP > 300 Kbps (%)',
            'HTTP DL (Bytes)', 'HTTP UL (Bytes)', 'Nb of sec with 125000B < Vol DL', 'Nb of sec with 1250B < Vol UL < 6250B',
            'Nb of sec with 31250B < Vol DL < 125000B', 'Nb of sec with 37500B < Vol UL', 'Nb of sec with 6250B < Vol DL < 31250B',
            'Nb of sec with 6250B < Vol UL < 37500B','Nb of sec with Vol DL < 6250B', 'Nb of sec with Vol UL < 1250B']


# In[12]:


# Numerical null values are filled based on the outlier and normal distribution
# Initialize the NullValueFiller class
filler = NullValueFiller(data, null_columns)
    
# Fill null values based on mean/median decision
filler.fill_nulls()


# In[13]:


null_counts = data.isnull().sum()
print(null_counts)


# # Task 2

# ##### Aggregate the metrics per customer id (MSISDN) and report the top 10 customers per engagement metric 

# In[14]:


analysis = TelecomEngagementAnalysis(dataframe=data)
top_sessions = analysis.get_top_customers(metric='session_count')
print(top_sessions)


# ##### Normalize each engagement metric and run a k-means (k=3) to classify customers in three groups of engagement.
# minimum, maximum, average & total non-normalized metrics for each cluster

# In[15]:


# clustered data
clustered_data, normalized_metrics, kmeans = analysis.normalize_and_cluster()
cluster_stats = analysis.compute_cluster_stats(clustered_data)
print(cluster_stats)


# In[ ]:


analysis = TelecomEngagementAnalysis(data)
analysis.plot_top_applications()


# In[ ]:


analysis.find_optimal_k()


# In[18]:


top_users_per_app = analysis.traffic_by_application()
print(top_users_per_app)


# In[ ]:




