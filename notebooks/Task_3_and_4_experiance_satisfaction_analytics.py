#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[7]:


import os 
import sys
import pandas as pd
import psycopg2 
from dotenv import load_dotenv
from sqlalchemy import create_engine


# In[8]:


import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")


# In[9]:


current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0,parent_dir)


# In[10]:


from importlib import reload
import scripts.data_loader, scripts.clean_and_transform, scripts.analysis_1, scripts.visualization, scripts.analysis_2, scripts.analysis_3
reload(scripts.data_loader)
reload(scripts.clean_and_transform)
reload(scripts.analysis_1)
reload(scripts.analysis_2)
reload(scripts.visualization)
reload(scripts.analysis_3)


# In[11]:


from scripts.data_loader import DataLoader, LoadSqlData
from scripts.clean_and_transform import DropNullRows, NullValueFiller, DropUndefined
from scripts.analysis_1 import UserSessionAggregator, DataDescriber, VariableTransformer, MetricsAnalyzer, DispersionAnalyzer, PCAAnalyzer, HandsetAnalysis
from scripts.analysis_2 import TelecomEngagementAnalysis
from scripts.analysis_3 import TelecomAnalysis, SatisfactionAnalysis, PostgreSQLExporter
from scripts.visualization import UnivariateAnalyzer, BivariateAnalyzer, CorrelationAnalyzer


# # Load data from postgreSQL

# In[12]:


# Define your SQL query
query = "SELECT * FROM xdr_data"
# Create an instance of the LoadSqlData class
data_loader = LoadSqlData(query)
# Load data using psycopg2
data= data_loader.load_data_using_sqlalchemy()
data.head()


# # Clean the data

# In[13]:


data.shape


# In[14]:


# Drop  undefined values from Handset Type
drop_undefined = DropUndefined(data)
data = drop_undefined.DeleteUndefined(column='Handset Type', value='undefined')


# In[15]:


# drop null rows for the follwing columns
# Bearer Id, Start, End, IMSI, MSISDN/Number, IMEI,Last Location Name, Handset Manufacturer, Handset Type
col_1 = ['Bearer Id', 'Start', 'End', 'IMSI', 'MSISDN/Number', 'IMEI', 'Last Location Name', 'Handset Manufacturer', 'Handset Type']
dropper = DropNullRows(columns_to_check=col_1)

# Drop rows where the specified column has null values
data = dropper.drop_if_null(data)


# In[16]:


null_columns = ['Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)', 
           'DL TP < 50 Kbps (%)','50 Kbps < DL TP < 250 Kbps (%)', '250 Kbps < DL TP < 1 Mbps (%)', 'DL TP > 1 Mbps (%)', 
           'UL TP < 10 Kbps (%)','10 Kbps < UL TP < 50 Kbps (%)', '50 Kbps < UL TP < 300 Kbps (%)', 'UL TP > 300 Kbps (%)',
            'HTTP DL (Bytes)', 'HTTP UL (Bytes)', 'Nb of sec with 125000B < Vol DL', 'Nb of sec with 1250B < Vol UL < 6250B',
            'Nb of sec with 31250B < Vol DL < 125000B', 'Nb of sec with 37500B < Vol UL', 'Nb of sec with 6250B < Vol DL < 31250B',
            'Nb of sec with 6250B < Vol UL < 37500B','Nb of sec with Vol DL < 6250B', 'Nb of sec with Vol UL < 1250B']


# In[17]:


# Numerical null values are filled based on the outlier and normal distribution
# Initialize the NullValueFiller class
filler = NullValueFiller(data, null_columns)
    
# Fill null values based on mean/median decision
filler.fill_nulls()


# In[18]:


data["Start"] = pd.to_datetime(data["Start"])
data["End"] = pd.to_datetime(data["End"])


# In[19]:


null_counts = data.isnull().sum()
if null_counts.sum() > 0:
    print('Null value present please check the dataframe')
else:
    print('All columns are not null')


# # Task 3

# In[20]:


telecom_analysis = TelecomAnalysis(data)
# Preprocess data calculate the average
telecom_analysis.preprocess_data()


# In[21]:


# Task 3 1: Aggregate metrics per customer
aggregated_data = telecom_analysis.aggregate_per_customer()
print("Aggregated Data:\n", aggregated_data.head())


# In[22]:


# Task 3.2: Compute top, bottom, and most frequent for metrics
for metric in ['TCP Retransmission', 'RTT', 'Throughput']:
    top_10, bottom_10, frequent = telecom_analysis.compute_top_bottom_frequent(metric)
    print(f"Top 10 {metric}: {top_10}")
    print(f"Bottom 10 {metric}: {bottom_10}")
    print(f"Most Frequent {metric}: {frequent}")


# In[23]:


# Task 3.3: Average Distribution analysis
throughput_distribution = telecom_analysis.distribution_analysis('Throughput', 'Handset Type')
print("Throughput Distribution:\n", throughput_distribution.head())

tcp_distribution = telecom_analysis.distribution_analysis('TCP Retransmission', 'Handset Type')
print("TCP Retransmission Distribution:\n", tcp_distribution.head())


# In[24]:


# Task 3.4: K-means clustering
features = ['TCP Retransmission', 'RTT', 'Throughput']
cluster_data, cluster_centers = telecom_analysis.kmeans_clustering(features)
print("Cluster Data:\n", cluster_data.head())
print("Cluster Centers:\n", cluster_centers)


# In[25]:


import matplotlib.pyplot as plt
# Visualize clusters
plt.scatter(telecom_analysis.data['TCP Retransmission'], telecom_analysis.data['Throughput'], c=telecom_analysis.data['Cluster'])
plt.title("Clusters Visualization")
plt.xlabel("TCP Retransmission")
plt.ylabel("Throughput")
plt.show()


# 
# # Task 4 Satisfaction Analysis

# In[26]:


# Satisfaction Analysis
satisfaction_analysis = SatisfactionAnalysis(telecom_analysis.data, cluster_centers)
satisfaction_analysis.assign_scores()

top_10_satisfied = satisfaction_analysis.calculate_satisfaction()
print("Top 10 Satisfied Customers:\n", top_10_satisfied)


# In[27]:


top_10_satisfied.info()


# In[28]:


# regression model
regression_model = satisfaction_analysis.regression_model()
print("Regression Model Coefficients:\n", regression_model.coef_)


# In[29]:


#  k-means (k=2) on the engagement & the experience score. 
score_clusters, score_cluster_centers = satisfaction_analysis.kmeans_on_scores(n_clusters=2)
print("Score Clusters:\n", score_clusters.head())


# In[30]:


# Aggregate the average satisfaction & experience score per cluster
aggregated_scores = satisfaction_analysis.aggregate_scores_per_cluster()
print("Aggregated Scores per Cluster:\n", aggregated_scores)


# In[31]:


# Export to PostgreSQL
exporter = PostgreSQLExporter()
exporter.create_table(table_name='Top_10_Exported')


# In[32]:


exporter.export_dataframe(table_name='Top_10_Exported', dataframe=top_10_satisfied)


# In[33]:


exporter.close_connection()


# In[ ]:




