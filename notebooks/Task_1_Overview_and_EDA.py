#!/usr/bin/env python
# coding: utf-8

# # Import

# In[1]:


import os 
import sys
import pandas as pd


# In[2]:


current_dir = os.getcwd()
print(current_dir)

parent_dir = os.path.dirname(current_dir)
print(parent_dir)

sys.path.insert(0,parent_dir)


# In[3]:


from importlib import reload
import scripts.data_loader, scripts.clean_and_transform, scripts.analysis_1, scripts.visualization
reload(scripts.data_loader)
reload(scripts.clean_and_transform)
reload(scripts.analysis_1)
reload(scripts.visualization)


# In[4]:


from scripts.data_loader import DataLoader
from scripts.clean_and_transform import DropNullRows, NullValueFiller, DropUndefined
from scripts.analysis_1 import UserSessionAggregator, DataDescriber, VariableTransformer, MetricsAnalyzer, DispersionAnalyzer, PCAAnalyzer, HandsetAnalysis
from scripts.visualization import UnivariateAnalyzer, BivariateAnalyzer, CorrelationAnalyzer


# #  Load data

# In[14]:


file_path = r"C:\ML and DS Files\Kifiya AI\Kaim-week-2\Week 2 data\Data\Copy of Week2_challenge_data_source(CSV).csv"
loader = DataLoader(file_path)
data = loader.load_data()


# In[6]:


data.head(2)


# In[7]:


data.shape


# In[8]:


num_undefined = data['Handset Type'].isin(['undefined']).sum()

print(f"Number of 'undefined' values: {num_undefined}")
percentage_undefined = (num_undefined / len(data)) * 100

print(f"Percentage of 'undefined' values: {percentage_undefined:.2f}%")


# In[9]:


# Drop  undefined values from Handset Type
drop_undefined = DropUndefined(data)
data = drop_undefined.DeleteUndefined(column='Handset Type', value='undefined')


# In[10]:


num_undefined = data['Handset Type'].isin(['undefined']).sum()

print(f"Number of 'undefined' values: {num_undefined}")
percentage_undefined = (num_undefined / len(data)) * 100

print(f"Percentage of 'undefined' values: {percentage_undefined:.2f}%")


# In[11]:


# drop null rows for the follwing columns
# Bearer Id, Start, End, IMSI, MSISDN/Number, IMEI,Last Location Name, Handset Manufacturer, Handset Type
col_1 = ['Bearer Id', 'Start', 'End', 'IMSI', 'MSISDN/Number', 'IMEI', 'Last Location Name', 'Handset Manufacturer', 'Handset Type']
dropper = DropNullRows(columns_to_check=col_1)

# Drop rows where the specified column has null values
data = dropper.drop_if_null(data)


# In[12]:


data.shape


# In[13]:


null_columns = ['Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)', 
           'DL TP < 50 Kbps (%)','50 Kbps < DL TP < 250 Kbps (%)', '250 Kbps < DL TP < 1 Mbps (%)', 'DL TP > 1 Mbps (%)', 
           'UL TP < 10 Kbps (%)','10 Kbps < UL TP < 50 Kbps (%)', '50 Kbps < UL TP < 300 Kbps (%)', 'UL TP > 300 Kbps (%)',
            'HTTP DL (Bytes)', 'HTTP UL (Bytes)', 'Nb of sec with 125000B < Vol DL', 'Nb of sec with 1250B < Vol UL < 6250B',
            'Nb of sec with 31250B < Vol DL < 125000B', 'Nb of sec with 37500B < Vol UL', 'Nb of sec with 6250B < Vol DL < 31250B',
            'Nb of sec with 6250B < Vol UL < 37500B','Nb of sec with Vol DL < 6250B', 'Nb of sec with Vol UL < 1250B']


# #  Numerical null values are filled based on the outlier and normal distribution

# In[14]:


# Initialize the NullValueFiller class
filler = NullValueFiller(data, null_columns)
    
# Fill null values based on mean/median decision
filler.fill_nulls()


# In[15]:


null_counts = data.isnull().sum()
print(null_counts.sum())


# In[16]:


data.columns


# In[17]:


# create an instance
analyzer = HandsetAnalysis(data)


# In[18]:


# Identify the top 10 handsets
print("Top 10 Handsets:")
top_10_handsets = analyzer.get_top_n('Handset Type', 10)
print(top_10_handsets)


# In[19]:


top_10_handsets.info()


# In[20]:


# Identify the top 3 handset manufacturers
print("\nTop 3 Handset Manufacturers:")
top_3_manufacturers = analyzer.get_top_n('Handset Manufacturer', 3)
print(top_3_manufacturers)


# In[21]:


# Identify the top 5 handsets per top 3 manufacturers
# Task 3: Identify the top 5 handsets per top 3 manufacturers
print("\nTop 5 Handsets Per Top 3 Manufacturers:")
top_5_per_top_3_manufacturers = analyzer.get_top_n_per_top_k_groups('Handset Manufacturer', 'Handset Type', 5, 3)
print(top_5_per_top_3_manufacturers)


# ## Task 1.1  Aggrigation

# In[22]:


# Instantiate the class
aggregator = UserSessionAggregator(data)

# Perform aggregation by IMSI
aggrigated = aggregator.aggregate_per_user(user_column='IMSI')

# View results
aggrigated.head(3)


# In[23]:


aggrigated.info()


# In[24]:


aggrigated.head(2)


# #### variable discription

# In[25]:


# Describe variables
describer = DataDescriber(aggrigated)
variable_description = describer.describe_variables()
print(variable_description)


# In[26]:


variable_description.info()


# #### Segmentation

# In[27]:


# Perform user segmentation and compute total data per decile
transformer = VariableTransformer(aggrigated)
decile_data = transformer.segment_users()
print(decile_data)


# In[28]:


decile_data.info()


# #### Analyze basic metrics

# In[29]:


# Analyze metrics
analyzer = MetricsAnalyzer(aggrigated)
basic_metrics = analyzer.analyze_metrics()
print(basic_metrics["metrics"])
print(basic_metrics["explanation"])


# #### Dispersion analyzer

# In[30]:


# Initialize the analyzer
analyzer = DispersionAnalyzer(aggrigated)

# Compute dispersion metrics
dispersion_metrics = analyzer.compute_dispersion()
print("Dispersion Metrics:\n", dispersion_metrics)

# Provide interpretation
interpretation = analyzer.interpret_dispersion(dispersion_metrics)
print("\nInterpretation:\n", interpretation)


# #### Graphical analysis

# In[31]:


uni_analyzer = UnivariateAnalyzer(aggrigated)
uni_analyzer.plot_variable(column='total_youtube_data')


# In[32]:


# Bivariate Analysis
bi_analyzer = BivariateAnalyzer(aggrigated)
bi_analyzer.plot_relationship(column_x='xdr_sessions', column_y='total_application_data_volume')


# #### Correlation Analysis

# In[33]:


aggrigated.columns


# In[34]:


corr_columns = ['IMSI', 'xdr_sessions', 'total_session_duration', 'total_dl_data',
       'total_ul_data', 'total_social_media_data', 'total_google_data',
       'total_email_data', 'total_youtube_data', 'total_netflix_data',
       'total_gaming_data', 'total_other_data',
       'total_application_data_volume']


# In[35]:


# Correlation Analysis
corr_analyzer = CorrelationAnalyzer(aggrigated)
correlation_matrix = corr_analyzer.compute_correlation_matrix(corr_columns)
corr_analyzer.plot_correlation_heatmap(correlation_matrix)


# #### Principal component analysis

# In[36]:


# PCA
pca_analyzer = PCAAnalyzer(aggrigated)
pca_results = pca_analyzer.perform_pca(corr_columns, n_components=2)
print("PCA Results:\n", pca_results)


# '''
# Here are four key insights from the PCA results:
# 
# 1. **Dominance of the First Principal Component (PC1)**:  
#    The first principal component (PC1) accounts for **63.37% of the variance**, indicating that it captures the majority of the variability in the dataset. This suggests that a single underlying factor heavily influences the data distribution.
# 
# 2. **Low Contribution from the Second Principal Component (PC2)**:  
#    The second principal component (PC2) explains only **7.70% of the variance**, demonstrating that it contributes much less to the overall variability. This indicates that PC2 captures nuances or secondary patterns in the data.
# 
# 3. **Cumulative Explained Variance**:  
#    Together, the first two components explain **71.0% of the total variance**, meaning a significant proportion of the data’s variability can be summarized using just these two dimensions. However, nearly **29% of the variance** remains unexplained, possibly requiring additional components for further analysis.
# 
# 4. **Potential for Dimensionality Reduction**:  
#    By retaining only two components, we achieve a **substantial reduction in dimensionality** while preserving most of the critical information. This simplification is useful for visualizing relationships between variables and reducing computational complexity in downstream tasks.
# '''

# 
