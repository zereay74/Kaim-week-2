# Telecom Data Analysis Tasks

## Overview
This repository contains scripts and notebooks for analyzing telecom user data. Tasks include exploratory data analysis (EDA), user engagement analysis, and experience and satisfaction analytics. The project follows a modular object-oriented programming (OOP) approach.

---

## Task Highlights

### Task 1: Telecommunication EDA
- **Goals**:
  - Preprocess and clean data.
  - Analyze user-level and session-level metrics.
  - Identify popular handsets and manufacturers.
  - Perform segmentation and dimensionality reduction.

- **Key Outputs**:
  - Top 10 handsets and top 3 manufacturers.
  - Aggregated user metrics (session count, duration, total traffic).
  - Correlation matrix and PCA results.
  - Visualizations (e.g., histograms, boxplots).

**Scripts**:  
`data_loader.py`, `clean_and_transform.py`, `analysis_1.py`, 'analysis_2.py', `visualization.py`   

---

### Task 2: User Engagement Analysis
- **Goals**:
  - Aggregate engagement metrics (session count, duration, traffic).
  - Cluster users into groups using K-Means.
  - Analyze application-specific traffic.

- **Key Outputs**:
  - Top 10 users by engagement metrics.
  - Clusters of light, average, and heavy users.
  - Top 3 most used applications.

**Key Methods**:  
`aggregate_metrics()`, `normalize_and_cluster()`, `traffic_by_application()`  

---

### Task 3 & 4: Experience and Satisfaction Analytics
- **Goals**:
  - Rank top 10 users by experience and satisfaction scores.
  - Export insights for database integration.
  - Analyze network performance and usage patterns.

- **Key Outputs**:
  - Top 10 users based on experience and satisfaction.
  - Database-ready tuples for exporting user insights.
  - Visualizations for RTT, throughput, and application usage.

### Streamlit app 
  - for visualization and diplaying plots
  - the app have the above 3 task pages in separat
  - for each task the app will display a data frame and it plots by choosing rows, columns.... 
---

## Results
1. **EDA**:
   - Top handsets and manufacturers identified.
   - Strong correlations between video-streaming applications.
   - PCA explained 71% of variance in top two components.

2. **User Engagement**:
   - Clustering revealed distinct user behavior patterns.
   - Top 3 applications: YouTube, Social Media, Google.

3. **Experience & Satisfaction**:
   - Top-performing users analyzed for database insertion.
   - Error-handling mechanisms implemented for robust analytics.
4. **Custom Visualization on Streamlit app**:
| | - selected dataframes from the tasks
| | - visualizations by choosing plot type, row, column..... 

---

## How to Run
1. Clone the repository:
2. cd to notebooks folder and run the notebooks
3. for futher analysis and customization the scripts are located in /scripts folder

## repo structure
Kaim week 2/
├── notebooks/
│   ├── Task_1_Overview_and_EDA.ipynb
│   ├── Task_2_User_Engagement_analysis.ipynb
│   └── Task_3_and_4_Experience_Satisfaction_anlalysis.ipynb
├── scripts/
│   ├── data_loader.py
│   ├── clean_and_transform.py
│   ├── analysis_1.py
|   ├── analysis_2.py
│   └── visualization.py
├── README.md
├── requirements.txt
| 
└── streamlit_app.py

