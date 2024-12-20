# Task Overview: Telecommunication Data Analysis

## Project Description
This project focuses on analyzing telecommunication user data to extract meaningful insights through data preprocessing, exploratory data analysis (EDA), and statistical techniques. The implementation follows a modular object-oriented programming (OOP) approach.

---

## Tasks Completed

### 1. Load Data and Preprocess
- **Objective**: Load the dataset and clean it by handling missing values and outliers.
- **Process**:
  - Loaded data from the source file.
  - Applied transformations to handle missing or erroneous values.
  - Ensured consistent data types for analysis.

**Script Used**: `data_loader.py`, `clean_and_transform.py`

---

### 2. Identify the Top 10 Handsets
- **Objective**: Identify the 10 most used handsets among users.
- **Process**:
  - Aggregated data based on the `Handset Type` column.
  - Counted occurrences to determine popularity.

**Script Used**: `analysis_1.py`

---

### 3. Identify the Top 3 Handset Manufacturers
- **Objective**: Identify the manufacturers producing the most popular handsets.
- **Process**:
  - Grouped data by `Handset Manufacturer`.
  - Counted occurrences to rank manufacturers.

**Script Used**: `analysis_1.py`

---

### 4. Aggregations and Key Variables
- **Objective**: Aggregate session-level data to create user-level metrics.
- **Key Variables**:
  - **IMSI**: Unique user identifier.
  - **xDR Sessions**: Total number of data sessions per user.
  - **Total Session Duration**: Summed session durations per user.
  - **Application-Specific Data**: Total data for Social Media, Google, YouTube, etc.

**Notebook Used**: `Task 1 Overview and EDA.ipynb`

**Script Used**: `clean_and_transform.py`, `analysis_1.py`

---

### 5. Variable Description and Segmentation
- **Objective**: 
  - Provide detailed descriptions for each variable.
  - Segment users into deciles based on total session duration.
- **Process**:
  - Created descriptive summaries of variables.
  - Applied decile segmentation using `pd.qcut`.

**Script Used**: `clean_and_transform.py`

---

### 6. Metrics and Graphical Analysis
- **Objective**: Compute descriptive statistics and visualize distributions.
- **Process**:
  - Analyzed central tendencies (mean, median) and dispersion (std, range).
  - Created visualizations such as histograms, boxplots, and scatter plots.

**Script Used**: `analysis_1.py`, `visualization.py`

---

### 7. Univariate and Bivariate Analysis
- **Objective**: 
  - Conduct univariate analysis to understand individual variable behavior.
  - Perform bivariate analysis to explore relationships between variables.
- **Key Activities**:
  - Computed measures of central tendency and dispersion.
  - Analyzed relationships between application data and total DL+UL data.

**Script Used**: `analysis_1.py`, `visualization.py`

---

### 8. Correlation Analysis
- **Objective**: Compute and interpret the correlation matrix for application-specific data.
- **Key Insights**:
  - Strong correlations observed between video-streaming applications like YouTube and Netflix.
  - Weak correlations for niche categories like Gaming Data.

**Script Used**: `analysis_1.py`

---

### 9. Dimensionality Reduction
- **Objective**: Reduce data dimensions using PCA while retaining critical information.
- **Key Findings**:
  - Principal Component 1 explained 63.37% of the variance.
  - Combined variance explained by the first two components was 71%.

**Script Used**: `analysis_1.py`

---

## Repository Structure
```
Kaim week 2/
├── notebooks/
│   └── Task 1 Overview and EDA.ipynb
├── scripts/
│   ├── data_loader.py
│   ├── clean_and_transform.py
│   ├── analysis_1.py
│   └── visualization.py
├──...
│  
└── README.md
```

---

## How to Run the Project
1. Clone the repository.
2. Ensure required Python libraries are installed.
3. Run the `Task 1 Overview and EDA.ipynb` notebook to reproduce results.
4. Explore individual scripts in the `scripts/` folder for specific functionalities.

---

## Key Insights
- Modular programming enables scalable and reusable code.
- Aggregation and segmentation reveal diverse user behaviors.
- Correlation and PCA provide actionable insights for feature selection and dimensionality reduction.

---

## Future Work
- Extend the analysis to include time-series trends.
- Apply clustering techniques to segment users based on behavior.
