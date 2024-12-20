import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class DropUndefined:
    def __init__(self, df):

        '''
        deletes rows if the conditions matched
        '''
        self.df = df
    def DeleteUndefined(self, column='Handset Type', value = 'undefined'):
        self.df.drop(self.df[self.df[column] == value].index, inplace=True)
        
        print(F"Sucessfully droped {value} columns")
        return self.df


class UserSessionAggregator:
    def __init__(self, df):
        """
        Initialize the UserSessionAggregator with a DataFrame.
        
        Parameters:
        df (pd.DataFrame): The input DataFrame containing session data.
        """
        self.df = df

    def aggregate_per_user(self, user_column='IMSI'):
        """
        Aggregate session information per user.
        
        Parameters:
        user_column (str): The column name to group by (e.g., 'IMSI', 'MSISDN/Number').
        
        Returns:
        pd.DataFrame: Aggregated data per user.
        """
        # Perform aggregation
        aggregated_data = self.df.groupby(user_column).agg(
            xdr_sessions=('Bearer Id', 'count'),  # Number of xDR sessions
            total_session_duration=('Dur. (ms)', 'sum'),  # Total session duration
            total_dl_data=('Total DL (Bytes)', 'sum'),  # Total download data
            total_ul_data=('Total UL (Bytes)', 'sum'),  # Total upload data
            total_social_media_data=('Social Media DL (Bytes)', 'sum'),  # Social Media DL
            total_google_data=('Google DL (Bytes)', 'sum'),  # Google DL
            total_email_data=('Email DL (Bytes)', 'sum'),  # Email DL
            total_youtube_data=('Youtube DL (Bytes)', 'sum'),  # Youtube DL
            total_netflix_data=('Netflix DL (Bytes)', 'sum'),  # Netflix DL
            total_gaming_data=('Gaming DL (Bytes)', 'sum'),  # Gaming DL
            total_other_data=('Other DL (Bytes)', 'sum'),  # Other DL
        ).reset_index()

        # Add total data volume for all applications
        aggregated_data['total_application_data_volume'] = (
            aggregated_data[['total_social_media_data', 'total_google_data', 
                             'total_email_data', 'total_youtube_data', 
                             'total_netflix_data', 'total_gaming_data', 
                             'total_other_data']].sum(axis=1)
        )

        return aggregated_data

''' Usage Example 
df = pd.DataFrame(data)

# Instantiate the class
aggregator = UserSessionAggregator(df)

# Perform aggregation
result = aggregator.aggregate_per_user(user_column='IMSI')

# View results
print(result)

'''


class DataDescriber:
    def __init__(self, df):
        """
        Initialize with the aggregated DataFrame.
        
        Parameters:
        df (pd.DataFrame): The input DataFrame to analyze.
        """
        self.df = df

    def describe_variables(self):
        """
        Describe variables and their data types.
        
        Returns:
        pd.DataFrame: Description of the variables.
        """
        # Define known descriptions
        known_descriptions = [
            "Bearer Id: Unique session identifier",
            "Start: Session start time",
            "End: Session end time",
            "Dur. (ms): Session duration in milliseconds",
            "IMSI: User identifier",
            "Total DL (Bytes): Total downloaded data",
            "Total UL (Bytes): Total uploaded data",
        ]

        # Generate placeholder descriptions for any extra columns
        additional_descriptions = [
            f"Description for {col}" for col in self.df.columns[len(known_descriptions):]
        ]

        # Combine known and additional descriptions
        all_descriptions = known_descriptions + additional_descriptions

        # Ensure the number of descriptions matches the number of columns
        all_descriptions = all_descriptions[:len(self.df.columns)]

        # Create the DataFrame
        description = pd.DataFrame({
            "Variable": self.df.columns,
            "DataType": self.df.dtypes.values,
            "Description": all_descriptions
        })

        return description



class VariableTransformer:
    def __init__(self, df):
        self.df = df.fillna(0)  # Handle missing values

    def segment_users(self):
        self.df['decile_class'] = pd.qcut(
            self.df['total_session_duration'], 10, labels=[f"Decile {i+1}" for i in range(10)]
        )

        # Aggregation with correct column names
        decile_data = self.df.groupby('decile_class').agg(
            total_users=('IMSI', 'count'),
            total_dl_data=('total_dl_data', 'sum'),
            total_ul_data=('total_ul_data', 'sum')
        ).reset_index()

        decile_data['total_data_dl_ul'] = decile_data['total_dl_data'] + decile_data['total_ul_data']
        return decile_data



class MetricsAnalyzer:
    def __init__(self, df):
        """
        Initialize with the aggregated DataFrame.
        
        Parameters:
        df (pd.DataFrame): The input DataFrame for analysis.
        """
        self.df = df

    def analyze_metrics(self):
        """
        Analyze basic metrics in the dataset.
        
        Returns:
        dict: A dictionary containing basic metrics and their explanations.
        """
        metrics = {
            "mean": self.df.mean(numeric_only=True).to_dict(),
            "median": self.df.median(numeric_only=True).to_dict(),
            "std": self.df.std(numeric_only=True).to_dict(),
            "min": self.df.min(numeric_only=True).to_dict(),
            "max": self.df.max(numeric_only=True).to_dict(),
        }
        
        explanation = (
            "Mean provides an average to understand typical user behavior. "
            "Median helps to identify the central tendency while minimizing the impact of outliers. "
            "Standard deviation (std) measures variability, which is important for understanding the spread in user behavior. "
            "Min and Max values help in detecting anomalies or defining the range of data."
        )
        return {"metrics": metrics, "explanation": explanation}



class DispersionAnalyzer:
    def __init__(self, df):
        """
        Initialize with the DataFrame.
        
        Parameters:
        df (pd.DataFrame): The input DataFrame for analysis.
        """
        self.df = df.select_dtypes(include='number')  # Keep only numeric columns

    def compute_dispersion(self):
        """
        Compute dispersion parameters for each quantitative variable.
        
        Returns:
        pd.DataFrame: Dispersion metrics for each numeric column.
        """
        dispersion_metrics = {
            "Variable": self.df.columns,
            "Range": self.df.max() - self.df.min(),
            "IQR": self.df.quantile(0.75) - self.df.quantile(0.25),
            "Variance": self.df.var(),
            "Standard Deviation": self.df.std(),
        }
        
        return pd.DataFrame(dispersion_metrics).set_index("Variable")

    def interpret_dispersion(self, metrics_df):
        """
        Provide interpretation for the dispersion metrics.
        
        Parameters:
        metrics_df (pd.DataFrame): Dispersion metrics DataFrame.
        
        Returns:
        str: Interpretation summary.
        """
        interpretation = (
            "The range indicates the spread between the smallest and largest values for each variable. "
            "A large range suggests a wide variation in values. The interquartile range (IQR) focuses on the middle 50% of data, "
            "helping identify the spread while minimizing the influence of outliers. Variance and standard deviation measure the "
            "overall spread around the mean, with higher values indicating greater variability. The mean absolute deviation provides "
            "a simpler measure of average deviation from the mean, less influenced by extreme values."
        )
        return interpretation



class PCAAnalyzer:
    def __init__(self, df):
        """
        Initialize with the DataFrame.
        
        Parameters:
        df (pd.DataFrame): The input DataFrame.
        """
        self.df = df

    def perform_pca(self, columns, n_components=2):
        """
        Perform PCA on the specified columns.
        
        Parameters:
        columns (list): List of column names for PCA.
        n_components (int): Number of principal components to retain.
        
        Returns:
        pd.DataFrame: DataFrame with principal components.
        """
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.df[columns])

        # Perform PCA
        pca = PCA(n_components=n_components)
        pca_components = pca.fit_transform(scaled_data)

        # Create a DataFrame for principal components
        pca_df = pd.DataFrame(
            pca_components,
            columns=[f"PC{i+1}" for i in range(n_components)]
        )
        
        explained_variance = pca.explained_variance_ratio_
        print(f"Explained Variance Ratios: {explained_variance}")
        print(f"Total Explained Variance: {sum(explained_variance):.2f}")
        
        return pca_df



class HandsetAnalysis:
    def __init__(self, df: pd.DataFrame):
        """
        Initializes the class with a DataFrame.

        Parameters:
        df (pd.DataFrame): The input DataFrame.
        """
        self.df = df

    def validate_column(self, column_name: str):
        """
        Validates that the specified column exists in the DataFrame.

        Parameters:
        column_name (str): The name of the column to validate.

        Raises:
        ValueError: If the column does not exist in the DataFrame.
        """
        if column_name not in self.df.columns:
            raise ValueError(f"Column '{column_name}' not found in the DataFrame.")

    def get_top_n(self, column_name: str, top_n: int = 10):
        """
        Identifies the top N items in the specified column.

        Parameters:
        column_name (str): The column to analyze.
        top_n (int): The number of top items to retrieve.

        Returns:
        pd.DataFrame: A DataFrame with the top N items and their counts.
        """
        self.validate_column(column_name)
        top_items = (
            self.df[column_name]
            .value_counts()
            .head(top_n)
            .reset_index()
        )
        top_items.columns = [column_name, 'Count']
        return top_items

    def get_top_n_per_top_k_groups(self, group_column: str, target_column: str, top_n: int = 5, top_k: int = 3):
        """
        Identifies the top N items in the target column for each of the top K groups in the group column.

        Parameters:
        group_column (str): The column to group by.
        target_column (str): The column to find the top items in for each group.
        top_n (int): The number of top items to retrieve per group.
        top_k (int): The number of top groups to consider.

        Returns:
        pd.DataFrame: A DataFrame with the top N items per group and their counts.
        """
        self.validate_column(group_column)
        self.validate_column(target_column)

        # Get top K manufacturers based on the count of handsets
        top_k_manufacturers = self.df[group_column].value_counts().head(top_k).index.tolist()

        # Filter the DataFrame to include only the top K manufacturers
        filtered_df = self.df[self.df[group_column].isin(top_k_manufacturers)]

        # Group by the group_column, calculate value counts within each group, and keep only the top_n
        top_items_per_group = (
            filtered_df.groupby(group_column)[target_column]
            .apply(lambda x: x.value_counts().head(top_n))
            .reset_index(name='Count')
        )
        return top_items_per_group

# Example Usage
'''

    analyzer = HandsetAnalysis(data)

    # Task 1: Identify the top 10 handsets
    print("Top 10 Handsets:")
    top_10_handsets = analyzer.get_top_n('Handset Type', 10)
    print(top_10_handsets)

    # Task 2: Identify the top 3 handset manufacturers
    print("\nTop 3 Handset Manufacturers:")
    top_3_manufacturers = analyzer.get_top_n('Manufacturer', 3)
    print(top_3_manufacturers)

    # Task 3: Identify the top 5 handsets per top 3 manufacturers
    print("\nTop 5 Handsets Per Top 3 Manufacturers:")
    top_5_per_manufacturer = analyzer.get_top_n_per_group('Manufacturer', 'Handset Type', 5)
    print(top_5_per_manufacturer)

'''

