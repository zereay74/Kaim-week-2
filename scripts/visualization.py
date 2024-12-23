import matplotlib.pyplot as plt
import seaborn as sns

class UnivariateAnalyzer:
    def __init__(self, df):
        """
        Initialize with the DataFrame.
        
        Parameters:
        df (pd.DataFrame): The input DataFrame.
        """
        self.df = df

    def plot_variable(self, column):
        """
        Plot a single variable using the most suitable method.
        
        Parameters:
        column (str): The column name to plot.
        """
        data = self.df[column]
        
        plt.figure(figsize=(8, 5))
        if data.nunique() > 10:  # Continuous variable
            sns.histplot(data, kde=True, bins=20)
            plt.title(f"Distribution of {column}")
            plt.xlabel(column)
            plt.ylabel("Frequency")
        else:  # Categorical or discrete variable
            sns.countplot(x=data, order=data.value_counts().index)
            plt.title(f"Frequency of {column}")
            plt.xlabel(column)
            plt.ylabel("Count")
        
        plt.tight_layout()
        plt.show()

    def analyze_all(self):
        """
        Generate plots for all columns and provide interpretation.
        """
        for column in self.df.select_dtypes(include='number').columns:
            self.plot_variable(column)
            print(f"Analysis for {column}:")
            print(
                f"- The distribution indicates {'high variability' if self.df[column].std() > 0.5 * self.df[column].mean() else 'low variability'}.\n"
            )


class BivariateAnalyzer:
    def __init__(self, df):
        """
        Initialize with the DataFrame.
        
        Parameters:
        df (pd.DataFrame): The input DataFrame.
        """
        self.df = df

    def plot_relationship(self, column_x, column_y):
        """
        Plot the relationship between two variables.
        
        Parameters:
        column_x (str): The independent variable (X-axis).
        column_y (str): The dependent variable (Y-axis).
        """
        plt.figure(figsize=(8, 5))
        sns.scatterplot(x=self.df[column_x], y=self.df[column_y])
        plt.title(f"Relationship between {column_x} and {column_y}")
        plt.xlabel(column_x)
        plt.ylabel(column_y)
        plt.tight_layout()
        plt.show()

    def analyze_relationships(self):
        """
        Analyze the relationship between each application-specific data column
        and the total download/upload data.
        """
        app_columns = [
            'total_social_media_data', 'total_google_data', 'total_email_data',
            'total_youtube_data', 'total_netflix_data', 'total_gaming_data',
            'total_other_data'
        ]
        for column in app_columns:
            self.plot_relationship(column, 'total_application_data_volume')
            correlation = self.df[column].corr(self.df['total_application_data_volume'])
            print(f"Correlation between {column} and total_application_data_volume: {correlation:.2f}")
            print(
                f"- {'Strong positive' if correlation > 0.7 else 'Weak or no'} relationship observed.\n"
            )


class CorrelationAnalyzer:
    def __init__(self, df):
        """
        Initialize with the DataFrame.
        
        Parameters:
        df (pd.DataFrame): The input DataFrame.
        """
        self.df = df

    def compute_correlation_matrix(self, columns):
        """
        Compute the correlation matrix for the given columns.
        
        Parameters:
        columns (list): List of column names for correlation analysis.
        
        Returns:
        pd.DataFrame: The correlation matrix.
        """
        correlation_matrix = self.df[columns].corr()
        return correlation_matrix

    def plot_correlation_heatmap(self, correlation_matrix):
        """
        Plot a heatmap of the correlation matrix.
        
        Parameters:
        correlation_matrix (pd.DataFrame): The correlation matrix to visualize.
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.show()
