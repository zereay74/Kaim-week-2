import pandas as pd
import numpy as np
from scipy import stats

class DropNullRows:
    def __init__(self, columns_to_check):
        """
        Initialize with the columns to check for null values.
        
        :param columns_to_check: List of column names to evaluate.
        """
        self.columns_to_check = columns_to_check

    def drop_if_null(self, df):
        """
        Drops rows where any of the specified columns have null values.
        
        :param df: The DataFrame to operate on.
        :return: A new DataFrame with rows dropped where the specified columns have null values.
        """
        print(f"Sucessfuly dropped null columns from {self.columns_to_check}")
        return df.dropna(subset=self.columns_to_check)
        

# Example Usage

# Initialize the class with columns to check
#dropper = DropNullRows(columns_to_check=['Column1', 'Column2'])

# Drop rows where any of the specified columns have null values
#df_cleaned = dropper.drop_if_null(df)



class NullValueFiller:
    def __init__(self, df: pd.DataFrame, columns: list):
        """
        Initializes the class with a dataframe and columns to be processed.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        columns (list): List of column names where null values should be filled.
        """
        self.df = df
        self.columns = columns
    
    def check_for_outliers(self, column: str):
        """
        Checks if a column has outliers by performing a normality test (Shapiro-Wilk).
        
        Returns:
        bool: True if the column has outliers (non-normal distribution), False otherwise.
        """
        # Drop null values for the test
        data = self.df[column].dropna()
        
        # Perform Shapiro-Wilk test for normality
        stat, p_value = stats.shapiro(data)
        
        # If p_value is small, it indicates the data is not normally distributed
        return p_value < 0.05

    def fill_nulls(self):
        """
        Fills null values in the specified columns using either mean or median based on distribution.
        """
        for column in self.columns:
            if self.df[column].isnull().sum() > 0:
                if self.check_for_outliers(column):
                    # If outliers are present, use median
                    fill_value = self.df[column].median()
                    self.df[column].fillna(fill_value, inplace=True)
                    print(f"Column '{column}': Filled null values with median.")
                else:
                    # If no outliers, use mean
                    fill_value = self.df[column].mean()
                    self.df[column].fillna(fill_value, inplace=True)
                    print(f"Column '{column}': Filled null values with mean.")
            else:
                print(f"Column '{column}': No null values to fill.")

    #Example Usage
    # Initialize the NullValueFiller class
    #filler = NullValueFiller(df, ['Age', 'Salary', 'Height'])
    
    # Fill null values based on mean/median decision
    #filler.fill_nulls()

    #print("\nUpdated DataFrame:")
    #print(df)
