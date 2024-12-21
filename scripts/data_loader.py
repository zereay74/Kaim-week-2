import os 
import sys
import pandas as pd
import psycopg2 
from dotenv import load_dotenv
from sqlalchemy import create_engine

class DataLoader:
    """
    A class to handle loading  data from CSV files.
    """

    def __init__(self, file_path):
        """
        Initializes the DataLoader with the file path to the CSV file.

        :param file_path: str, path to the CSV file
        """
        self.file_path = file_path

    def load_data(self):
        """
        Loads the CSV file into a pandas DataFrame.

        :return: pd.DataFrame containing the data from the CSV file
        """
        try:
            data = pd.read_csv(self.file_path)
            print(f"Data successfully loaded from {self.file_path}")
            return data
        except FileNotFoundError:
            print(f"Error: File not found at {self.file_path}")
            return None
        except pd.errors.EmptyDataError:
            print(f"Error: No data in file at {self.file_path}")
            return None
        except Exception as e:
            print(f"An error occurred while loading the file: {e}")
            return None

# Example usage:
# loader = DataLoader("path_to_your_file.csv")
# df = loader.load_data()

 
# Load environment variables
load_dotenv()

# Fetch database connection parameters
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

class LoadSqlData:
    """
    A class to load SQL data from PostgreSQL using psycopg2 or SQLAlchemy.
    """
    
    def __init__(self, query):
        self.query = query

    def load_data_from_postgres(self):
        """
        Load data from PostgreSQL using psycopg2.

     
        """
        try:
            # Establish connection to the database
            connection = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD
            )

            
            # Fetch data using pandas
            df = pd.read_sql_query(self.query, connection)
            
            # Close connection
            connection.close()
            print('Sucessfully Loaded')
            return df
        except Exception as e:
            print(f"An error occurred while loading data with psycopg2: {e}")
            print("Connection parameters:")
            print(f"Host: {DB_HOST}, Port: {DB_PORT}, DB: {DB_NAME}, User: {DB_USER}, Password: {DB_PASSWORD}")
            return None


    def load_data_using_sqlalchemy(self):
        """
        Load data from PostgreSQL using SQLAlchemy.
        """
        try:
            # Create connection string
            connection_string = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
            
            # Create engine
            engine = create_engine(connection_string)

            # Fetch data using pandas
            df = pd.read_sql_query(self.query, engine)
            print('Sucessfully Loaded')
            return df
        except Exception as e:
            print(f"An error occurred while loading data with SQLAlchemy: {e}")
            return None
