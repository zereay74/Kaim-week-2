import os
import psycopg2
import pandas as pd
from psycopg2 import sql
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.linear_model import LinearRegression

class TelecomAnalysis:
    def __init__(self, dataframe):
        self.data = dataframe
        
    def preprocess_data(self):
        # Create additional computed columns for convenience
        self.data['TCP Retransmission'] = self.data['TCP DL Retrans. Vol (Bytes)'] + self.data['TCP UL Retrans. Vol (Bytes)']
        self.data['RTT'] = (self.data['Avg RTT DL (ms)'] + self.data['Avg RTT UL (ms)']) / 2
        self.data['Throughput'] = (self.data['Avg Bearer TP DL (kbps)'] + self.data['Avg Bearer TP UL (kbps)']) / 2

    def aggregate_per_customer(self):
        # Aggregate metrics by customer
        aggregated = self.data.groupby('MSISDN/Number').agg({
            'TCP Retransmission': 'mean',
            'RTT': 'mean',
            'Handset Type': 'first',
            'Throughput': 'mean'
        }).reset_index()
        aggregated.rename(columns={
            'TCP Retransmission': 'Avg TCP Retransmission',
            'RTT': 'Avg RTT',
            'Throughput': 'Avg Throughput'
        }, inplace=True)
        return aggregated

    def compute_top_bottom_frequent(self, column_name):
        # Compute top, bottom, and most frequent values
        sorted_data = self.data[column_name].sort_values()
        top_10 = sorted_data.tail(10).values
        bottom_10 = sorted_data.head(10).values
        frequent = self.data[column_name].value_counts().head(10).index.tolist()
        return top_10, bottom_10, frequent

    def distribution_analysis(self, metric, group_by):
        # Analyze distribution grouped by a specific column
        grouped = self.data.groupby(group_by)[metric].mean().reset_index()
        grouped.sort_values(by=metric, ascending=False, inplace=True)
        return grouped

    def kmeans_clustering(self, features, n_clusters=3):
        # Perform K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.data['Cluster'] = kmeans.fit_predict(self.data[features])
        cluster_centers = kmeans.cluster_centers_
        return self.data[['MSISDN/Number', 'Cluster']], cluster_centers
    
''' def visualize_cluster(self, x_col, y_col, cluster_col, title = 'Cluster Visualization'):

        plt.scatter(
            self.data[x_col], self.data[y_col], c=self.data[cluster_col], cmap='viridis',alpha=0.7
        )
        plt.title(title)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.colorbar(label='cluster')
        plt.show()

'''

class SatisfactionAnalysis:
    def __init__(self, telecom_data, cluster_centers):
        self.data = telecom_data
        self.cluster_centers = cluster_centers

    def assign_scores(self):
        # Task 4.1: Assign engagement and experience scores
        less_engaged_center = self.cluster_centers[0]  # Assuming cluster 0 is the less engaged
        worst_experience_center = self.cluster_centers[0]  # Assuming cluster 0 is the worst experience
        
        self.data['Engagement Score'] = self.data.apply(
            lambda row: euclidean_distances(
                [[row['TCP Retransmission'], row['RTT'], row['Throughput']]],
                [less_engaged_center]
            )[0][0], axis=1
        )
        self.data['Experience Score'] = self.data.apply(
            lambda row: euclidean_distances(
                [[row['TCP Retransmission'], row['RTT'], row['Throughput']]],
                [worst_experience_center]
            )[0][0], axis=1
        )

    def calculate_satisfaction(self):
        # Task 4.2: Calculate satisfaction score
        self.data['Satisfaction Score'] = (self.data['Engagement Score'] + self.data['Experience Score']) / 2
        top_10_satisfied = self.data.nlargest(10, 'Satisfaction Score')

        return top_10_satisfied

    def regression_model(self):
        # Task 4.3: Build a regression model
        features = self.data[['Engagement Score', 'Experience Score']]
        target = self.data['Satisfaction Score']
        model = LinearRegression()
        model.fit(features, target)
        return model

    def kmeans_on_scores(self, n_clusters=2):
        # Task 4.4: K-means clustering on scores
        scores = self.data[['Engagement Score', 'Experience Score']]
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.data['Score Cluster'] = kmeans.fit_predict(scores)
        return self.data[['MSISDN/Number', 'Score Cluster']], kmeans.cluster_centers_

    def aggregate_scores_per_cluster(self):
        # Task 4.5: Aggregate average satisfaction and experience scores per cluster
        aggregated = self.data.groupby('Score Cluster').agg({
            'Satisfaction Score': 'mean',
            'Experience Score': 'mean'
        }).reset_index()
        return aggregated



class PostgreSQLExporter:
    def __init__(self):
        """
        Initialize the connection to the PostgreSQL database using credentials from .env file.
        """
        load_dotenv()
        self.connection = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT")
        )
        self.cursor = self.connection.cursor()

    def create_table(self, table_name):
        """
        Create a table for storing user scores if it does not already exist.
        """
        create_table_query = sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {table_name} (
                user_id BIGINT PRIMARY KEY,
                engagement_score FLOAT,
                experience_score FLOAT,
                satisfaction_score FLOAT
            );
            """
        ).format(table_name=sql.Identifier(table_name))

        self.cursor.execute(create_table_query)
        self.connection.commit()

    def insert_data(self, table_name, data):
        """
        Insert data into the specified PostgreSQL table.
        """
        try:
            insert_query = sql.SQL(
                """
                INSERT INTO {table_name} (user_id, engagement_score, experience_score, satisfaction_score)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE
                SET
                    engagement_score = EXCLUDED.engagement_score,
                    experience_score = EXCLUDED.experience_score,
                    satisfaction_score = EXCLUDED.satisfaction_score;
                """
            ).format(table_name=sql.Identifier(table_name))

            self.cursor.executemany(insert_query, data)
            self.connection.commit()
        except Exception as e:
            print(f"Error during data insertion into table '{table_name}':", e)

    def export_dataframe(self, table_name, dataframe):
        """
        Export a pandas DataFrame to the specified PostgreSQL table.
        """
        try:
            # Select the relevant columns (adjust column names to match your schema)
            transformed_df = dataframe[['IMSI', 'Engagement Score', 'Experience Score', 'Satisfaction Score']].copy()
            transformed_df.columns = ['user_id', 'engagement_score', 'experience_score', 'satisfaction_score']

            # Convert to list of tuples
            data_tuples = [tuple(row) for row in transformed_df.itertuples(index=False, name=None)]
            
            # Debugging output
            print(f"Prepared data tuples for insertion into table '{table_name}':", data_tuples[:5])

            # Insert data
            self.insert_data(table_name, data_tuples)
        except Exception as e:
            print("Error in exporting DataFrame:", e)
    def close_connection(self):
        self.cursor.close()
        self.connection.close()





