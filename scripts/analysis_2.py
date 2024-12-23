import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer

class TelecomEngagementAnalysis:
    def __init__(self, dataframe):
        """
        Initializes the TelecomEngagementAnalysis class with the prepared dataframe.

        Args:
            dataframe (pd.DataFrame): Cleaned and prepared dataframe.
        """
        self.data = dataframe

    def aggregate_metrics(self):
        """
        Aggregate the engagement metrics per customer ID (MSISDN).

        Returns:
            pd.DataFrame: Aggregated metrics.
        """
        self.data = self.data.copy()  # Create a copy to avoid SettingWithCopyWarning
        self.data['total_traffic'] = self.data['Total UL (Bytes)'] + self.data['Total DL (Bytes)']
        agg_data = self.data.groupby('MSISDN/Number').agg(
            session_count=('MSISDN/Number', 'count'),
            total_duration=('Dur. (ms)', 'sum'),
            total_traffic=('total_traffic', 'sum')
        ).reset_index()
        return agg_data

    def get_top_customers(self, metric, top_n=10):
        """
        Get the top customers based on a specific metric.

        Args:
            metric (str): The metric to rank customers by ('session_count', 'total_duration', 'total_traffic').
            top_n (int): Number of top customers to return.

        Returns:
            pd.DataFrame: Top customers for the specified metric.
        """
        agg_data = self.aggregate_metrics()
        return agg_data.nlargest(top_n, metric)

    def normalize_and_cluster(self, n_clusters=3):
        """
        Normalize engagement metrics and run k-means clustering.

        Args:
            n_clusters (int): Number of clusters for k-means.

        Returns:
            tuple: Cluster labels and the dataframe with normalized metrics.
        """
        agg_data = self.aggregate_metrics()
        metrics = agg_data[['session_count', 'total_duration', 'total_traffic']]
        scaler = MinMaxScaler()
        normalized_metrics = scaler.fit_transform(metrics)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        agg_data['cluster'] = kmeans.fit_predict(normalized_metrics)

        return agg_data, normalized_metrics, kmeans

    def compute_cluster_stats(self, clustered_data):
        """
        Compute cluster statistics for non-normalized metrics.

        Args:
            clustered_data (pd.DataFrame): Dataframe with cluster labels.

        Returns:
            pd.DataFrame: Cluster statistics.
        """
        cluster_stats = clustered_data.groupby('cluster').agg(
            min_session_count=('session_count', 'min'),
            max_session_count=('session_count', 'max'),
            avg_session_count=('session_count', 'mean'),
            total_session_count=('session_count', 'sum'),
            min_total_duration=('total_duration', 'min'),
            max_total_duration=('total_duration', 'max'),
            avg_total_duration=('total_duration', 'mean'),
            total_total_duration=('total_duration', 'sum'),
            min_total_traffic=('total_traffic', 'min'),
            max_total_traffic=('total_traffic', 'max'),
            avg_total_traffic=('total_traffic', 'mean'),
            total_total_traffic=('total_traffic', 'sum')
        ).reset_index()
        return cluster_stats

    def traffic_by_application(self):
        """
        Aggregate user total traffic per application and derive top 10 users per application.

        Returns:
            dict: Top 10 users per application.
        """
        application_columns = [
            'Social Media DL (Bytes)', 'Social Media UL (Bytes)',
            'Google DL (Bytes)', 'Google UL (Bytes)',
            'Email DL (Bytes)', 'Email UL (Bytes)',
            'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
            'Netflix DL (Bytes)', 'Netflix UL (Bytes)',
            'Gaming DL (Bytes)', 'Gaming UL (Bytes)',
            'Other DL (Bytes)', 'Other UL (Bytes)'
        ]

        top_users_per_app = {}
        for app in application_columns:
            app_traffic = self.data.groupby('MSISDN/Number')[app].sum().reset_index()
            top_users = app_traffic.nlargest(10, app)
            top_users_per_app[app] = top_users

        return top_users_per_app
    ''' 
    def aggregate_metrics(self):
        """
        Aggregate the engagement metrics per customer ID (MSISDN).

        Returns:
            pd.DataFrame: Aggregated metrics.
        """
        self.data = self.data.copy()  # Create a copy to avoid SettingWithCopyWarning
        self.data['total_traffic'] = self.data['Total UL (Bytes)'] + self.data['Total DL (Bytes)']
        agg_data = self.data.groupby('MSISDN/Number').agg(
            session_count=('MSISDN/Number', 'count'),
            total_duration=('Dur. (ms)', 'sum'),
            total_traffic=('total_traffic', 'sum')
        ).reset_index()
        return agg_data
    def __init__(self, dataframe):
        """
        Initializes the TelecomEngagementAnalysis class with the prepared dataframe.

        Args:
            dataframe (pd.DataFrame): Cleaned and prepared dataframe.
        """
        self.data = dataframe
        '''
    def plot_top_applications(self):
        """
        Plot the top 3 most used applications using appropriate charts.

        Returns:
            None
        """
        application_columns = [
            'Social Media DL (Bytes)', 'Google DL (Bytes)', 'Youtube DL (Bytes)',
            'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)'
        ]

        app_traffic = self.data[application_columns].sum().sort_values(ascending=False)
        top_3_apps = app_traffic.head(3)

        plt.figure(figsize=(10, 6))
        top_3_apps.plot(kind='bar', color=['skyblue', 'orange', 'green'])
        plt.title('Top 3 Most Used Applications')
        plt.ylabel('Total Traffic (Bytes)')
        plt.xlabel('Application')
        plt.xticks(rotation=45)
        plt.show()

    def find_optimal_k(self):
        """
        Use the elbow method to find the optimal number of clusters for k-means.

        Returns:
            None
        """
       
        aggregate_metrics = TelecomEngagementAnalysis.aggregate_metrics(self)
        agg_data = self.aggregate_metrics()
        metrics = agg_data[['session_count', 'total_duration', 'total_traffic']]
        scaler = MinMaxScaler()
        normalized_metrics = scaler.fit_transform(metrics)

        model = KMeans(random_state=42)
        visualizer = KElbowVisualizer(model, k=(2, 10))
        visualizer.fit(normalized_metrics)
        visualizer.show()

# Example usage:
# analysis = TelecomEngagementAnalysis(dataframe=cleaned_dataframe)
# top_sessions = analysis.get_top_customers(metric='session_count')
# clustered_data, normalized_metrics, kmeans = analysis.normalize_and_cluster()
# cluster_stats = analysis.compute_cluster_stats(clustered_data)
# analysis.plot_top_applications()
# analysis.find_optimal_k()
# top_users_per_app = analysis.traffic_by_application()
# print(cluster_stats)
