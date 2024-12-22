import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

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




