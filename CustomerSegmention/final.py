"""
Customer Segmentation Analysis
Grouping customers based on their purchasing behavior and demographics to target marketing strategies effectively.
Resources: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

This script performs comprehensive customer segmentation using:
1. K-Means Clustering
2. Hierarchical Clustering
3. DBSCAN Clustering
4. Data visualization and analysis
"""

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CustomerSegmentation:
    def __init__(self, data_file):
        """Initialize with customer data"""
        self.data = pd.read_csv(data_file)
        self.scaled_data = None
        self.clusters = {}
        self.scaler = StandardScaler()
        
    def load_and_explore_data(self):
        """Load and perform initial data exploration"""
        print("=== CUSTOMER SEGMENTATION ANALYSIS ===\n")
        print("1. Dataset Overview:")
        print(f"Dataset shape: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")
        print("\nFirst 5 rows:")
        print(self.data.head())
        
        print("\n2. Data Information:")
        print(self.data.info())
        
        print("\n3. Statistical Summary:")
        print(self.data.describe())
        
        print("\n4. Missing Values:")
        print(self.data.isnull().sum())
        
        return self.data
    
    def visualize_data_distribution(self):
        """Create visualizations for data distribution"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Customer Data Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Age distribution
        axes[0, 0].hist(self.data['Age'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Age Distribution')
        axes[0, 0].set_xlabel('Age')
        axes[0, 0].set_ylabel('Frequency')
        
        # Annual Income distribution
        axes[0, 1].hist(self.data['Annual Income (k$)'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('Annual Income Distribution')
        axes[0, 1].set_xlabel('Annual Income (k$)')
        axes[0, 1].set_ylabel('Frequency')
        
        # Spending Score distribution
        axes[0, 2].hist(self.data['Spending Score (1-100)'], bins=20, alpha=0.7, color='salmon', edgecolor='black')
        axes[0, 2].set_title('Spending Score Distribution')
        axes[0, 2].set_xlabel('Spending Score (1-100)')
        axes[0, 2].set_ylabel('Frequency')
        
        # Gender distribution
        gender_counts = self.data['Gender'].value_counts()
        axes[1, 0].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', colors=['lightblue', 'pink'])
        axes[1, 0].set_title('Gender Distribution')
        
        # Income vs Spending Score
        axes[1, 1].scatter(self.data['Annual Income (k$)'], self.data['Spending Score (1-100)'], 
                          alpha=0.6, c='purple')
        axes[1, 1].set_title('Income vs Spending Score')
        axes[1, 1].set_xlabel('Annual Income (k$)')
        axes[1, 1].set_ylabel('Spending Score (1-100)')
        
        # Age vs Spending Score
        axes[1, 2].scatter(self.data['Age'], self.data['Spending Score (1-100)'], 
                          alpha=0.6, c='orange')
        axes[1, 2].set_title('Age vs Spending Score')
        axes[1, 2].set_xlabel('Age')
        axes[1, 2].set_ylabel('Spending Score (1-100)')
        
        plt.tight_layout()
        plt.show()
        
    def prepare_data_for_clustering(self):
        """Prepare and scale data for clustering"""
        # Select features for clustering (excluding CustomerID and Gender for now)
        features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
        clustering_data = self.data[features]
        
        # Scale the data
        self.scaled_data = self.scaler.fit_transform(clustering_data)
        
        print("Data prepared for clustering:")
        print(f"Features used: {features}")
        print(f"Scaled data shape: {self.scaled_data.shape}")
        
        return self.scaled_data
    
    def find_optimal_clusters_kmeans(self, max_k=10):
        """Find optimal number of clusters using Elbow method and Silhouette analysis"""
        sse = []
        silhouette_scores = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.scaled_data)
            sse.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.scaled_data, kmeans.labels_))
        
        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Elbow plot
        ax1.plot(k_range, sse, 'bo-', linewidth=2, markersize=8)
        ax1.set_title('Elbow Method for Optimal K')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Sum of Squared Errors (SSE)')
        ax1.grid(True, alpha=0.3)
        
        # Silhouette plot
        ax2.plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
        ax2.set_title('Silhouette Score for Different K')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Find optimal k
        optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"Optimal number of clusters based on Silhouette Score: {optimal_k}")
        print(f"Best Silhouette Score: {max(silhouette_scores):.3f}")
        
        return optimal_k
    
    def perform_kmeans_clustering(self, n_clusters=5):
        """Perform K-Means clustering"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.data['KMeans_Cluster'] = kmeans.fit_predict(self.scaled_data)
        self.clusters['kmeans'] = kmeans
        
        print(f"K-Means clustering completed with {n_clusters} clusters")
        print("Cluster distribution:")
        print(self.data['KMeans_Cluster'].value_counts().sort_index())
        
        return self.data['KMeans_Cluster']
    
    def perform_hierarchical_clustering(self, n_clusters=5):
        """Perform Hierarchical clustering"""
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        self.data['Hierarchical_Cluster'] = hierarchical.fit_predict(self.scaled_data)
        self.clusters['hierarchical'] = hierarchical
        
        print(f"Hierarchical clustering completed with {n_clusters} clusters")
        print("Cluster distribution:")
        print(self.data['Hierarchical_Cluster'].value_counts().sort_index())
        
        return self.data['Hierarchical_Cluster']
    
    def perform_dbscan_clustering(self, eps=0.5, min_samples=5):
        """Perform DBSCAN clustering"""
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.data['DBSCAN_Cluster'] = dbscan.fit_predict(self.scaled_data)
        self.clusters['dbscan'] = dbscan
        
        n_clusters = len(set(self.data['DBSCAN_Cluster'])) - (1 if -1 in self.data['DBSCAN_Cluster'] else 0)
        n_noise = list(self.data['DBSCAN_Cluster']).count(-1)
        
        print(f"DBSCAN clustering completed")
        print(f"Number of clusters: {n_clusters}")
        print(f"Number of noise points: {n_noise}")
        print("Cluster distribution:")
        print(self.data['DBSCAN_Cluster'].value_counts().sort_index())
        
        return self.data['DBSCAN_Cluster']
    
    def visualize_clusters(self):
        """Create comprehensive cluster visualizations"""
        # Create subplots for different clustering methods
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['K-Means Clustering', 'Hierarchical Clustering', 
                          'DBSCAN Clustering', 'Cluster Comparison'],
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}]]
        )
        
        # K-Means visualization
        for cluster in self.data['KMeans_Cluster'].unique():
            cluster_data = self.data[self.data['KMeans_Cluster'] == cluster]
            fig.add_trace(
                go.Scatter(
                    x=cluster_data['Annual Income (k$)'],
                    y=cluster_data['Spending Score (1-100)'],
                    mode='markers',
                    name=f'K-Means Cluster {cluster}',
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # Hierarchical visualization
        for cluster in self.data['Hierarchical_Cluster'].unique():
            cluster_data = self.data[self.data['Hierarchical_Cluster'] == cluster]
            fig.add_trace(
                go.Scatter(
                    x=cluster_data['Annual Income (k$)'],
                    y=cluster_data['Spending Score (1-100)'],
                    mode='markers',
                    name=f'Hierarchical Cluster {cluster}',
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # DBSCAN visualization
        for cluster in self.data['DBSCAN_Cluster'].unique():
            cluster_data = self.data[self.data['DBSCAN_Cluster'] == cluster]
            cluster_name = 'Noise' if cluster == -1 else f'DBSCAN Cluster {cluster}'
            fig.add_trace(
                go.Scatter(
                    x=cluster_data['Annual Income (k$)'],
                    y=cluster_data['Spending Score (1-100)'],
                    mode='markers',
                    name=cluster_name,
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Comparison plot (using K-Means)
        for cluster in self.data['KMeans_Cluster'].unique():
            cluster_data = self.data[self.data['KMeans_Cluster'] == cluster]
            fig.add_trace(
                go.Scatter(
                    x=cluster_data['Age'],
                    y=cluster_data['Spending Score (1-100)'],
                    mode='markers',
                    name=f'Cluster {cluster}',
                    showlegend=True
                ),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="Customer Segmentation Results")
        fig.update_xaxes(title_text="Annual Income (k$)", row=1, col=1)
        fig.update_xaxes(title_text="Annual Income (k$)", row=1, col=2)
        fig.update_xaxes(title_text="Annual Income (k$)", row=2, col=1)
        fig.update_xaxes(title_text="Age", row=2, col=2)
        fig.update_yaxes(title_text="Spending Score", row=1, col=1)
        fig.update_yaxes(title_text="Spending Score", row=1, col=2)
        fig.update_yaxes(title_text="Spending Score", row=2, col=1)
        fig.update_yaxes(title_text="Spending Score", row=2, col=2)
        
        fig.show()
    
    def analyze_clusters(self):
        """Analyze cluster characteristics"""
        print("\n=== CLUSTER ANALYSIS ===\n")
        
        # Analyze K-Means clusters
        print("K-Means Cluster Analysis:")
        print("-" * 40)
        
        cluster_analysis = self.data.groupby('KMeans_Cluster').agg({
            'Age': ['mean', 'std'],
            'Annual Income (k$)': ['mean', 'std'],
            'Spending Score (1-100)': ['mean', 'std'],
            'CustomerID': 'count'
        }).round(2)
        
        cluster_analysis.columns = ['Age_Mean', 'Age_Std', 'Income_Mean', 'Income_Std', 
                                   'Spending_Mean', 'Spending_Std', 'Count']
        
        print(cluster_analysis)
        
        # Gender distribution by cluster
        print("\nGender Distribution by K-Means Cluster:")
        print("-" * 45)
        gender_cluster = pd.crosstab(self.data['KMeans_Cluster'], self.data['Gender'], normalize='index') * 100
        print(gender_cluster.round(1))
        
        return cluster_analysis
    
    def create_cluster_profiles(self):
        """Create detailed cluster profiles with business insights"""
        print("\n=== CUSTOMER SEGMENT PROFILES ===\n")
        
        cluster_stats = self.data.groupby('KMeans_Cluster').agg({
            'Age': 'mean',
            'Annual Income (k$)': 'mean', 
            'Spending Score (1-100)': 'mean',
            'CustomerID': 'count'
        }).round(1)
        
        # Define cluster profiles based on characteristics
        profiles = {}
        for cluster in cluster_stats.index:
            age = cluster_stats.loc[cluster, 'Age']
            income = cluster_stats.loc[cluster, 'Annual Income (k$)']
            spending = cluster_stats.loc[cluster, 'Spending Score (1-100)']
            count = cluster_stats.loc[cluster, 'CustomerID']
            
            # Create profile description
            if income < 40 and spending < 40:
                profile = "Budget-Conscious Customers"
                description = "Low income, low spending - Price-sensitive segment"
                strategy = "Focus on value propositions, discounts, and affordable products"
            elif income < 40 and spending > 60:
                profile = "Young Spenders"
                description = "Low income but high spending - Potentially younger customers"
                strategy = "Target with trendy, affordable products and payment plans"
            elif income > 60 and spending < 40:
                profile = "Conservative High-Earners"
                description = "High income but low spending - Conservative spenders"
                strategy = "Focus on quality, investment products, and long-term value"
            elif income > 60 and spending > 60:
                profile = "Premium Customers"
                description = "High income, high spending - Most valuable segment"
                strategy = "Premium products, exclusive offers, VIP treatment"
            else:
                profile = "Moderate Customers"
                description = "Balanced income and spending patterns"
                strategy = "Diverse product portfolio, targeted promotions"
            
            profiles[cluster] = {
                'Profile': profile,
                'Description': description,
                'Strategy': strategy,
                'Age': age,
                'Income': income,
                'Spending': spending,
                'Count': count,
                'Percentage': (count / len(self.data) * 100)
            }
        
        # Display profiles
        for cluster, profile in profiles.items():
            print(f"CLUSTER {cluster}: {profile['Profile']}")
            print(f"Size: {profile['Count']} customers ({profile['Percentage']:.1f}%)")
            print(f"Average Age: {profile['Age']:.1f} years")
            print(f"Average Income: ${profile['Income']:.1f}k")
            print(f"Average Spending Score: {profile['Spending']:.1f}/100")
            print(f"Description: {profile['Description']}")
            print(f"Marketing Strategy: {profile['Strategy']}")
            print("-" * 60)
        
        return profiles
    
    def create_3d_visualization(self):
        """Create 3D visualization of clusters"""
        fig = go.Figure()
        
        for cluster in self.data['KMeans_Cluster'].unique():
            cluster_data = self.data[self.data['KMeans_Cluster'] == cluster]
            
            fig.add_trace(go.Scatter3d(
                x=cluster_data['Age'],
                y=cluster_data['Annual Income (k$)'],
                z=cluster_data['Spending Score (1-100)'],
                mode='markers',
                marker=dict(size=8, opacity=0.8),
                name=f'Cluster {cluster}',
                text=[f'Customer {id}<br>Age: {age}<br>Income: ${income}k<br>Spending: {spending}' 
                      for id, age, income, spending in zip(
                          cluster_data['CustomerID'], 
                          cluster_data['Age'],
                          cluster_data['Annual Income (k$)'],
                          cluster_data['Spending Score (1-100)']
                      )],
                hovertemplate='%{text}<extra></extra>'
            ))
        
        fig.update_layout(
            title='3D Customer Segmentation Visualization',
            scene=dict(
                xaxis_title='Age',
                yaxis_title='Annual Income (k$)',
                zaxis_title='Spending Score (1-100)'
            ),
            width=900,
            height=700
        )
        
        fig.show()
    
    def export_results(self):
        """Export clustering results to CSV"""
        output_file = 'customer_segmentation_results.csv'
        self.data.to_csv(output_file, index=False)
        print(f"\nResults exported to: {output_file}")
        
        # Create summary report
        summary_file = 'segmentation_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("CUSTOMER SEGMENTATION ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total Customers: {len(self.data)}\n")
            f.write(f"Features Used: Age, Annual Income, Spending Score\n")
            f.write(f"Clustering Methods: K-Means, Hierarchical, DBSCAN\n\n")
            
            # Cluster distribution
            f.write("K-Means Cluster Distribution:\n")
            for cluster in sorted(self.data['KMeans_Cluster'].unique()):
                count = len(self.data[self.data['KMeans_Cluster'] == cluster])
                percentage = count / len(self.data) * 100
                f.write(f"Cluster {cluster}: {count} customers ({percentage:.1f}%)\n")
        
        print(f"Summary report saved to: {summary_file}")

def main():
    """Main function to run the customer segmentation analysis"""
    # Initialize the segmentation analysis
    segmentation = CustomerSegmentation('Mall_Customers.csv')
    
    # Step 1: Load and explore data
    segmentation.load_and_explore_data()
    
    # Step 2: Visualize data distribution
    segmentation.visualize_data_distribution()
    
    # Step 3: Prepare data for clustering
    segmentation.prepare_data_for_clustering()
    
    # Step 4: Find optimal number of clusters
    optimal_k = segmentation.find_optimal_clusters_kmeans()
    
    # Step 5: Perform different clustering algorithms
    segmentation.perform_kmeans_clustering(n_clusters=optimal_k)
    segmentation.perform_hierarchical_clustering(n_clusters=optimal_k)
    segmentation.perform_dbscan_clustering(eps=0.6, min_samples=4)
    
    # Step 6: Visualize clusters
    segmentation.visualize_clusters()
    
    # Step 7: Analyze clusters
    segmentation.analyze_clusters()
    
    # Step 8: Create business profiles
    segmentation.create_cluster_profiles()
    
    # Step 9: Create 3D visualization
    segmentation.create_3d_visualization()
    
    # Step 10: Export results
    segmentation.export_results()
    
    print("\n" + "="*60)
    print("CUSTOMER SEGMENTATION ANALYSIS COMPLETED!")
    print("="*60)
    print("\nKey Deliverables:")
    print("1. Comprehensive data analysis and visualization")
    print("2. Multiple clustering algorithm results")
    print("3. Business-oriented customer profiles")
    print("4. Marketing strategy recommendations")
    print("5. Exported results for further analysis")

if __name__ == "__main__":
    main()