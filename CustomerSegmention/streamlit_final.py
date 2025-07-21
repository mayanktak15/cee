import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import io
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
        return fig
        
    def prepare_data_for_clustering(self):
        """Prepare and scale data for clustering"""
        features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
        clustering_data = self.data[features]
        self.scaled_data = self.scaler.fit_transform(clustering_data)
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
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(k_range, sse, 'bo-', linewidth=2, markersize=8)
        ax1.set_title('Elbow Method for Optimal K')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Sum of Squared Errors (SSE)')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
        ax2.set_title('Silhouette Score for Different K')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        optimal_k = k_range[np.argmax(silhouette_scores)]
        return fig, optimal_k
    
    def perform_kmeans_clustering(self, n_clusters=5):
        """Perform K-Means clustering"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.data['KMeans_Cluster'] = kmeans.fit_predict(self.scaled_data)
        self.clusters['kmeans'] = kmeans
        return self.data['KMeans_Cluster']
    
    def perform_hierarchical_clustering(self, n_clusters=5):
        """Perform Hierarchical clustering"""
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        self.data['Hierarchical_Cluster'] = hierarchical.fit_predict(self.scaled_data)
        self.clusters['hierarchical'] = hierarchical
        return self.data['Hierarchical_Cluster']
    
    def perform_dbscan_clustering(self, eps=0.5, min_samples=5):
        """Perform DBSCAN clustering"""
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.data['DBSCAN_Cluster'] = dbscan.fit_predict(self.scaled_data)
        self.clusters['dbscan'] = dbscan
        return self.data['DBSCAN_Cluster']
    
    def visualize_clusters(self):
        """Create comprehensive cluster visualizations"""
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
        
        return fig
    
    def analyze_clusters(self):
        """Analyze cluster characteristics"""
        cluster_analysis = self.data.groupby('KMeans_Cluster').agg({
            'Age': ['mean', 'std'],
            'Annual Income (k$)': ['mean', 'std'],
            'Spending Score (1-100)': ['mean', 'std'],
            'CustomerID': 'count'
        }).round(2)
        
        cluster_analysis.columns = ['Age_Mean', 'Age_Std', 'Income_Mean', 'Income_Std', 
                                   'Spending_Mean', 'Spending_Std', 'Count']
        
        gender_cluster = pd.crosstab(self.data['KMeans_Cluster'], self.data['Gender'], normalize='index') * 100
        
        return cluster_analysis, gender_cluster.round(1)
    
    def create_cluster_profiles(self):
        """Create detailed cluster profiles with business insights"""
        cluster_stats = self.data.groupby('KMeans_Cluster').agg({
            'Age': 'mean',
            'Annual Income (k$)': 'mean', 
            'Spending Score (1-100)': 'mean',
            'CustomerID': 'count'
        }).round(1)
        
        profiles = {}
        for cluster in cluster_stats.index:
            age = cluster_stats.loc[cluster, 'Age']
            income = cluster_stats.loc[cluster, 'Annual Income (k$)']
            spending = cluster_stats.loc[cluster, 'Spending Score (1-100)']
            count = cluster_stats.loc[cluster, 'CustomerID']
            
            if income < 40 and spending < 40:
                profile_name = "Budget-Conscious Customers"
                description = "Low income, low spending - Price-sensitive segment"
                strategy = "Focus on value propositions, discounts, and affordable products"
            elif income < 40 and spending > 60:
                profile_name = "Young Spenders"
                description = "Low income but high spending - Potentially younger customers"
                strategy = "Target with trendy, affordable products and payment plans"
            elif income > 60 and spending < 40:
                profile_name = "Conservative High-Earners"
                description = "High income but low spending - Conservative spenders"
                strategy = "Focus on quality, investment products, and long-term value"
            elif income > 60 and spending > 60:
                profile_name = "Premium Customers"
                description = "High income, high spending - Most valuable segment"
                strategy = "Premium products, exclusive offers, VIP treatment"
            else:
                profile_name = "Moderate Customers"
                description = "Balanced income and spending patterns"
                strategy = "Diverse product portfolio, targeted promotions"
            
            profiles[cluster] = {
                'Profile': profile_name,
                'Description': description,
                'Strategy': strategy,
                'Age': age,
                'Income': income,
                'Spending': spending,
                'Count': count,
                'Percentage': (count / len(self.data) * 100)
            }
        
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
        
        return fig

def main():
    """Main function to run the Streamlit application"""
    st.set_page_config(layout="wide", page_title="Customer Segmentation Analysis")
    st.title("Interactive Customer Segmentation Analysis")

    st.sidebar.title("Controls")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        segmentation = CustomerSegmentation(uploaded_file)
        
        st.sidebar.header("Analysis Steps")
        show_data = st.sidebar.checkbox("Show Raw Data", True)
        show_dist = st.sidebar.checkbox("Show Data Distributions", True)
        run_clustering = st.sidebar.checkbox("Perform Clustering", True)

        if show_data:
            st.header("1. Dataset Overview")
            data = segmentation.load_and_explore_data()
            st.dataframe(data.head())
            
            st.subheader("Data Information")
            buffer = io.StringIO()
            data.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)

            st.subheader("Statistical Summary")
            st.dataframe(data.describe())

            st.subheader("Missing Values")
            st.dataframe(data.isnull().sum())

        if show_dist:
            st.header("2. Data Distribution Visualizations")
            fig = segmentation.visualize_data_distribution()
            st.pyplot(fig)

        if run_clustering:
            st.header("3. Clustering Analysis")
            segmentation.prepare_data_for_clustering()
            
            st.subheader("Finding Optimal Number of Clusters (K-Means)")
            max_k = st.sidebar.slider("Max K for Elbow Method", 2, 15, 10)
            fig_optimal_k, optimal_k = segmentation.find_optimal_clusters_kmeans(max_k)
            st.pyplot(fig_optimal_k)
            st.success(f"Optimal number of clusters found: {optimal_k}")

            st.sidebar.header("Clustering Parameters")
            n_clusters = st.sidebar.number_input("Number of Clusters (K-Means/Hierarchical)", 2, 15, optimal_k)
            dbscan_eps = st.sidebar.slider("DBSCAN Epsilon", 0.1, 2.0, 0.6, 0.1)
            dbscan_min_samples = st.sidebar.slider("DBSCAN Min Samples", 2, 10, 4)

            segmentation.perform_kmeans_clustering(n_clusters)
            segmentation.perform_hierarchical_clustering(n_clusters)
            segmentation.perform_dbscan_clustering(eps=dbscan_eps, min_samples=dbscan_min_samples)

            st.header("4. Cluster Visualization")
            cluster_fig = segmentation.visualize_clusters()
            st.plotly_chart(cluster_fig, use_container_width=True)

            st.header("5. Cluster Analysis (K-Means)")
            cluster_analysis, gender_cluster = segmentation.analyze_clusters()
            st.subheader("Cluster Characteristics")
            st.dataframe(cluster_analysis)
            st.subheader("Gender Distribution by Cluster")
            st.dataframe(gender_cluster)

            st.header("6. Customer Segment Profiles")
            profiles = segmentation.create_cluster_profiles()
            for cluster, profile in profiles.items():
                with st.expander(f"CLUSTER {cluster}: {profile['Profile']}"):
                    st.write(f"**Size:** {profile['Count']} customers ({profile['Percentage']:.1f}%)")
                    st.write(f"**Average Age:** {profile['Age']:.1f} years")
                    st.write(f"**Average Income:** ${profile['Income']:.1f}k")
                    st.write(f"**Average Spending Score:** {profile['Spending']:.1f}/100")
                    st.write(f"**Description:** {profile['Description']}")
                    st.write(f"**Marketing Strategy:** {profile['Strategy']}")

            st.header("7. 3D Cluster Visualization")
            fig_3d = segmentation.create_3d_visualization()
            st.plotly_chart(fig_3d, use_container_width=True)
            
            st.header("8. Download Results")
            csv = segmentation.data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Segmentation Results as CSV",
                data=csv,
                file_name='customer_segmentation_results.csv',
                mime='text/csv',
            )

    else:
        st.info("Awaiting for CSV file to be uploaded.")
        st.warning("Please upload a CSV file with 'Age', 'Annual Income (k$)', and 'Spending Score (1-100)' columns.")

if __name__ == "__main__":
    import io
    main()
