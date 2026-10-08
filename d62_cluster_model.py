import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

df = pd.read_csv('data/Customer-Churn.csv')
rfm = pd.DataFrame()
rfm['recency']   = df['tenure']
rfm['frequency'] = (df[['PhoneService', 'StreamingTV', 'StreamingMovies', 
                          'OnlineSecurity', 'OnlineBackup']].apply(
                          lambda col: col.map({'Yes': 1, 'No': 0, 'No internet service': 0})).sum(axis=1))
rfm['monetary']  = df['MonthlyCharges']

scaler = StandardScaler()
scaled_data = scaler.fit_transform(rfm.values)
kmeans_model = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_model.fit(scaled_data)

rfm["cluster"] = kmeans_model.labels_

pca = PCA(n_components=2)
reduced = pca.fit_transform(scaled_data)

plt.figure(figsize=(10, 6))
colors = ['#2196F3', '#4CAF50', '#F44336']
for cluster_id in [0, 1, 2]:
    mask = rfm['cluster'] == cluster_id
    plt.scatter(reduced[mask, 0], reduced[mask, 1],
                c=colors[cluster_id], label=f'Cluster {cluster_id}', alpha=0.5, s=20)

plt.title('Customer Segments (PCA)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.tight_layout()
plt.savefig('segments_plot.png')
print('Plot saved to segments_plot.png')

cluster_stats = rfm.groupby('cluster')[['recency', 'frequency', 'monetary']].mean().round(2)

print("\n=== Cluster Stats ===")
print(cluster_stats)

print("\n=== LLM Business Descriptions ===")
for cluster_id in [0, 1, 2]:
    stats = cluster_stats.loc[cluster_id]
    response = llm.invoke(f"""You are a business analyst. Describe this customer segment in 2 sentences of plain business English.
                          Cluster {cluster_id} average stats:
                          - Recency (tenure in months): {stats['recency']}
                          - Frequency (number of services used): {stats['frequency']}
                          - Monetary (monthly charges $): {stats['monetary']}
                          Give the cluster a business name and explain what action the business should take.""")
    print(f"\nCluster {cluster_id}: {response.content}")