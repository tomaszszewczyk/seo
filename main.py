import requests
import numpy as np
import pandas as pd
import argparse
from sklearn.cluster import AgglomerativeClustering

def get_embeddings(keywords, model='mxbai-embed-large', url='http://localhost:11434/api/embeddings'):
    embeddings = []
    for keyword in keywords:
        response = requests.post(url, json={
            'model': model,
            'prompt': keyword
        }).json()
        embeddings.append(response['embedding'])
    return np.array(embeddings)

def cluster(embeddings, distance_threshold, n_clusters=None):
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        distance_threshold=distance_threshold,
        metric='cosine',
        linkage='average'
    )
    clustering.fit(embeddings)

    return clustering.labels_

def group_keywords(df, embeddings, is_pl, distance_threshold=0.5, n_clusters=None):
    # Extract keywords
    keywords = df['Keyword'].tolist()
    
    # Cluster
    labels = cluster(embeddings, distance_threshold, n_clusters)
    
    # Create result dictionary with cluster metrics
    clusters = {}
    for (keyword, volume, difficulty), label in zip(
        df[['Keyword', 'Volume', 'Keyword Difficulty']].itertuples(index=False, name=None), 
        labels
    ):
        if label not in clusters:
            clusters[label] = {
                'keywords': [],
                'volumes': [],
                'difficulties': []
            }
        clusters[label]['keywords'].append(keyword)
        clusters[label]['volumes'].append(int(volume) if pd.notna(volume) else 0)
        clusters[label]['difficulties'].append(float(difficulty) if pd.notna(difficulty) else 0)
    
    # Calculate metrics for each cluster
    result_clusters = []
    for label, data in clusters.items():
        total_volume = sum(data['volumes'])
        avg_difficulty = sum(data['difficulties']) / len(data['difficulties']) if data['difficulties'] else 0
        result_clusters.append({
            'cluster_id': label,
            'is_pl': is_pl,
            'keywords': data['keywords'],
            'total_volume': total_volume,
            'avg_difficulty': avg_difficulty
        })
    
    return result_clusters

def main(input_file, output_file):
    df = pd.read_csv(input_file)
    
    pl_df = df[df['Database'].apply(lambda x: 'PL' in x.upper())]
    non_pl_df = df[~df['Database'].apply(lambda x: 'PL' in x.upper())]

    pl_keywords = pl_df['Keyword'].tolist()
    non_pl_keywords = non_pl_df['Keyword'].tolist()
    
    print(f"Getting embeddings for {len(pl_keywords)} PL keywords...")
    pl_embeddings = get_embeddings(pl_keywords)
    
    print(f"Getting embeddings for {len(non_pl_keywords)} non-PL keywords...")
    non_pl_embeddings = get_embeddings(non_pl_keywords)

    for t in [0.5, 0.4, 0.3, 0.2, 0.1]:
        print(f"Clustering with threshold {t}...")

        pl_clusters = group_keywords(pl_df, pl_embeddings, is_pl=True, distance_threshold=t)
        non_pl_clusters = group_keywords(non_pl_df, non_pl_embeddings, is_pl=False, distance_threshold=t)
        
        all_clusters = pl_clusters + non_pl_clusters
        
        with open(output_file + f'_t{t}.txt', 'w') as f:
            for cluster in all_clusters:
                f.write(f"Group {cluster['cluster_id']} details:\n")
                f.write(f"Type: {'PL' if cluster['is_pl'] else 'Non-PL'}\n")
                f.write(f"Total Volume: {cluster['total_volume']}\n")
                f.write(f"Avg Difficulty: {cluster['avg_difficulty']:.2f}\n")
                f.write("Keywords:\n")
                for keyword in cluster['keywords']:
                    f.write(f" - {keyword}\n")
                f.write("\n")
        
        print(f"Clustering complete. Results written to {output_file + f'_t{t}.txt'}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cluster keywords from a CSV file.')
    parser.add_argument('input_file', help='Path to the input CSV file')
    parser.add_argument('--output_file', default='clusters_output', help='Path to the output file')
    args = parser.parse_args()

    main(args.input_file, args.output_file)
