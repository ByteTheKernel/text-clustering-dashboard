import plotly.express as px
from umap import UMAP
import pandas as pd


def generate_clustering_figures(messages_embeddings, kmeans_labels, dbscan_labels, messages):
    # Создание 2D и 3D проекций с помощью UMAP
    reducer_2d = UMAP(n_components=2)
    embedding_2d = reducer_2d.fit_transform(messages_embeddings)

    reducer_3d = UMAP(n_components=3)
    embedding_3d = reducer_3d.fit_transform(messages_embeddings)

    # Создание DataFrame для K-means
    df_2d_kmeans = create_dataframe(embedding_2d, kmeans_labels, messages, n_components=2)
    df_3d_kmeans = create_dataframe(embedding_3d, kmeans_labels, messages, n_components=3)

    # Создание DataFrame для DBSCAN
    df_2d_hdbscan = create_dataframe(embedding_2d, dbscan_labels, messages, n_components=2)
    df_3d_hdbscan = create_dataframe(embedding_3d, dbscan_labels, messages, n_components=3)

    # Фильтрация шума для DBSCAN
    cleaned_labels_2d, cleaned_embedding_2d, cleaned_messages_2d = filter_noise(dbscan_labels, embedding_2d, messages)
    df_2d_hdbscan_cleaned = create_dataframe(cleaned_embedding_2d, cleaned_labels_2d, cleaned_messages_2d,
                                             n_components=2)

    cleaned_labels_3d, cleaned_embedding_3d, cleaned_messages_3d = filter_noise(dbscan_labels, embedding_3d, messages)
    df_3d_hdbscan_cleaned = create_dataframe(cleaned_embedding_3d, cleaned_labels_3d, cleaned_messages_3d,
                                             n_components=3)

    # Создание графиков для K-means
    kmeans_2d_fig = visualize_clustering(df_2d_kmeans, n_components=2, title='K-means Clustering')
    kmeans_3d_fig = visualize_clustering(df_3d_kmeans, n_components=3, title='K-means Clustering')

    # Создание графиков для DBSCAN
    dbscan_2d_fig = visualize_clustering(df_2d_hdbscan, n_components=2, title='DBSCAN Clustering')
    dbscan_3d_fig = visualize_clustering(df_3d_hdbscan, n_components=3, title='DBSCAN Clustering')

    # Создание графиков для DBSCAN (без шума)
    dbscan_2d_cleaned_fig = visualize_clustering(df_2d_hdbscan_cleaned, n_components=2, title='DBSCAN Clustering (без шума)')
    dbscan_3d_cleaned_fig = visualize_clustering(df_3d_hdbscan_cleaned, n_components=3, title='DBSCAN Clustering (без шума)')

    # Возвращаем словарь с графиками
    figures = {
        'kmeans': {
            '2D': kmeans_2d_fig,
            '3D': kmeans_3d_fig,
        },
        'dbscan': {
            '2D': dbscan_2d_fig,
            '3D': dbscan_3d_fig,
            '2D (без шума)': dbscan_2d_cleaned_fig,
            '3D (без шума)': dbscan_3d_cleaned_fig,
        }
    }

    return figures


def create_dataframe(embedding, labels, messages, n_components=2):
    data = {'Cluster': labels, 'Message': messages}
    for i in range(n_components):
        data[f'UMAP{i + 1}'] = embedding[:, i]
    return pd.DataFrame(data)


def visualize_clustering(df, n_components=2, title='Clustering', dimensions=None):
    if n_components == 3:
        fig = px.scatter_3d(df, x='UMAP1', y='UMAP2', z='UMAP3', color='Cluster', title=title,
                            labels={'x': 'UMAP Dimension 1', 'y': 'UMAP Dimension 2', 'z': 'UMAP Dimension 3'},
                            category_orders={'Cluster': sorted(df['Cluster'].unique())},
                            color_discrete_sequence=px.colors.qualitative.Plotly)
        fig.update_layout(
            scene=dict(
                xaxis_title='UMAP Dimension 1',
                yaxis_title='UMAP Dimension 2',
                zaxis_title='UMAP Dimension 3',
                camera=dict(eye=dict(x=0, y=2, z=0.1))  # Начальный угол обзора для более горизонтального отображения
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            hovermode='closest'
        )
    else:
        fig = px.scatter(df, x='UMAP1', y='UMAP2', color='Cluster', title=title,
                         labels={'x': 'UMAP Dimension 1', 'y': 'UMAP Dimension 2'},
                         category_orders={'Cluster': sorted(df['Cluster'].unique())},
                         color_discrete_sequence=px.colors.qualitative.Plotly)

    fig.update_traces(marker=dict(size=6, opacity=0.8), selector=dict(mode='markers'))
    fig.update_traces(
        hovertemplate="<b>Кластер:</b> %{customdata[0]}<br>"
                      "<b>Сообщение:</b> %{customdata[1]}",
        customdata=df[['Cluster', 'Message']]
    )
    return fig


def filter_noise(labels, embedding, messages):
    cleaned_labels = [label for label in labels if label != -1]
    cleaned_embedding = embedding[labels != -1]
    cleaned_messages = messages[labels != -1]
    return cleaned_labels, cleaned_embedding, cleaned_messages
