from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
from kneed import KneeLocator


def find_optimal_clusters(data, max_k=10):
    # Список для хранения суммы квадратов расстояний
    sse = []
    k_values = range(2, max_k + 1)

    # Применяем K-means с разным количеством кластеров и считаем SSE и силуэтный коэффициент
    for k in k_values:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)

    # Вычисляем "изгиб" графика, где уменьшение SSE замедляется
    kneedle = KneeLocator(k_values, sse, curve='convex', direction='decreasing')
    knee_loc = kneedle.elbow

    plot = plot_elbow_and_silhouette(k_values, sse, knee_loc)

    return knee_loc, plot


def plot_elbow_and_silhouette(k_values, sse, knee_loc):
    k_values = list(k_values)

    fig = go.Figure()

    # График метода локтя
    fig.add_trace(go.Scatter(x=k_values, y=sse, mode='lines+markers', name='SSE'))
    fig.add_vline(x=knee_loc, line=dict(color='red', dash='dash'), name='Elbow Point')

    # Настройка осей
    fig.update_layout(
        title='Метод локтя',
        xaxis=dict(title='Количество кластеров'),
        yaxis=dict(title='SSE'),
        legend=dict(x=0.1, y=1.1)
    )

    return fig
