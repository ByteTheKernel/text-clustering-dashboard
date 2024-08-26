import plotly.graph_objects as go
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator


def calculate_kn_distance(data, min_pts):
    knn = NearestNeighbors(n_neighbors=min_pts)
    knn.fit(data)
    distances, _ = knn.kneighbors(data)
    return distances[:, -1]


def plot_kn_distance(kn_distance, elbow_point):
    sorted_kn_distance = sorted(kn_distance)
    fig = go.Figure()

    # График k-расстояний
    fig.add_trace(go.Scatter(x=list(range(len(sorted_kn_distance))), y=sorted_kn_distance, mode='lines+markers',
                             name='k-distance', line=dict(color='royalblue'), marker=dict(color='royalblue')))
    fig.add_vline(x=elbow_point, line=dict(color='red', dash='dash'), name='Elbow Point')

    # Настройка осей и стиля
    fig.update_layout(
        title='k-distance plot for DBSCAN',
        xaxis=dict(title='Data points'),
        yaxis=dict(title='k-distance'),
        legend=dict(x=0.1, y=1.1),
    )

    return fig


def find_optimal_epsilon(data, min_pts=4):
    kn_distance = calculate_kn_distance(data, min_pts)
    kneedle = KneeLocator(range(len(kn_distance)), kn_distance, curve='convex', direction='decreasing')
    optimal_epsilon = kn_distance[kneedle.elbow]

    # Построение графика k-расстояний
    plot = plot_kn_distance(kn_distance, kneedle.elbow)

    return optimal_epsilon, plot
