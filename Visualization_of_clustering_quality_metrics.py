import plotly.graph_objs as go


def generate_metric_figures(kmeans_time, dbscan_time, kmeans_memory, dbscan_memory, kmeans_silhouette,
                            dbscan_silhouette):
    # Создание графика сравнения времени и памяти
    time_memory_fig = create_comparison_visualization(kmeans_time, dbscan_time, kmeans_memory, dbscan_memory)

    # Создание графика сравнения силуэтных коэффициентов
    silhouette_fig = create_silhouette_comparison(kmeans_silhouette, dbscan_silhouette)

    # Словарь для быстрого доступа к графикам
    metric_figures = {
        'time_memory': time_memory_fig,
        'silhouette': silhouette_fig,
    }

    return metric_figures


def create_comparison_visualization(kmeans_time, hdbscan_time, kmeans_memory, hdbscan_memory):
    # Создаем данные для графиков
    data = [
        go.Bar(
            x=['K-Means', 'DBSCAN'],
            y=[kmeans_time, hdbscan_time],
            name='Время (секунды)',
            marker=dict(color='blue'),
            text=[kmeans_time, hdbscan_time],
            textposition='auto'
        ),
        go.Bar(
            x=['K-Means', 'DBSCAN'],
            y=[kmeans_memory, hdbscan_memory],
            name='Память (МиБ)',
            marker=dict(color='green'),
            text=[kmeans_memory, hdbscan_memory],
            textposition='auto'
        )
    ]

    # Создаем декорации для графиков
    layout = go.Layout(
        title='Сравнение времени и памяти',
        xaxis=dict(title='Алгоритмы'),
        yaxis=dict(title='Значения'),
        barmode='group'
    )

    # Собираем все в один график
    fig = go.Figure(data=data, layout=layout)

    return fig


def create_silhouette_comparison(kmeans_silhouette, dbscan_silhouette):
    # Создаем данные для графика
    data = [
        go.Bar(
            x=['K-Means', 'DBSCAN'],
            y=[kmeans_silhouette, dbscan_silhouette],
            name='Silhouette Score',
            marker=dict(color=['blue', 'green']),
            width=0.4,  # Устанавливаем ширину столбцов
            text=[kmeans_silhouette, dbscan_silhouette],  # Добавляем значения столбцов
            textposition='auto'  # Автоматическое позиционирование текста
        )
    ]

    # Создаем декорации для графика
    layout = go.Layout(
        title='Сравнение силуэтных коэффициентов',
        xaxis=dict(title='Алгоритмы'),
        yaxis=dict(title='Силуэтный коэффициент'),
        barmode='group'
    )

    # Собираем все в один график
    fig = go.Figure(data=data, layout=layout)

    return fig
