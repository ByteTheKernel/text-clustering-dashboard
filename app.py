import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from evaluate_clustering_performance import evaluate_clustering_performance
from Text_preparation import text_prep
from Tokenizer import get_sentence_embeddings
from Find_optimal_KMeans_Klusters import find_optimal_clusters
from Find_optimal_DBSCAN_params import find_optimal_epsilon
from Analyzing_resources import perform_clustering
from Summarize_clusters import create_cluster_visualizations
import plotly.graph_objects as go
from umap import UMAP
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from Visualize_clusters import generate_clustering_figures
from Visualization_of_clustering_quality_metrics import generate_metric_figures

if __name__ == '__main__':
    # Загрузка и предварительная обработка данных
    data = pd.read_csv('social_media_messages.csv')
    messages = data['message'].apply(text_prep)

    # Преобразование текстов в векторные представления с использованием ai-forever/sbert_large_nlu_ru
    # Получение векторных представлений сообщений
    messages_embeddings = get_sentence_embeddings(messages.tolist())

    reducer_512d = UMAP(n_components=512)
    embedding_512d = reducer_512d.fit_transform(messages_embeddings)

    # Применение K-means
    optimal_k, kmeans_fig = find_optimal_clusters(embedding_512d)
    kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
    kmeans_labels, kmeans_time, kmeans_memory = perform_clustering(kmeans, embedding_512d)

    # Применение DBSCAN
    best_min_samples = 4
    optimal_epsilon, dbscan_fig = find_optimal_epsilon(embedding_512d, best_min_samples)
    dbscan_clusterer = DBSCAN(eps=optimal_epsilon, min_samples=best_min_samples)
    hdbscan_labels, hdbscan_time, hdbscan_memory = perform_clustering(dbscan_clusterer, embedding_512d)

    # Словари для быстрого доступа к графикам
    optimization_figures = {
        'kmeans': {
            'Метод локтя': kmeans_fig,
        },
        'dbscan': {
            'График k-расстояний': dbscan_fig,
        }
    }

    # Генерация всех визуализаций для каждого метода кластеризации
    kmeans_visualizations = create_cluster_visualizations(messages, kmeans_labels)
    dbscan_visualizations = create_cluster_visualizations(messages, hdbscan_labels)

    # Получение сортированного списка кластеров для каждого метода
    sorted_kmeans_clusters = sorted(kmeans_visualizations.keys())
    sorted_dbscan_clusters = sorted(dbscan_visualizations.keys())

    # Основное приложение Dash
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    # Определение layout приложения
    app.layout = dbc.Container([
        # dbc.Card для визуализации подбора оптимальных параметров кластеризации
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Подбор оптимальных параметров кластеризации")),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label('Выберите метод кластеризации'),
                                dcc.Dropdown(
                                    id='optimization-method-dropdown',
                                    options=[
                                        {'label': 'K-means', 'value': 'kmeans'},
                                        {'label': 'DBSCAN', 'value': 'dbscan'},
                                    ],
                                    value='kmeans'  # Значение по умолчанию
                                )
                            ], md=6),
                            dbc.Col([
                                dbc.Label('Выберите график для отображения'),
                                dcc.Dropdown(
                                    id='optimization-plot-dropdown',
                                    options=[],  # Опции будут динамически обновляться
                                    value=None  # Значение будет установлено после выбора метода кластеризации
                                )
                            ], md=6)
                        ], className="mb-4"),

                        dbc.Row([
                            dbc.Col(
                                dcc.Graph(id='optimization-visualization')
                            )
                        ])
                    ])
                ], className="mb-4")
            ])
        ]),

        # dbc.Card для визуализации результатов кластеризации
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Визуализация результатов кластеризации")),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label('Выберите метод кластеризации для графиков'),
                                dcc.Dropdown(
                                    id='visualization-method-dropdown',
                                    options=[
                                        {'label': 'K-means', 'value': 'kmeans'},
                                        {'label': 'DBSCAN', 'value': 'dbscan'},
                                    ],
                                    value='kmeans'  # Значение по умолчанию
                                )
                            ], md=6),
                            dbc.Col([
                                dbc.Label('Выберите тип визуализации'),
                                dcc.Dropdown(
                                    id='visualization-dropdown',
                                    options=[
                                        {'label': '2D', 'value': '2D'},
                                        {'label': '3D', 'value': '3D'},
                                        {'label': '2D (без шума)', 'value': '2D (без шума)'},
                                        {'label': '3D (без шума)', 'value': '3D (без шума)'},
                                    ],
                                    value='2D'  # Значение по умолчанию
                                )
                            ], md=6)
                        ], className="mb-4"),

                        dbc.Row([
                            dbc.Col(
                                dcc.Graph(id='clustering-visualization')  # График отображается моментально
                            )
                        ])
                    ])
                ], className="mb-4")
            ])
        ]),

        # dbc.Card для подробного анализа кластеров
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Подробный анализ кластеров")),
                    dbc.CardBody(
                        [
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Выберите метод кластеризации для графиков:"),
                                    dcc.Dropdown(
                                        id="method-dropdown",
                                        options=[
                                            {'label': 'K-means', 'value': 'kmeans'},
                                            {'label': 'DBSCAN', 'value': 'dbscan'}
                                        ],
                                        value='kmeans',
                                        style={'marginBottom': '10px'}
                                    )
                                ], md=6),  # Используем md=6, чтобы Dropdown занимал половину ширины строки
                                dbc.Col([
                                    dbc.Label("Выберите кластер:"),
                                    dcc.Dropdown(
                                        id="cluster-dropdown",
                                        style={'marginBottom': '10px'}
                                    )
                                ], md=6),  # Используем md=6 для второго Dropdown
                            ], className="mb-4"),

                            dbc.Row([
                                dbc.Col(dcc.Loading(
                                    id="loading-bar-chart",
                                    children=[dcc.Graph(id='bar-chart')],
                                    type="default"
                                ), md=4),
                                dbc.Col([
                                    dcc.Tabs(
                                        id='tabs-visualizations',
                                        children=[
                                            dcc.Tab(
                                                label="Treemap",
                                                children=[dcc.Loading(
                                                    id="loading-treemap",
                                                    children=[dcc.Graph(id='treemap')],
                                                    type="default"
                                                )]
                                            ),
                                            dcc.Tab(
                                                label="Wordcloud",
                                                children=[dcc.Loading(
                                                    id="loading-wordcloud",
                                                    children=[dcc.Graph(id='wordcloud')],
                                                    type="default"
                                                )]
                                            )
                                        ],
                                    )
                                ], md=8),
                            ]),
                            dbc.Row([
                                dbc.Button(
                                    "Текст кластера",
                                    id="collapse-button",
                                    className="mb-3",
                                    color="primary"
                                ),
                                dbc.Collapse(
                                    dbc.CardBody(id="cluster-table"),
                                    id="collapse",
                                    is_open=False,
                                ),
                            ]),
                        ]
                    ),
                ], className="mb-4")
            ])
        ]),

        # Новый dbc.Card для метрик качества кластеризации
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Метрики качества кластеризации")),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label('Выберите метрику для отображения'),
                                dcc.Dropdown(
                                    id='metrics-method-dropdown',
                                    options=[
                                        {'label': 'Сравнение времени и памяти', 'value': 'time_memory'},
                                        {'label': 'Сравнение силуэтных коэффициентов', 'value': 'silhouette'},
                                    ],
                                    value='time_memory',  # Значение по умолчанию
                                )
                            ], md=12)
                        ], className="mb-4"),

                        dbc.Row([
                            dbc.Col(
                                dcc.Graph(id='metrics-visualization')  # График отображается моментально
                            )
                        ])
                    ])
                ])
            ])
        ])
    ])

    # Callback для обновления доступных графиков в зависимости от метода кластеризации
    @app.callback(
        Output('optimization-plot-dropdown', 'options'),
        Output('optimization-plot-dropdown', 'value'),
        Input('optimization-method-dropdown', 'value')
    )
    def set_plot_options(selected_method):
        if selected_method == 'kmeans':
            return [{'label': 'Метод локтя', 'value': 'Метод локтя'}], 'Метод локтя'
        elif selected_method == 'dbscan':
            return [{'label': 'График k-расстояний', 'value': 'График k-расстояний'}], 'График k-расстояний'

    @app.callback(
        Output('optimization-visualization', 'figure'),
        Input('optimization-method-dropdown', 'value'),
        Input('optimization-plot-dropdown', 'value')
    )
    def update_optimization_visualization(selected_method, selected_plot):
        # Проверяем наличие графика в словаре и возвращаем его
        if selected_method in optimization_figures and selected_plot in optimization_figures[selected_method]:
            return optimization_figures[selected_method][selected_plot]
        else:
            # Возвращаем пустую фигуру, если нет соответствующего графика
            return go.Figure()


    @app.callback(
        Output('cluster-dropdown', 'options'),
        Output('cluster-dropdown', 'value'),
        Input('method-dropdown', 'value')
    )
    def update_cluster_dropdown(selected_method):
        if selected_method == 'kmeans':
            options = [{'label': f'Cluster {i}', 'value': i} for i in sorted_kmeans_clusters]
            value = sorted_kmeans_clusters[0]
        elif selected_method == 'dbscan':
            options = [{'label': f'Cluster {i}', 'value': i} for i in sorted_dbscan_clusters]
            value = sorted_dbscan_clusters[0]
        else:
            options = []
            value = None
        return options, value

    # Callback для обновления визуализаций на основе выбора пользователя
    @app.callback(
        [Output('bar-chart', 'figure'),
         Output('treemap', 'figure'),
         Output('wordcloud', 'figure'),
         Output('cluster-table', 'children')],
        [Input('method-dropdown', 'value'),
         Input('cluster-dropdown', 'value')]
    )
    def update_visualizations(selected_method, selected_cluster_id):
        if selected_method == 'kmeans':
            selected_visualizations = kmeans_visualizations[selected_cluster_id]
        elif selected_method == 'dbscan':
            selected_visualizations = dbscan_visualizations[selected_cluster_id]
        else:
            selected_visualizations = {
                'bar_chart': {},
                'treemap': {},
                'wordcloud': {},
                'cluster_table': None
            }
        return (selected_visualizations['bar_chart'],
                selected_visualizations['treemap'],
                selected_visualizations['wordcloud'],
                selected_visualizations['cluster_table'])

    @app.callback(
        Output("collapse", "is_open"),
        [Input("collapse-button", "n_clicks")],
        [State("collapse", "is_open")]
    )
    def toggle_collapse(n_clicks, is_open):
        if n_clicks:
            return not is_open
        return is_open

    # Словари для быстрого доступа к графикам
    figures = generate_clustering_figures(messages_embeddings, kmeans_labels, hdbscan_labels, messages)

    @app.callback(
        Output('clustering-visualization', 'figure'),
        Input('visualization-method-dropdown', 'value'),
        Input('visualization-dropdown', 'value')
    )
    def update_visualization(selected_method, selected_visualization):
        return figures[selected_method][selected_visualization]

    kmeans_silhouette, hdbscan_silhouette = evaluate_clustering_performance(hdbscan_labels, kmeans_labels,
                                                                            embedding_512d)

    # Создание графиков метрик качества
    metric_figures = generate_metric_figures(kmeans_time, hdbscan_time, kmeans_memory, hdbscan_memory,
                                             kmeans_silhouette, hdbscan_silhouette)

    @app.callback(
        Output('metrics-visualization', 'figure'),
        Input('metrics-method-dropdown', 'value')
    )
    def update_metrics_visualization(selected_metric):
        return metric_figures[selected_metric]

    app.run_server(debug=True, use_reloader=False)
