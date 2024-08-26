import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from extract_keywords import extract_keywords
from wordcloud import WordCloud
import plotly.express as px
from io import BytesIO
from PIL import Image, ImageFilter
from dash import dash_table


def create_cluster_visualizations(messages, labels):
    clusters_df = pd.DataFrame({
        'Cluster': labels,
        'Texts': messages
    })

    cluster_visualizations = {}

    for cluster_id in clusters_df['Cluster'].unique():
        cluster_data = clusters_df[clusters_df['Cluster'] == cluster_id]
        cluster_texts = cluster_data['Texts'].tolist()
        cluster_keywords = extract_keywords(cluster_texts)

        # Генерация Bar Chart
        bar_chart = generate_bar_chart(cluster_keywords)

        # Генерация Treemap
        treemap = generate_treemap(cluster_keywords, cluster_id)

        # Создание и отображение Wordcloud
        wordcloud_image = wordcloud_to_plotly(cluster_keywords)
        fig_wordcloud = px.imshow(wordcloud_image, binary_string=True)
        fig_wordcloud.update_layout(
            xaxis={'visible': False},
            yaxis={'visible': False},
            margin=dict(l=20, r=20, t=30, b=20),
            height=400
        )

        # Преобразование текстов кластера в формат таблицы
        cluster_table = dash_table.DataTable(
            columns=[{"name": "", "id": "Texts"}],
            data=[{"Texts": text} for text in cluster_texts],
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '5px'},
            style_header={'display': 'none'},  # Скрытие заголовка таблицы
            style_as_list_view=True,  # Удаление границ таблицы
        )

        # Сохранение визуализаций для каждого кластера
        cluster_visualizations[cluster_id] = {
            'bar_chart': bar_chart,
            'treemap': treemap,
            'wordcloud': fig_wordcloud,
            'cluster_table': cluster_table
        }

    return cluster_visualizations


# Функция для создания Wordcloud из списка ключевых слов
def wordcloud_to_plotly(cluster_keywords):
    # Создание строки с повторением каждого ключевого слова в соответствии с его частотой
    words = ' '.join([f"{word} " * freq for word, freq in cluster_keywords.items()])

    wordcloud = WordCloud(
        width=3200,
        height=1800,
        max_words=50,  # Установка максимального количества слов
        min_font_size=50,  # Установка минимального размера шрифта
        collocations=False,
        background_color="white",
        contour_width=1,
        contour_color='steelblue',  # Цвет контура
        prefer_horizontal=1.0,  # Предпочтение горизонтального расположения слов
        relative_scaling=0.5,  # Настройка относительного масштабирования
    ).generate(words)

    # Использование matplotlib для создания изображения
    plt.figure(figsize=(24, 14))
    plt.imshow(wordcloud.recolor(), interpolation='bilinear')
    plt.axis('off')

    # Сохранение изображения в BytesIO
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    # Преобразование Wordcloud в массив NumPy
    # Чтение изображения из буфера и конвертация в Pillow Image
    image = Image.open(buf)

    # Применение анти-алиасинга для сглаживания изображения
    image = image.filter(ImageFilter.SMOOTH_MORE)
    image = image.filter(ImageFilter.DETAIL)

    # Усиление контраста для улучшения четкости
    # enhancer = ImageEnhance.Contrast(image)
    # image = enhancer.enhance(2)

    # Конвертация изображения обратно в массив NumPy
    wordcloud_image = np.array(image)
    return wordcloud_image


# Функция для отображения часто используемых слов в кластере с помощью Bar Chart
def generate_bar_chart(cluster_keywords):
    # Сортируем ключевые слова по частоте и берем только первые 15
    sorted_keywords = dict(sorted(cluster_keywords.items(), key=lambda item: item[1], reverse=True)[:15])

    # Инвертируем порядок для отображения самых частых слов вверху
    sorted_keywords = dict(reversed(sorted_keywords.items()))

    # Создаем столбчатую диаграмму со словами по оси Y и частотой по оси X
    fig_bar = px.bar(
        x=list(sorted_keywords.values()),
        y=list(sorted_keywords.keys()),
        orientation='h'
    )
    fig_bar.update_layout(
        xaxis_title='',  # Убираем подпись оси X
        yaxis_title='',  # Убираем подпись оси Y
        margin=dict(l=20, r=20, t=30, b=20),
        height=550
    )
    return fig_bar


# Функция для создания Treemap из списка ключевых слов
def generate_treemap(cluster_keywords, cluster_id):
    # Сортируем ключевые слова по частоте и берем только первые 25
    sorted_keywords = dict(sorted(cluster_keywords.items(), key=lambda item: item[1], reverse=True)[:25])

    # Преобразование словаря ключевых слов в DataFrame
    keywords_df = pd.DataFrame(list(sorted_keywords.items()), columns=['Keyword', 'Frequency'])

    fig_treemap = px.treemap(keywords_df, path=['Keyword'], values='Frequency', color='Frequency')
    fig_treemap.update_layout(xaxis={'visible': False}, yaxis={'visible': False})
    fig_treemap.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=400)
    return fig_treemap
