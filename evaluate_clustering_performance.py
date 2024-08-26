from sklearn.metrics import silhouette_score


def evaluate_clustering_performance(dbscan_labels, kmeans_labels, embedding):
    # Вычисление силуэтного коэффициента для K-means
    kmeans_silhouette = silhouette_score(embedding, kmeans_labels)

    # Подсчет количества кластеров и шумовых точек
    n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise = list(dbscan_labels).count(-1)

    # Вычисление силуэтного коэффициента для DBSCAN, если есть более одного кластера
    if n_clusters > 1:
        # Фильтрация шумовых точек перед вычислением силуэтного коэффициента
        filtered_labels = dbscan_labels[dbscan_labels != -1]
        filtered_data = embedding[dbscan_labels != -1]
        dbscan_silhouette = silhouette_score(filtered_data, filtered_labels)
        print(f"DBSCAN Silhouette Score: {dbscan_silhouette}")
    else:
        dbscan_silhouette = None
        print("Не удалось вычислить силуэтный коэффициент, так как кластеров меньше двух.")

    # Возврат значений
    return kmeans_silhouette, dbscan_silhouette
