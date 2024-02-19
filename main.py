import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import matplotlib.pyplot as plt

# Генерация случайных данных для примера
np.random.seed(42)
X = np.random.randn(300, 2)

# Добавим несколько выбросов
outliers = np.array([[8, 8], [10, 10]])
X = np.concatenate([X, outliers])

# Применение K-means к данным
n_clusters = 3  # Используемое количество кластеров
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X)

# Вычисление расстояний от центроидов до точек в каждом кластере
distances = pairwise_distances_argmin_min(X, kmeans.cluster_centers_)[1]

# Определение выбросов по метрике (например, используем 95-й перцентиль)
threshold = np.percentile(distances, 95)
outliers_indices = np.where(distances > threshold)[0]

# Построение графика
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis', label='Точки данных')
plt.scatter(X[outliers_indices, 0], X[outliers_indices, 1], color='red', marker='x', label='Выбросы')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='black', marker='s', s=200, alpha=0.7, label='Кластерные центры')
plt.legend()
plt.title('K-означает кластеризацию с обнаружением выбросов')
plt.show()
