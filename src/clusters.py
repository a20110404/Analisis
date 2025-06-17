import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

df = pd.read_csv('london_houses.csv')
x = df[['Square Meters', 'Price (£)']].values

m, n = x.shape

def on_click_factory(data_displayed, data_original):
    def on_click(event):
        if event.inaxes:
            # Buscar el punto más cercano en el espacio mostrado
            distances = np.sqrt((data_displayed[:, 0] - event.xdata) ** 2 + (data_displayed[:, 1] - event.ydata) ** 2)
            idx = np.argmin(distances)
            print(f"Tamaño de la casa: {data_original[idx, 0]:.2f} m², Precio: £{data_original[idx, 1]:,.2f}")
    return on_click

# Visualización inicial
plt.figure(1)
plt.plot(x[:, 0], x[:, 1], 'ko', markersize=7, markerfacecolor='y')
plt.title('Datos originales')
plt.xlabel('Tamaño de la casa (m²)')
plt.ylabel('Precio de la casa (pesos)')
plt.grid(True)
plt.gcf().canvas.mpl_connect('button_press_event', on_click_factory(x, x))
plt.show(block=False)
time.sleep(1)

# Normalización tipo z-score
x_norm = (x - np.mean(x, axis=0)) / np.std(x, axis=0, ddof=0)

plt.figure(2)
plt.plot(x_norm[:, 0], x_norm[:, 1], 'ko', markersize=7, markerfacecolor='y')
plt.title('Datos normalizados')
plt.xlabel('Tamaño de la casa (m²)')
plt.ylabel('Precio de la casa (pesos)')
plt.grid(True)
plt.gcf().canvas.mpl_connect('button_press_event', on_click_factory(x_norm, x))
plt.show(block=False)
time.sleep(1)

# Parámetros de K-means
k = 3 # Número de clusters
lim_inf = np.min(x_norm, axis=0)
lim_sup = np.max(x_norm, axis=0)
Mu_t = np.random.rand(k, n) * (lim_sup - lim_inf) + lim_inf # Centroides aleatorios
Mu_tmas1 = np.zeros((k, n))
colors = ['m', 'c', 'g', 'b']

while True:
    C = [[] for _ in range(k)]
    # Asignar cada punto al cluster más cercano
    for xi in x_norm:
        dist = np.sum((Mu_t - xi) ** 2, axis=1)
        ind = np.argmin(dist)
        C[ind].append(xi)
    # Visualización de clusters y centroides
    plt.clf()
    plt.plot(x_norm[:, 0], x_norm[:, 1], 'ko', markersize=7, markerfacecolor='y')
    plt.plot(Mu_t[:, 0], Mu_t[:, 1], 'r*', markersize=12)
    for j in range(k):
        if len(C[j]) > 0:
            cluster_points = np.array(C[j])
            plt.plot(cluster_points[:, 0], cluster_points[:, 1], 'ko', markersize=7, markerfacecolor=colors[j])
            Mu_tmas1[j, :] = np.mean(cluster_points, axis=0)
        else:
            Mu_tmas1[j, :] = Mu_t[j, :]  # Si el cluster está vacío, no mover el centroide
    plt.plot(Mu_tmas1[:, 0], Mu_tmas1[:, 1], 'b*', markersize=12)
    plt.title('K-means 2D')
    plt.xlabel('Tamaño de la casa (m²)')
    plt.ylabel('Precio de la casa (pesos)')
    plt.grid(True)
    plt.pause(0.5)
    # Criterio de paro
    distMu = np.sqrt(np.sum((Mu_tmas1 - Mu_t) ** 2, axis=1))
    if np.sum(distMu) <= np.finfo(float).eps:
        break
    Mu_t = Mu_tmas1.copy()

plt.show(block=True)

print('Centroides finales:')
for j in range(k):
    print(f'Centroide {j+1} = [{Mu_t[j, 0]:.4f}, {Mu_t[j, 1]:.4f}]')

print('\nNúmero de datos en cada clúster:')
for j in range(k):
    print(f'C{j+1} = {len(C[j])} datos')

# Paso final: detectar anomalías con distancia al centroide
anomaly_threshold = 2.5  # Cambia esto según lo agresivo que quieras ser
anomalies = []

for cluster_idx in range(k):
    centroid = Mu_t[cluster_idx]
    cluster_points = np.array(C[cluster_idx])
    distances = np.linalg.norm(cluster_points - centroid, axis=1)
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)

    for i, dist in enumerate(distances):
        if dist > mean_dist + anomaly_threshold * std_dist:
            anomalies.append(cluster_points[i])

# Visualización de anomalías
if anomalies:
    anomalies = np.array(anomalies)
    plt.figure()
    plt.plot(x_norm[:, 0], x_norm[:, 1], 'ko', markersize=7, markerfacecolor='y')
    plt.plot(anomalies[:, 0], anomalies[:, 1], 'ro', markersize=10, markerfacecolor='none', label='Anomalías precios')
    plt.title('Anomalías detectadas con K-Means')
    plt.xlabel('Tamaño normalizado')
    plt.ylabel('Precio normalizado')
    plt.grid(True)
    plt.legend()
    plt.gcf().canvas.mpl_connect('button_press_event', on_click_factory(x_norm, x))
    plt.show()
else:
    print("No se detectaron anomalías significativas.")

anomaly_info = []

for cluster_idx in range(k):
    centroid = Mu_t[cluster_idx]
    cluster_points = np.array(C[cluster_idx])
    distances = np.linalg.norm(cluster_points - centroid, axis=1)
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)

    for i, dist in enumerate(distances):
        if dist > mean_dist + anomaly_threshold * std_dist:
            # Buscar el índice original en el DataFrame
            idx = df[(df['Square Meters'] == cluster_points[i][0]*np.std(x[:,0])+np.mean(x[:,0])) &
                     (df['Price (£)'] == cluster_points[i][1]*np.std(x[:,1])+np.mean(x[:,1]))].index
            if len(idx) > 0:
                idx = idx[0]
                anomaly_info.append({
                    'Address': df.iloc[idx]['Address'],
                    'Neighborhood': df.iloc[idx]['Neighborhood'],
                    'Price (£)': df.iloc[idx]['Price (£)'],
                    'Square Meters': df.iloc[idx]['Square Meters']
                })

# Ahora anomaly_info tiene la información de las casas anómalas
# las usaremos para hacer web scraping

# Ejemplo de búsqueda web para cada anomalía
from duckduckgo_search import DDGS

for anomaly in anomaly_info:
    query = f"{anomaly['Address']} {anomaly['Neighborhood']} house price"
    print(f"\nBuscando información web para: {query}")
    with DDGS() as ddgs:
        resultados = ddgs.text(query, region="wt-wt", safesearch="Moderate", max_results=3)
        for r in resultados:
            print(f"Título: {r['title']}\nURL: {r['href']}\nResumen: {r['body']}\n")


# ------------------------------------- WEB SCRAPING -------------------------------------
from duckduckgo_search import DDGS

for anomaly in anomaly_info:
    query = f"{anomaly['Address']} {anomaly['Neighborhood']} house price"
    print(f"\nBuscando información web para: {query}")
    with DDGS() as ddgs:
        resultados = ddgs.text(query, region="wt-wt", safesearch="Moderate", max_results=3)
        for r in resultados:
            print(f"Título: {r['title']}\nURL: {r['href']}\nResumen: {r['body']}\n")