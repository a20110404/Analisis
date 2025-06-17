import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore

# Cargar el dataset
df = pd.read_csv('london_houses.csv')

# Normaliza el precio por edad
df['zscore_price'] = zscore(df['Price (£)'])
# Marca como anómala si el z-score es mayor a 2 o menor a -2
df['Es_Anomalia'] = (np.abs(df['zscore_price']) > 2).astype(int)

# Seleccionar variables
X = df[['Building Age', 'Price (£)']].values

y = df['Es_Anomalia'].values  # Debemoss tener esta columna

# Normalización tipo z-score
media = np.mean(X, axis=0)
sigma = np.std(X, axis=0)
X_norm = (X - media) / sigma

# Agregar columna de unos para el bias
m, n = X_norm.shape
X_aug = np.hstack([np.ones((m, 1)), X_norm])

# Inicialización de parámetros
a = np.ones(n)
a = np.insert(a, 0, 1)
a[2] = 0 # Inicializar el parámetro de sesgo a 0

beta = 0.4
iterMax = 300
convergencia = []

def g(z):
    return 1 / (1 + np.exp(-z))

# Entrenamiento
for iter in range(iterMax):
    h = g(X_aug @ a)
    J = -np.mean(y * np.log(h + 1e-8) + (1 - y) * np.log(1 - h + 1e-8))
    convergencia.append(J)
    grad = (1/m) * (X_aug.T @ (h - y))
    a = a - beta * grad

# Visualización de la frontera de decisión
x1_plot = np.linspace(-2, 2, 100)
x2_plot = (-a[0] - a[1]*x1_plot) / a[2]
plt.figure()
plt.plot(X_norm[y==0, 0], X_norm[y==0, 1], 'ok', markerfacecolor='y', label='Normal')
plt.plot(X_norm[y==1, 0], X_norm[y==1, 1], 'sk', markerfacecolor='m', label='Anómala')
plt.plot(x1_plot, x2_plot, 'r', label='Frontera')
plt.xlabel('Build age')
plt.ylabel('Price (£)')
plt.title('Detección de anomalías con regresión logística')
plt.legend()

def on_click(event):
    if event.inaxes:
        # Encuentra el punto más cercano al clic
        distances = np.sqrt((X_norm[:, 0] - event.xdata) ** 2 + (X_norm[:, 1] - event.ydata) ** 2)
        idx = np.argmin(distances)
        edad = X[idx, 0]
        precio = X[idx, 1]
        print(f"Edad del edificio: {edad} años, Precio: £{precio:,.2f}")

# Después de tu plt.plot(...), conecta el evento:
fig = plt.gcf()
cid = fig.canvas.mpl_connect('button_press_event', on_click)

plt.show()

# Curva de convergencia
plt.figure()
plt.plot(convergencia)
plt.xlabel('Iteración')
plt.ylabel('Costo J')
plt.show()

# ------------------------------------- WEB SCRAPING -------------------------------------
from duckduckgo_search import DDGS

# Encuentra los índices de las anomalías detectadas por regresión logística
anomaly_indices = np.where(y == 1)[0]

anomaly_info = []
for idx in anomaly_indices:
    anomaly_info.append({
        'Address': df.iloc[idx]['Address'],
        'Neighborhood': df.iloc[idx]['Neighborhood'],
        'Price (£)': df.iloc[idx]['Price (£)'],
        'Building Age': df.iloc[idx]['Building Age']
    })

for anomaly in anomaly_info:
    query = f"{anomaly['Address']} {anomaly['Neighborhood']} house price"
    print(f"\nBuscando información web para: {query}")
    with DDGS() as ddgs:
        resultados = ddgs.text(query, region="wt-wt", safesearch="Moderate", max_results=3)
        for r in resultados:
            print(f"Título: {r['title']}\nURL: {r['href']}\nResumen: {r['body']}\n")