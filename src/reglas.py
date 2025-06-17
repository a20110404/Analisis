import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from duckduckgo_search import DDGS

# Cargar datos
df = pd.read_csv('london_houses.csv')

# Calcular precio por metro cuadrado
df['Precio por m2'] = df['Price (£)'] / df['Square Meters']

# Reglas para anomalías
limite_bajo = 5000
limite_alto = 15000
df['Es_Anomalia'] = ((df['Precio por m2'] < limite_bajo) | (df['Precio por m2'] > limite_alto)).astype(int)

# Seleccionar variables para graficar
X = df[['Square Meters', 'Price (£)']].values
y = df['Es_Anomalia'].values

# Normalización tipo z-score
media = np.mean(X, axis=0)
sigma = np.std(X, axis=0)
X_norm = (X - media) / sigma

# Visualización
plt.figure()
plt.plot(X_norm[y==0, 0], X_norm[y==0, 1], 'ok', markerfacecolor='y', label='Normal')
plt.plot(X_norm[y==1, 0], X_norm[y==1, 1], 'sk', markerfacecolor='m', label='Anómala')
plt.xlabel('Tamaño normalizado')
plt.ylabel('Precio normalizado')
plt.title('Anomalías por reglas en precio por m2')
plt.legend()

def on_click(event):
    if event.inaxes:
        distances = np.sqrt((X_norm[:, 0] - event.xdata) ** 2 + (X_norm[:, 1] - event.ydata) ** 2)
        idx = np.argmin(distances)
        metros = X[idx, 0]
        precio = X[idx, 1]
        print(f"Tamaño: {metros} m², Precio: £{precio:,.2f}")

fig = plt.gcf()
fig.canvas.mpl_connect('button_press_event', on_click)
plt.show()

# Web scraping para anomalías
anomaly_indices = np.where(y == 1)[0]
anomaly_info = []
for idx in anomaly_indices:
    anomaly_info.append({
        'Address': df.iloc[idx]['Address'],
        'Neighborhood': df.iloc[idx]['Neighborhood'],
        'Price (£)': df.iloc[idx]['Price (£)'],
        'Square Meters': df.iloc[idx]['Square Meters']
    })

for anomaly in anomaly_info:
    query = f"{anomaly['Address']} {anomaly['Neighborhood']} house price"
    print(f"\nBuscando información web para: {query}")
    with DDGS() as ddgs:
        resultados = ddgs.text(query, region="wt-wt", safesearch="Moderate", max_results=3)
        for r in resultados:
            print(f"Título: {r['title']}\nURL: {r['href']}\nResumen: {r['body']}\n")
