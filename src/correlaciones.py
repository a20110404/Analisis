import numpy as np
import matplotlib.pyplot as plt
import time

# Cargar datos
x = np.loadtxt('dataSet_kmeans2D_casas.txt')

plt.figure(figsize=(7, 5))
plt.xlabel('Tamaño de la casa (m²)')
plt.ylabel('Precio de la casa (pesos)')
plt.title('Ajuste progresivo de la tendencia lineal')
plt.grid(True)

for i in range(2, len(x) + 1):
    plt.clf()
    plt.scatter(x[:i, 0], x[:i, 1], color='orange', edgecolor='k', alpha=0.7)
    plt.xlabel('Tamaño de la casa (m²)')
    plt.ylabel('Precio de la casa (pesos)')
    plt.title('Ajuste progresivo de la tendencia lineal')
    plt.grid(True)

    # Ajuste lineal solo con los puntos actuales
    m, b = np.polyfit(x[:i, 0], x[:i, 1], 1)
    plt.plot(x[:i, 0], m * x[:i, 0] + b, color='blue', linewidth=2, label='Tendencia lineal')
    plt.legend()
    plt.tight_layout()
    plt.pause(0.01)

plt.show()