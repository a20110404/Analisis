import numpy as np

# Parámetros para la generación de datos
np.random.seed(42)
num_casas = 100

# Generar tamaños de casas (m^2), por ejemplo entre 40 y 250 m^2
tamanios = np.random.uniform(40, 250, num_casas)

# Generar precios en pesos, suponiendo que el precio depende del tamaño más algo de ruido
precio_base = 8000  # precio por m^2 en pesos
ruido = np.random.normal(0, 200000, num_casas)  # ruido aleatorio
precios = tamanios * precio_base + ruido

# Debo asegurarme de que no haya precios negativos >w>
precios = np.maximum(precios, 100000)

datos = np.column_stack((tamanios, precios))
np.savetxt('dataSet_kmeans2D_casas.txt', datos, fmt='%.2f', delimiter='\t')

print("Archivo 'dataSet_kmeans2D_casas.txt' generado correctamente.")