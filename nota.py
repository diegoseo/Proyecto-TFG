import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Paso 1: Cargar y preparar los datos
datos = pd.read_csv('limpio.csv')
datos_numeric = datos.select_dtypes(include=['float64', 'int64'])
datos_numeric = datos_numeric.dropna()  # Eliminar valores faltantes

# Escalar los datos
scaler = StandardScaler()
datos_escalados = scaler.fit_transform(datos_numeric)

# Paso 2: Calcular el PCA
n_componentes = 2  # Número de componentes principales
pca = PCA(n_components=n_componentes)
pca_resultados = pca.fit_transform(datos_escalados)

# Crear un DataFrame para los scores
pca_scores = pd.DataFrame(pca_resultados, columns=[f'PC{i+1}' for i in range(n_componentes)])

# Crear un DataFrame para los loadings
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(n_componentes)],
    index=datos_numeric.columns
)

# Paso 3: Interpretar resultados
print("Varianza explicada:", pca.explained_variance_ratio_)
print("Scores:")
print(pca_scores.head())
print("Loadings:")
print(loadings.head())

# Paso 4: Graficar los resultados
# a) Gráfico de Scores
plt.figure(figsize=(8, 6))
plt.scatter(pca_scores['PC1'], pca_scores['PC2'], alpha=0.7, color='blue')
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Gráfico de Scores (Muestras)')
plt.grid()
plt.show()

# b) Gráfico de Loadings
plt.figure(figsize=(8, 6))
plt.scatter(loadings['PC1'], loadings['PC2'], alpha=0.7, color='green')
for i, variable in enumerate(loadings.index):
    plt.text(loadings['PC1'][i], loadings['PC2'][i], variable, fontsize=9)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Gráfico de Loadings (Variables)')
plt.grid()
plt.show()
