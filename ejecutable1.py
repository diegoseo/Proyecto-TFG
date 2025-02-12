import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 📊 Simulación de datos espectrales (100 muestras, 6 variables)
np.random.seed(42)
data = np.random.rand(100, 6)

# 🔹 Estandarizar los datos
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# 🔹 Aplicar PCA
pca = PCA(n_components=6)
dato_pca = pca.fit_transform(data_scaled)

# 📊 Crear DataFrame con resultados PCA
df_pca = pd.DataFrame(dato_pca, columns=[f'PC{i+1}' for i in range(6)])

# 🎨 Asignación de colores a los puntos (opcional)
asignacion_colores = ['red', 'blue', 'green', 'orange', 'purple'] * 20  # Repite colores para 100 muestras
df_pca["Color"] = asignacion_colores[:len(df_pca)]  # Asegurar que la lista coincide con las muestras

# 📌 Seleccionar los componentes a graficar
componentes_x = [0, 1]  # PC1 + PC2
componentes_y = [2]     # PC3
componentes_z = [3]     # PC4

# 📊 Calcular los valores para cada eje
df_pca['Eje_X'] = df_pca.iloc[:, componentes_x].sum(axis=1)
df_pca['Eje_Y'] = df_pca.iloc[:, componentes_y].sum(axis=1)
df_pca['Eje_Z'] = df_pca.iloc[:, componentes_z].sum(axis=1)

# 🏷 Etiquetas de los ejes
label_x = "+".join([f"PC{c+1}" for c in componentes_x])  # PC1 + PC2
label_y = "+".join([f"PC{c+1}" for c in componentes_y])  # PC3
label_z = "+".join([f"PC{c+1}" for c in componentes_z])  # PC4

# 📊 Gráfico 3D interactivo con Plotly
fig = px.scatter_3d(df_pca, x='Eje_X', y='Eje_Y', z='Eje_Z',
                     color=df_pca["Color"],
                     title="Análisis de Componentes Principales 3D",
                     labels={"Eje_X": label_x, "Eje_Y": label_y, "Eje_Z": label_z},
                     opacity=0.8)

fig.show()


















 elif visualizar_pca == 2:
     
     label_x = "".join([f"PC{c+1}" for c in componentes_x])  
     label_y = "".join([f"PC{c+1}" for c in componentes_y]) 
     label_z = "".join([f"PC{c+1}" for c in componentes_z]) 
     
    
                        
      colores_pca_original = [asignacion_colores.get(type_, 'black') for type_ in types]
      fig = plt.figure(figsize=(8, 6))
      ax = fig.add_subplot(111, projection='3d')

     
      ax.scatter(dato_pca[:, 0], dato_pca[:, 1], dato_pca[:, 2], c=colores_pca_original, alpha=0.7) # Graficamos por cada columna de dato_pca
     
     
      ax.set_xlabel(label_x)# Etiquetamos de los ejes
      ax.set_ylabel(label_y)
      ax.set_zlabel(label_z)
     
      plt.tight_layout()

      # Título del plot
      ax.set_title('Análisis de Componentes Principales 3D de ' + archivo_nombre)
      plt.show()
















