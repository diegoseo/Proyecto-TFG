import numpy as np
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import pandas as pd

# 🔹 Generar una prueba más pequeña (5 muestras con 3 características)
np.random.seed(42)
data_small = np.random.rand(5, 3)

# 🔹 Crear un DataFrame pequeño
df_small = pd.DataFrame(data_small, columns=[f'Feature_{i}' for i in range(3)])
df_small.index = [f"Muestra {i}" for i in range(5)]  # Etiquetar muestras

# 🔹 Mostrar los datos de prueba
print("📊 Datos de prueba (pequeño):\n", df_small)

# 🔹 Calcular la matriz de distancias euclidianas
distancias_small = sch.distance.pdist(df_small.values, metric='euclidean')

# 🔹 Convertir a formato cuadrado para visualizar mejor
dist_matrix_small = sch.distance.squareform(distancias_small)
df_distancias_small = pd.DataFrame(dist_matrix_small, index=df_small.index, columns=df_small.index)

# 🔹 Mostrar la matriz de distancias
print("\n📏 Matriz de Distancias Euclidianas (pequeño):\n", df_distancias_small)

# 🔹 Construcción de la Jerarquía de Clusters con Ward
dendrograma_small = sch.linkage(distancias_small, method='ward')

# 🔹 Convertir la matriz linkage a DataFrame
df_linkage_small = pd.DataFrame(dendrograma_small, columns=["Cluster 1", "Cluster 2", "Distancia", "Elementos"])

# 🔹 Mostrar la matriz de fusiones de clusters
print("\n🔗 Matriz de Linkage (pequeño):\n", df_linkage_small)

# 🔹 Graficar el Dendrograma Pequeño
plt.figure(figsize=(8, 4))
sch.dendrogram(dendrograma_small, labels=df_small.index, leaf_rotation=90)
plt.title("Dendrograma con distancia Euclidiana y método de Ward (Pequeño)")
plt.xlabel("Muestras")
plt.ylabel("Distancia")
plt.show()






def normalizado_media():
        
    intensity = df.iloc[1:, 1:] 
    cabecera = df.iloc[[0]].copy() 
    scaler = StandardScaler() 
    cal_nor = scaler.fit_transform(intensity) 
    dato_normalizado = pd.DataFrame(cal_nor, columns=intensity.columns) # lo convertimos de vuelta en un DataFrame
    df_concatenado = pd.concat([cabecera,dato_normalizado], axis=0, ignore_index=True)
    df_concatenado.columns = df_concatenado.iloc[0]  # Asigna la primera fila como nombres de columna
    df_concatenado_cabecera_nueva = df_concatenado[1:].reset_index(drop=True)
    df_media_pca= pd.DataFrame(df_concatenado_cabecera_nueva.iloc[:,1:])
    return  df_media_pca











def grafico_loading(pca, raman_shift, op_pca):

    plt.figure(figsize=(10, 6))


    loadings = pca.components_  

    n_componentes = loadings.shape[0]  # Número de componentes principales
   
    for i in op_pca:  
        if not isinstance(i, int):  
            raise ValueError(f"El índice de componente debe ser un entero, pero se recibió {type(i)}.")
        if i >= n_componentes:
            print(f"Advertencia: PC {i+1} no existe en los datos de PCA.")
            continue

       
        plt.plot(raman_shift, loadings[i, :], label=f'PC {i+1}')
       
    ax = plt.gca()  # Obtener el objeto de los ejes
    ax.xaxis.set_major_locator(ticker.MultipleLocator(200)) # PARA QUE SE ENTIENDA EL EJE X Y SALGA EN INTERVALO

    ay = plt.gca()  # Obtener el objeto de los ejes
    ay.yaxis.set_major_locator(ticker.MultipleLocator(0.05)) # PARA QUE SE ENTIENDA EL EJE X Y SALGA EN INTERVALO


    plt.xlabel('Raman Shift (cm⁻¹)')
    plt.ylabel('Loading')
    plt.title('Loading Plot para PCA y Raman Shift')
    plt.legend()
    plt.grid()
    plt.show()
   
    







