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







        #import webbrowser
        
        label_x = "".join([f"PC{c+1}" for c in componentes_x])  
        label_y = "".join([f"PC{c+1}" for c in componentes_y]) 
        label_z = "".join([f"PC{c+1}" for c in componentes_z]) 
        

                           
        colores_pca_original = [asignacion_colores.get(type_, 'black') for type_ in types]

    # Verificar si hay datos suficientes para graficar
    if dato_pca.shape[1] < 3:
        print("Error: No hay suficientes componentes principales para graficar en 3D.")
    else:
        # Extraer los datos de las componentes principales
        x_data = dato_pca[:, 0]  # PC1
        y_data = dato_pca[:, 1]  # PC2
        z_data = dato_pca[:, 2]  # PC3
    
        print("X DATA")
        print(x_data[:5])
        print("Y DATA")
        print(y_data[:5])
        print("Z DATA")
        print(z_data[:5])
    
        # Verificar que los colores coincidan con la cantidad de datos
        if len(colores_pca_original) != len(x_data):
            colores_pca_original = ['blue'] * len(x_data)  # Usar azul si hay desajuste
    
        # Crear la figura en Plotly
        fig = go.Figure()
    
        # Agregar el gráfico de dispersión 3D
        fig.add_trace(go.Scatter3d(
            x=x_data,
            y=y_data,
            z=z_data,
            mode='markers',
            marker=dict(
                size=5,
                color=colores_pca_original,  # Colores asignados a cada punto
                opacity=0.7
            ),
            hovertext=[f"Punto {i}" for i in range(len(x_data))]  # Información al pasar el mouse
        ))
    
        # Configurar los ejes y el título
        fig.update_layout(
            title=f'Análisis de Componentes Principales 3D de {archivo_nombre}',
            scene=dict(
                xaxis_title=label_x,
                yaxis_title=label_y,
                zaxis_title=label_z
            ),
            margin=dict(l=0, r=0, b=0, t=40)  # Ajustar los márgenes
        )
    
        # Mostrar la gráfica interactiva
        fig.show(renderer="browser")  # Forzar apertura en navegador













































import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2


def pca(dato,raman_shift):
    
    num_muestras , num_variables = dato.shape # EL MENOR VALOR ES LA CANTIDAD MAXIMA DE COMPORNENTES PRINCIPALES POSIBLES
    
    if num_muestras >= num_variables :
        max_pc = num_variables
    else:
        max_pc = num_muestras
    
    
    #MOSTRAMOS UN MENO DE LAS POSIBLES FORMAS DE MOSTRAR EL PCA
    print("Como deseas Visualizar los componentes principales")
    print("1-Elegir 2 Componentes para graficar en 2D")
    print("2-Elegir 3 Componentes para graficar en 3D")
    print("5-Salir")
    visualizar_pca = int(input("Opcion = "))
    

    componentes_x = []
    componentes_y = []
    componentes_z = []
    
    n_componentes = 0
    while n_componentes > max_pc or n_componentes <= 0 :
        print("Ingrese la cantidad de componentes principales:")
        n_componentes = int(input("n = "))
        if n_componentes > max_pc or n_componentes <= 0:
            print("n se encuentra fuera del rango permitido ( ", max_pc , " )")
            n_componentes = 0
        elif n_componentes == 1:
            print("N debe ser distinto de 1")
            n_componentes = 0

    dato = dato.dropna() #eliminamos las filas con valores NAN
    datos_df = dato.transpose() #PASAMOS LA CABECERA DE TIPOS A LA COLUMNA
    escalado = StandardScaler() 
    dato_escalado = escalado.fit_transform(datos_df)
    cov_matrix = np.cov(dato_escalado, rowvar=False)  # rowvar=False para covarianza entre variables
    pca = PCA(n_components=n_componentes)
    # Ajustar y transformar los datos
    dato_pca = pca.fit_transform(dato_escalado) # fit_transform ya hace el calcilo de los eigenvectores y eigenvalores y matriz de covarianza
    eigenvalores = pca.explained_variance_
    suma_eigenvalores = sum(eigenvalores) # Hallamos su suma solo para despues ver el % de importancia
    porcentaje_varianza = (eigenvalores / suma_eigenvalores) * 100
    # Obtener Eigenvectores (Componentes principales)
    eigenvectores = pca.components_ * -1 # en realidad multiplique por -1 por que quiero cambiar el orden de los signos, No afecta el resultado final
    if visualizar_pca == 1:
        print("INGRESE LOS COMPONENTES PRINCIPALES QUE DESEAS VISUALIZAR EN 2D")
        while True:
            pc = input(f"Ingrese un número de Componente Principal para el eje X (1-{n_componentes}) o 'salir' para continuar: ")
            if pc.lower() == "salir":
                break
            if pc.isdigit() and 1 <= int(pc) <= n_componentes:
                componentes_x.append(int(pc) - 1)
                break
            else:
                print("Número fuera de rango. Intente de nuevo.")

        # Selección de componentes para el eje Y
        while True:
            pc = input(f"Ingrese un número de Componente Principal para el eje Y (1-{n_componentes}) o 'salir' para finalizar: ")
            if pc.lower() == "salir":
                break
            if pc.isdigit() and 1 <= int(pc) <= n_componentes:
                componentes_y.append(int(pc) - 1)
                break
            else:
                print("Número fuera de rango. Intente de nuevo.")
    elif visualizar_pca == 2:
        print("INGRESE LOS COMPONENTES PRINCIPALES QUE DESEAS VISUALIZAR EN 3D")
        while True:
            pc = input(f"Ingrese un número de Componente Principal para el eje X (1-{n_componentes}) o 'salir' para continuar: ")
            if pc.lower() == "salir":
                break
            if pc.isdigit() and 1 <= int(pc) <= n_componentes:
                componentes_x.append(int(pc) - 1)
                break
            else:
                print("Número fuera de rango. Intente de nuevo.")

        # Selección de componentes para el eje Y
        while True:
            pc = input(f"Ingrese un número de Componente Principal para el eje Y (1-{n_componentes}) o 'salir' para continuar: ")
            if pc.lower() == "salir":
                break
            if pc.isdigit() and 1 <= int(pc) <= n_componentes:
                componentes_y.append(int(pc) - 1)
                break
            else:
                print("Número fuera de rango. Intente de nuevo.")

        # Selección de componentes para el eje Z
        while True:
            pc = input(f"Ingrese un número de Componente Principal para el eje Z (1-{n_componentes}) o 'salir' para finalizar: ")
            if pc.lower() == "salir":
                break
            if pc.isdigit() and 1 <= int(pc) <= n_componentes:
                componentes_z.append(int(pc) - 1)
                break
            else:
                print("Número fuera de rango. Intente de nuevo.")


    if visualizar_pca == 1:
         eje_x = dato_pca[:,componentes_x]  
         eje_y = dato_pca[:,componentes_y] 
      
         dato_pca = np.column_stack((eje_x,eje_y)) 

    elif visualizar_pca == 2:
        eje_x = dato_pca[:,componentes_x]  
        eje_y = dato_pca[:,componentes_y]
        eje_z = dato_pca[:,componentes_z]

        dato_pca = np.column_stack((eje_x,eje_y,eje_z)) 

    if visualizar_pca == 1:
        
        label_x = "".join([f"PC{c+1}" for c in componentes_x])  
        label_y = "".join([f"PC{c+1}" for c in componentes_y]) 
        colores_pca_original = [asignacion_colores.get(type_, 'black') for type_ in types]
        plt.figure(figsize=(10, 6))
        plt.scatter(dato_pca[:, 0], dato_pca[:, 1], c=colores_pca_original, alpha=0.7)
        
        plt.xlabel(label_x)
        plt.ylabel(label_y)
        plt.title('Análisis de Componentes Principales 2D de ' + archivo_nombre) 
        plt.grid()
        plt.show()  
    elif visualizar_pca == 2:    
        plot_pca_3d_with_ellipsoids(datos_df, types, asignacion_colores, "archivo.csv", n_componentes)

       


def plot_pca_3d_with_ellipsoids(dato, types, asignacion_colores, archivo_nombre, n_componentes=3):
    """
    Realiza PCA, grafica en 3D y dibuja elipsoides de confianza para cada tipo.
    """
    # Normalizar los datos
    escalado = StandardScaler()
    dato_escalado = escalado.fit_transform(dato)
    
    # Aplicar PCA
    pca = PCA(n_components=n_componentes)
    dato_pca = pca.fit_transform(dato_escalado)
    
    # Convertir a DataFrame
    df_pca = pd.DataFrame(dato_pca, columns=[f'PC{i+1}' for i in range(n_componentes)])
    df_pca['Tipo'] = types  # Agregar tipo de cada punto
    
    # Crear la figura en Plotly
    fig = go.Figure()
    
    # Agregar puntos de dispersión
    for tipo in np.unique(types):
        indices = df_pca['Tipo'] == tipo
        fig.add_trace(go.Scatter3d(
            x=df_pca.loc[indices, 'PC1'],
            y=df_pca.loc[indices, 'PC2'],
            z=df_pca.loc[indices, 'PC3'],
            mode='markers',
            marker=dict(size=5, color=asignacion_colores[tipo], opacity=0.7),
            name=f'Tipo {tipo}'
        ))
    
        # Calcular el centro y la covarianza del grupo
        datos_tipo = df_pca.loc[indices, ['PC1', 'PC2', 'PC3']].to_numpy()
        centro = np.mean(datos_tipo, axis=0)
        cov = np.cov(datos_tipo.T)
        
        # Generar el elipsoide de confianza
        if datos_tipo.shape[0] > 3:  # Asegurar que haya suficientes puntos
            elipsoide = generar_elipsoide(centro, cov, color=asignacion_colores[tipo])
            fig.add_trace(elipsoide)
    
    # Configurar los ejes y el título
    fig.update_layout(
        title=f'Análisis de Componentes Principales 3D de {archivo_nombre}',
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    # Mostrar la gráfica
    fig.show()

def generar_elipsoide(centro, cov, num_puntos=30, color='rgba(150,150,150,0.3)'):
    """Genera un trazo de Plotly con un elipsoide de confianza basado en la covarianza."""
    U, S, _ = np.linalg.svd(cov)
    radii = np.sqrt(chi2.ppf(0.95, df=3) * S)  # Intervalo de confianza del 95%
    
    u = np.linspace(0, 2 * np.pi, num_puntos)
    v = np.linspace(0, np.pi, num_puntos)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    
    # Transformar coordenadas para adaptarlas a la covarianza
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = centro + np.dot(U, np.multiply(radii, [x[i, j], y[i, j], z[i, j]]))
    
    return go.Surface(x=x, y=y, z=z, opacity=0.3, colorscale=[[0, color], [1, color]], showscale=False)

















