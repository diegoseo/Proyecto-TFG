from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QFileDialog, QMessageBox
#from hilo import HiloCargarArchivo
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
import plotly.graph_objects as go
import plotly.express as px
import scipy.cluster.hierarchy as sch # PARA EL HCA
import matplotlib.ticker as ticker
import time
from scipy.signal import savgol_filter # Para suavizado de Savitzky Golay
from scipy.ndimage import gaussian_filter1d # PARA EL FILTRO GAUSSIANO
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import chi2 # PARA GRAFICAR LOS ELIPSOIDES
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform # PARA EL HCA

def columna_con_menor_filas(df):

    # Calcular el número de valores no nulos en cada columna
    valores_no_nulos = df.notna().sum()

    # Encontrar la columna con la menor cantidad de valores no nulos
    columna_menor = valores_no_nulos.idxmin()
    cantidad_menor = valores_no_nulos.min()

    return columna_menor, cantidad_menor


#corregir_base_lineal, corregir_shirley, normalizar_por_media, normalizar_por_area, suavizar_sg, suavizar_gaussiano, suavizar_media_movil, primera_derivada, segunda_derivada


# CORROBORAR RESULTADOS, NO COINCIDE CON ORANGE
def normalizar_por_media(df,metodo): # NORMALIZAMOS  COLUMNA EN VEZ DE HACER POR FILA (LA NORMALIZACION POR FILA ESTA COMENTADA ABAJO)
    print("Normalizar Media")
    if metodo == "Standardize u=0, v2=1": # NORMALIZACION Z-SCORE = (x - μ) / σ , RESTA LA MEDIA Y DIVIDE POR LA DESVIACION ESTANDAR DE SU COLUMNA 
        # print("ENTRO EN Standardize u=0, v2=1")
        # df_numerico = df.apply(pd.to_numeric, errors='coerce')  # por si hay strings
        # print(df)
        # df_zscore = (df_numerico - df_numerico.mean()) / df_numerico.std()
        # return df_zscore
        
        
        df_transpuesta = df.T  # Muestras como filas

        # Normalización Z-score
        df_normalizado = (df_transpuesta - df_transpuesta.mean(axis=0)) / df_transpuesta.std(axis=0)

        # Opcional: volver al formato original con columna Raman Shift
        df_normalizado = df_normalizado.T

        return df_normalizado
            
    elif metodo == "Center to u=0": # RESTAMOS LA MEDIA DE CADA COLUMNA SIN ESCALAR LA VARIANZA x′=x−μ
        print("ENTRO EN Center to u=0")
        return df - df.mean() # simplemente retorna la reste de las intensidades por su media
    
    elif metodo == "Scale to v2=1": # ESCALAMOS PARA TENER VARIANZA IGUAL A 1, DIVIDIMOS CADA COLUMNA POR SU DESVIACION ESTANDAR
        print("ENTREO EN SCALE TO V2=1")
        return df / df.std()
    
    elif metodo == "Normalize to interval [-1,1]": # SE HACE UNA TRANSFORMACION LINEAL QUE SE BASA EN EL MINIMO Y MAXIMO DE CADA COLUMNA
        print("ENTRO EN Normalize to interval [-1,1]")
        min_vals = df.min()
        max_vals = df.max()
        rango = max_vals - min_vals
        rango_reemplazo = rango.replace(0, 1)  #  para el caso de que min = max
        return 2 * ((df - min_vals) / rango_reemplazo) - 1
    elif metodo == "Normalize to interval [0,1]": # ESCALMOS PARA QUE LOS VALORES ESTE ENTRE 0 Y 1 , MIN-MAX
        print("ENTRO EN Normalize to interval [0,1]")
        min_vals = df.min()
        max_vals = df.max()
        rango = max_vals - min_vals
        rango_reemplazo = rango.replace(0, 1)  # evita la división por cero si todos los valores son iguales
        return (df - min_vals) / rango_reemplazo
    
    
    
    
def normalizar_por_area(df,raman_shift): # RETORNA UN DF PERO FALTA AGREGAR LA COLUMNA DEL RAMANSHIFT ANTES DE PASAR A LA SIGUIENTE TRANSFORMACION
    print("Normalizar Area")
    columnas_normalizadas = []
    np_array = raman_shift.to_numpy()  # eje x para integrar

    for col in df.columns:
        y = df[col].to_numpy()
        area = np.trapezoid(y, np_array) * -1  # El -1 es para corregir si el eje está invertido

        if area != 0:
            normalizado = y / area
        else:
            print(f"El área de la columna {col} es cero. Se conserva sin normalizar.")
            normalizado = y

        columnas_normalizadas.append(pd.Series(normalizado, name=col))

    df_normalizado = pd.concat(columnas_normalizadas, axis=1)
    return df_normalizado

    # intensity = df.iloc[1:, 1:] # EXTRAEMOS TODAS DEMAS COLUMNAS EXCEPTO LA PRIMERA FILA Y PRIMERA COLUMNA
    # #print(intensity)  
    
    # cabecera = df.iloc[[0]].copy() # EXTRAEMOS LA PRIMERA FILA 
    # #print(cabecera)
    
    # df3 = pd.DataFrame(intensity)
    # #print("DataFrame de Intensidades:")
    # #print(df3)
    # df3 = df3.apply(pd.to_numeric, errors='coerce')  # Convierte a numérico, colocando NaN donde haya problemas
    # #print(df3)
    # np_array = raman_shift.astype(float).to_numpy() #CONVERTIMOS INTENSITY AL TIPO NUMPY POR QUE POR QUE NP.TRAPZ UTILIZA ESE TIPO DE DATOS
    # #print("valor de np_array: ")
    # #print(np_array)
    
    # df3_normalizado = df3.copy()
    # for col in df.columns:
    #     #print(df3[col])
    #     #print(df3_normalizado[col])
        
    #     # np.trapz para hallar el area bajo la curva por el metodo del trapecio
    #     area = (np.trapezoid(df3[col], np_array)) *-1  #MULTIPLIQUE POR -1 PARA QUE EL GRAFICO SALGA TODO HACIA ARRIBA ESTO SE DEBE A QUE EL RAMAN_SHIFT ESTA EN FORMA DECRECIENTE
    #     if area != 0:
    #         df3_normalizado[col] = df3[col] / area
    #     else:
    #         print(f"Advertencia: El área de la columna {col} es cero y no se puede normalizar.") #seguro contra errores de división por cero 
    # return df3_normalizado
    
    
# CORROBORAR RESULTADOS , POCA DIFERENCIA CON EL DF ORIGINAL
def suavizar_sg(df, ventana, orden):
    print("Suavizado sg entrada")
    # Convertimos a numpy array
    dato = df.to_numpy()

    suavizado = np.apply_along_axis(lambda x: savgol_filter(x, window_length=ventana, polyorder=orden), axis=0, arr=dato) # Aplicamos suavizado por columnas (axis=0)

    suavizado_df = pd.DataFrame(suavizado, columns=df.columns) # Volvemos a DataFrame, con los mismos nombres de columna
    
    diferencia = df - suavizado_df
    print(diferencia.abs().sum().sum())
    
    return suavizado_df
    
    
    
# CORROBORAR RESULTADOS , POCA DIFERENCIA CON EL DF ORIGINAL    
def suavizar_gaussiano(df,sigma):
    print("Suavizado fg")
    #cabecera = df.iloc[[0]].copy()  # Fila de nombres de muestras

    # Convertir a numpy y asegurarse que es float
    dato = df.to_numpy(dtype=float)

    # Aplicar filtro gaussiano por columna
    suavizado_gaussiano = np.apply_along_axis(lambda x: gaussian_filter1d(x, sigma=sigma), axis=0, arr=dato)

    # Reconstruir DataFrame
    suavizado_gaussiano_pd = pd.DataFrame(suavizado_gaussiano)
    #suavizado_gaussiano_pd.columns = cabecera.iloc[0, 1:].values  # Recuperar nombres de columnas

    return suavizado_gaussiano_pd


# CORROBORAR RESULTADOS , POCA DIFERENCIA CON EL DF ORIGINAL
def suavizar_media_movil(df, ventana):
    print("Suavizado mm")
    suavizado_media_movil = df.rolling(window=ventana, min_periods = 1 ,center=True).mean() # mean() es para hallar el promedio, Si usamos min_periods=3, entonces las dos primeras posiciones serían NaN porque no hay 3 datos disponibles todavía.
    # EN EL CODIGO  DE SPYDER TENGO LA FUNCION suavizado_mediamovil_paraPCA POR QUE ME GENERABA VALORES NAN, AHORA DEBE DE SOLUCIONARCE CON ESTE MIN_PERIODS
    # print("SUAVIZADO MEDIAMOVIL",suavizado_media_movil.shape)
    # print("SUAVIZADO MEDIAMOVIL",suavizado_media_movil)
    
    return suavizado_media_movil
    

def corregir_base_lineal(df,raman_shift):
    print("Correccion Base")
    # Filtrar valores válidos
    valid_idx = df.dropna().index.intersection(raman_shift.dropna().index)
    df_filtrado = df.loc[valid_idx].reset_index(drop=True)
    raman_shift_filtrado = raman_shift.loc[valid_idx].reset_index(drop=True)

    # Corrección de línea base usando diccionario
    dict_corregidos = {}
    for col in df_filtrado.columns:
        y = df_filtrado[col]
        coef = np.polyfit(raman_shift_filtrado, y, 1)  # ajuste lineal
        base_lineal = coef[0] * raman_shift_filtrado + coef[1]
        dict_corregidos[col] = y - base_lineal

    df_corregido = pd.DataFrame(dict_corregidos)

    return df_corregido

def correccion_de_shirley(y):
    print("Correcion Shirley")
    tol=1e-5
    max_iter=100 # tol= tolerancia del error , max_iter = número máximo de iteraciones que se permitirán para ajustar la línea base de forma progresiva.
    y = np.asarray(y)
    n = len(y)
    baseline = np.zeros(n)
    diff = np.inf
    iteration = 0

    while diff > tol and iteration < max_iter:
        previous = baseline.copy()
        for i in range(n):
            numerador = np.trapezoid(y[i:] - baseline[i:], dx=1)
            denominador = np.trapezoid(y - baseline, dx=1)
            if denominador == 0:
                baseline[i] = y[0]
            else:
                baseline[i] = y[0] + (y[-1] - y[0]) * (numerador / denominador)
        diff = np.linalg.norm(baseline - previous)
        iteration += 1

    return y - baseline

def corregir_shirley(df, raman_shift=None):
    print("Aplicando corrección de Shirley")
    columnas_corregidas = []

    for col in df.columns:
        y = df[col].to_numpy()
        corregido = correccion_de_shirley(y)
        columnas_corregidas.append(pd.Series(corregido, name=col))

    # Unir todas las series corregidas en un solo DataFrame
    df_shirley = pd.concat(columnas_corregidas, axis=1)
    return df_shirley

# CON DIFF SI GENERA VALORES NAN , PERO CON GRADIENT NO GENERA NAN
def primera_derivada(df,raman_shift):
    print("Primera Derivada")
    print(raman_shift)
    df_derivada = pd.DataFrame(index=df.index, columns=df.columns)

    for col in df.columns:
        y = df[col].values
        primer_der = np.gradient(y, raman_shift)  # Derivada de y respecto a x
        df_derivada[col] = primer_der

    return df_derivada

def segunda_derivada(df,raman_shift):
    print("Segunda Derivada")
    df_derivada2 = pd.DataFrame(index=df.index, columns=df.columns)
    for col in df.columns:
        y = df[col].values
        primer_der = np.gradient(y, raman_shift)# Primera derivada
        segundo_der = np.gradient(primer_der, raman_shift)# Segunda derivada
        df_derivada2[col] = segundo_der

    return df_derivada2

    
    
    

# PREGUNTAR ACA QUE PASA CUANDO ENTRE LOS ARCHIVOS FUSIONADOS TENEMOS MAS DE 3 COMPONENTES PRINCIPALES
# NO VA A FUNCIONAR AHORA EL MID-LEVEL POR QUE a visualizar_pca no se le carga un valor
#POR LO QUE ENTENDI EL RESULTADO DE LOS COMPONENTES PRINCIPALES ES EL MISMO YA SEA UN n=2 O UN n=3
def pca(df,componentes):
    print("DF DENTRO DEL PCA PRUEBAAA")
    print(df)
    #tipos,asignacion_colores,raman_shift = calculos_colores(df)
    print("PCA Componentes: ", componentes)
    componentes = int(componentes)
    num_muestras, num_variables = df.shape #OBTENEMOS LA CANTIDAD DE  FILAS Y COLUMNAS
    print("num_muestras: ",num_muestras,"num_variables: ",num_variables)
    max_pc = min(num_muestras, num_variables) #CANTIDAD MAXIMA DE N ES EL MENOR NUMERO

    if 1 < componentes <= max_pc:
        #df = df.iloc[1:,1:].T
        pca = PCA(n_components=componentes) 
        dato_pca = pca.fit_transform(df)  # fit_transform ya hace el calcilo de los eigenvectores y eigenvalores y matriz de covarianza, (cambiar dato_centralizado por dato_escalado si quiero usar el otro metodo)
        print("DATO PCA")
        print(dato_pca)
        print("Shape:", dato_pca.shape)  # (n_filas, n_componentes)
        # Calculamos porcentaje de varianza
        varianza_ratio = pca.explained_variance_ratio_  # en [0, 1]
        varianza_porcentaje = varianza_ratio * 100  # convertir a %
        print("VARIANZA PORCENTAJE")
        print(varianza_porcentaje)
        return dato_pca, varianza_porcentaje
    else:
        raise ValueError(f"El número de componentes debe estar entre 2 y {max_pc}")



def plot_pca_2d(dato_pca,varianza_porcentaje,asignacion_colores,types,componentes_x,componentes_y,intervalo_confianza):
    """
    Realiza PCA, grafica en 2D y dibuja elipses de confianza para cada tipo.
    """
    print("DENTRO DE PLOT_PCA_2D")
    print("VARIANZA_PORCENTAJE")
    print(varianza_porcentaje)
    print("COMPONNTES_X")
    print(componentes_x)
    print("COMPONNTES_Y")
    print(componentes_y)
    print("TYPES")
    types = types.reset_index(drop=True)
    print(types)
    # Asegurar que los componentes sean enteros
    idx_x = componentes_x[0] if isinstance(componentes_x, (list, np.ndarray)) else componentes_x
    idx_y = componentes_y[0] if isinstance(componentes_y, (list, np.ndarray)) else componentes_y
    print("idx_x = ",idx_x)
    print("idx_y = ",idx_y)
    
    
    
    porcentaje_varianza_x = varianza_porcentaje[idx_x-1]
    porcentaje_varianza_y = varianza_porcentaje[idx_y-1]
    print("porcentaje_varianza_x = ",porcentaje_varianza_x)
    print("porcentaje_varianza_y = ",porcentaje_varianza_y)

    eje_x = dato_pca[:, idx_x-1]
    eje_y = dato_pca[:, idx_y-1]
    print("eje_x:")
    print(eje_x)
    print("eje_x:", len(eje_x))
    print("eje_y:")
    print(eje_y)
    print("eje_y:", len(eje_y))
    
    dato_2d = np.column_stack((eje_x, eje_y))
    print("DATO_2D:")
    print(dato_2d)

    df_pca = pd.DataFrame(dato_2d, columns=['PC1', 'PC2'])
    df_pca['Tipo'] = types
    print("df_pca:")
    print(df_pca)

    fig = go.Figure()
    intervalo = float(intervalo_confianza) / 100  # convertir a decimal
    print("Intervalo:",intervalo)
    
    for tipo in np.unique(types):
        indices = df_pca['Tipo'] == tipo
        fig.add_trace(go.Scatter(
            x=df_pca.loc[indices, 'PC1'],
            y=df_pca.loc[indices, 'PC2'],
            mode='markers',
            marker=dict(size=5, color=asignacion_colores[tipo], opacity=0.7),
            name=f'Tipo {tipo}'
        ))

        datos_tipo = df_pca.loc[indices, ['PC1', 'PC2']].to_numpy()

        if datos_tipo.shape[0] > 2 and not np.allclose(datos_tipo.std(axis=0), 0):
            centro = np.mean(datos_tipo, axis=0)
            cov = np.cov(datos_tipo.T)
            elipse = generar_elipse(centro, cov, color=asignacion_colores[tipo], intervalo_confianza=intervalo)
            fig.add_trace(elipse)
        else:
            print(f"⚠️ Grupo '{tipo}' con datos insuficientes o varianza nula para elipse.")

    fig.update_layout(
        title='Análisis de Componentes Principales 2D',
        xaxis_title=f'PC{idx_x} ({porcentaje_varianza_x:.2f}%)',
        yaxis_title=f'PC{idx_y} ({porcentaje_varianza_y:.2f}%)',
        margin=dict(l=0, r=0, b=0, t=40)
    )

    return fig
    
    
def generar_elipse(centro, cov, num_puntos=100, color='rgba(150,150,150,0.3)', intervalo_confianza=0.95):
    try:
        U, S, _ = np.linalg.svd(cov)
        radii = np.sqrt(chi2.ppf(intervalo_confianza, df=2) * S)

        theta = np.linspace(0, 2 * np.pi, num_puntos)
        x = np.cos(theta)
        y = np.sin(theta)

        elipse = np.array([x, y]).T @ np.diag(radii) @ U.T + centro

        return go.Scatter(
            x=elipse[:, 0],
            y=elipse[:, 1],
            mode='lines',
            line=dict(color=color, width=2),
            showlegend=False
        )
    except Exception as e:
        print(f"❌ Error generando elipse: {e}")
        return go.Scatter(x=[], y=[], mode='lines', showlegend=False)

    
    
    


# RECIBE MUCHOS PARAMETROS SOLO POR QUE QUIERO QUE SALGA LINDO EL NOMBRE DE LOS EJES
def plot_pca_3d(dato_pca,varianza_porcentaje,asignacion_colores,types,componentes_x,componentes_y,componentes_z,intervalo_confianza):
    
    idx_x = componentes_x[0] if isinstance(componentes_x, (list, np.ndarray)) else componentes_x
    idx_y = componentes_y[0] if isinstance(componentes_y, (list, np.ndarray)) else componentes_y
    idx_z = componentes_z[0] if isinstance(componentes_z, (list, np.ndarray)) else componentes_z
    
    porcentaje_varianza_x = varianza_porcentaje[idx_x-1]
    porcentaje_varianza_y = varianza_porcentaje[idx_y-1]
    porcentaje_varianza_z = varianza_porcentaje[idx_z-1]
    print("porcentaje_varianza_x = ",porcentaje_varianza_x)
    print("porcentaje_varianza_y = ",porcentaje_varianza_y)
    print("porcentaje_varianza_y = ",porcentaje_varianza_z)
    
    eje_x = dato_pca[:, idx_x-1]
    eje_y = dato_pca[:, idx_y-1]
    eje_z = dato_pca[:, idx_z-1]
    print("eje_x:")
    print(eje_x)
    print("eje_x:", len(eje_x))
    print("eje_y:")
    print(eje_y)
    print("eje_y:", len(eje_y))
    print(eje_z)
    print("eje_z:", len(eje_z))
  
  
    dato_3d = np.column_stack((eje_x, eje_y,eje_z))
    print("DATO_2D:")
    print(dato_3d)

    df_pca = pd.DataFrame(dato_3d, columns=['PC1', 'PC2', 'PC3'])
    df_pca['Tipo'] = types

    # print("Cantidad de puntos para graficar:", len(df_pca))
    # print("Tipos:", df_pca['Tipo'].unique())

    print("% Varianza x")
    print(porcentaje_varianza_x)
    
    print("% Varianza y")
    print(porcentaje_varianza_y)
    
    print("% Varianza z")
    print(porcentaje_varianza_z)
      
    print("Componentes x")
    print(componentes_x)
    
    print("Componentes y")
    print(componentes_y)
    
    print("Componentes z")
    print(componentes_z)

    fig = go.Figure() #Usas Plotly
    intervalo = float(intervalo_confianza) / 100  # convertir a decimal
    print("Intervalo:",intervalo)
    

    for tipo in np.unique(types):
        indices = df_pca['Tipo'] == tipo
        fig.add_trace(go.Scatter3d(
            x=df_pca.loc[indices, 'PC1'], # Usa los valores de PC1 del tipo actual. Selecciona solo las filas donde indices es True, es decir, solo los puntos de ese tipo
            y=df_pca.loc[indices, 'PC2'],
            z=df_pca.loc[indices, 'PC3'],
            mode='markers',
            marker=dict(size=5, color=asignacion_colores[tipo], opacity=0.7),
            name=f'Tipo {tipo}'
        ))

        # Generar elipsoide de confianza
        datos_tipo = df_pca.loc[indices, ['PC1', 'PC2', 'PC3']].to_numpy()
        if datos_tipo.shape[0] > 3:
            centro = np.mean(datos_tipo, axis=0)
            cov = np.cov(datos_tipo.T)
            elipsoide = generar_elipsoide(centro, cov, asignacion_colores[tipo],intervalo)
            fig.add_trace(elipsoide)


    fig.update_layout(
        legend=dict(
                font=dict(
                size=18  # Aumenta el tamaño de la leyenda (puedes probar con 16, 18, etc.)
                ),
                title=dict(
                            text="Tipos de Muestras",  # Título de la leyenda
                            font=dict(size=16, family="Arial", color="black")  # Configuración del título
                        ),
                itemsizing="constant",  # Mantiene el tamaño de los íconos proporcional
                bordercolor="black",  # Color del borde de la leyenda
                borderwidth=2,  # Grosor del borde
                bgcolor="rgba(255,255,255,0.7)"  # Fondo semitransparente para la leyenda
        ),
        title=dict(
                    text=f'<b><u>Análisis de Componentes Principales 3D</u></b>',  # Negrita y subrayado
                    x=0.5,  # Centrar el título (0 izquierda, 1 derecha, 0.5 centro)
                    xanchor="center",  # Asegura que esté alineado al centro
                    font=dict(
                    family="Arial",  # Tipo de letra
                    size=20,  # Tamaño del título
                    color="black"  # Color del título
                    )),
        scene=dict(
            xaxis_title= f'PC{componentes_x} {porcentaje_varianza_x:.2f}%',# PARA LA ETIQUETAS
            yaxis_title= f'PC{componentes_y} {porcentaje_varianza_y:.2f}%',
            zaxis_title= f'PC{componentes_z} {porcentaje_varianza_z:.2f}%',
            # PARA QUE EL CUBO SEA DE COLOR GRIS
            #xaxis=dict(backgroundcolor="rgb(240, 240, 240)", gridcolor="gray", showbackground=True),
            #yaxis=dict(backgroundcolor="rgb(240, 240, 240)", gridcolor="gray", showbackground=True),
            #zaxis=dict(backgroundcolor="rgb(240, 240, 240)", gridcolor="gray", showbackground=True)
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    #print("LLEGO HASTA ACA")
    #fig.show(renderer="browser") 
    return fig

def generar_elipsoide(centro, cov, color='rgba(150,150,150,0.3)',intervalo=0.95):
    intervalo_confianza = intervalo
    print("INTERVALO CONFIANZA")
    print(intervalo_confianza)
    
    U, S, _ = np.linalg.svd(cov)
    radii = np.sqrt(chi2.ppf(intervalo_confianza, df=3) * S) # 0.999 para que encierre lo mas que pueda todas las muestras dentro del elipsoide

    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = centro + np.dot(U, np.multiply(radii, [x[i, j], y[i, j], z[i, j]]))

    return go.Surface(x=x, y=y, z=z, opacity=0.3, colorscale=[[0, color], [1, color]], showscale=False)


############## COMIENZA LA IMPLEMENTACION DEL TSNE ################################

def tsne(df, n_componentes,perplexity=30 ,learning_rate=200, max_iter=1000):
    print("DENTRO DE TSNE COMPONENTES : ",n_componentes)
    print(df.shape)  
    componentes = n_componentes
    
    # n_samples = df.shape[0]
    # perplexity = min(30, max(5, n_samples // 3)) # ESTE ES UN PARAMETRO, PERPLEXITY TIENE QUE SER UN VALOR ESPECIAL, CREO QUE MENOR A LA MUESTRA

    tsne = TSNE(n_components=componentes,perplexity=perplexity,learning_rate=learning_rate,max_iter=max_iter,init='pca',random_state=42)
    datos_transformados = tsne.fit_transform(df)
    
    return datos_transformados

def plot_tsne_2d(dato_tsne, tipos, asignacion_colores, intervalo=0.95):
    df = pd.DataFrame(dato_tsne, columns=["Eje X", "Eje Y"])
    df["Tipo"] = tipos
    df["Color"] = [asignacion_colores[t] for t in tipos]

    fig = px.scatter(
        df, x="Eje X", y="Eje Y",
        color="Tipo",
        color_discrete_map=asignacion_colores,
        title="t-SNE 2D",
        hover_name="Tipo"
    )

    # Agregar elipses de confianza
    for tipo in df["Tipo"].unique():
        grupo = df[df["Tipo"] == tipo][["Eje X", "Eje Y"]].values
        if grupo.shape[0] < 3:
            continue
        centro = grupo.mean(axis=0)
        cov = np.cov(grupo.T)
        valores, vectores = np.linalg.eigh(cov)
        orden = valores.argsort()[::-1]
        valores = valores[orden]
        vectores = vectores[:, orden]
        chi2_val = chi2.ppf(intervalo, df=2)
        angulos = np.linspace(0, 2 * np.pi, 100)
        elipse = np.array([np.cos(angulos), np.sin(angulos)])
        #escala = np.sqrt(valores[:, None] * chi2_val)
        escala = np.diag(np.sqrt(valores * chi2_val)) 
        print("vectores:", vectores.shape)
        print("escala:", escala.shape)
        print("elipse:", elipse.shape)
        print("centro:", centro.shape)

        elipse_transf = vectores @ escala @ elipse + centro[:, None]

        fig.add_trace(go.Scatter(
            x=elipse_transf[0],
            y=elipse_transf[1],
            mode="lines",
            line=dict(color=asignacion_colores[tipo], dash="solid"),
            name=f"Elipse {tipo}",
            showlegend=False
        ))

    fig.update_layout(width=800, height=600)
    return fig



def generar_elipsoide_tsne(centro, cov, color='rgba(150,150,150,0.3)', intervalo=0.95):
    U, S, _ = np.linalg.svd(cov)
    radii = np.sqrt(chi2.ppf(intervalo, df=3) * S)

    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = centro + np.dot(U, np.multiply(radii, [x[i, j], y[i, j], z[i, j]]))

    return go.Surface(x=x, y=y, z=z, opacity=0.3, colorscale=[[0, color], [1, color]], showscale=False)

def plot_tsne_3d(dato_tsne, tipos, asignacion_colores, intervalo=0.95):
    tipos = tipos.reset_index(drop=True)
    df = pd.DataFrame(dato_tsne, columns=["Eje X", "Eje Y", "Eje Z"])
    df["Tipo"] = tipos
    df["Color"] = [asignacion_colores[t] for t in tipos]

    fig = go.Figure()

    for tipo in df["Tipo"].unique():
        grupo = df[df["Tipo"] == tipo][["Eje X", "Eje Y", "Eje Z"]].values

        fig.add_trace(go.Scatter3d(
            x=grupo[:, 0],
            y=grupo[:, 1],
            z=grupo[:, 2],
            mode='markers',
            marker=dict(size=5, color=asignacion_colores[tipo], opacity=0.7),
            name=f"Tipo {tipo}"
        ))

        if grupo.shape[0] >= 4:
            centro = grupo.mean(axis=0)
            cov = np.cov(grupo.T)
            elipsoide = generar_elipsoide_tsne(centro, cov, asignacion_colores[tipo], intervalo)
            fig.add_trace(elipsoide)

    fig.update_layout(
        title="t-SNE 3D",
        scene=dict(
            xaxis_title="Eje X",
            yaxis_title="Eje Y",
            zaxis_title="Eje Z",
            camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.5),  # posición de la cámara
            center=dict(x=0, y=0, z=0),     # centro de enfoque
            )
        ),
        margin=dict(l=50, r=50, b=50, t=50),
        width=1000,
        height=800
    )

    return fig



#def pca(df,componentes):

#def tsne(df, n_componentes,perplexity=30 ,learning_rate=200, max_iter=1000):
    
#self.tsne_pca_resultado = tsne_pca(df_intensidades, self.cp_pca, self.cp_tsne)


# # HACER t-SNE(PCA(X)) Y SUS GRAFICOS
def tsne_pca(df,cp_pca,cp_tsne,perplexity=30, learning_rate=200, max_iter=1000):
    """
    Aplica PCA seguido de t-SNE al dataframe dado.
    """
    print("CP_PCA:",cp_pca)
    print("CP_TSNE:",cp_tsne)
    
    print("AHORA LLAMARA A LA FUNCION PCA")
    # Aplicar PCA primero (usás tu función ya hecha)
    dato_pca, _ = pca(df, cp_pca) # LLAMAMOS A LA FUNCION PCA Y RETORNA DOS VALORES PERO NOSOTROS SOLO NECESITAMOS DE LOS PCA Y NO SU VARIANZA

    print("AHORA LLAMARA A LA FUNCION TSNE")
    # Luego aplicar t-SNE sobre ese resultado (USAMOS LA FUNCION HECHA DE TSNE)
    tsne_resultado = tsne(dato_pca,n_componentes=cp_tsne,perplexity=perplexity,learning_rate=learning_rate,max_iter=max_iter)
    
    return tsne_resultado



def generar_informe(nombre_informe,opciones,componentes,intervalo,cp_pca,cp_tsne,componentes_seleccionados,asignacion_colores,pca_resultado,varianza_porcentaje,tsne_resultado,tsne_pca_resultado):

    nombre_archivo = f"{nombre_informe}.txt"
    with open(nombre_archivo, 'w', encoding='utf-8') as f:
        f.write("INFORME DE REDUCCIÓN DE DIMENSIONALIDAD\n")
        f.write("========================================\n\n")

        f.write(">> Parámetros Generales\n")
        f.write(f"Nombre del informe: {nombre_informe}\n")
        f.write(f"Componentes seleccionados para visualización: {componentes_seleccionados}\n")
        f.write(f"Intervalo de confianza: {intervalo}\n")
        f.write(f"Componentes para PCA en t-SNE(PCA(X)): {cp_pca}\n")
        f.write(f"Componentes para t-SNE en t-SNE(PCA(X)): {cp_tsne}\n")
        f.write(f"Componentes Principales: {componentes}\n")
        f.write(f"Varianza porcentaje: {varianza_porcentaje}\n")
        f.write(f"Opciones activadas: {opciones}\n\n")

        f.write(">> Colores asignados por tipo:\n")
        for tipo, color in asignacion_colores.items():
            f.write(f"{tipo}: {color}\n")
        f.write("\n")

        if pca_resultado is not None:
            f.write(">> Resultado de PCA:\n")
            f.write(str(pca_resultado))
            f.write("\n\n")

        if tsne_resultado is not None:
            f.write(">> Resultado de t-SNE:\n")
            f.write(str(tsne_resultado))
            f.write("\n\n")

        if tsne_pca_resultado is not None:
            f.write(">> Resultado de t-SNE(PCA(X)):\n")
            f.write(str(tsne_pca_resultado))
            f.write("\n\n")

    print(f"Informe generado: {nombre_archivo}")


def calculo_hca(dato, raman_shift,opciones):
    dato = dato.dropna()
    dato = dato.apply(pd.to_numeric, errors='coerce').dropna().astype(float)

    datos = dato.iloc[:,1:]
    print("DATO CALCULO HCA")
    print(datos)
    print("RAMAN SHIFT HCA")
    print(raman_shift)
    print("OPCIONES HCA")
    print(opciones)
    
    # SEPERAMOS LOS DATOS QUE TIENE OPCIONES DOS VARIABLES.
    claves = list(opciones.keys()) 
    metodo_distancia = claves[0] if len(claves) > 0 else None # Con el if validamos de que tenga al menos un elemento
    metodo_enlace = claves[1] if len(claves) > 0 else None # Con el if validamos de que tenga el segundo elemento, en caso de no  tener asigna None
    
    if metodo_distancia != "None":
        if metodo_distancia == "Euclidiana":
            nombre_plot = "Euclidiana"
            distancia = pdist(datos.T, metric='euclidean')
            #df_distancias = pd.DataFrame(squareform(distancia), index=datos.T.index, columns=datos.T.index)
            # Guardamos la matriz de distancias en un archivo de texto
            #ruta_archivo = "matriz_distancias.txt"
            #df_distancias.to_csv(ruta_archivo, sep='\t', index=True)
        elif metodo_distancia == "Manhattan":
            nombre_plot = "Manhattan"
            distancia = pdist(datos.T, metric='cityblock')
        elif metodo_distancia == "Coseno":
            nombre_plot = "Coseno"
            distancia = pdist(datos.T, metric='cosine')
        elif metodo_distancia == "Chebyshev":
            nombre_plot = "Chebyshev"
            distancia = pdist(datos.T, metric='chebyshev')
        elif metodo_distancia == "Pearson":
            nombre_plot = "Pearson"
            correlacion = datos.corr(method='pearson')
            distancia = squareform(1 - correlacion)
        elif metodo_distancia == "Spearman":
            nombre_plot = "Spearman"
            correlacion = datos.corr(method='spearman')
            distancia = squareform(1 - correlacion)
        elif metodo_distancia == "Jaccard":
            nombre_plot = "Jaccard"
            distancia = pdist(datos.T, metric='jaccard')
    else:
        print("Opción inválida. Ingrese un metodo de distancia")

    if metodo_enlace != "None":
        if metodo_enlace == "Ward":
            nombre_enlace = "ward"
            dendrograma = sch.linkage(distancia, method='ward')
                    
            # #  Convertirmos la matriz de linkage a un DataFrame
            # df_linkage = pd.DataFrame(dendrograma, columns=["Cluster 1", "Cluster 2", "Distancia", "Elementos fusionados"])              
            # # Guardamos la matriz de linkage en un archivo de texto
            # ruta_archivo = "matriz_linkage.txt"
            # df_linkage.to_csv(ruta_archivo, sep='\t', index=False)
                    
        elif metodo_enlace == "Single Linkage":
            nombre_enlace = "single"
            dendrograma = sch.linkage(distancia, method='single')
        elif metodo_enlace == "Complete Linkage":
            nombre_enlace = "complete"
            dendrograma = sch.linkage(distancia, method='complete')
        elif metodo_enlace == "Average Linkage":
            nombre_enlace = "average"
            dendrograma = sch.linkage(distancia, method='average')
    else:
        print("Opción inválida. Ingrese un metodo de cluster")
            
    fig = plt.figure(figsize=(16, 8))
    sch.dendrogram(
        dendrograma,
        labels=datos.columns,
        truncate_mode='lastp',
        p=12,
        leaf_rotation=90
    )
    plt.title(f'Dendrograma usando {nombre_enlace} linkage con distancia de {nombre_plot} (HCA)')
    plt.xlabel('Muestras')
    plt.ylabel('Distancia')

    return fig        
    # plt.figure(figsize=(16, 8))
    # sch.dendrogram(
    # dendrograma,
    # labels=datos.columns,
    # truncate_mode='lastp',
    # p=12,
    # leaf_rotation=90
    # )
    # plt.title(f'Dendrograma usando {nombre_enlace} linkage con distancia de {nombre_plot} (HCA)')
    # plt.xlabel('Muestras')
    # plt.ylabel('Distancia')

    # ruta_imagen = "dendrograma.png"
    # plt.savefig(ruta_imagen, dpi=300, bbox_inches='tight')
    # webbrowser.open(ruta_imagen)
    # plt.show()

    # return fig
    
    

# HCA 
# LOANDINS
# VER EL PCA DE RICARDO POR QUE EL ARCHIVO CARBON2.CSV NO ME SALE JUNTOS LAS MUESTRA E7 E6




def grafico_loading(pca, raman_shift, op_pca):
    
    print("ENTRO EN GRAFICO DE LOADING")
    print("selected_components=",op_pca) # op_pca son los cp seleccionados
    
    if op_pca[2] == 0:
        op_pca.remove(0)
    print("selected_components=",op_pca) # op_pca son los cp seleccionados
    
    print("PCA en LOADING")
    print(pca)
    
    print("RAMAN SHIFT LOADING")
    print(raman_shift)
    # print("PCA dentro de la función de loading")
    # print(pca)
    # print("Raman shift =",raman_shift.shape)
    # print(raman_shift)
 
    # Calcular el PCA dentro de la función
    modelo_pca = PCA(n_components=max(op_pca)+1)
    modelo_pca.fit(pca)
    loadings = modelo_pca.components_
    print("Loadings shape:", loadings.shape)

    # Crear DataFrame con Raman Shift
    df_loadings = pd.DataFrame({"Raman_Shift": raman_shift})

    # Crear gráfico
    plt.figure(figsize=(10, 6))
    for i in op_pca:
        if i >= loadings.shape[0]:
            print(f"Advertencia: El componente PC{i+1} no existe.")
            continue
        plt.plot(raman_shift, loadings[i], label=f'PC{i}')
        df_loadings[f'PC{i+1}'] = loadings[i]

    # Ajuste de ejes
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))

    plt.xlabel('Raman Shift (cm⁻¹)')
    plt.ylabel('Loading')
    plt.title('Loading Plot para PCA y Raman Shift')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Pregunta por exportar CSV (opcional si estás en GUI)
    try:
        print("Descargar CSV de loadings?")
        print("1. Sí")
        print("2. No")
        des = int(input("Opción: "))
        if des == 1:
            df_loadings.to_csv("loadings.csv", index=False)
            print("Archivo 'loadings.csv' descargado.")
            time.sleep(1)
    except Exception:
        print("Modo silencioso: no se pidió CSV.")

    return plt.gcf()  # retorna la figura actual
