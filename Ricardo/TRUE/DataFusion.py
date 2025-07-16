#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 11:55:37 2025

@author: rick
"""

import pandas as pd
import csv
import os
import re  # Para manejo de expresiones regulares
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter, medfilt
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d
import sys
from scipy.interpolate import UnivariateSpline
import plotly.graph_objects as go
from scipy.stats import chi2
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import seaborn as sns
from scipy.interpolate import interp1d






## Función para detectar qué tipo de delimitador tiene el CSV
def detectar_separador(archivo):
    try:
        with open(archivo, 'r', newline='', encoding='utf-8') as f:
            muestra = f.read(1024)  # Leer solo una parte del archivo para detección rápida
            sniffer = csv.Sniffer()
            if sniffer.has_header(muestra):
                print("✔ Se detectó que el archivo tiene encabezado.")
            separador = sniffer.sniff(muestra).delimiter
            #print(f"🔍 Separador detectado: '{separador}'")
            return separador
    except Exception as e:
        print(f"❌ Error al analizar el archivo: {e}")
        return None


## Función para limpiar encabezados (elimina sufijos numéricos y caracteres especiales)
def limpiar_encabezados(columnas):
    """
    Limpia los encabezados eliminando sufijos numéricos (".1", ".2", ...) y 
    caracteres especiales como tabulaciones, saltos de línea y espacios extra.
    """
    columnas_limpias = []
    for col in columnas:
        col_limpia = re.sub(r'\.\d+$', '', col)  # Elimina sufijos numéricos
        col_limpia = re.sub(r'[\t\n\r]+', '', col_limpia)  # Elimina \t, \n, \r
        col_limpia = col_limpia.strip()  # Elimina espacios extra al inicio y al final
        columnas_limpias.append(col_limpia)
    
    return columnas_limpias


## Función para leer el CSV, detectar separador y limpiar encabezados
def lectura_archivo():
    archivo = input("Ingrese la ruta o el nombre del archivo: ")
    separador = detectar_separador(archivo)
    if separador:
        try:
            df = pd.read_csv(archivo, sep=separador)
            print("✔ Archivo leído correctamente.")
            df.columns = limpiar_encabezados(df.columns) # Limpieza de nombres de columnas
            print("🛠 Encabezados limpios:", df.columns.tolist())
            return df
        except Exception as e:
            print(f"❌ Error al leer CSV: {e}")
            return None
    return None

def mostrar_espectros(df): 
    
    x_column = df.columns[0]
    
    unique_types = encabezados(df) 
    
    colors = plt.cm.tab20.colors  # Paleta de colores suficientemente grande
    color_map = {unique: colors[i % len(colors)] for i, unique in enumerate(unique_types)}

    # Configurar la figura
    plt.figure(figsize=(14, 10))
    
    # Graficar cada tipo una sola vez en la leyenda
    for unique_type in unique_types:
        # Filtrar las columnas correspondientes al tipo actual
        columns = [col for col in df.columns if col.startswith(unique_type)]
        
        # Graficar todas las columnas del tipo actual
        for col in columns:
            plt.plot(df[x_column], df[col], color=color_map[unique_type], alpha=0.6)
        
        # Agregar una entrada en la leyenda solo para el tipo (una vez)
        plt.plot([], [], label=unique_type, color=color_map[unique_type])  # Dummy plot for legend
    
    # Etiquetas y leyendas
    plt.title("Espectros Raman", fontsize=16)
    plt.xlabel(f"{x_column} (cm⁻¹)", fontsize=14)  # Se usa el nombre de la primera columna
    plt.ylabel("Intensidad", fontsize=14)
    plt.legend(title="Tipos", fontsize=12, loc='upper right', frameon=False)
    plt.grid(True)
    
    # Mostrar la gráfica
    plt.show()
      
def encabezados(df):
    # Obtener los encabezados únicos
    unique_headers = df.columns.unique()
    print("\n🔹 Encabezados únicos:")
    print(unique_headers)
    unique_types = set(df.columns[1:])  # Toma todas las columnas excepto la primera
    print(unique_types)
    return unique_types 

def minmax(df):
    x_column_index = 0  # asumimos que la primera columna es X
    df_norm = df.copy()

    for i in range(len(df.columns)):
        if i == x_column_index:
            continue  # no normalizar la columna X
        try:
            col_data = pd.to_numeric(df.iloc[:, i], errors='coerce')
            col_min = col_data.min()
            col_max = col_data.max()
            rango = col_max - col_min

            if pd.notna(rango) and rango != 0:
                df_norm.iloc[:, i] = (col_data - col_min) / rango
            else:
                df_norm.iloc[:, i] = 0
        except Exception as e:
            print(f"❌ Error al normalizar columna #{i}: {e}")

    print("✅ Min-Max aplicado")
    print(df_norm)
    return df_norm

def normalizar_area(df):
    """
    Normaliza cada columna (excepto la primera) dividiendo por el área bajo la curva.
    Se asume que la primera columna es el eje X (e.g. Raman shift).
    """
    x_column = df.columns[0]
    df_norm = df.copy()

    x = df[x_column].values

    for i in range(1, len(df.columns)):
        y = pd.to_numeric(df.iloc[:, i], errors='coerce')

        # Calcular área con integración numérica
        area = np.trapz(y, x)

        if pd.notna(area) and area != 0:
            df_norm.iloc[:, i] = y / area
        else:
            df_norm.iloc[:, i] = 0
            print(f"⚠ Área nula o inválida en columna {df.columns[i]}")

    print("✅ Normalización por área aplicada.")
    return df_norm

def normalizar_zscore(df):
    """
    Normaliza cada columna (excepto la primera) usando Z-score:
    (valor - media) / desviación estándar.
    """
    df_norm = df.copy()
    #x_column = df.columns[0]

    for i in range(1, len(df.columns)):
        y = pd.to_numeric(df.iloc[:, i], errors='coerce')
        media = y.mean()
        std = y.std()

        if pd.notna(std) and std != 0:
            df_norm.iloc[:, i] = (y - media) / std
        else:
            df_norm.iloc[:, i] = 0
            print(f"⚠ Desviación estándar nula en columna {df.columns[i]}")

    print("✅ Normalización Z-score aplicada.")
    return df_norm   
    
def normalizar_media_limpiar(df):
    """
    Normaliza cada columna (excepto la primera) dividiendo por su media.

    Parámetros:
    - df: DataFrame con espectros. La primera columna es el eje X.

    Retorna:
    - df_norm: DataFrame normalizado por media.
    """
    df_norm = df.copy()
    #x_column = df.columns[0]

    for i in range(1, len(df.columns)):
        y = pd.to_numeric(df.iloc[:, i], errors='coerce')
        media = y.mean()

        if pd.notna(media) and media != 0:
            df_norm.iloc[:, i] = y / media
        else:
            df_norm.iloc[:, i] = 0
            print(f"⚠ Media nula o inválida en columna {df.columns[i]}")

    print("✅ Normalización por media aplicada.")
    return df_norm

def normalizar_media(df):

    eje_x = df.iloc[:, 0]

    intensidades = df.iloc[:, 1:]

    intensidades_centradas = intensidades - intensidades.mean()

    df_normalizado = pd.concat([eje_x, intensidades_centradas], axis=1)
    
    return df_normalizado

def normalizar(df):
    print("""
          1. Normalizar por Min-Max
          2. Normalizar por Area
          3. Normalizar por Z-Score
          4. Normalizar por media
          0. Volver
          """)
    opt = int(input("ingrese opcion: "))
    if opt == 0:
        print("Volviendo...")
    if opt == 1:
        df = minmax(df)
    elif opt == 2:
        df = normalizar_area(df)
    elif opt == 3: 
        df = normalizar_zscore(df)
    elif opt == 4:
        df = normalizar_media(df)
    elif opt==0:
       print("""
             volviendo al menu...
             {}
             """.format("-" * 32)) 
    print(df.head())
    return df


def savitzky(df, window_length=11, polyorder=2):
    """
    Aplica suavizado Savitzky-Golay a todas las columnas del DataFrame excepto la primera.
    Parámetros:
    - df: DataFrame con la primera columna como eje X.
    - window_length: Tamaño de ventana (debe ser impar y >= polyorder + 2).
    - polyorder: Orden del polinomio (típicamente 2 o 3).

    Retorna:
    - df_suavizado: nuevo DataFrame con los espectros suavizados.
    """
    window_length = int(input("Ingrese tamaño de la ventana: "))
    polyorder = int(input("Ingrese orden polinomico: "))
    while window_length % 2 == 0 or polyorder > window_length - 2:
        window_length= int(input("""
                    Por favor ingrese un numero impar en el tamaño
                    de la ventana
                    -> """))
        polyorder = int(input(f"""
                    Por favor ingrese un numero menor a igual a {window_length-2}
                    -> """))          
    
        
    df_suavizado = df.copy()
    n_cols = df.shape[1]

    for i in range(1, n_cols):  # excluye la primera columna (X)
        y = pd.to_numeric(df.iloc[:, i], errors='coerce')
        # Ajuste automático si window_length es mayor que los datos
        win = min(window_length, len(y) - 1 if len(y) % 2 == 0 else len(y))
        if win % 2 == 0:
            win -= 1  # asegurarse de que sea impar
        if win > polyorder:
            df_suavizado.iloc[:, i] = savgol_filter(y, window_length=win, polyorder=polyorder)
        else:
            print(f"⚠ Columna '{df.columns[i]}' no se suavizó (window_length <= polyorder)")
            

    print("✅ Suavizado Savitzky-Golay aplicado.")
    print(df.head())
    return df_suavizado

def media_movil(df): #TODO: CORREGIR LOS PRIMEROS 
    """
    Aplica suavizado por media móvil a todas las columnas excepto la primera.

    Parámetros:
    - df: DataFrame con la primera columna como eje X.
    - window_size: Tamaño de la ventana de promedio (entero impar recomendado).

    Retorna:
    - df_suavizado: DataFrame con los espectros suavizados.
    """
    window_size = int(input("\t\t\tIngrese tamaño de ventana:"))
    while window_size % 2 == 0:
        window_size = int(input("""
                                 Por favor ingrese un numero impar, de preferencia 5, 7 o 9 
                                 para que el punto central esté bien definido.
                                 -> """))
    df_suavizado = df.copy()
    n_cols = df.shape[1]

    for i in range(1, n_cols):  # Omitimos la primera columna (eje X)
        y = pd.to_numeric(df.iloc[:, i], errors='coerce')
        # Aplicar media móvil con padding en bordes
        suavizado = np.convolve(y, np.ones(window_size)/window_size, mode='same')
        df_suavizado.iloc[:, i] = suavizado

    print("✅ Suavizado por media móvil aplicado.")
    print(df.head())
    return df_suavizado

def filtro_gaussiano(df):
    """
    Aplica suavizado gaussiano a todas las columnas del DataFrame excepto la primera.

    Parámetros:
    - df: DataFrame con espectros. La primera columna es el eje X.
    Valor de sigma	Resultado en el espectro
        sigma pequeño (1 o 2)	Suavizado leve (conserva detalles)
        sigma grande (5 o más)	Suavizado intenso (pierde detalles finos)
    - sigma: Desviación estándar de la gaussiana (mayor = más suavizado).

    Retorna:
    - df_suavizado: DataFrame con los espectros suavizados.
    """
    sigma = int(input("\t\t\tIngrese sigma:")) #considerar usar sigma 2 o 3 
    
    df_suavizado = df.copy()
    #x_column = df.columns[0]

    for i in range(1, len(df.columns)):
        y = pd.to_numeric(df.iloc[:, i], errors='coerce')
        df_suavizado.iloc[:, i] = gaussian_filter1d(y, sigma=sigma)

    print("✅ Suavizado Gaussiano aplicado.")
    print(df.head())
    return df_suavizado

def mediana(df):
    """
    Aplica suavizado por mediana a todas las columnas excepto la primera.

    Parámetros:
    - df: DataFrame con espectros. La primera columna es el eje X.

    Retorna:
    - df_suavizado: DataFrame suavizado por mediana.
    """
    ventana = int(input("\t\t\tIngrese tamaño de la ventana (impar recomendado): "))

    # Asegurar que la ventana sea impar
    if ventana % 2 == 0:
        ventana += 1
        print(f"\t\t\tVentana ajustada a {ventana} (impar obligatorio)")

    df_suavizado = df.copy()

    for i in range(1, len(df.columns)):
        y = pd.to_numeric(df.iloc[:, i], errors='coerce')
        suavizado = medfilt(y, kernel_size=ventana)
        df_suavizado.iloc[:, i] = suavizado

    print("✅ Suavizado por mediana aplicado.")
    return df_suavizado

def spline(df):
    """
    Aplica una interpolación spline suave a todas las columnas excepto la primera.

    Parámetros:
    - df: DataFrame con espectros. La primera columna es el eje X.

    Retorna:
    - df_suavizado: DataFrame con curvas suavizadas por spline.
    """
    print("""
                s bajo (≈ 0.1 – 1) → se ajusta más estrictamente a los datos (menos suave).
                s alto (≈ 2 – 5 o más) → curva más flexible, más suavizada.
                Un buen valor inicial: s = 1.5
          """)
    suavizado = float(input("\t\t\tIngrese factor de suavizado (s): "))  # Ej: 1.0 a 5.0

    df_suavizado = df.copy()
    x = pd.to_numeric(df.iloc[:, 0], errors='coerce')  # eje X

    for i in range(1, len(df.columns)):
        y = pd.to_numeric(df.iloc[:, i], errors='coerce')
        try:
            spline = UnivariateSpline(x, y, s=suavizado)
            df_suavizado.iloc[:, i] = spline(x)
        except Exception as e:
            print(f"❌ Error en columna {df.columns[i]}: {e}")
            df_suavizado.iloc[:, i] = y  # dejar los datos originales si falla

    print("✅ Suavizado por interpolación spline aplicado.")
    return df_suavizado

       
def correccion_polinomial(df, grado=3):
    df_corregido = df.copy()
    x = np.arange(df.shape[0])
    for i in range(1, len(df.columns)):
        y = pd.to_numeric(df.iloc[:, i], errors='coerce').fillna(0)
        coef = np.polyfit(x, y, grado)
        baseline = np.polyval(coef, x)
        df_corregido.iloc[:, i] = y - baseline
    print(f"✅ Corrección Polinomial aplicada (grado={grado})")
    return df_corregido

def correccion_asls(df, lam=1e5, p=0.01, niter=10):
    """
    Implementación manual del algoritmo ALS (Asymmetric Least Squares Smoothing)
    basado en Eilers & Boelens (2005).
    """
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    def baseline_als(y, lam, p, niter):
        L = len(y)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
        D = lam * D.dot(D.transpose())
        w = np.ones(L)
        for _ in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + D
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)
        return z

    df_corregido = df.copy()
    for i in range(1, len(df.columns)):
        y = pd.to_numeric(df.iloc[:, i], errors='coerce').fillna(0)
        baseline = baseline_als(y.values, lam=lam, p=p, niter=niter)
        df_corregido.iloc[:, i] = y - baseline
    print(f"✅ Corrección ALS aplicada (lam={lam}, p={p}, niter={niter})")
    return df_corregido

def correccion_airpls(df, lam=1e5, niter=10):
    """
    Implementación simplificada de airPLS basada en Zhang et al. (2010).
    """
    def airpls_baseline(y, lam=1e5, niter=10):
        from scipy import sparse
        from scipy.sparse.linalg import spsolve

        L = len(y)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
        D = D.dot(D.transpose())
        H = lam * D
        w = np.ones(L)
        for i in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + H
            z = spsolve(Z, w * y)
            d = y - z
            m = np.mean(d[d < 0]) if np.any(d < 0) else 0
            s = np.std(d[d < 0]) if np.any(d < 0) else 0.01
            w = np.exp(-(d - m)**2 / (2 * s**2))
            w[d > z] = 1
        return z

    df_corregido = df.copy()
    for i in range(1, len(df.columns)):
        y = pd.to_numeric(df.iloc[:, i], errors='coerce').fillna(0)
        baseline = airpls_baseline(y.values, lam=lam, niter=niter)
        df_corregido.iloc[:, i] = y - baseline
    print(f"✅ Corrección airPLS aplicada (lam={lam}, iter={niter})")
    return df_corregido

def correccion_shirley(df, tol=1e-3, max_iter=100):
    """
    Corrección de fondo tipo Shirley basada en la formulación clásica.
    """
    def shirley_baseline(y, tol=1e-3, max_iter=100):
        y = np.asarray(y, dtype=float)
        N = len(y)
        b = np.zeros_like(y)
        y0 = y[0]
        yn = y[-1]
        b_old = b.copy()
        for _ in range(max_iter):
            for i in range(N):
                integral = np.trapz(y[i:] - b_old[i:], dx=1)
                b[i] = y0 + (yn - y0) * (integral / np.trapz(y - b_old, dx=1))
            if np.linalg.norm(b - b_old) < tol:
                break
            b_old = b.copy()
        return b

    df_corregido = df.copy()
    for i in range(1, len(df.columns)):
        y = pd.to_numeric(df.iloc[:, i], errors='coerce').fillna(0)
        baseline = shirley_baseline(y.values, tol=tol, max_iter=max_iter)
        df_corregido.iloc[:, i] = y - baseline
    print(f"✅ Corrección Shirley aplicada (tol={tol}, iter={max_iter})")
    return df_corregido

def correccion_modpoly(df, grado=3):
    """
    Implementación básica de ModPoly: ajuste polinomial iterativo ignorando picos.
    """
    from numpy.polynomial import Polynomial

    def modpoly_baseline(y, grado=3, max_iter=100, tol=1e-6):
        mask = np.ones_like(y, dtype=bool)
        for _ in range(max_iter):
            p = Polynomial.fit(np.arange(len(y))[mask], y[mask], grado)
            base = p(np.arange(len(y)))
            nueva_mask = y < base
            if np.all(mask == nueva_mask):
                break
            mask = nueva_mask
        return base

    df_corregido = df.copy()
    for i in range(1, len(df.columns)):
        y = pd.to_numeric(df.iloc[:, i], errors='coerce').fillna(0)
        baseline = modpoly_baseline(y.values, grado=grado)
        df_corregido.iloc[:, i] = y - baseline
    print(f"✅ Corrección ModPoly aplicada (grado={grado})")
    return df_corregido

def correccion_lineal(df):
    """
    Corrección de fondo lineal: resta una recta entre los extremos del espectro.
    """
    df_corregido = df.copy()
    for i in range(1, len(df.columns)):
        y = pd.to_numeric(df.iloc[:, i], errors='coerce').fillna(0)
        #x = np.arange(len(y))
        y0, y1 = y.iloc[0], y.iloc[-1]
        baseline = np.linspace(y0, y1, len(y))
        df_corregido.iloc[:, i] = y - baseline
    print("✅ Corrección Lineal aplicada")
    return df_corregido

def correccion_rolling_ball(df, radio=50):
    from scipy.ndimage import minimum_filter1d, maximum_filter1d

    def rolling_ball_baseline(y, radius):
        y = np.asarray(y, dtype=float)
        y_smooth = maximum_filter1d(y, size=radius)
        baseline = minimum_filter1d(y_smooth, size=radius)
        return baseline

    df_corregido = df.copy()
    for i in range(1, len(df.columns)):
        y = pd.to_numeric(df.iloc[:, i], errors='coerce').fillna(0)
        baseline = rolling_ball_baseline(y.values, radio)
        df_corregido.iloc[:, i] = y - baseline
    print(f"✅ Corrección Rolling Ball aplicada (radio={radio})")
    return df_corregido


 
def suavizado(df):
    print("""
          1. Smoothing por Savitzky-Golay
          2. Smoothing por Media Movil 
          3. Smoothing por Filtro Gaussiano
          4. Smoothing por Mediana
          5. Smoothing por Interpolacion suave 
          0. Volver
          """)
    opt = int(input("ingrese opcion: "))
    if opt == 1:
        df = savitzky(df)
    elif opt == 2:
        df = media_movil(df)
    elif opt == 3: 
        df = filtro_gaussiano(df)
    elif opt == 4:
        df = mediana(df)
    elif opt == 5 :
        df = spline(df)
    elif opt==0:
       print("""
             volviendo al menu...
             {}
             """.format("-" * 32))
    return df
          

def derivada(df):
    """
    Aplica la primera o segunda derivada a todos los espectros (columnas) del DataFrame, excepto la primera (eje X).

    Parámetros:
    - df: DataFrame con la primera columna como eje X (e.g., Raman shift).
    Retorna:
    - df_derivada: DataFrame con derivadas aplicadas a los espectros.
    """

    try:
        orden = int(input("\t\t\tIngrese orden (1 o 2): "))
        while orden not in [1, 2]:
            orden = int(input("\t⚠️ Orden inválido. Ingrese 1 o 2: "))
    except Exception as e:
        print("❌ Error en la entrada:", e)
        return None

    df_derivada = df.copy()
    x = pd.to_numeric(df.iloc[:, 0], errors='coerce')

    for i in range(1, df.shape[1]):
        y = pd.to_numeric(df.iloc[:, i], errors='coerce')

        # Verifica que no haya NaNs en x o y
        if x.isnull().any() or y.isnull().any():
            print(f"⚠️ Columna {df.columns[i]} contiene valores nulos. Se omite.")
            continue

        if orden == 1:
            deriv = np.gradient(y.values, x.values)
        else:
            deriv = np.gradient(np.gradient(y.values, x.values), x.values)

        df_derivada.iloc[:, i] = deriv

    print(f"\n✅ Derivada de orden {orden} aplicada correctamente.")
    print(df_derivada.head())
    return df_derivada

def obtener_types(df):
    """
    Extrae el tipo base de cada espectro a partir del nombre de la columna.
    Asume que los nombres son del tipo 'Ibuprofeno_1', 'Aspirina_2', etc.
    """
    return [col.split('_')[0] for col in df.columns[1:]]  # omite la primera columna (eje X)

def aplicar_pca(df, types, n_componentes=2):
    n_componentes = int(input("ingrese xdlol"))
    datos = df.iloc[:, 1:].T
    
    pca = PCA(n_components=n_componentes)
    dato_pca = pca.fit_transform(datos)
    varianza = pca.explained_variance_ratio_ * 100

    if n_componentes == 2:
        plot_pca_2d_interactivo(types, dato_pca, varianza)
    elif n_componentes == 3:
        plot_pca_3d_interactivo(types, dato_pca, varianza)
    else:
        print(f"PCA aplicado con {n_componentes} componentes.")
        print("⚠️ Visualización no disponible para más de 3 dimensiones. Puedes analizar la matriz resultante manualmente o reducir a 2-3 componentes para visualizar.")
    print(dato_pca)
    return dato_pca, varianza, pca


def generar_elipsoide(centro, cov, color='rgba(150,150,150,0.3)', intervalo_confianza=0.95):
    U, S, _ = np.linalg.svd(cov)
    radii = np.sqrt(chi2.ppf(intervalo_confianza, df=3) * S)

    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = centro + np.dot(U, np.multiply(radii, [x[i, j], y[i, j], z[i, j]]))

    return go.Surface(x=x, y=y, z=z, opacity=0.3, colorscale=[[0, color], [1, color]], showscale=False)


def plot_pca_2d_interactivo(types, dato_pca, varianza):
    df_pca = pd.DataFrame(dato_pca, columns=['PC1', 'PC2'])
    df_pca['Tipo'] = types

    colores_unicos = plt.cm.tab20.colors
    tipos_unicos = sorted(df_pca['Tipo'].unique())
    asignacion_colores = {tipo: f'rgba({int(r*255)},{int(g*255)},{int(b*255)},0.7)' for tipo, (r, g, b) in zip(tipos_unicos, colores_unicos)}

    fig = go.Figure()

    for tipo in tipos_unicos:
        indices = df_pca['Tipo'] == tipo
        fig.add_trace(go.Scatter(
            x=df_pca.loc[indices, 'PC1'],
            y=df_pca.loc[indices, 'PC2'],
            mode='markers',
            marker=dict(size=8, color=asignacion_colores[tipo], opacity=0.7),
            name=f'Tipo {tipo}'
        ))

    fig.update_layout(
        title=f'<b><u>PCA 2D - Agrupamiento por Tipo</u></b>',
        xaxis_title=f'PC1 ({varianza[0]:.2f}%)',
        yaxis_title=f'PC2 ({varianza[1]:.2f}%)',
        legend=dict(font=dict(size=14), bgcolor="rgba(255,255,255,0.7)", bordercolor="black", borderwidth=1),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.show(renderer="browser")


def plot_pca_3d_interactivo(types, dato_pca, varianza):
    df_pca = pd.DataFrame(dato_pca, columns=['PC1', 'PC2', 'PC3'])
    df_pca['Tipo'] = types

    intervalo_confianza = float(input("Ingrese el intervalo de confianza (%): ")) / 100

    colores_unicos = plt.cm.tab20.colors
    tipos_unicos = sorted(df_pca['Tipo'].unique())
    asignacion_colores = {tipo: f'rgba({int(r*255)},{int(g*255)},{int(b*255)},0.7)' for tipo, (r, g, b) in zip(tipos_unicos, colores_unicos)}

    fig = go.Figure()

    for tipo in tipos_unicos:
        indices = df_pca['Tipo'] == tipo
        fig.add_trace(go.Scatter3d(
            x=df_pca.loc[indices, 'PC1'],
            y=df_pca.loc[indices, 'PC2'],
            z=df_pca.loc[indices, 'PC3'],
            mode='markers',
            marker=dict(size=5, color=asignacion_colores[tipo], opacity=0.7),
            name=f'Tipo {tipo}'
        ))

        datos_tipo = df_pca.loc[indices, ['PC1', 'PC2', 'PC3']].to_numpy()
        if datos_tipo.shape[0] > 3:
            centro = np.mean(datos_tipo, axis=0)
            cov = np.cov(datos_tipo.T)
            elipsoide = generar_elipsoide(centro, cov, asignacion_colores[tipo], intervalo_confianza)
            fig.add_trace(elipsoide)

    fig.update_layout(
        legend=dict(
            font=dict(size=14),
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="black",
            borderwidth=1
        ),
        title=dict(
            text=f'<b><u>PCA 3D - Agrupamiento por Tipo</u></b>',
            x=0.5,
            xanchor="center",
            font=dict(size=20)
        ),
        scene=dict(
            xaxis_title=f'PC1 ({varianza[0]:.2f}%)',
            yaxis_title=f'PC2 ({varianza[1]:.2f}%)',
            zaxis_title=f'PC3 ({varianza[2]:.2f}%)'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.show(renderer="browser")



def correccion_base(df):
    print("""
          1. Correccion Lineal  
          2. Correccion Polinomial
          3. Correccion AsLS
          4. Correccion airPLS
          5. Correccion Shirley
          6. Correccion Modpoly 
          7. Correccion Rolling Ball
          0. Volver
          """)
    opt = int(input("ingrese opcion: "))
    if opt == 1:
        df = correccion_lineal(df)
    elif opt == 2:
        df = correccion_polinomial(df)
    elif opt == 3:
        df = correccion_asls(df)
    elif opt == 4:
        df = correccion_airpls(df)
    elif opt == 5:
        df = correccion_shirley(df)
    elif opt == 6:
        df = correccion_modpoly(df)
    elif opt == 7:
        df = correccion_rolling_ball(df)
    elif opt == 0:
        print("Volviendo ...")
    return df

def aplicar_hca(df, cortar_en=3, usar_maxclust=True):
    """
    Aplica Análisis de Agrupamiento Jerárquico (HCA), muestra mapa de calor de distancia y dendrograma.

    Parámetros:
    - df: DataFrame con espectros. Primera columna = eje X.
    - cortar_en: Número de grupos o distancia de corte
    - usar_maxclust: True para criterio por cantidad de clusters, False para distancia
    """

    opciones_distancia = {
        '1': 'euclidean',
        '2': 'cityblock',
        '3': 'cosine',
        '4': 'chebyshev',
        '5': 'correlation',
        '6': 'spearman',
        '7': 'jaccard'
    }

    opciones_enlace = {
        '1': 'ward',
        '2': 'single',
        '3': 'complete',
        '4': 'average',
        '5': 'weighted',
        '6': 'centroid',
        '7': 'median'
    }

    print("\nSeleccione el tipo de distancia:")
    print("1. Euclidiana\n2. Manhattan\n3. Coseno\n4. Chebyshev\n5. Correlación (Pearson)\n6. Correlación de Rangos (Spearman)\n7. Jaccard (solo binario)")
    seleccion_d = input("Ingrese el número correspondiente: ").strip()
    metodo_distancia = opciones_distancia.get(seleccion_d, 'euclidean')

    print("\nSeleccione el método de enlace:")
    print("1. Ward\n2. Single\n3. Complete\n4. Average\n5. Weighted\n6. Centroid\n7. Median")
    seleccion_e = input("Ingrese el número correspondiente: ").strip()
    metodo_enlace = opciones_enlace.get(seleccion_e, 'ward')

    print(f"\n📌 Método de distancia: {metodo_distancia}")
    print(f"📌 Método de enlace: {metodo_enlace}")

    nombres = df.columns[1:]
    datos = df.iloc[:, 1:].T

    if metodo_distancia == 'spearman':
        rho, _ = spearmanr(datos)
        distancia_matriz = 1 - rho
        distancia = squareform(distancia_matriz, checks=False)
    else:
        distancia = pdist(datos, metric=metodo_distancia)
        distancia_matriz = squareform(distancia)

    plt.figure(figsize=(10, 8))
    sns.heatmap(distancia_matriz, xticklabels=nombres, yticklabels=nombres, cmap='viridis', annot=False)
    plt.title(f"🗺️ Matriz de Distancias ({metodo_distancia})")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    Z = linkage(distancia, method=metodo_enlace)

    plt.figure(figsize=(12, 6))
    dendrogram(Z, labels=nombres, leaf_rotation=90)
    plt.title("📊 Dendrograma - HCA")
    plt.xlabel("Espectros")
    plt.ylabel("Distancia")
    if not usar_maxclust:
        plt.axhline(y=cortar_en, color='red', linestyle='--')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    if usar_maxclust:
        grupos = fcluster(Z, t=cortar_en, criterion='maxclust')
    else:
        grupos = fcluster(Z, t=cortar_en, criterion='distance')

    print("\n🔹 Grupos asignados:")
    for nombre, grupo in zip(nombres, grupos):
        print(f"{nombre}: Grupo {grupo}")

    return grupos

def inspeccionar_archivo(df, nombre_dataset="Datos"):
    """
    Inspecciona un DataFrame espectroscópico para saber si necesita ajuste de formato.
    - Verifica si es necesario transponer.
    - Verifica si es necesario interpolar.

    Parámetros:
    - df: DataFrame cargado.
    - nombre_dataset: nombre para impresión más clara.
    """
    print(f"\n🔍 Inspección del archivo: {nombre_dataset}") 
    print("Dimensiones (filas, columnas):", df.shape)
    print("Primeras filas:")
    print(df.head())

    if df.shape[0] < df.shape[1]:
        print("\n✅ Las muestras están en columnas. NO se necesita transponer.")
    else:
        print("\n⚠️ Las muestras parecen estar en filas. Puede ser necesario transponer (.T).")

    print("\nVerificando la primera columna:")
    print(df.iloc[:, 0].head())

    if np.all(np.diff(df.iloc[:, 0]) >= 0):
        print("\n✅ El eje X (Raman Shift o Wavenumber) está en orden ascendente.")
    else:
        print("\n⚠️ El eje X NO está en orden ascendente. Puede ser necesario reordenar.")

def analisis_datos(df): 
    #TODO: PCA INPUT DE DIMENSIONES
    #TODO: HCA VER QUE HACER CON GRUPOS
    print("""
          1. PCA   
          2. HCA
          3. K-Means Clustering
          4. DBSCAN
          0. Volver
          """)
    opt = int(input("ingrese opcion: "))
    if opt == 1:
        types = obtener_types(df)
        dato_pca, varianza, pca = aplicar_pca(df, types)
    elif opt == 2:
        grupos = aplicar_hca(df)
    elif opt == 3:
        print( """
              NO RESUELTO 
                1.	Elegís un número de clusters k.
            	2.	El algoritmo selecciona k centroides iniciales aleatorios.
            	3.	Cada espectro se asigna al centroide más cercano.
            	4.	Los centroides se recalculan como el promedio de los espectros asignados a ellos.
            	5.	Se repiten pasos 3-4 hasta que:
            	•	Las asignaciones no cambian.
            	•	O alcanza un número máximo de iteraciones.
              
              
              """)
    elif opt == 4: 
        print("""
              NO RESULETO DBSCAN 
              
              
              """)
    elif opt == 0: 
        print("Volviendo ...")
        
          
def exportar_dataset_csv(df, carpeta_destino='./csv_exportados'):
    """
    Exporta un DataFrame preprocesado a un archivo CSV.
    
    Parámetros:
    - df: DataFrame a guardar.
    - nombre_archivo: Nombre del archivo CSV (sin extensión).
    - carpeta_destino: Carpeta donde guardar los CSVs. Se crea si no existe.
    """
    nombre_archivo = input("ingrese nombre del archivo: ")
    if not os.path.exists(carpeta_destino):
        os.makedirs(carpeta_destino)
    
    
    # Ruta completa
    ruta_completa = os.path.join(carpeta_destino, f'{nombre_archivo}.csv')
    
    try:
        df.to_csv(ruta_completa, index=False)
        print(f"✅ Dataset exportado exitosamente a: {ruta_completa}")
    except Exception as e:
        print(f"❌ Error al exportar el dataset: {e}")
    
        
def listar_archivos_csv(directorio):
    archivos = [f for f in os.listdir(directorio) if f.endswith('.csv')]
    return archivos


#ESTA FUNCION NOS SIRVE PARA ESE ARCHIVO QUE TENIAMOS DEL FTIR DONDE TENIAMOS LOS SUBDIJOS POR CADA COMPONENTE QUIMICO 
def limpiar_etiquetas_columnas(df):
    """
    Limpia los nombres de las columnas eliminando sufijos como '10-B', '1L-E', etc.,
    dejando solo el nombre base. También convierte los nombres a minúsculas
    y elimina espacios extra.
    """
    nuevas_columnas = []
    for col in df.columns:
        if col == df.columns[0]:
            nuevas_columnas.append(col)  # eje X
        else:
            base = re.sub(r'\s*\d+[a-zA-Z-]*$', '', str(col))  # elimina sufijo como '10-B'
            base = re.sub(r'\s+', '', base)  # elimina espacios internos
            base = base.lower().strip()  # minúsculas y sin espacios externos
            nuevas_columnas.append(base)
    df.columns = nuevas_columnas
    return df
        

def interpolar_dataframe(df, nuevo_eje_x):
    eje_original = df.iloc[:, 0]
    columnas = df.columns[1:]
    df_interp = pd.DataFrame({df.columns[0]: nuevo_eje_x})

    for col in columnas:
        x = eje_original
        y = df[col]

        validos = ~(pd.isna(x) | pd.isna(y))
        x_valid = x[validos]
        y_valid = y[validos]

        if len(x_valid) < 2:
            print(f"⚠️ Columna {col} no tiene suficientes puntos válidos para interpolar.")
            df_interp[col] = np.nan
            continue

        f = interp1d(x_valid, y_valid, kind='linear', bounds_error=False, fill_value='extrapolate')
        df_interp[col] = f(nuevo_eje_x)

    if df_interp.shape[1] <= 1:
        return None

    return df_interp

def fusionar_interpolados(df_ftir, df_raman):
    """
    Fusiona dos DataFrames interpolados (FTIR y Raman) y los exporta en formato tradicional.
    """
    df_ftir_t = df_ftir.set_index(df_ftir.columns[0]).T
    df_raman_t = df_raman.set_index(df_raman.columns[0]).T
    df_fusionado = pd.concat([df_ftir_t, df_raman_t], axis=1)
    df_fusionado.index.name = 'Muestra'

    # Convertir al formato tradicional: columnas = muestras, filas = eje X
    df_fusionado_t = df_fusionado.T
    df_fusionado_t.insert(0, 'eje_x', range(1, len(df_fusionado_t)+1))
    columnas = ['eje_x'] + df_fusionado_t.columns[1:].tolist()
    df_tradicional = df_fusionado_t[columnas]

    # Guardar archivo en formato tradicional
    ruta_salida = './csv_exportados/fusion_formato_tradicional.csv'
    df_tradicional.to_csv(ruta_salida, index=False)
    print(f"✅ Fusion exportada en formato tradicional como: {ruta_salida}")

    return df_tradicional



def datafusion():
    base_path = './csv_exportados'
    archivo_ftir = input("Ingrese el nombre del archivo FTIR (con .csv): ").strip()
    ruta_ftir = os.path.join(base_path, archivo_ftir)
    archivo_raman = input("Ingrese el nombre del archivo RAMAN (con .csv): ").strip()
    ruta_raman = os.path.join(base_path, archivo_raman)

    try:
        df_ftir = pd.read_csv(ruta_ftir)
        df_raman = pd.read_csv(ruta_raman)
    except Exception as e:
        print(f"❌ Error al leer los archivos: {e}")
        return

    df_ftir = limpiar_etiquetas_columnas(df_ftir)
    df_raman = limpiar_etiquetas_columnas(df_raman)
    
    df_ftir = df_ftir.loc[:, ~df_ftir.columns.duplicated()]
    df_raman = df_raman.loc[:, ~df_raman.columns.duplicated()]

    df_ftir.iloc[:, 0] = df_ftir.iloc[:, 0].astype(str).str.replace('.', '', regex=False).astype(float)
    df_raman.iloc[:, 0] = df_raman.iloc[:, 0].astype(str).str.replace('.', '', regex=False).astype(float)

    print("\n📌 Elija el tipo de fusión de datos:")
    print("1. Nivel bajo (Low-Level Fusion)")
    print("2. Nivel medio (Feature-Level Fusion) [pendiente]")
    print("3. Nivel alto (Decision-Level Fusion) [pendiente]")
    opcion_fusion = input("Ingrese una opción (1/2/3): ").strip()

    if opcion_fusion == '1':
        fusion_low_level(df_ftir, df_raman, archivo_ftir, archivo_raman, base_path)
    elif opcion_fusion == '2':
        fusion_feature_level(df_ftir, df_raman)
    elif opcion_fusion == '3':
        fusion_decision_level(df_ftir, df_raman)
    else:
        print("❌ Opción inválida.")


def fusion_low_level(df_ftir, df_raman, archivo_ftir, archivo_raman, base_path):
    eje_ftir = df_ftir.iloc[:, 0]
    eje_raman = df_raman.iloc[:, 0]

    if not eje_ftir.equals(eje_raman):
        print("\n⚠️ Los ejes X no coinciden. Se necesita interpolación.")
        opcion = input("¿Desea interpolar ambos datasets a un mismo eje común? (s/n): ").strip().lower()
        if opcion == 's':
            min_comun = max(eje_ftir.min(), eje_raman.min())
            max_comun = min(eje_ftir.max(), eje_raman.max())
            puntos = int(input("Ingrese cantidad de puntos para interpolar (ej. 1500): "))
            nuevo_eje = np.linspace(min_comun, max_comun, puntos)

            df_ftir_interp = interpolar_dataframe(df_ftir, nuevo_eje)
            df_raman_interp = interpolar_dataframe(df_raman, nuevo_eje)

            if df_ftir_interp is None or df_raman_interp is None:
                print("❌ Error: la interpolación no produjo columnas válidas.")
                return

            nombre_ftir = os.path.splitext(archivo_ftir)[0] + '_interpolado.csv'
            df_ftir_interp.to_csv(os.path.join(base_path, nombre_ftir), index=False)

            nombre_raman = os.path.splitext(archivo_raman)[0] + '_interpolado.csv'
            df_raman_interp.to_csv(os.path.join(base_path, nombre_raman), index=False)

            fusion = fusionar_interpolados(df_ftir_interp, df_raman_interp)
        else:
            print("⏭️ No se aplicó interpolación. No es posible continuar con la fusión de bajo nivel.")
            return
    else:
        print("\n✅ Ejes X coinciden. Se puede fusionar directamente.")
        fusion = fusionar_interpolados(df_ftir, df_raman)

    fusion.to_csv(os.path.join(base_path, "fusion_ftir_raman_lowlevel.csv"), index=False)
    print("✅ Fusion (Low-Level) guardada como fusion_ftir_raman_lowlevel.csv")


def fusion_feature_level(df_ftir, df_raman):
    print("🛠️ Funcionalidad para Feature-Level Fusion en desarrollo...")


def fusion_decision_level(df_ftir, df_raman):
    print("🛠️ Funcionalidad para Decision-Level Fusion en desarrollo...")

def menu():
    print("-" * 50) 
    #texto_desplazamiento("MENU", 10, 0.1)
    print("****MENU****")
    print("11. leer otro dataset")
    print("1. Mostrar espectros ")
    print("2. Normalizar Espectro")
    print("3. Suavizar Espectro")
    print("4. Derivar ")
    print("5. Correccion Linea Base")
    print("6. Analisis de Datos y Agrupamiento")
    print("7. Inspeccionar Archivo")
    print("8. Exportar Dataframe")
    print("9. Data Fusion")
    print("0. Salir del programa")     

## Función principal
def main():
    df = lectura_archivo()
    df_original = df
    if df is not None:
        print("\n🔹 Primeras filas del archivo CSV:")
        print(df.head())
    while True:
        menu()
        opt = int(input("Ingrese opcion: "))    
        if opt == 11:
            df=lectura_archivo()
        elif opt == 1:
            print("""
                  1. Mostrar dataframe original
                  2. Mostrar dataframe filtrado 
                  0. Volver
                  """)
            opc_espectros = int(input("Ingrese opcion: "))
            if opc_espectros == 1:
                mostrar_espectros(df_original)
            elif opc_espectros == 2:
                mostrar_espectros(df)
        elif opt == 2:
            df = normalizar(df)
            #print(df)
        elif opt == 3: 
            df = suavizado(df) 
        elif opt == 4:
            df = derivada(df)
        elif opt == 5:
            df = correccion_base(df)
        elif opt == 6: 
            analisis_datos(df)
        elif opt == 7:
            inspeccionar_archivo(df)
        elif opt == 8: 
            exportar_dataset_csv(df)
        elif opt == 9:
            datafusion()
            
        elif opt==0:
            print("""
                saliendo del programa...
                {}
                """.format("-" * 32))
            sys.exit()       
            
    
  

if __name__ == "__main__":
    main()