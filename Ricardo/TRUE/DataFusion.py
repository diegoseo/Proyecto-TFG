#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 11:55:37 2025

@author: rick
"""

import pandas as pd
import csv
import re  # Para manejo de expresiones regulares
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.signal import savgol_filter, medfilt
from sklearn.decomposition import PCA
from numpy import trapz
from scipy.ndimage import gaussian_filter1d
import sys
import time
from scipy.interpolate import UnivariateSpline
from pybaselines.whittaker import asls
from pybaselines import  morphological
from pybaselines.airpls import airpls 
from pybaselines.shirley import shirley
from pybaselines.modpoly import modpoly







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
    
def normalizar_media(df):
    """
    Normaliza cada columna (excepto la primera) dividiendo por su media.

    Parámetros:
    - df: DataFrame con espectros. La primera columna es el eje X.

    Retorna:
    - df_norm: DataFrame normalizado por media.
    """
    df_norm = df.copy()
    x_column = df.columns[0]

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
    return df_suavizado

def media_movil(df):
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
        windows_size = int(input("""
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
    x_column = df.columns[0]

    for i in range(1, len(df.columns)):
        y = pd.to_numeric(df.iloc[:, i], errors='coerce')
        df_suavizado.iloc[:, i] = gaussian_filter1d(y, sigma=sigma)

    print("✅ Suavizado Gaussiano aplicado.")
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
    #print("4. Aplicar PCA al espectro")
    #print("5. aux validar_eje_x")
    print("0. Salir del programa")
    

def derivada(df): # aca tenemos que tener en cuenta el tema del orden y los valores nulos
    """
    Aplica la primera o segunda derivada a todos los espectros (columnas) del DataFrame, excepto la primera (eje X).

    Parámetros:
    - df: DataFrame con la primera columna como eje X (e.g., Raman shift).
    - orden: 1 para primera derivada, 2 para segunda.

    Retorna:
    - df_derivada: DataFrame con derivadas aplicadas a los espectros.
    """
    orden = int(input("\t\t\tIngrese orden:"))
    while orden != 1 and orden != 2:
        orden = int(input("""
              El orden de la derivada solo puede ser 1 o 2
              ->
              """))
    df_derivada = df.copy()
    x = pd.to_numeric(df.iloc[:, 0], errors='coerce')  # eje X

    for i in range(1, len(df.columns)):
        y = pd.to_numeric(df.iloc[:, i], errors='coerce')

        if orden == 1:
            derivada = np.gradient(y, x)
        elif orden == 2:
            derivada = np.gradient(np.gradient(y, x), x)
        else:
            raise ValueError("Solo se permite orden 1 o 2.") #no entrara aqui

        df_derivada.iloc[:, i] = derivada

    print(f"✅ Derivada de orden {orden} aplicada.")
    return df_derivada
       
def correccion_polinomial(df):
    """Aplica corrección de línea base por ajuste polinomial a cada espectro (columna) del DataFrame.
    Parámetros:
    - df: DataFrame con la primera columna como eje X.
    - grado: grado del polinomio para el ajuste (por defecto: 3)

    Retorna:
    - df_corregido: DataFrame con la línea base corregida.
    """
    grado = int(input("""
          Ingrese el grado del polinomio
          Grado del polinomio	Cuándo usarlo
              2 o 3             Fondo ligeramente curvo
              4 a 6	            Fondo más complejo
              >6	           ⚠️ Riesgo de sobreajuste
          ->     
          """))
    df_corregido = df.copy()
    x = pd.to_numeric(df.iloc[:, 0], errors='coerce')

    for i in range(1, len(df.columns)):
        y = pd.to_numeric(df.iloc[:, i], errors='coerce')

        # Ajustar un polinomio al espectro completo
        coef = np.polyfit(x, y, deg=grado)
        base = np.polyval(coef, x)

        # Corregir el espectro
        df_corregido.iloc[:, i] = y - base

    print(f"✅ Corrección polinomial (grado {grado}) aplicada.")
    return df_corregido


def correccion_asls(df, lam=1e5, p=0.01):
    """
    Aplica corrección de línea base con el método ASLS a cada espectro del DataFrame.

    Parámetros:
    - df: DataFrame con la primera columna como eje X.
    - lam: Parámetro de suavizado (lambda). Mayor = línea base más suave.
    - p: Asimetría. Más bajo = más sensible a picos.

    Retorna:
    - df_corregido: DataFrame con fondo corregido por ASLS.
    """
    print("""
          por defecto
          LAMBDA = 1e5 
          Asimetria (p) = 0.01
          """)
    df_corregido = df.copy()
    for i in range(1, len(df.columns)):
        y = pd.to_numeric(df.iloc[:, i], errors='coerce').fillna(0)
        baseline, _ = als.asls(y, lam=lam, p=p)
        corregido = y - baseline
        df_corregido.iloc[:, i] = corregido

    print(f"✅ Corrección ASLS aplicada (λ={lam}, p={p})")
    return df_corregido

def correccion_airpls(df, lam=1e5, max_iter=50):
    """
    Aplica corrección de línea base con el método airPLS a cada espectro del DataFrame.

    Parámetros:
    - df: DataFrame con la primera columna como eje X.
    - lam: Parámetro de suavizado (lambda). Mayor = línea base más suave.
    - max_iter: Número máximo de iteraciones.

    Retorna:
    - df_corregido: DataFrame con fondo corregido.
    """
    print("""
          por defecto
          LAMBDA = 1e5 
          maximas iteraciones  = 50
          """)
    df_corregido = df.copy()

    for i in range(1, len(df.columns)):
        y = pd.to_numeric(df.iloc[:, i], errors='coerce').fillna(0)
        baseline, _ = airpls.airpls(y, lam=lam, max_iter=max_iter)
        corregido = y - baseline
        df_corregido.iloc[:, i] = corregido

    print(f"✅ Corrección airPLS aplicada (λ={lam}, iter={max_iter})")
    return df_corregido



def correccion_shirley(df, max_iter=100): #TODO: ver el tema de las impresiones de max iter 
    """
    Aplica corrección de línea base tipo Shirley a cada espectro del DataFrame.

    Parámetros:
    - df: DataFrame con la primera columna como eje X.
    - max_iter: Número máximo de iteraciones para convergencia.

    Retorna:
    - df_corregido: DataFrame con fondo corregido.
    """
    df_corregido = df.copy()

    for i in range(1, len(df.columns)):
        y = pd.to_numeric(df.iloc[:, i], errors='coerce').fillna(0)
        baseline, _ = shirley.shirley(y, max_iter=max_iter)
        corregido = y - baseline
        df_corregido.iloc[:, i] = corregido

    print(f"✅ Corrección Shirley aplicada (máx. {max_iter} iteraciones)")
    return df_corregido

def correccion_modpoly(df, grado=3, max_iter=100): #TODO ver el tema de grado y max iter 
    """
    Aplica corrección de línea base usando el método ModPoly.

    Parámetros:
    - df: DataFrame con espectros. La primera columna es el eje X.
    - grado: Grado del polinomio (típico: 2 a 6)
    - max_iter: Número máximo de iteraciones.

    Retorna:
    - df_corregido: DataFrame con los espectros corregidos.
    """
    df_corregido = df.copy()

    for i in range(1, len(df.columns)):
        y = pd.to_numeric(df.iloc[:, i], errors='coerce').fillna(0)
        baseline, _ = modpoly.modpoly(y, poly_order=grado, max_iter=max_iter)
        corregido = y - baseline
        df_corregido.iloc[:, i] = corregido

    print(f"✅ Corrección ModPoly aplicada (grado {grado}, {max_iter} iteraciones).")
    return df_corregido

def correccion_lineal(df):
    """
    Aplica corrección lineal de fondo a todos los espectros en el DataFrame,
    restando una línea recta entre los extremos de cada espectro.

    Parámetros:
    - df: DataFrame con la primera columna como eje X.

    Retorna:
    - df_corregido: DataFrame con el fondo lineal corregido.
    """
    df_corregido = df.copy()
    #x = df.iloc[:, 0]

    for i in range(1, len(df.columns)):
        y = pd.to_numeric(df.iloc[:, i], errors='coerce')

        # Línea base entre los extremos
        y0 = y.iloc[0]
        y1 = y.iloc[-1]
        baseline = np.linspace(y0, y1, len(y))

        # Corregir
        corregido = y - baseline
        df_corregido.iloc[:, i] = corregido

    print("✅ Corrección lineal aplicada.")
    return df_corregido


def correccion_rolling_ball(df, radius=50):
    """
    Aplica corrección de línea base usando el método Rolling Ball (morphological filter).

    Parámetros:
    - df: DataFrame con la primera columna como eje X.
    - radius: Radio de la "pelota" (más grande = fondo más amplio, menos detalle).

    Retorna:
    - df_corregido: DataFrame con los espectros corregidos.
    """
    df_corregido = df.copy()

    for i in range(1, len(df.columns)):
        y = pd.to_numeric(df.iloc[:, i], errors='coerce').fillna(0)
        baseline, _ = morphological.rolling_ball(y, radius=radius)
        corregido = y - baseline
        df_corregido.iloc[:, i] = corregido

    print(f"✅ Corrección Rolling Ball aplicada (radio = {radius})")
    return df_corregido

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
        return df

        
    

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
        elif opt==0:
            print("""
                saliendo del programa...
                {}
                """.format("-" * 32))
            sys.exit()       
            
    
    

if __name__ == "__main__":
    main()