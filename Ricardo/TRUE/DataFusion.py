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
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from numpy import trapz
from scipy.ndimage import gaussian_filter1d
import sys
import time


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
    
# =============================================================================
#     print(df['Ibuprofen'].head(10))
#     print(type(df['Ibuprofen'].iloc[0]))
# =============================================================================
    
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

    print("✅ Min-Max aplicado sin necesidad de renombrar columnas.")
    print(df_norm)
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
        
    return df
        
    

def menu():
    print("-" * 50) 
    #texto_desplazamiento("MENU", 10, 0.1)
    print("****MENU****")
    print("0. leer otro dataset")
    print("1. Mostrar espectros ")
    print("2. Normalizar Espectro")
    
    
    
    
    

## Función principal
def main():
    df = lectura_archivo()
    if df is not None:
        print("\n🔹 Primeras filas del archivo CSV:")
        print(df.head())
    while True:
        menu()
        opt = int(input("Ingrese opcion: "))    
        if opt == 0:
            df=lectura_archivo()
        if opt == 1:
            mostrar_espectros(df)
        if opt == 2:
            df = normalizar(df)
            #print(df)
        if opt >10:
            break
            
    
    

if __name__ == "__main__":
    main()