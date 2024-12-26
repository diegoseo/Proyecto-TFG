#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 08:04:08 2024

@author: diego
"""


# main.py
import os
# import numpy as np
import pandas as pd
import csv # PARA ENCONTRAR EL TIPO DE DELIMITADOR DEL ARCHIVO .CSV
import re # PARA LA EXPRECION REGULAR DE LOS SUFIJOS
import time
#import matplotlib
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# from sklearn.preprocessing import StandardScaler # PARA LA NORMALIZACION POR LA MEDIA 
# from sklearn.decomposition import PCA
# from scipy.signal import savgol_filter # Para suavizado de Savitzky Golay
# from scipy.ndimage import gaussian_filter # PARA EL FILTRO GAUSSIANO
# from sklearn.cluster import AgglomerativeClustering
# from scipy.cluster.hierarchy import linkage, cut_tree, dendrogram

def archivo_existe(ruta_archivo):
    print("Buscando archivo.")
    time.sleep(3)
    print("Buscando archivo..")
    time.sleep(2)
    print("Buscando archivo...")
    time.sleep(1)
    return os.path.isfile(ruta_archivo)

nombre = input("Por favor, ingresa tu nombre: ")
print(f"Hola, {nombre}!")

existe = False
archivo_nombre = input("Ingrese el nombre del archivo: ")

 

# Función para detectar el delimitador automáticamente por que los archivos pueden estar ceparados por , o ; etc
def identificar_delimitador(archivo):
    with open(archivo, 'r') as file:
        muestra_csv = file.read(4096)  # Lee una muestra de 4096 bytes
        #print("LA MUESTRA DEL CSV ES:")
        #print(muestra_csv)
        caracter = csv.Sniffer()
        delimitador = caracter.sniff(muestra_csv).delimiter
    return delimitador




def detectar_labels(df): #Detecta si los labels están en la fila o en la columna para ver si hacemos la transpuesta  o no

    # Verificar la primera fila (si contiene strings)
    if df.iloc[0].apply(lambda x: isinstance(x, str)).all():
        return "fila" #si los labels están en la primera fila
    
    # Verificar la primera columna (si contiene strings)
    elif df.iloc[:, 0].apply(lambda x: isinstance(x, str)).all():
        return "columna" #si los labels están en la primera columna
    
    # Si no hay etiquetas detectadas
    return "ninguno" #si no se detectan labels.



while existe == False:   
    if archivo_existe(archivo_nombre):  
        # print("Encontrado!.")
        # print("Analizando archivo.")
        # time.sleep(3)
        # print("Analizando archivo..")
        # time.sleep(2)
        # print("Analizando archivo...")
        time.sleep(1)
        bd_name = archivo_nombre #Este archivo contiene los datos espectroscópicos que serán leídos
        delimitador = identificar_delimitador(bd_name)
        print("EL DELIMITADOR ES: ", delimitador)
        df = pd.read_csv(bd_name, delimiter = delimitador , header=None)
        existe = True
        if detectar_labels(df) == "columna" :
            print("SE HIZO LA TRASPUESTA")
            df = df.T
        else:
            print("NO SE HIZO LA TRANSPUESTA")
    else:
        print("El archivo no existe.")
        archivo_nombre = input("Ingrese el nombre del archivo: ")

# print("DF ANTES DEL CORTE")
# print(df)
# print(df.shape)

# print("LOGRO LEER EL ARCHIVO")


def columna_con_menor_filas(df):
    
    # Calcular el número de valores no nulos en cada columna
    valores_no_nulos = df.notna().sum()
    
    # Encontrar la columna con la menor cantidad de valores no nulos
    columna_menor = valores_no_nulos.idxmin()
    cantidad_menor = valores_no_nulos.min()
    
    return columna_menor, cantidad_menor


col,fil = columna_con_menor_filas(df)
if len(df) == fil:
    print("EL DATASET TIENE LA MISMA CANTIDAD DE FILAS EN CADA COLUMNA")
else:
    print("LA COLUMNA CON MENOR CANTIDAD DE DATOS ES:")
    print("TIPO: ",df.iloc[0,col+1])
    print("FILA: ",fil)
    print("COLUMNA: ",col+1)    
    print("COMO DESEAS ARREGLAR EL DATAFRAME")
    print("1- ELIMINAR TODAS LAS FILAS HASTA IGUALAR A LA MENOR")
    print("2- ELIMINAR LA COLUMNA CON MENOR NUMERO DE FILAS")
    opcion= int(input("OPCION: "))
    
    if opcion == 1:
        print(df.shape)
        menor_cant_filas = df.dropna().shape[0] # Buscamos la columna con menor cantidad de intensidades
        # print("menor cantidad de filas:", menor_cant_filas)

        df_truncado = df.iloc[:menor_cant_filas] # Hacemos los cortes para igualar las columnas

        df = df_truncado
        print(df.shape)
    else:
        print(df.shape)
        df.drop(columns=[col], inplace=True)
        print(df.shape)
        



# renombramos la celda [0,0]

print("Cambiar a cero: ",df.iloc[0,0])

df.iloc[0,0] = float(0)

print("Cambiar a cero: ",df.iloc[0,0])

print(df)



# HACEMOS LA ELIMINACION DE LOS SUFIJOS EN CASO DE TENER


for col in df.columns:
    valor = re.sub(r'[_\.]\d+$', '', str(df.at[0, col]).strip())  # Eliminar sufijos con _ o .
    try:
        df.at[0, col] = float(valor)  # Convertir de nuevo a float si es posible
    except ValueError:
        df.at[0, col] = valor  # Mantener como string si no es convertible


print("Luego de eliminar los sufijos")
print(df)


