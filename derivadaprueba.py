#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 18:49:38 2024

@author: diego
"""



# main.py
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler # PARA LA NORMALIZACION POR LA MEDIA 
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter # Para suavizado de Savitzky Golay
from scipy.ndimage import gaussian_filter # PARA EL FILTRO GAUSSIANO


def archivo_existe(ruta_archivo):
    return os.path.isfile(ruta_archivo)

nombre = input("Por favor, ingresa tu nombre: ")
print(f"Hola, {nombre}!")

existe = False
archivo_nombre = input("Ingrese el nombre del archivo: ")
 
while existe == False:   
    if archivo_existe(archivo_nombre):
        bd_name = archivo_nombre #Este archivo contiene los datos espectroscópicos que serán leídos
        df = pd.read_csv(bd_name, delimiter = ',' , header=None)
        existe = True
    else:
        print("El archivo no existe.")
        archivo_nombre = input("Ingrese el nombre del archivo: ")
    
print(df)


   

df_derivada = df.copy() #creamos un nuevo dataframe 
df_derivada.columns = range(len(df_derivada.columns)) #nos aseguramos que los nombres sean unicos en la cabecera
#print("xxxxxxxxxxxxxxx")
#print(df_derivada)
df_derivada = df_derivada.drop(0) #eliminamos la cabecera
# print("wwwwwwwwwwwwwwwwwwwwww")
# print(df_derivada)
df_derivada = df_derivada.apply(pd.to_numeric, errors='coerce') #convertimos a numericos
# print("rrrrrrrrrrrrrrrrrrrrrrrrrrrr")
# print(df_derivada)
df_derivada = df_derivada.dropna() #eliminamos las filas con NaN
# print("ppppppppppppppppppppppppppppppp")
# print(df_derivada)
df_derivada_diff = df_derivada.diff() #calculo de la derivada
df_derivada_diff.columns = df.columns # volvemos a colocar los nombres originales de la columnas

# Mostrar el resultado
print("DataFrame con la primera derivada:")
print(df_derivada_diff)

   

