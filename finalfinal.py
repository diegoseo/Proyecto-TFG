#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 08:22:48 2024

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
