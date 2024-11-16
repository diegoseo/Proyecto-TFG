#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 20:26:09 2024

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

'''
    PREPARAMOS EL SIGUIENTE MENU
'''


def main():
    while True:
        mostrar_menu()
        opcion = input("Selecciona una opción: ")
        
        if opcion == '1':
            metodo = 1
            print("Procesando los datos")
            print("Por favor espere un momento...")
            #mostrar_espectros(df2,metodo,0)
        # elif opcion == '2':
        #     metodo = 2
        #     print("Procesando los datos")
        #     print("Por favor espere un momento...")
        #     #mostrar_espectros(df_media_pca,metodo,0)
        # elif opcion == '3':
        #     metodo = 3
        #     print("Procesando los datos")
        #     print("Por favor espere un momento...")
        #     #mostrar_espectros(df_concatenado_cabecera_nueva_area,metodo,0)
        # elif opcion == '4':
        #     #suavizado_saviztky_golay(0,0)          
        # elif opcion == '5':
        #     #suavizado_filtroGausiano(0,0)
        #     # print("Procesando los datos")
        #     # print("Por favor espere un momento...")        
        # elif opcion == '6':
        #      suavizado_mediamovil(0,0)
        #      # print("Procesando los datos")
        #      # print("Por favor espere un momento...")     
        # elif opcion == '7':
        #      # print("Procesando los datos")
        #      # print("Por favor espere un momento...")
        #      mostrar_pca()       
        # elif opcion == '8':
        #      primera_derivada(0,0)
        # elif opcion == '9':
        #      segunda_derivada(0,0)
        # # elif opcion == '10':
        # #     #correcion_LineaB()
        # # elif opcion == '11':
        # #     #correcion_shirley()
        # # elif opcion == '12':
        # #     #espectro_escalado()
        # # elif opcion == '13':
        # #     #espectro_acotado()
        elif opcion == '14':
            print("Saliendo del programa...")
            break
        else:
            print("Opción no válida. Inténtalo de nuevo.")

def mostrar_menu():
      print("\n--- Menú Principal ---")
      print("1. MOSTRAR ESPECTROS")
      print("2. NORMALIZAR POR MEDIA")
      print("3. NORMALIZAR POR AREA")
      print("4. SUAVIZADO POR SAVIZTKY-GOLAY")
      print("5. SUAVIZADO POR FILTRO GAUSIANO")
      print("6. SUAVIZADO POR MEDIA MOVIL")
      print("7. PCA")
      print("8. PRIMERA DERIVADA")
      print("9. SEGUNDA DERIVADA")
      print("10. CORRECCION LINEA BASE")
      print("11. CORRECION SHIRLEY")
      print("12. ESPECTRO ESCALADO")
      print("13. ESPECTRO ACOTADO")
      print("14. Salir")
      
      

 
#GRAFICAMOS LOS ESPECTROS SIN NORMALIZAR#

raman_shift = df.iloc[1:, 0].reset_index(drop=True)  # EXTRAEMOS TODA LA PRIMERA COLUMNA, reset_index(drop=True) SIRVE PARA QUE EL INDICE COMIENCE EN 0 Y NO EN 1
print(raman_shift)

intensity = df.iloc[1:, 1:] # EXTRAEMOS TODAS DEMAS COLUMNAS EXCEPTO LA PRIMERA FILA Y PRIMERA COLUMNA
print(intensity)   

tipos = df.iloc[0, 1:] # EXTRAEMOS LA PRIMERA FILA MENOS DE LA PRIMERA COLUMNA
print(tipos)

cabecera = df.iloc[[0]].copy() # EXTRAEMOS LA PRIMERA FILA 
print(cabecera)

cant_tipos = tipos.nunique() # PARA EL EJEMPLO DE LIMPIO.CSV CANT_TIPOS TENDRA VALOR 4 YA QUE HAY 4 TIPOS (collagen,lipids,glycogen,DNA)
print(cant_tipos)

tipos_nombres = df.iloc[0, 1:].unique() # OBTENEMOS LOS NOMBRES DE LOS TIPOS
print(tipos_nombres)

# Obtenemos el colormap sin especificar el número de colores
cmap = plt.colormaps['hsv']  # Usamos solo el nombre del colormap

# Nos aseguramos de que `colores` es una lista
colores = [cmap(i) for i in np.linspace(0, 1, len(tipos_nombres))]  # Genera una lista de colores

# Crear el diccionario de asignación de colores
asignacion_colores = {tipo: mcolors.to_hex(colores[i]) for i, tipo in enumerate(tipos_nombres)}

diccionario=pd.DataFrame(asignacion_colores.items())
print(diccionario)

#AHORA QUE YA TENGO ASIGNADO UN COLOR POR CADA TIPO TENGO QUE GRAFICAR LOS ESPECTROS#


"""
VARIABLES DE MOSTRAR ESPECTROS
"""

















if __name__ == "__main__":
     main()


    
    
    
    
    
    
    
    
    
