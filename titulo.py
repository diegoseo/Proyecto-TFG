#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 20:47:33 2025

@author: diego
"""

from manipulacion_archivos import archivo_nombre
import matplotlib.pyplot as plt

bd_name = archivo_nombre

def titulo_plot_mostrar(metodo,nor_op):
    # TODO ESTE CONDICIONAL ES SOLO PARA QUE EL TITULO DEL GRAFICO MUESTRE POR CUAL METODO SE NORMALIZO
    if metodo == 1:
        plt.xlabel('Longitud de onda / Frecuencia')
        plt.ylabel('Intensidad')
        plt.title(f'Espectros del archivo {bd_name}')
        plt.show()
    elif metodo == 2:
        plt.xlabel('Longitud de onda / Frecuencia')
        plt.ylabel('Intensidad')
        plt.title(f'Espectros del archivo {bd_name} Normalizado por la Media')
        plt.show()
    elif metodo == 3:
        plt.xlabel('Longitud de onda / Frecuencia')
        plt.ylabel('Intensidad')
        plt.title(f'Espectros del archivo {bd_name} Normalizado por Area')
        plt.show()
    elif metodo == 4:
        if nor_op == 1:
               plt.xlabel('Longitud de onda / Frecuencia')
               plt.ylabel('Intensidad')
               plt.title(f'Espectros del archivo {bd_name} Suavizado por Saviztky_golay y Normalizado por la media')
               plt.show()   
        elif nor_op == 2:
               plt.xlabel('Longitud de onda / Frecuencia')
               plt.ylabel('Intensidad')
               plt.title(f'Espectros del archivo {bd_name} Suavizado por Saviztky_golay y Normalizado Area')
               plt.show() 
        else:
               plt.xlabel('Longitud de onda / Frecuencia')
               plt.ylabel('Intensidad')
               plt.title(f'Espectros del archivo {bd_name} Suavizado por Saviztky_golay y sin Normalizar ')
               plt.show()  
    elif metodo == 5:
        if nor_op == 1:
               plt.xlabel('Longitud de onda / Frecuencia')
               plt.ylabel('Intensidad')
               plt.title(f'Espectros del archivo {bd_name} Suavizado por Filtro Gausiano y Normalizado por la media')
               plt.show()   
        elif nor_op == 2:
               plt.xlabel('Longitud de onda / Frecuencia')
               plt.ylabel('Intensidad')
               plt.title(f'Espectros del archivo {bd_name} Suavizado por Filtro Gausiano y Normalizado Area')
               plt.show() 
        else:
               plt.xlabel('Longitud de onda / Frecuencia')
               plt.ylabel('Intensidad')
               plt.title(f'Espectros del archivo {bd_name} Suavizado por Filtro Gausiano y sin Normalizar ')
               plt.show()  
    elif metodo == 6:
        if nor_op == 1:
               plt.xlabel('Longitud de onda / Frecuencia')
               plt.ylabel('Intensidad')
               plt.title(f'Espectros del archivo {bd_name} Suavizado por Media Movil y Normalizado por la media')
               plt.show()   
        elif nor_op == 2:
               plt.xlabel('Longitud de onda / Frecuencia')
               plt.ylabel('Intensidad')
               plt.title(f'Espectros del archivo {bd_name} Suavizado por Media Movil y Normalizado Area')
               plt.show() 
        else:
               plt.xlabel('Longitud de onda / Frecuencia')
               plt.ylabel('Intensidad')
               plt.title(f'Espectros del archivo {bd_name} Suavizado por Media Movil y sin Normalizar ')
               plt.show()  
                                