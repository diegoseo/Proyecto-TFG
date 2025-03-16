#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 11:55:37 2025

@author: rick
"""

import pandas as pd 

def lectura_archivo (ruta, separador, encabezado, n_filas):
        try:
            ruta = input('Ingrese ruta o nombre del archivo: ')
            df = df.read_csv(ruta)
            
            
    
def reconocer_separador(archivo):
    seperador = detectar_separador(arhivo)
    if separador:
        try:
            df = pd.read_csv(arhivo, sep = separador)
            print("archivo leido correctamente ✔")
            return df
        except Exception as e:
            print(f"❌ Error al leer csv: {e}")