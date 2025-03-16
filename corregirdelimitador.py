#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 20:47:57 2025

@author: diego
"""

import pandas as pd
import re

# 📌 Asegúrate de que el archivo tiene la extensión correcta
archivo = "analgesicos.csv"  # Si el archivo es .xlsx, cámbialo a "analgesicos.xlsx"

# 📌 Si es un CSV, usa pd.read_csv(). Si es un Excel, usa pd.read_excel().
try:
    if archivo.endswith(".csv"):
        df = pd.read_csv(archivo)  # Leer CSV
    elif archivo.endswith(".xlsx"):
        df = pd.read_excel(archivo)  # Leer Excel
    else:
        raise ValueError("❌ Formato de archivo no compatible.")
    
    # Función para limpiar nombres de las columnas
    def limpiar_nombres(nombre):
        return re.sub(r"\d+-?[A-Za-z]?", "", nombre)  # Elimina números y sufijos

    # Aplicar limpieza a los nombres de las columnas
    df.columns = [limpiar_nombres(col) for col in df.columns]

    # Guardar el archivo limpio en el formato original
    if archivo.endswith(".csv"):
        df.to_csv("datos_limpios.csv", index=False)  # Guardar como CSV
    else:
        df.to_excel("datos_limpios.xlsx", index=False)  # Guardar como Excel

    print(f"✅ Archivo limpio guardado como 'datos_limpios.{archivo.split('.')[-1]}'")

except Exception as e:
    print(f"❌ Error: {e}")
