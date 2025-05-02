# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 11:40:06 2025

@author: ricardo.leguizamon
"""

import pandas as pd
import os

def convertir_xlsx_a_csv(archivo_xlsx):
    # Obtener el nombre base del archivo (sin extensión)
    nombre_base = os.path.splitext(os.path.basename(archivo_xlsx))[0]
    archivo_csv = f"{nombre_base}.csv"
    
    # Leer el archivo Excel
    df = pd.read_excel(archivo_xlsx)

    # Guardar como CSV
    df.to_csv(archivo_csv, index=False)

    print(f"Archivo convertido exitosamente: {archivo_csv}")
    
    
def listar_archivos_xlsx(directorio):
    archivos = [f for f in os.listdir(directorio) if f.endswith('.xlsx')]
    return archivos

# Ejemplo de uso:
if __name__ == "__main__":
    carpeta = "."  # "." significa el directorio actual
    archivos_xlsx = listar_archivos_xlsx(carpeta)
    print("Archivos .xlsx encontrados:")
    for archivo in archivos_xlsx:
        print(archivo)
    archivo_entrada = input("archivo entrada: ")  # cambia aquí por tu archivo
    convertir_xlsx_a_csv(archivo_entrada)
    
