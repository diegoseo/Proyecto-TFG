#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 11:55:37 2025

@author: rick
"""

import pandas as pd
import csv
import re  # Para manejo de expresiones regulares


## Función para detectar qué tipo de delimitador tiene el CSV
def detectar_separador(archivo):
    try:
        with open(archivo, 'r', newline='', encoding='utf-8') as f:
            muestra = f.read(1024)  # Leer solo una parte del archivo para detección rápida
            sniffer = csv.Sniffer()
            if sniffer.has_header(muestra):
                print("✔ Se detectó que el archivo tiene encabezado.")
            separador = sniffer.sniff(muestra).delimiter
            print(f"🔍 Separador detectado: '{separador}'")
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
def lectura_archivo(archivo):
    separador = detectar_separador(archivo)
    if separador:
        try:
            df = pd.read_csv(archivo, sep=separador)
            print("✔ Archivo leído correctamente.")

            # Limpieza de nombres de columnas
            df.columns = limpiar_encabezados(df.columns)
            print("🛠 Encabezados limpios:", df.columns.tolist())

            return df
        except Exception as e:
            print(f"❌ Error al leer CSV: {e}")
            return None
    return None


## Función principal
def main():
    archivo_csv = input("Ingrese la ruta o el nombre del archivo: ")
    df = lectura_archivo(archivo_csv)
    
    if df is not None:
        print("\n🔹 Primeras filas del archivo CSV:")
        print(df.head())  # Muestra las primeras filas para verificar

if __name__ == "__main__":
    main()