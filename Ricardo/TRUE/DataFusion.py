#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 11:55:37 2025

@author: rick
"""

import pandas as pd 

def detectar_separador (archivo):
        try: 
            with open(archivo, r, newline= '', enconding = 'utf-8') as f :
                muestra = f.read(1024)
                sniffer = csv.Sniffer()
                if sniffer.has_header(muestra):
                    print("Archivo con cabecera")
                separador = sniffer.sniff(muestra).delimiter
                print(f"el separador es: '{separador}'")
                return separador 
        except Exception as e: 
            print(f"Error al analizar el archivo{e}")
            return None 
            
            
    
def lectura_archivo(archivo):
    seperador = detectar_separador(archivo)
    if separador:
        try:
            df = pd.read_csv(archivo, sep = separador)
            print("archivo leido correctamente ✔")
            return df
        except Exception as e:
            print(f"❌ Error al leer csv: {e}")
            return None
    return None 


def main():
    
    archivo_csv = input ("Ongrese la ruta o el nombre del archivo:")
    df = lectura_archivo(archivo_csv)
    df.head()
    
if __name__ == "__main__":
    main()
    
    
