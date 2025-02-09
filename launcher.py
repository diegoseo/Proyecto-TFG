#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 22:10:07 2025

@author: diego
"""

import subprocess

def ejecutar_script(script):
    try:
        subprocess.run(["python", script], check=True)
    except Exception as e:
        print(f"Error al ejecutar {script}: {e}")

while True:
    archivo = input("Ingrese el nombre del archivo .py que desea ejecutar (o 'salir' para terminar): ").strip()
    
    if archivo.lower() == "salir":
        print("Saliendo del programa...")
        break

    if not archivo.endswith(".py"):
        print("Por favor, ingrese un archivo con extensión .py")
        continue

    ejecutar_script(archivo)
