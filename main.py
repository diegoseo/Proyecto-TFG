#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 14:26:25 2025

@author: diego
"""
# ACA SERA LA LLAMADA A LA FUNCIONES CON SUS MENUS Y TITULO PLOT

import manipulacion_archivos
import sys  # Para usar sys.exit()
#from funciones import * # NO SE RECOMIENDA IMPORTAR TODO DE UNA CON EL * 
from funciones import mostrar_espectros, datos_sin_normalizar, mostrar_leyendas # SE LLAMA DE A UNO A LAS FUNCIONES PARA NO TENER QUE HACER CADA RATO funcion.raman_shift
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

def menu_principal():
    
    # PROXIMA MEJOR SERIA ELIMINAR LA OPCION 1 Y QUE SOLO QUEDE LA OPCION 2 Y QUE EN CANTIDAD DE ARCHIVOS SE PONGA 1 Y SI ES MAYOR A 1 QUE HAGA EL DATA FUSION, SERIA PARA OPTIMIZAR MAS EL CODIGO
    
    while True:
        print("\n--- Menú Principal ---")
        print("1- Cargar Archivo") # ESTA OPCION SERA UTIL PARA EL CASO QUE QUERAMOS LEER SOLO UN ARCHIVO Y HACERLE LOS ANALISIS NECESARIOS
        print("2- Data Fusion") # SI EL USUARIO QUIERE HACER UN DATA FUSION DEBERA DECIR POR CUAL METODO Y CUANTOS ARCHIVOS DESEA FUSIONAR
        print("3- Salir")
        
        opcion = int(input("Ingrese  una opcion: "))
        
        if opcion == 1:
            manipulacion_archivos.pila_df = [] # VACIAMOS LA PILA POR QUE PUEDE ESTAR CARGADO SI PASA DE LA OPCION 2 AL 1 , AUN QUE PODRIA SERVIR ELIMINAR ESTA LINEA PARA QUE PODAMOS SEGIR AGREGANDO ARCHIVOS DE A UNO EN LA PILA POR SI HALLAMOS ELEGIDO MAL LA CANTIDAD TOTAL DE ARCHIVOS A LEER
            #manipulacion_archivos.cargar_archivo(1) #CARGARA UN UNICO ARCHIVO Y FUNCIONARA COMO EL CODIGO DE ENERO2025 NORMALMENTE
            # print("Ingrese el Metodo de reduccion de dimensionalidad")
            # print("1- PCA") #POR EL MOMENTO SOLO ESTE ESTARA IMPLEMENTADO, LAS DEMAS OPCIONES LA USAREMOS UNA VEZ QUE DEMOSTREMOS QUE EL DATAFUSION SI FUNCIONE
            # print("2- T-SNE")
            # print("3- OTROS")
            # m_dim = int(input("Opcion: "))
            print("DF DENTRO DE MAIN.PY")
            print(manipulacion_archivos.pila_df)
            df = manipulacion_archivos.cargar_archivo(1) #CARGARA UN UNICO ARCHIVO Y FUNCIONARA COMO EL CODIGO DE ENERO2025 NORMALMENTE
            variables_utiles(df)
            main(df)
        elif opcion == 2:
           cant_archivos = int(input("Ingrese la cantidad de archivos que desea fusionar: "))
           # print("Ingrese el Metodo de reduccion de dimensionalidad")
           # print("1- PCA") #POR EL MOMENTO SOLO ESTE ESTARA IMPLEMENTADO, LAS DEMAS OPCIONES LA USAREMOS UNA VEZ QUE DEMOSTREMOS QUE EL DATAFUSION SI FUNCIONE
           # print("2- T-SNE")
           # print("3- OTROS")
           # m_dim = int(input("Opcion: "))
           print("Ingrese el Metodo de datafusion: ")
           print("1- Low Level (Concatenacion)")
           print("2- Mid level (Caracterizacion)")
           print("3- Otros (Por si se necesite mas metodos)")
           m_df = int(input("Opcion: "))
           
           manipulacion_archivos.cargar_archivo(cant_archivos) # ESTA FUNCION LEERA DE A UNO LOS ARCHIVOS Y PRIMERAMENTE POR CADA ARCHIVO LEIDO PREGUNTARA SI ELIMINAR FILA O COL PARA IGUALAR ESO PARA CADA ARCHIVO ANTES DE HACER EL DATAFUSION, AL HACER EL DATAFUSION PUEDE EXISTIR UNA DIFERENCIA DE DIMENCIONES NUEVAMENTE.
           print("DF DENTRO DE MAIN.PY")
           print(manipulacion_archivos.pila_df)
           
           # ACA HAY QUE FUSIONAR PRIMERO ANTES DE LLAMAR A main()
           
        elif opcion == 3:
             print("Saliendo del programa...")
             sys.exit()  # Termina completamente el programa
            


def mostrar_menu():
     print("1. MOSTRAR ESPECTROS")
     print("2. NORMALIZAR POR MEDIA")
     print("3. NORMALIZAR POR AREA")
     print("4. SUAVIZADO POR SAVIZTKY-GOLAY")
     print("5. SUAVIZADO POR FILTRO GAUSIANO")
     print("6. SUAVIZADO POR MEDIA MOVIL")
     print("7. PRIMERA DERIVADA")
     print("8. SEGUNDA DERIVADA")
     print("9. CORRECCION BASE LINEAL")
     print("10. CORRECION SHIRLEY")
     print("11. REDUCIR DIMENSIONALIDAD") #print("1- PCA") print("2- T-SNE")print("3- OTROS") IMPLEMENTAR SU MENU
     print("12. GRAFICO HCA")
     print("13. CAMBIAR ARCHIVO")
     print("14. Salir")
      
def sub_menu():
    print("Como deseas ver el espectro")
    print("1- Grafico completo")
    print("2- Grafico acotado")
    print("3- Grafico por tipo") 
    print("4- Grafico acotado por tipo") 
    print("5- Descargar .csv ") 
    print("6- Descargar .csv acotado") 
    print("7- Descargar .csv por tipo") 
    print("8- Descargar .csv acotado por tipo")
    print("9- Volver") 
   
 
def variables_utiles(df):  
    global raman_shift,tipos,types,cant_tipos,tipos_nombres,cmap,colores,asignacion_colores,diccionario
    
    raman_shift = df.iloc[1:, 0].reset_index(drop=True)  # EXTRAEMOS TODA LA PRIMERA COLUMNA, reset_index(drop=True) SIRVE PARA QUE EL INDICE COMIENCE EN 0 Y NO EN 1
    print(raman_shift)
    
    tipos = df.iloc[0, 1:] # EXTRAEMOS LA PRIMERA FILA MENOS DE LA PRIMERA COLUMNA
    #print(tipos)
    types=tipos.tolist()
    
    cant_tipos = tipos.nunique() # PARA EL EJEMPLO DE LIMPIO.CSV CANT_TIPOS TENDRA VALOR 4 YA QUE HAY 4 TIPOS (collagen,lipids,glycogen,DNA)
    #print(cant_tipos)
    tipos_nombres = df.iloc[0, 1:].unique() # OBTENEMOS LOS NOMBRES DE LOS TIPOS
    #print(tipos_nombres)
    
    # Seleccionar un colormap distintivo
    cmap = plt.cm.Spectral  # Puedes probar con "hsv", "Set3", "Spectral", etc.
    
    #cmap = plt.colormaps['hsv']  # Usamos solo el nombre del colormap, Obtenemos el colormap sin especificar el número de colores
    
    # Nos aseguramos de que `colores` es una lista
    colores = [cmap(i) for i in np.linspace(0, 1, len(tipos_nombres))]  # Genera una lista de colores
    
    # Crear el diccionario de asignación de colores
    asignacion_colores = {tipo: mcolors.to_hex(colores[i]) for i, tipo in enumerate(tipos_nombres)}
    #print(asignacion_colores)
    
    diccionario=pd.DataFrame(asignacion_colores.items())
     #print(diccionario)

    
    

'''
    PREPARAMOS EL SIGUIENTE MENU
'''


def main(df):
    
    #df = manipulacion_archivos.pila_df[0] # HACE AL PRIMER ELEMENTO DE LA LISTA, SE ASUME QUE SOLO DEBERIA DE HABER UN SOLO ELEMENTO LUEGO DENTRO DEL DF YA QUE YA se funsionaron
    print("DF DENTRO DEL MAIN")
    print(df)
    #raman_shift = df.iloc[1:, 0].reset_index(drop=True)  # EXTRAEMOS TODA LA PRIMERA COLUMNA, reset_index(drop=True) SIRVE PARA QUE EL INDICE COMIENCE EN 0 Y NO EN 1
    mostrar_leyendas(df,diccionario,cant_tipos)
    while True:
        mostrar_menu()
        opcion = input("Selecciona una opción: ")
        #print("volvio a salir")
        if opcion == '1':
            print("entro 1")
            metodo = 1
            sub_menu()
            metodo_grafico = int(input("Opcion: "))
            if metodo_grafico == 1:
                 print("Procesando los datos")
                 print("Por favor espere un momento...")
                 mostrar_espectros(datos_sin_normalizar(df),raman_shift,asignacion_colores,metodo,0,0,0)
            # elif metodo_grafico == 2:
            #     espectro_acotado(datos_sin_normalizar(),0, 0,1,0,0)
            # elif metodo_grafico == 3:
            #     grafico_tipo(datos_sin_normalizar(),raman_shift,0,metodo,0,0)
            # elif metodo_grafico == 4:
            #     grafico_acotado_tipo(datos_sin_normalizar(),raman_shift,metodo,0,0,0)
            # elif metodo_grafico == 5: 
            #     descargar_csv(1,datos_sin_normalizar(),raman_shift) # 1 PARA SABER QUE VIENE SIN NORMALIZAR  
            # elif metodo_grafico == 6:
            #     descargar_csv_acotado(datos_sin_normalizar(),1,raman_shift) # 1 PARA SABER QUE VIENE DE ACA Y PODER ELEGIR EL NOMBRE DE LA CARPETA DE SALIDA
            # elif metodo_grafico == 7:
            #     descargar_csv_tipo(datos_sin_normalizar(),1,raman_shift) # 1 PARA SABER QUE VIENE SIN NORMALIZAR  
            # elif metodo_grafico == 8:
            #     descargar_csv_acotado_tipo(datos_sin_normalizar(),1,raman_shift) # 1 PARA SABER QUE VIENE SIN NORMALIZAR  
            # elif metodo_grafico == 9:
            #     main()
                
                
        elif opcion == '2':
            print("entro 2")
            # metodo = 2
            # sub_menu()
            # metodo_grafico = int(input("Opcion: "))
            # if metodo_grafico == 1:
            #     print("Procesando los datos")
            #     print("Por favor espere un momento...")
            #     mostrar_espectros(normalizado_media(),raman_shift,metodo,0,0,0)
            # elif metodo_grafico == 2:
            #     espectro_acotado(normalizado_media(),0, 0,2,0,0)
            # elif metodo_grafico == 3:
            #     grafico_tipo(normalizado_media(),raman_shift,0,metodo,0,0)
            # elif metodo_grafico == 4:
            #     grafico_acotado_tipo(normalizado_media(),raman_shift,metodo,0,0,0)
            # elif metodo_grafico == 5:
            #     descargar_csv(2,normalizado_media(),raman_shift) # 2 PARA SABER QUE VIENE DE LA MEDIA  
            # elif metodo_grafico == 6:
            #     descargar_csv_acotado(normalizado_media(),2,raman_shift) # 2 PARA SABER QUE VIENE DE ACA Y PODER ELEGIR EL NOMBRE DE LA CARPETA DE SALIDA
            # elif metodo_grafico == 7:
            #     descargar_csv_tipo(normalizado_media(),2,raman_shift) # 2 PARA SABER QUE VIENE  NORMALIZAR MEDIA
            # elif metodo_grafico == 8:
            #     descargar_csv_acotado_tipo(normalizado_media(),2,raman_shift) # 2 PARA SABER QUE VIENE SIN NORMALIZAR 
            # elif metodo_grafico == 9:
            #     main()
                
                
        elif opcion == '3': 
             print("entro 3")
             # metodo = 3
             # sub_menu()
             # metodo_grafico = int(input("Opcion: "))
                 
             # if metodo_grafico == 1:
             #     print("Procesando los datos")
             #     print("Por favor espere un momento...")
             #     mostrar_espectros(normalizado_area(),raman_shift,metodo,0,0,0)
             # elif metodo_grafico == 2:
             #     espectro_acotado(normalizado_area(), 0,0,3,0,0)
             # elif metodo_grafico == 3:
             #     grafico_tipo(normalizado_area(),raman_shift,0,metodo,0,0)
             # elif metodo_grafico == 4:
             #     grafico_acotado_tipo(normalizado_area(),raman_shift,metodo,0,0,0)
             # elif metodo_grafico == 5:
             #     descargar_csv(3,normalizado_area(),raman_shift) # 3 PARA SABER QUE VIENE DEL AREA   
             # elif metodo_grafico == 6:
             #    descargar_csv_acotado(normalizado_area(),3,raman_shift) # 3 PARA SABER QUE VIENE DE ACA Y PODER ELEGIR EL NOMBRE DE LA CARPETA DE SALIDA
             # elif metodo_grafico == 7:
             #     descargar_csv_tipo(normalizado_area(),3,raman_shift) # 3 PARA SABER QUE VIENE  NORMALIZAR AREA
             # elif metodo_grafico == 8:
             #     descargar_csv_acotado_tipo(normalizado_area(),3,raman_shift) # 3 PARA SABER QUE VIENE SIN NORMALIZAR 
             # elif metodo_grafico == 9:
             #     main()
         
        elif opcion == '4':  
            print("entro 4")
            # while True:  # Bucle para mantener al usuario en el submenú
            #   metodo = 4
              
            #   sub_menu()
            #   metodo_grafico = int(input("Opcion: "))               
            #   if metodo_grafico == 1:
            #       dato,nor_op = suavizado_menu()
            #       if dato is None:  # Manejar la opción "Volver"
            #           continue # EVITA QUE CONTINUE LA EJECUCION DE LAS LINEAS RESTANTES Y SALTA AL SIGUIENTE CICLO
            #       dato_suavizado = suavizado_saviztky_golay(dato)
            #       mostrar_espectros(dato_suavizado,raman_shift,metodo,nor_op,0,0)
            #   elif metodo_grafico == 2:
            #       dato,nor_op = suavizado_menu()
            #       if dato is None:  # Manejar la opción "Volver"
            #           continue 
            #       dato_suavizado = suavizado_saviztky_golay(dato)
            #       if nor_op == 1:
            #           metodo = 4
            #       elif nor_op == 2:
            #           metodo = 5
            #       elif nor_op == 3:
            #           metodo = 6
            #       espectro_acotado(dato_suavizado,0,0,metodo,0,0)
            #   elif metodo_grafico == 3:
            #       dato,nor_op = suavizado_menu()
            #       if dato is None:  # Manejar la opción "Volver"
            #           continue 
            #       dato_suavizado = suavizado_saviztky_golay(dato)
            #       grafico_tipo(dato_suavizado,raman_shift,nor_op,metodo,0)
            #   elif metodo_grafico == 4:
            #       dato,nor_op = suavizado_menu()
            #       if dato is None:  # Manejar la opción "Volver"
            #           continue 
            #       dato_suavizado = suavizado_saviztky_golay(dato)
            #       grafico_acotado_tipo(dato_suavizado,raman_shift,metodo,nor_op,0,0)
            #   elif metodo_grafico == 5:
            #       dato,nor_op = suavizado_menu()
            #       if dato is None:  # Manejar la opción "Volver"
            #           continue 
            #       dato_suavizado = suavizado_saviztky_golay(dato)
            #       descargar_csv(4, dato_suavizado,raman_shift) # 4 PARA SABER QUE VIENE DEL SUAVIZADO POR suavizado_saviztky_golay
            #   elif metodo_grafico == 6:
            #       dato,nor_op = suavizado_menu()
            #       if dato is None:  # Manejar la opción "Volver"
            #           continue 
            #       dato_suavizado = suavizado_saviztky_golay(dato)
            #       descargar_csv_acotado(dato_suavizado,4,raman_shift) # 4 PARA SABER QUE VIENE DE ACA Y PODER ELEGIR EL NOMBRE DE LA CARPETA DE SALIDA
            #   elif metodo_grafico == 7:
            #       dato,nor_op = suavizado_menu()
            #       if dato is None:  # Manejar la opción "Volver"
            #           continue 
            #       dato_suavizado = suavizado_saviztky_golay(dato)
            #       descargar_csv_tipo(dato_suavizado,4,raman_shift) # 4 PARA SABER QUE VIENE DE ACA Y PODER ELEGIR EL NOMBRE DE LA CARPETA DE SALIDA
            #   elif metodo_grafico == 8:
            #       dato,nor_op = suavizado_menu()
            #       if dato is None:  # Manejar la opción "Volver"
            #           continue 
            #       dato_suavizado = suavizado_saviztky_golay(dato)
            #       descargar_csv_acotado_tipo(dato_suavizado,4,raman_shift) # 4 PARA SABER QUE VIENE DE ACA Y PODER ELEGIR EL NOMBRE DE LA CARPETA DE SALIDA
            #   elif metodo_grafico == 9:
            #       main()
        elif opcion == '5':  
              print("entro 5")
        #       while True:  # Bucle para mantener al usuario en el submenú
        #         metodo = 5           
        #         sub_menu()
        #         metodo_grafico = int(input("Opcion: "))               
        #         if metodo_grafico == 1:
        #             dato,nor_op = suavizado_menu()
        #             if dato is None:  # Manejar la opción "Volver"
        #                 continue # EVITA QUE CONTINUE LA EJECUCION DE LAS LINEAS RESTANTES Y SALTA AL SIGUIENTE CICLO
        #             dato_suavizado = suavizado_filtroGausiano(dato)
        #             mostrar_espectros(dato_suavizado,raman_shift,metodo,nor_op,0,0)
        #         elif metodo_grafico == 2:
        #             dato,nor_op = suavizado_menu()
        #             if dato is None:  # Manejar la opción "Volver"
        #                 continue 
        #             dato_suavizado = suavizado_filtroGausiano(dato)
        #             if nor_op == 1:
        #                 metodo = 7
        #             elif nor_op == 2:
        #                 metodo = 8
        #             elif nor_op == 3:
        #                 metodo = 9
        #             espectro_acotado(dato_suavizado,0,0,metodo,0,0)
        #         elif metodo_grafico == 3:
        #             dato,nor_op = suavizado_menu()
        #             if dato is None:  # Manejar la opción "Volver"
        #                 continue 
        #             dato_suavizado = suavizado_filtroGausiano(dato)
        #             grafico_tipo(dato_suavizado,raman_shift,nor_op,metodo,0,0)
        #         elif metodo_grafico == 4:
        #             dato,nor_op = suavizado_menu()
        #             if dato is None:  # Manejar la opción "Volver"
        #                 continue 
        #             dato_suavizado = suavizado_filtroGausiano(dato)
        #             grafico_acotado_tipo(dato_suavizado,raman_shift,metodo,nor_op,0,0)
        #         elif metodo_grafico == 5:
        #             dato,nor_op = suavizado_menu()
        #             if dato is None:  # Manejar la opción "Volver"
        #                 continue 
        #             dato_suavizado = suavizado_filtroGausiano(dato)
        #             descargar_csv(5, dato_suavizado,raman_shift) # 5 PARA SABER QUE VIENE DEL SUAVIZADO POR Filtro Gaussiano
        #         elif metodo_grafico == 6:
        #             dato,nor_op = suavizado_menu()
        #             if dato is None:  # Manejar la opción "Volver"
        #                 continue 
        #             dato_suavizado = suavizado_filtroGausiano(dato)
        #             descargar_csv_acotado(dato_suavizado,5,raman_shift) # 5 PARA SABER QUE VIENE DE ACA Y PODER ELEGIR EL NOMBRE DE LA CARPETA DE SALIDA
        #         elif metodo_grafico == 7:
        #             dato,nor_op = suavizado_menu()
        #             if dato is None:  # Manejar la opción "Volver"
        #                 continue 
        #             dato_suavizado = suavizado_filtroGausiano(dato)
        #             descargar_csv_tipo(dato_suavizado,5,raman_shift) # 5 PARA SABER QUE VIENE DE ACA Y PODER ELEGIR EL NOMBRE DE LA CARPETA DE SALIDA
        #         elif metodo_grafico == 8:
        #             dato,nor_op = suavizado_menu()
        #             if dato is None:  # Manejar la opción "Volver"
        #                 continue 
        #             dato_suavizado = suavizado_filtroGausiano(dato)
        #             descargar_csv_acotado_tipo(dato_suavizado,5,raman_shift) # 5 PARA SABER QUE VIENE DE ACA Y PODER ELEGIR EL NOMBRE DE LA CARPETA DE SALIDA
        #         elif metodo_grafico == 9:
        #             main()    
        elif opcion == '6':  
                 print("entro 6")
        #         while True:  # Bucle para mantener al usuario en el submenú
        #           metodo = 6          
        #           sub_menu()
        #           metodo_grafico = int(input("Opcion: "))               
        #           if metodo_grafico == 1:
        #               dato,nor_op = suavizado_menu()
        #               if dato is None:  # Manejar la opción "Volver"
        #                   continue # EVITA QUE CONTINUE LA EJECUCION DE LAS LINEAS RESTANTES Y SALTA AL SIGUIENTE CICLO
        #               dato_suavizado = suavizado_mediamovil(dato)
        #               mostrar_espectros(dato_suavizado,raman_shift,metodo,nor_op,0,0)
        #           elif metodo_grafico == 2:
        #               dato,nor_op = suavizado_menu()
        #               if dato is None:  # Manejar la opción "Volver"
        #                   continue 
        #               dato_suavizado = suavizado_mediamovil(dato)
        #               if nor_op == 1:
        #                   metodo = 10
        #               elif nor_op == 2:
        #                   metodo = 11
        #               elif nor_op == 3:
        #                   metodo = 12
        #               espectro_acotado(dato_suavizado,0,0,metodo,0,0)
        #           elif metodo_grafico == 3:
        #               dato,nor_op = suavizado_menu()
        #               if dato is None:  # Manejar la opción "Volver"
        #                   continue 
        #               dato_suavizado = suavizado_mediamovil(dato)
        #               grafico_tipo(dato_suavizado,raman_shift,nor_op,metodo,0,0)
        #           elif metodo_grafico == 4:
        #               dato,nor_op = suavizado_menu()
        #               if dato is None:  # Manejar la opción "Volver"
        #                   continue 
        #               dato_suavizado = suavizado_mediamovil(dato)
        #               grafico_acotado_tipo(dato_suavizado,raman_shift,metodo,nor_op,0,0)
        #           elif metodo_grafico == 5:
        #               dato,nor_op = suavizado_menu()
        #               if dato is None:  # Manejar la opción "Volver"
        #                   continue 
        #               dato_suavizado = suavizado_mediamovil(dato)
        #               descargar_csv(6, dato_suavizado,raman_shift) # 6 PARA SABER QUE VIENE DEL SUAVIZADO POR media movil
        #           elif metodo_grafico == 6:
        #               dato,nor_op = suavizado_menu()
        #               if dato is None:  # Manejar la opción "Volver"
        #                   continue 
        #               dato_suavizado = suavizado_mediamovil(dato)
        #               descargar_csv_acotado(dato_suavizado,6,raman_shift) # 6 PARA SABER QUE VIENE DE ACA Y PODER ELEGIR EL NOMBRE DE LA CARPETA DE SALIDA
        #           elif metodo_grafico == 7:
        #               dato,nor_op = suavizado_menu()
        #               if dato is None:  # Manejar la opción "Volver"
        #                   continue 
        #               dato_suavizado = suavizado_mediamovil(dato)
        #               descargar_csv_tipo(dato_suavizado,6,raman_shift) # 6 PARA SABER QUE VIENE DE ACA Y PODER ELEGIR EL NOMBRE DE LA CARPETA DE SALIDA
        #           elif metodo_grafico == 8:
        #               dato,nor_op = suavizado_menu()
        #               if dato is None:  # Manejar la opción "Volver"
        #                   continue 
        #               dato_suavizado = suavizado_mediamovil(dato)
        #               descargar_csv_acotado_tipo(dato_suavizado,6,raman_shift) # 6 PARA SABER QUE VIENE DE ACA Y PODER ELEGIR EL NOMBRE DE LA CARPETA DE SALIDA
        #           elif metodo_grafico == 9:
        #               main()                   
        elif opcion == '7':           
                    print("entro 7")
                    # while True:  # Bucle para mantener al usuario en el submenú
                    #   metodo = 7          
                    #   sub_menu()
                    #   metodo_grafico = int(input("Opcion: "))               
                    #   if metodo_grafico == 1:
                    #       dato,nor_op ,m_suavi = suavizado_menu_derivadas()
                    #       if dato is None:  # Manejar la opción "Volver"
                    #           continue # EVITA QUE CONTINUE LA EJECUCION DE LAS LINEAS RESTANTES Y SALTA AL SIGUIENTE CICLO
                    #       dato_suavizado = primera_derivada(dato)
                    #       mostrar_espectros(dato_suavizado,raman_shift,metodo,nor_op,m_suavi,1)
                    #   elif metodo_grafico == 2:
                    #       dato,nor_op,m_suavi= suavizado_menu_derivadas()
                    #       if dato is None:  # Manejar la opción "Volver"
                    #           continue 
                    #       dato_suavizado = primera_derivada(dato)
                    #       espectro_acotado(dato_suavizado,0,0,nor_op,m_suavi,1)
                    #   elif metodo_grafico == 3:
                    #       dato,nor_op,m_suavi = suavizado_menu_derivadas()
                    #       if dato is None:  # Manejar la opción "Volver"
                    #           continue 
                    #       dato_suavizado = primera_derivada(dato)
                    #       grafico_tipo(dato_suavizado,raman_shift,nor_op,metodo,m_suavi,1)
                    #   elif metodo_grafico == 4:
                    #       dato,nor_op,m_suavi = suavizado_menu_derivadas()
                    #       if dato is None:  # Manejar la opción "Volver"
                    #           continue 
                    #       dato_suavizado = primera_derivada(dato)
                    #       grafico_acotado_tipo(dato_suavizado,raman_shift,metodo,nor_op,m_suavi,1)
                    #   elif metodo_grafico == 5:
                    #       dato,nor_op,m_suavi = suavizado_menu_derivadas()
                    #       if dato is None:  # Manejar la opción "Volver"
                    #           continue 
                    #       dato_suavizado = primera_derivada(dato)
                    #       descargar_csv(7, dato_suavizado,raman_shift) # 7 PARA SABER QUE VIENE DE LA PRIMERA DERIVADA
                    #   elif metodo_grafico == 6:
                    #       dato,nor_op,m_suavi = suavizado_menu_derivadas()
                    #       if dato is None:  # Manejar la opción "Volver"
                    #           continue 
                    #       dato_suavizado = primera_derivada(dato)
                    #       descargar_csv_acotado(dato_suavizado,7,raman_shift) # 7 PARA SABER QUE VIENE DE ACA Y PODER ELEGIR EL NOMBRE DE LA CARPETA DE SALIDA
                    #   elif metodo_grafico == 7:
                    #       dato,nor_op,m_suavi = suavizado_menu_derivadas()
                    #       if dato is None:  # Manejar la opción "Volver"
                    #           continue 
                    #       dato_suavizado = primera_derivada(dato)
                    #       descargar_csv_tipo(dato_suavizado,7,raman_shift) # 7 PARA SABER QUE VIENE DE ACA Y PODER ELEGIR EL NOMBRE DE LA CARPETA DE SALIDA
                    #   elif metodo_grafico == 8:
                    #       dato,nor_op,m_suavi = suavizado_menu_derivadas()
                    #       if dato is None:  # Manejar la opción "Volver"
                    #           continue 
                    #       dato_suavizado = primera_derivada(dato)
                    #       descargar_csv_acotado_tipo(dato_suavizado,7,raman_shift) # 7 PARA SABER QUE VIENE DE ACA Y PODER ELEGIR EL NOMBRE DE LA CARPETA DE SALIDA
                    #   elif metodo_grafico == 9:
                    #       main()                     
        elif opcion == '8':           
                    print("entro 8")
                    # while True:  # Bucle para mantener al usuario en el submenú
                    #   metodo = 8          
                    #   sub_menu()
                    #   metodo_grafico = int(input("Opcion: "))               
                    #   if metodo_grafico == 1:
                    #       dato,nor_op ,m_suavi = suavizado_menu_derivadas()
                    #       if dato is None:  # Manejar la opción "Volver"
                    #           continue # EVITA QUE CONTINUE LA EJECUCION DE LAS LINEAS RESTANTES Y SALTA AL SIGUIENTE CICLO
                    #       dato_suavizado = segunda_derivada(dato)
                    #       mostrar_espectros(dato_suavizado,raman_shift,metodo,nor_op,m_suavi,2)
                    #   elif metodo_grafico == 2:
                    #       dato,nor_op,m_suavi= suavizado_menu_derivadas()
                    #       if dato is None:  # Manejar la opción "Volver"
                    #           continue 
                    #       dato_suavizado = segunda_derivada(dato)
                    #       espectro_acotado(dato_suavizado,0,0,nor_op,m_suavi,2)
                    #   elif metodo_grafico == 3:
                    #       dato,nor_op,m_suavi = suavizado_menu_derivadas()
                    #       if dato is None:  # Manejar la opción "Volver"
                    #           continue 
                    #       dato_suavizado = segunda_derivada(dato)
                    #       grafico_tipo(dato_suavizado,raman_shift,nor_op,metodo,m_suavi,2)
                    #   elif metodo_grafico == 4:
                    #       dato,nor_op,m_suavi = suavizado_menu_derivadas()
                    #       if dato is None:  # Manejar la opción "Volver"
                    #           continue 
                    #       dato_suavizado = segunda_derivada(dato)
                    #       grafico_acotado_tipo(dato_suavizado,raman_shift,metodo,nor_op,m_suavi,2)
                    #   elif metodo_grafico == 5:
                    #       dato,nor_op,m_suavi = suavizado_menu_derivadas()
                    #       if dato is None:  # Manejar la opción "Volver"
                    #           continue 
                    #       dato_suavizado = segunda_derivada(dato)
                    #       descargar_csv(8, dato_suavizado,raman_shift) # 8 PARA SABER QUE VIENE DE LA SEGUNDA DERIVADA
                    #   elif metodo_grafico == 6:
                    #       dato,nor_op,m_suavi = suavizado_menu_derivadas()
                    #       if dato is None:  # Manejar la opción "Volver"
                    #           continue 
                    #       dato_suavizado = segunda_derivada(dato)
                    #       descargar_csv_acotado(dato_suavizado,8,raman_shift) # 8 PARA SABER QUE VIENE DE ACA Y PODER ELEGIR EL NOMBRE DE LA CARPETA DE SALIDA
                    #   elif metodo_grafico == 7:
                    #       dato,nor_op,m_suavi = suavizado_menu_derivadas()
                    #       if dato is None:  # Manejar la opción "Volver"
                    #           continue 
                    #       dato_suavizado = segunda_derivada(dato)
                    #       descargar_csv_tipo(dato_suavizado,8,raman_shift) # 8 PARA SABER QUE VIENE DE ACA Y PODER ELEGIR EL NOMBRE DE LA CARPETA DE SALIDA
                    #   elif metodo_grafico == 8:
                    #       dato,nor_op,m_suavi = suavizado_menu_derivadas()
                    #       if dato is None:  # Manejar la opción "Volver"
                    #           continue 
                    #       dato_suavizado = segunda_derivada(dato)
                    #       descargar_csv_acotado_tipo(dato_suavizado,8,raman_shift) # 8 PARA SABER QUE VIENE DE ACA Y PODER ELEGIR EL NOMBRE DE LA CARPETA DE SALIDA
                    #   elif metodo_grafico == 9:
                    #       main()     
        elif opcion == '9':
                      print("entro 9")
                      # while True:  # Bucle para mantener al usuario en el submenú
                      #   metodo = 9          
                      #   sub_menu()
                      #   metodo_grafico = int(input("Opcion: "))               
                      #   if metodo_grafico == 1:
                      #       dato,nor_op ,m_suavi = menu_correccion()
                      #       if dato is None:  # Manejar la opción "Volver"
                      #           continue # EVITA QUE CONTINUE LA EJECUCION DE LAS LINEAS RESTANTES Y SALTA AL SIGUIENTE CICLO
                      #       dato_suavizado ,raman_shift_corregido = correcion_LineaB(dato) # RAMAN_SHIFT_CORREGIDO ES POR QUE YA SON ELIMANDOS LOS VALORES NAN
                      #       mostrar_espectros(dato_suavizado,raman_shift_corregido,metodo,nor_op,m_suavi,3)
                      #   elif metodo_grafico == 2:
                      #       dato,nor_op,m_suavi= menu_correccion()
                      #       if dato is None:  # Manejar la opción "Volver"
                      #           continue 
                      #       dato_suavizado ,raman_shift_corregido = correcion_LineaB(dato)
                      #       espectro_acotado(dato_suavizado,raman_shift_corregido,0,nor_op,m_suavi,3)
                      #   elif metodo_grafico == 3:
                      #       dato,nor_op,m_suavi = menu_correccion()
                      #       if dato is None:  # Manejar la opción "Volver"
                      #           continue 
                      #       dato_suavizado ,raman_shift_corregido = correcion_LineaB(dato)
                      #       grafico_tipo(dato_suavizado,raman_shift_corregido,nor_op,metodo,m_suavi,3)
                      #   elif metodo_grafico == 4:
                      #       dato,nor_op,m_suavi = menu_correccion()
                      #       if dato is None:  # Manejar la opción "Volver"
                      #           continue 
                      #       dato_suavizado ,raman_shift_corregido = correcion_LineaB(dato)
                      #       grafico_acotado_tipo(dato_suavizado,raman_shift_corregido,metodo,nor_op,m_suavi,3)
                      #   elif metodo_grafico == 5:
                      #           dato,nor_op,m_suavi = menu_correccion()
                      #           if dato is None:  # Manejar la opción "Volver"
                      #               continue 
                      #           dato_suavizado ,raman_shift_corregido = correcion_LineaB(dato)
                      #           descargar_csv(9, dato_suavizado,raman_shift_corregido) # 9 PARA SABER QUE VIENE DE CORRECCION LINEAL
                      #   elif metodo_grafico == 6:
                      #       dato,nor_op,m_suavi = menu_correccion()
                      #       if dato is None:  # Manejar la opción "Volver"
                      #           continue 
                      #       dato_suavizado ,raman_shift_corregido = correcion_LineaB(dato)
                      #       descargar_csv_acotado(dato_suavizado,9,raman_shift_corregido) # 9 PARA SABER QUE VIENE DE LA CORRECCION BASE LINEAL                          
                      #   elif metodo_grafico == 7:
                      #       dato,nor_op,m_suavi = menu_correccion()
                      #       if dato is None:  # Manejar la opción "Volver"
                      #           continue 
                      #       dato_suavizado ,raman_shift_corregido = correcion_LineaB(dato)
                      #       descargar_csv_tipo(dato_suavizado,9,raman_shift_corregido) # 9 PARA SABER QUE VIENE DE ACA Y PODER ELEGIR EL NOMBRE DE LA CARPETA DE SALIDA
                      #   elif metodo_grafico == 8:
                      #       dato,nor_op,m_suavi = menu_correccion()
                      #       if dato is None:  # Manejar la opción "Volver"
                      #           continue 
                      #       dato_suavizado ,raman_shift_corregido = correcion_LineaB(dato)
                      #       descargar_csv_acotado_tipo(dato_suavizado,9,raman_shift_corregido) # 9 PARA SABER QUE VIENE DE ACA Y PODER ELEGIR EL NOMBRE DE LA CARPETA DE SALIDA
                      #   elif metodo_grafico == 9:
                      #       main() 
        elif opcion == '10':
                     print("entro 10")
                     # while True:  # Bucle para mantener al usuario en el submenú
                     #   metodo = 10         
                     #   sub_menu()
                     #   metodo_grafico = int(input("Opcion: "))               
                     #   if metodo_grafico == 1:
                     #       dato,nor_op ,m_suavi = menu_correccion()
                     #       if dato is None:  # Manejar la opción "Volver"
                     #           continue # EVITA QUE CONTINUE LA EJECUCION DE LAS LINEAS RESTANTES Y SALTA AL SIGUIENTE CICLO
                     #       dato_suavizado ,raman_shift_corregido = correcion_Shirley(dato,raman_shift) # RAMAN_SHIFT_CORREGIDO ES POR QUE YA SON ELIMANDOS LOS VALORES NAN
                     #       mostrar_espectros(dato_suavizado,raman_shift_corregido,metodo,nor_op,m_suavi,4)
                     #   elif metodo_grafico == 2:
                     #       dato,nor_op,m_suavi= menu_correccion()
                     #       if dato is None:  # Manejar la opción "Volver"
                     #           continue 
                     #       dato_suavizado ,raman_shift_corregido = correcion_Shirley(dato,raman_shift)
                     #       espectro_acotado(dato_suavizado,raman_shift_corregido,0,nor_op,m_suavi,4)
                     #   elif metodo_grafico == 3:
                     #       dato,nor_op,m_suavi = menu_correccion()
                     #       if dato is None:  # Manejar la opción "Volver"
                     #           continue 
                     #       dato_suavizado ,raman_shift_corregido = correcion_Shirley(dato,raman_shift)
                     #       grafico_tipo(dato_suavizado,raman_shift_corregido,nor_op,metodo,m_suavi,4)
                     #   elif metodo_grafico == 4:
                     #       dato,nor_op,m_suavi = menu_correccion()
                     #       if dato is None:  # Manejar la opción "Volver"
                     #           continue 
                     #       dato_suavizado ,raman_shift_corregido = correcion_Shirley(dato,raman_shift)
                     #       grafico_acotado_tipo(dato_suavizado,raman_shift_corregido,metodo,nor_op,m_suavi,4)
                     #   elif metodo_grafico == 5:
                     #           dato,nor_op,m_suavi = menu_correccion()
                     #           if dato is None:  # Manejar la opción "Volver"
                     #               continue 
                     #           dato_suavizado ,raman_shift_corregido = correcion_Shirley(dato,raman_shift)
                     #           descargar_csv(10, dato_suavizado,raman_shift_corregido) # 9 PARA SABER QUE VIENE DE CORRECCION LINEAL
                     #   elif metodo_grafico == 6:
                     #       dato,nor_op,m_suavi = menu_correccion()
                     #       if dato is None:  # Manejar la opción "Volver"
                     #           continue 
                     #       dato_suavizado ,raman_shift_corregido = correcion_Shirley(dato,raman_shift)
                     #       descargar_csv_acotado(dato_suavizado,10,raman_shift_corregido) # 9 PARA SABER QUE VIENE DE LA CORRECCION BASE LINEAL                          
                     #   elif metodo_grafico == 7:
                     #       dato,nor_op,m_suavi = menu_correccion()
                     #       if dato is None:  # Manejar la opción "Volver"
                     #           continue 
                     #       dato_suavizado ,raman_shift_corregido = correcion_Shirley(dato,raman_shift)
                     #       descargar_csv_tipo(dato_suavizado,10,raman_shift_corregido) # 9 PARA SABER QUE VIENE DE ACA Y PODER ELEGIR EL NOMBRE DE LA CARPETA DE SALIDA
                     #   elif metodo_grafico == 8:
                     #       dato,nor_op,m_suavi = menu_correccion()
                     #       if dato is None:  # Manejar la opción "Volver"
                     #           continue 
                     #       dato_suavizado ,raman_shift_corregido = correcion_Shirley(dato,raman_shift)
                     #       descargar_csv_acotado_tipo(dato_suavizado,10,raman_shift_corregido) # 9 PARA SABER QUE VIENE DE ACA Y PODER ELEGIR EL NOMBRE DE LA CARPETA DE SALIDA
                     #   elif metodo_grafico == 9:
                     #       main() 
        
        elif opcion == '11':
                    print("entro 11")
                    # while True:  # Bucle para mantener al usuario en el submenú
                    #   metodo = 11          
                    #   print("1. Generar grafico PCA")
                    #   print("2. Volver")
                    #   metodo_grafico = int(input("Opcion: "))               
                    #   if metodo_grafico == 1:
                    #       print("Deseas realizar alguna correcion?")
                    #       print("1. CORRECCION BASE LINEAL")
                    #       print("2. CORRECION SHIRLEY")
                    #       print("3. No")
                    #       correcion = int(input("Opcion: "))
                    #       if correcion == 1:
                    #           dato,nor_op,m_suavi = menu_correccion()
                    #           if dato is None:  # Manejar la opción "Volver"
                    #               continue 
                    #           dato_suavizado ,raman_shift_corregido = correcion_LineaB(dato)
                    #           dato = dato_suavizado
                    #           pca(dato,raman_shift) 
                    #       elif correcion == 2:
                    #           print("FALTA IMPLEMENTAR CORRECION DE SHIRLEY")
                    #           pca(dato,raman_shift)
                    #       else: 
                    #           dato,nor_op,m_suavi = menu_correccion()
                    #           pca(dato,raman_shift)
                              
                    #       # if dato is None:  # Manejar la opción "Volver"
                    #       #     continue # EVITA QUE CONTINUE LA EJECUCION DE LAS LINEAS RESTANTES Y SALTA AL SIGUIENTE CICLO
                    #       # PCA(dato,raman_shift) # RAMAN_SHIFT_CORREGIDO ES POR QUE YA SON ELIMANDOS LOS VALORES NAN
                    #   elif metodo_grafico == 2:
                    #       main() 
    
        elif opcion == '12':  #PARA EL HCA DA LAS OPCIONES DE NORMALIZAR,SUAVIZAR,DERIVAR PERO NO LA DE CORREGIR QUE ESTARIA FALTANDO
            print("entro 12")
        #               while True:  # Bucle para mantener al usuario en el submenú
        #                 metodo = 9          
        #                 print("1. Generar grafico Dendrograma")
        #                 print("2. Volver")
        #                 metodo_grafico = int(input("Opcion: "))               
        #                 if metodo_grafico == 1:
        #                     dato,nor_op ,m_suavi = menu_correccion()
        #                     if dato is None:  # Manejar la opción "Volver"
        #                         continue # EVITA QUE CONTINUE LA EJECUCION DE LAS LINEAS RESTANTES Y SALTA AL SIGUIENTE CICLO
        #                     hca(dato,raman_shift) # RAMAN_SHIFT_CORREGIDO ES POR QUE YA SON ELIMANDOS LOS VALORES NAN
        #                 elif metodo_grafico == 2:
        #                     main() 
        elif opcion == "13":
            print("entro 13")
        #     python = sys.executable  # Obtiene la ruta del ejecutable de Python
        #     os.execl(python, python, *sys.argv)  # Reinicia el script                          
        elif opcion == '14':
            print("entro 14")
        #     print("Saliendo del programa...")
        #     sys.exit()  # Termina completamente el programa
        # else:
        #     print("Opción no válida. Inténtalo de nuevo.")

    
    
    
    
    
# Para ejecutar el menú principal
if __name__ == "__main__":
    menu_principal()
