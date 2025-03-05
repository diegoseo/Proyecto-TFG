#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 20:47:33 2025

@author: diego
"""

import matplotlib.pyplot as plt



def titulo_plot_mostrar(archivo_nombre,metodo,nor_op):
    # TODO ESTE CONDICIONAL ES SOLO PARA QUE EL TITULO DEL GRAFICO MUESTRE POR CUAL METODO SE NORMALIZO
    bd_name = archivo_nombre
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
                
                

def titulo_plot_acotado(archivo_nombre,nor_op,min_rango,max_rango):
     bd_name = archivo_nombre
     if nor_op == 1:
         plt.xlabel('Longitud de onda / Frecuencia')
         plt.ylabel('Intensidad')
         plt.title(f'Espectros del archivo {bd_name} en el rango de {min_rango} a {max_rango}')
         plt.show()
     elif nor_op == 2:
         plt.xlabel('Longitud de onda / Frecuencia')
         plt.ylabel('Intensidad')
         plt.title(f'Espectros del archivo {bd_name} normalizado por la media en el rango de {min_rango} a {max_rango}')
         plt.show()
     elif nor_op == 3:
         plt.xlabel('Longitud de onda / Frecuencia')
         plt.ylabel('Intensidad')
         plt.title(f'Espectros del archivo {bd_name} normalizado por la area en el rango de {min_rango} a {max_rango}')
         plt.show() 
     elif nor_op == 4:
         plt.xlabel('Longitud de onda / Frecuencia')
         plt.ylabel('Intensidad')
         plt.title(f'Espectros del archivo {bd_name} Suavizado por Saviztky_golay y Normalizado por la media')
         plt.show()   
     elif nor_op == 5:
         plt.xlabel('Longitud de onda / Frecuencia')
         plt.ylabel('Intensidad')
         plt.title(f'Espectros del archivo {bd_name} Suavizado por Saviztky_golay y Normalizado Area')
         plt.show() 
     elif nor_op == 6:
         plt.xlabel('Longitud de onda / Frecuencia')
         plt.ylabel('Intensidad')
         plt.title(f'Espectros del archivo {bd_name} Suavizado por Saviztky_golay y sin Normalizar ')
         plt.show()  
     elif nor_op == 7:
         plt.xlabel('Longitud de onda / Frecuencia')
         plt.ylabel('Intensidad')
         plt.title(f'Espectros del archivo {bd_name} Suavizado por Filtro Gausiano y Normalizado por la media')
         plt.show()   
     elif nor_op == 8:
         plt.xlabel('Longitud de onda / Frecuencia')
         plt.ylabel('Intensidad')
         plt.title(f'Espectros del archivo {bd_name} Suavizado por Filtro Gausiano y Normalizado Area')
         plt.show() 
     elif nor_op == 9:
         plt.xlabel('Longitud de onda / Frecuencia')
         plt.ylabel('Intensidad')
         plt.title(f'Espectros del archivo {bd_name} Suavizado por Filtro Gausiano y sin Normalizar ')
         plt.show()  
     elif nor_op == 10:
         plt.xlabel('Longitud de onda / Frecuencia')
         plt.ylabel('Intensidad')
         plt.title(f'Espectros del archivo {bd_name} Suavizado por Media Movil y Normalizado por la media')
         plt.show()   
     elif nor_op == 11:
         plt.xlabel('Longitud de onda / Frecuencia')
         plt.ylabel('Intensidad')
         plt.title(f'Espectros del archivo {bd_name} Suavizado por Media Movil y Normalizado Area')
         plt.show() 
     elif nor_op == 12:
         plt.xlabel('Longitud de onda / Frecuencia')
         plt.ylabel('Intensidad')
         plt.title(f'Espectros del archivo {bd_name} Suavizado por Media Movil y sin Normalizar ')
         plt.show()      
         
                                
         
            
         

def titulo_plot_tipo(archivo_nombre,metodo,mostrar_tipo,opcion,m_suavi):
    bd_name = archivo_nombre
    print("titulo_plot_tipo")
    # TODO ESTE CONDICIONAL ES SOLO PARA QUE EL TITULO DEL GRAFICO MUESTRE POR CUAL METODO SE NORMALIZO
    if metodo == 1:
        plt.xlabel('Longitud de onda / Frecuencia')
        plt.ylabel('Intensidad')
        plt.title(f'Espectro del tipo: {mostrar_tipo} del archivo {bd_name} sin normalizar')
        plt.show()
    elif metodo == 2:
        plt.xlabel('Longitud de onda / Frecuencia')
        plt.ylabel('Intensidad')
        plt.title(f'Espectro del tipo: {mostrar_tipo} del archivo {bd_name} Normalizado por la Media')
        plt.show()
    elif metodo == 3:
        plt.xlabel('Longitud de onda / Frecuencia')
        plt.ylabel('Intensidad')
        plt.title(f'Espectro del tipo: {mostrar_tipo} del archivo {bd_name} Normalizado por Area')
        plt.show()
    elif metodo == 4:
        if opcion == 1:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Espectro del tipo: {mostrar_tipo}  Suavizado por Saviztky_golay y Normalizado por la media')
            plt.show()   
        elif opcion == 2:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Espectro del tipo: {mostrar_tipo} Suavizado por Saviztky_golay y Normalizado Area')
            plt.show() 
        else:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Espectro del tipo: {mostrar_tipo}  Suavizado por Saviztky_golay y sin Normalizar ')
            plt.show()          
    elif metodo == 5:
        if opcion == 1:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Espectro del tipo: {mostrar_tipo}  Suavizado por Filtro Gaussiano y Normalizado por la media')
            plt.show()   
        elif opcion == 2:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Espectro del tipo: {mostrar_tipo}  Suavizado por Filtro Gaussiano y Normalizado Area')
            plt.show() 
        else:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Espectro del tipo: {mostrar_tipo} Suavizado por Filtro Gaussiano y sin Normalizar ')
            plt.show() 
    elif metodo == 6:
        if opcion == 1:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Espectro del tipo: {mostrar_tipo}  Suavizado por Media Movil y Normalizado por la media')
            plt.show()   
        elif opcion == 2:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Espectro del tipo: {mostrar_tipo}  Suavizado por Media Movil y Normalizado Area')
            plt.show() 
        else:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Espectro del tipo: {mostrar_tipo}  Suavizado por Media Movil y sin Normalizar ')
            plt.show() 
    elif metodo == 7:
            print("hola PCA")
    elif metodo == 8:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Espectro del tipo: {mostrar_tipo}  por la primera derivada ')
            plt.show() 
    elif metodo == 9:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Espectro del tipo: {mostrar_tipo}  por la primera derivada ')
            plt.show() 
    elif metodo == 10:
            if opcion == '1':
                if m_suavi == 1:
                    plt.xlabel('Longitud de onda / Frecuencia')
                    plt.ylabel('Intensidad')
                    plt.title(f'Espectros del archivo {bd_name} Suavizado por SAVIZTKY-GOLAY y Normalizado por la media')
                    plt.show()
                elif m_suavi == 2:
                    plt.xlabel('Longitud de onda / Frecuencia')
                    plt.ylabel('Intensidad')
                    plt.title(f'Espectros del archivo {bd_name} Suavizado por FILTRO GAUSIANO y Normalizado por la media')
                    plt.show()
                elif m_suavi == 3:
                    plt.xlabel('Longitud de onda / Frecuencia')
                    plt.ylabel('Intensidad')
                    plt.title(f'Espectros del archivo {bd_name} Suavizado por MEDIA MOVIL y Normalizado por la media')
                    plt.show()
                else:
                    plt.xlabel('Longitud de onda / Frecuencia')
                    plt.ylabel('Intensidad')
                    plt.title(f'Espectros del archivo {bd_name} sin suavizar y Normalizado por la media')
                    plt.show()
            elif opcion == '2':
                if m_suavi == 1:
                    plt.xlabel('Longitud de onda / Frecuencia')
                    plt.ylabel('Intensidad')
                    plt.title(f'Espectros del archivo {bd_name} Suavizado por SAVIZTKY-GOLAY y Normalizado por Area')
                    plt.show()
                elif m_suavi == 2:
                    plt.xlabel('Longitud de onda / Frecuencia')
                    plt.ylabel('Intensidad')
                    plt.title(f'Espectros del archivo {bd_name} Suavizado por FILTRO GAUSIANO y Normalizado por Area')
                    plt.show()
                elif m_suavi == 3:
                    plt.xlabel('Longitud de onda / Frecuencia')
                    plt.ylabel('Intensidad')
                    plt.title(f'Espectros del archivo {bd_name} Suavizado por MEDIA MOVIL y Normalizado por Area')
                    plt.show()
                else:
                    plt.xlabel('Longitud de onda / Frecuencia')
                    plt.ylabel('Intensidad')
                    plt.title(f'Espectros del archivo {bd_name} sin Suavizar y Normalizado por Area')
                    plt.show()
            else:
                if m_suavi == 1:
                    plt.xlabel('Longitud de onda / Frecuencia')
                    plt.ylabel('Intensidad')
                    plt.title(f'Espectros del archivo {bd_name} Suavizado por SAVIZTKY-GOLAY y Sin Normalizar')
                    plt.show()
                elif m_suavi == 2:
                    plt.xlabel('Longitud de onda / Frecuencia')
                    plt.ylabel('Intensidad')
                    plt.title(f'Espectros del archivo {bd_name} Suavizado por FILTRO GAUSIANO y Sin Normalizar')
                    plt.show()
                elif m_suavi == 3:
                    plt.xlabel('Longitud de onda / Frecuencia')
                    plt.ylabel('Intensidad')
                    plt.title(f'Espectros del archivo {bd_name} Suavizado por MEDIA MOVIL y Sin Normalizar')
                    plt.show()
                else:
                    plt.xlabel('Longitud de onda / Frecuencia')
                    plt.ylabel('Intensidad')
                    plt.title(f'Espectros del archivo {bd_name} sin Suavizar y Sin Normalizar')
                    plt.show()
    # elif metodo == 11:
    # elif metodo == 12:
    elif metodo == 13:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Espectros del archivo {bd_name} Acotado ')
            plt.show() 
    else:
        print("NO HAY GRAFICA DISPONIBLE PARA ESTA OPCION")

 


def titulo_plot_tipo_acotado(archivo_nombre,metodo,mostrar_tipo,min_rango,max_rango,m_suavi):
 bd_name = archivo_nombre
 print("titulo_plot_tipo_acotado")
 # TODO ESTE CONDICIONAL ES SOLO PARA QUE EL TITULO DEL GRAFICO MUESTRE POR CUAL METODO SE NORMALIZO
 if metodo == 1:
     plt.xlabel('Longitud de onda / Frecuencia')
     plt.ylabel('Intensidad')
     plt.title(f'Espectro del tipo: {mostrar_tipo} del archivo {bd_name} sin normalizar en el rango:[{min_rango},{max_rango}]')
     plt.show()
 elif metodo == 2:
     plt.xlabel('Longitud de onda / Frecuencia')
     plt.ylabel('Intensidad')
     plt.title(f'Espectro del tipo: {mostrar_tipo} del archivo {bd_name} Normalizado por la Media en el rango:[{min_rango},{max_rango}]')
     plt.show()
 elif metodo == 3:
     plt.xlabel('Longitud de onda / Frecuencia')
     plt.ylabel('Intensidad')
     plt.title(f'Espectro del tipo: {mostrar_tipo} del archivo {bd_name} Normalizado por Area en el rango:[{min_rango},{max_rango}]')
     plt.show()
 elif metodo == 4:
     if m_suavi == 1:
         plt.xlabel('Longitud de onda / Frecuencia')
         plt.ylabel('Intensidad')
         plt.title(f'Espectro del tipo: {mostrar_tipo}  Suavizado por Saviztky_golay y Normalizado por la media en el rango:[{min_rango},{max_rango}]')
         plt.show()   
     elif m_suavi == 2:
         plt.xlabel('Longitud de onda / Frecuencia')
         plt.ylabel('Intensidad')
         plt.title(f'Espectro del tipo: {mostrar_tipo} Suavizado por Saviztky_golay y Normalizado Area en el rango:[{min_rango},{max_rango}]')
         plt.show() 
     else:
         plt.xlabel('Longitud de onda / Frecuencia')
         plt.ylabel('Intensidad')
         plt.title(f'Espectro del tipo: {mostrar_tipo}  Suavizado por Saviztky_golay y sin Normalizar en el rango:[{min_rango},{max_rango}]')
         plt.show()          
 elif metodo == 5:
     if m_suavi == 1:
         plt.xlabel('Longitud de onda / Frecuencia')
         plt.ylabel('Intensidad')
         plt.title(f'Espectro del tipo: {mostrar_tipo}  Suavizado por Filtro Gaussiano y Normalizado por la media en el rango:[{min_rango},{max_rango}]')
         plt.show()   
     elif m_suavi == 2:
         plt.xlabel('Longitud de onda / Frecuencia')
         plt.ylabel('Intensidad')
         plt.title(f'Espectro del tipo: {mostrar_tipo}  Suavizado por Filtro Gaussiano y Normalizado Area en el rango:[{min_rango},{max_rango}]')
         plt.show() 
     else:
         plt.xlabel('Longitud de onda / Frecuencia')
         plt.ylabel('Intensidad')
         plt.title(f'Espectro del tipo: {mostrar_tipo} Suavizado por Filtro Gaussiano y sin Normalizar en el rango:[{min_rango},{max_rango}]')
         plt.show() 
 elif metodo == 6:
     if m_suavi == 1:
         plt.xlabel('Longitud de onda / Frecuencia')
         plt.ylabel('Intensidad')
         plt.title(f'Espectro del tipo: {mostrar_tipo}  Suavizado por Media Movil y Normalizado por la media en el rango:[{min_rango},{max_rango}]')
         plt.show()   
     elif m_suavi == 2:
         plt.xlabel('Longitud de onda / Frecuencia')
         plt.ylabel('Intensidad')
         plt.title(f'Espectro del tipo: {mostrar_tipo}  Suavizado por Media Movil y Normalizado Area en el rango:[{min_rango},{max_rango}]')
         plt.show() 
     else:
         plt.xlabel('Longitud de onda / Frecuencia')
         plt.ylabel('Intensidad')
         plt.title(f'Espectro del tipo: {mostrar_tipo}  Suavizado por Media Movil y sin Normalizar en el rango:[{min_rango},{max_rango}]')
         plt.show() 
 elif metodo == 7:
         print("hola PCA")
 elif metodo == 8:
         plt.xlabel('Longitud de onda / Frecuencia')
         plt.ylabel('Intensidad')
         plt.title(f'Espectro del tipo: {mostrar_tipo}  por la primera derivada ')
         plt.show() 
 elif metodo == 9:
         plt.xlabel('Longitud de onda / Frecuencia')
         plt.ylabel('Intensidad')
         plt.title(f'Espectro del tipo: {mostrar_tipo}  por la primera derivada ')
         plt.show() 
 # elif metodo == 10:
 #         if opcion == '1':
 #             if m_suavi == 1:
 #                 plt.xlabel('Longitud de onda / Frecuencia')
 #                 plt.ylabel('Intensidad')
 #                 plt.title(f'Espectros del archivo {bd_name} Suavizado por SAVIZTKY-GOLAY y Normalizado por la media')
 #                 plt.show()
 #             elif m_suavi == 2:
 #                 plt.xlabel('Longitud de onda / Frecuencia')
 #                 plt.ylabel('Intensidad')
 #                 plt.title(f'Espectros del archivo {bd_name} Suavizado por FILTRO GAUSIANO y Normalizado por la media')
 #                 plt.show()
 #             elif m_suavi == 3:
 #                 plt.xlabel('Longitud de onda / Frecuencia')
 #                 plt.ylabel('Intensidad')
 #                 plt.title(f'Espectros del archivo {bd_name} Suavizado por MEDIA MOVIL y Normalizado por la media')
 #                 plt.show()
 #             else:
 #                 plt.xlabel('Longitud de onda / Frecuencia')
 #                 plt.ylabel('Intensidad')
 #                 plt.title(f'Espectros del archivo {bd_name} sin suavizar y Normalizado por la media')
 #                 plt.show()
 #         elif opcion == '2':
 #             if m_suavi == 1:
 #                 plt.xlabel('Longitud de onda / Frecuencia')
 #                 plt.ylabel('Intensidad')
 #                 plt.title(f'Espectros del archivo {bd_name} Suavizado por SAVIZTKY-GOLAY y Normalizado por Area')
 #                 plt.show()
 #             elif m_suavi == 2:
 #                 plt.xlabel('Longitud de onda / Frecuencia')
 #                 plt.ylabel('Intensidad')
 #                 plt.title(f'Espectros del archivo {bd_name} Suavizado por FILTRO GAUSIANO y Normalizado por Area')
 #                 plt.show()
 #             elif m_suavi == 3:
 #                 plt.xlabel('Longitud de onda / Frecuencia')
 #                 plt.ylabel('Intensidad')
 #                 plt.title(f'Espectros del archivo {bd_name} Suavizado por MEDIA MOVIL y Normalizado por Area')
 #                 plt.show()
 #             else:
 #                 plt.xlabel('Longitud de onda / Frecuencia')
 #                 plt.ylabel('Intensidad')
 #                 plt.title(f'Espectros del archivo {bd_name} sin Suavizar y Normalizado por Area')
 #                 plt.show()
 #         else:
 #             if m_suavi == 1:
 #                 plt.xlabel('Longitud de onda / Frecuencia')
 #                 plt.ylabel('Intensidad')
 #                 plt.title(f'Espectros del archivo {bd_name} Suavizado por SAVIZTKY-GOLAY y Sin Normalizar')
 #                 plt.show()
 #             elif m_suavi == 2:
 #                 plt.xlabel('Longitud de onda / Frecuencia')
 #                 plt.ylabel('Intensidad')
 #                 plt.title(f'Espectros del archivo {bd_name} Suavizado por FILTRO GAUSIANO y Sin Normalizar')
 #                 plt.show()
 #             elif m_suavi == 3:
 #                 plt.xlabel('Longitud de onda / Frecuencia')
 #                 plt.ylabel('Intensidad')
 #                 plt.title(f'Espectros del archivo {bd_name} Suavizado por MEDIA MOVIL y Sin Normalizar')
 #                 plt.show()
 #             else:
 #                 plt.xlabel('Longitud de onda / Frecuencia')
 #                 plt.ylabel('Intensidad')
 #                 plt.title(f'Espectros del archivo {bd_name} sin Suavizar y Sin Normalizar')
 #                 plt.show()
 # elif metodo == 11:
 # elif metodo == 12:
 elif metodo == 13:
         plt.xlabel('Longitud de onda / Frecuencia')
         plt.ylabel('Intensidad')
         plt.title(f'Espectros del archivo {bd_name} Acotado ')
         plt.show() 
 else:
     print("NO HAY GRAFICA DISPONIBLE PARA ESTA OPCION")








def titulo_plot_primera_derivada(archivo_nombre,opcion,metodo_suavizado):
    print("OPCION = ", opcion)
    print("METODO_SUAVIZADO = ", metodo_suavizado)
    bd_name = archivo_nombre
    if opcion == 1:
        if metodo_suavizado == 1:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Primera Derivada del archivo {bd_name} Suavizado por Saviztky_golay y Normalizado por la media ')
            plt.show()
        elif metodo_suavizado == 2:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Primera Derivada del archivo {bd_name} Suavizado por FILTRO GAUSIANO y Normalizado por la media ')
            plt.show()
        elif metodo_suavizado == 3:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Primera Derivada  del archivo {bd_name} Suavizado por MEDIA MOVIL y Normalizado por la media ')
            plt.show()
        else:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Primera Derivada  del archivo {bd_name} Sin Suavizar y Normalizado por la media ')
            plt.show()
    elif opcion == 2:
        if metodo_suavizado == 1:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Primera Derivada  del archivo {bd_name} Suavizado por Saviztky_golay y Normalizado por Area ')
            plt.show()
        elif metodo_suavizado == 2:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Primera Derivada del archivo {bd_name} Suavizado por FILTRO GAUSIANO y Normalizado por Area ')
            plt.show()
        elif metodo_suavizado == 3:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Primera Derivada  del archivo {bd_name} Suavizado por MEDIA MOVIL y Normalizado por Area')
            plt.show()
        else:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Primera Derivada  del archivo {bd_name} Sin Suavizar y Normalizado por Area')
            plt.show()
    else:
        if metodo_suavizado == 1:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Primera Derivada del archivo {bd_name} Suavizado por Saviztky_golay y sin Normalizar')
            plt.show()
        elif metodo_suavizado == 2:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Primera Derivada del archivo {bd_name} Suavizado por FILTRO GAUSIANO y sin Normalizar ')
            plt.show()
        elif metodo_suavizado == 3:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Primera Derivada del archivo {bd_name} Suavizado por MEDIA MOVIL y sin Normalizar')
            plt.show()
        else:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Primera Derivada del archivo {bd_name} Sin Suavizar y sin Normalizar')
            plt.show()









def titulo_plot_segunda_derivada(archivo_nombre,opcion,metodo_suavizado):
     bd_name = archivo_nombre
     if opcion == 1:
         if metodo_suavizado == 1:
             plt.xlabel('Longitud de onda / Frecuencia')
             plt.ylabel('Intensidad')
             plt.title(f'Segunda Derivada del archivo {bd_name} Suavizado por Saviztky_golay y Normalizado por la media ')
             plt.show()
         elif metodo_suavizado == 2:
             plt.xlabel('Longitud de onda / Frecuencia')
             plt.ylabel('Intensidad')
             plt.title(f'Segunda Derivada del archivo {bd_name} Suavizado por FILTRO GAUSIANO y Normalizado por la media ')
             plt.show()
         elif metodo_suavizado == 3:
             plt.xlabel('Longitud de onda / Frecuencia')
             plt.ylabel('Intensidad')
             plt.title(f'Segunda Derivada  del archivo {bd_name} Suavizado por MEDIA MOVIL y Normalizado por la media ')
             plt.show()
         else:
             plt.xlabel('Longitud de onda / Frecuencia')
             plt.ylabel('Intensidad')
             plt.title(f'Segunda Derivada  del archivo {bd_name} Sin Suavizar y Normalizado por la media ')
             plt.show()
     elif opcion == 2:
         if metodo_suavizado == 1:
             plt.xlabel('Longitud de onda / Frecuencia')
             plt.ylabel('Intensidad')
             plt.title(f'Segunda Derivada  del archivo {bd_name} Suavizado por Saviztky_golay y Normalizado por Area ')
             plt.show()
         elif metodo_suavizado == 2:
             plt.xlabel('Longitud de onda / Frecuencia')
             plt.ylabel('Intensidad')
             plt.title(f'Segunda Derivada del archivo {bd_name} Suavizado por FILTRO GAUSIANO y Normalizado por Area ')
             plt.show()
         elif metodo_suavizado == 3:
             plt.xlabel('Longitud de onda / Frecuencia')
             plt.ylabel('Intensidad')
             plt.title(f'Segunda Derivada  del archivo {bd_name} Suavizado por MEDIA MOVIL y Normalizado por Area')
             plt.show()
         else:
             plt.xlabel('Longitud de onda / Frecuencia')
             plt.ylabel('Intensidad')
             plt.title(f'Segunda Derivada  del archivo {bd_name} Sin Suavizar y Normalizado por Area')
             plt.show()
     else:
         if metodo_suavizado == 1:
             plt.xlabel('Longitud de onda / Frecuencia')
             plt.ylabel('Intensidad')
             plt.title(f'Segunda Derivada del archivo {bd_name} Suavizado por Saviztky_golay y sin Normalizar')
             plt.show()
         elif metodo_suavizado == 2:
             plt.xlabel('Longitud de onda / Frecuencia')
             plt.ylabel('Intensidad')
             plt.title(f'Segunda Derivada del archivo {bd_name} Suavizado por FILTRO GAUSIANO y sin Normalizar ')
             plt.show()
         elif metodo_suavizado == 3:
             plt.xlabel('Longitud de onda / Frecuencia')
             plt.ylabel('Intensidad')
             plt.title(f'Segunda Derivada del archivo {bd_name} Suavizado por MEDIA MOVIL y sin Normalizar')
             plt.show()
         else:
             plt.xlabel('Longitud de onda / Frecuencia')
             plt.ylabel('Intensidad')
             plt.title(f'Segunda Derivada del archivo {bd_name} Sin Suavizar y sin Normalizar')
             plt.show()
         



def titulo_plot_correcion_base(archivo_nombre,nor_op,metodo_suavizado):
    bd_name = archivo_nombre
    print("entro aca")
    print("NOR_op=",nor_op)
    print("Metodo_suavizado=",metodo_suavizado)
    if nor_op == 1:
        if metodo_suavizado == 1:
            print("correccopn xDDDD")
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Correccion Linea base del archivo {bd_name} Suavizado por Saviztky_golay y Normalizado por la media ')
            plt.show()
        elif metodo_suavizado == 2:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Correccion Linea base  del archivo {bd_name} Suavizado por FILTRO GAUSIANO y Normalizado por la media ')
            plt.show()
        elif metodo_suavizado == 3:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Correccion Linea base  del archivo {bd_name} Suavizado por MEDIA MOVIL y Normalizado por la media ')
            plt.show()
        else:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Correccion Linea base del archivo {bd_name} Sin Suavizar y Normalizado por la media ')
            plt.show()
    elif nor_op  == 2:
        if metodo_suavizado == 1:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Correccion Linea base del archivo {bd_name} Suavizado por Saviztky_golay y Normalizado por Area ')
            plt.show()
        elif metodo_suavizado == 2:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Correccion Linea base  del archivo {bd_name} Suavizado por FILTRO GAUSIANO y Normalizado por Area ')
            plt.show()
        elif metodo_suavizado == 3:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Correccion Linea base  del archivo {bd_name} Suavizado por MEDIA MOVIL y Normalizado por Area')
            plt.show()
        else:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Correccion Linea base  del archivo {bd_name} Sin Suavizar y Normalizado por Area')
            plt.show()
    else:
        if metodo_suavizado == 1:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Correccion Linea base  del archivo {bd_name} Suavizado por Saviztky_golay y sin Normalizar')
            plt.show()
        elif metodo_suavizado == 2:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Correccion Linea base  del archivo {bd_name} Suavizado por FILTRO GAUSIANO y sin Normalizar ')
            plt.show()
        elif metodo_suavizado == 3:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Correccion Linea base  del archivo {bd_name} Suavizado por MEDIA MOVIL y sin Normalizar')
            plt.show()
        else:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Correccion Linea base  del archivo {bd_name} Sin Suavizar y sin Normalizar')
            plt.show()
        
