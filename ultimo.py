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
    
#print(df)

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
            mostrar_espectros(df2,metodo,0,0)
        elif opcion == '2':
            metodo = 2
            print("Procesando los datos")
            print("Por favor espere un momento...")
            mostrar_espectros(df_media_pca,metodo,0,0)
        elif opcion == '3':
            metodo = 3
            print("Procesando los datos")
            print("Por favor espere un momento...")
            mostrar_espectros(df_concatenado_cabecera_nueva_area,metodo,0,0)
        elif opcion == '4':
            suavizado_saviztky_golay(0,0)          
        elif opcion == '5':
            suavizado_filtroGausiano(0,0)
            print("Procesando los datos")
            print("Por favor espere un momento...")        
        elif opcion == '6':
              suavizado_mediamovil(0,0)
              print("Procesando los datos")
              print("Por favor espere un momento...")     
        elif opcion == '7':
              print("Procesando los datos")
              print("Por favor espere un momento...")
              mostrar_pca()       
        elif opcion == '8':
              primera_derivada(0,0)
        elif opcion == '9':
              segunda_derivada(0,0)
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
#print(intensity)   

tipos = df.iloc[0, 1:] # EXTRAEMOS LA PRIMERA FILA MENOS DE LA PRIMERA COLUMNA
#print(tipos)
types=tipos.tolist() #OJO AUN NO AGREGAMOS ESTA LINEA A ULTIMO.PY
print(types)

cabecera = df.iloc[[0]].copy() # EXTRAEMOS LA PRIMERA FILA 
#print(cabecera)

cant_tipos = tipos.nunique() # PARA EL EJEMPLO DE LIMPIO.CSV CANT_TIPOS TENDRA VALOR 4 YA QUE HAY 4 TIPOS (collagen,lipids,glycogen,DNA)
#print(cant_tipos)

tipos_nombres = df.iloc[0, 1:].unique() # OBTENEMOS LOS NOMBRES DE LOS TIPOS
#print(tipos_nombres)

# Obtenemos el colormap sin especificar el número de colores
cmap = plt.colormaps['hsv']  # Usamos solo el nombre del colormap

# Nos aseguramos de que `colores` es una lista
colores = [cmap(i) for i in np.linspace(0, 1, len(tipos_nombres))]  # Genera una lista de colores

# Crear el diccionario de asignación de colores
asignacion_colores = {tipo: mcolors.to_hex(colores[i]) for i, tipo in enumerate(tipos_nombres)}
print(asignacion_colores)

diccionario=pd.DataFrame(asignacion_colores.items())
print(diccionario)

#AHORA QUE YA TENGO ASIGNADO UN COLOR POR CADA TIPO TENGO QUE GRAFICAR LOS ESPECTROS#


"""
VARIABLES DE MOSTRAR ESPECTROS
"""
df2 = df.copy()
df2.columns = df2.iloc[0]
#print(df2)
df2 = df2.drop(0).reset_index(drop=True) #eliminamos la primera fila
df2 = df2.drop(df2.columns[0], axis=1) #eliminamos la primera columna el del rama_shift
#print(df2) # aca ya tenemos la tabla de la manera que necesitamos, fila cero es la cabecera con los nombres de los tipos anteriormente eran indice numericos consecutivos
df2 = df2.apply(pd.to_numeric, errors='coerce') #CONVERTIMOS A NUMERICO
#print("EL DATAFRAME DEL ESPECTRO SIN NORMALIZAR ES")
#print(df2) #ESTA VARIABLE SE USA PARA EL PCA TAMBIEN
#print(df2.shape)



"""
VARIABLES DE NORMALIZAR POR LA MEDIA    tratar de hacer por la forma del ejemplo y no por z-core para ver si se soluciona lo de la raya
"""

global df_media_pca
scaler = StandardScaler() 
cal_nor = scaler.fit_transform(intensity) #calcula la media y desviación estándar
#print(cal_nor)
dato_normalizado = pd.DataFrame(cal_nor, columns=intensity.columns) # lo convertimos de vuelta en un DataFrame
#print(dato_normalizado)
df_concatenado = pd.concat([cabecera,dato_normalizado], axis=0, ignore_index=True)
#print(df_concatenado)
# Paso 1: Convertir la primera fila en cabecera
df_concatenado.columns = df_concatenado.iloc[0]  # Asigna la primera fila como nombres de columna
# Paso 2: Eliminar la primera fila (ahora es la cabecera) y resetear el índice
df_concatenado_cabecera_nueva = df_concatenado[1:].reset_index(drop=True)
#print(df_concatenado_cabecera_nueva.head(50))
df_media_pca= pd.DataFrame(df_concatenado_cabecera_nueva.iloc[:,1:])
#print("EL ESPECTRO NORMALIZADO POR LA MEDIA ES")
#print(df_media_pca) #ESTA VARIABLE SE USA PARA EL PCA TAMBIEN
#print('normalizacion media')
df_media_pca.to_csv("nor_media_df.csv", index=False)


"""
 VARIABLES DE NORMALIZAR POR AREA    tratar de hacer otro sin np.trap para ver si se soluciona lo de la raya
"""

global df_concatenado_cabecera_nueva_area
df3 = pd.DataFrame(intensity)
#print("DataFrame de Intensidades:")
#print(df3)
df3 = df3.apply(pd.to_numeric, errors='coerce')  # Convierte a numérico, colocando NaN donde haya problemas
#print(df3)
np_array = raman_shift.to_numpy() #CONVERTIMOS INTENSITY AL TIPO NUMPY POR QUE POR QUE NP.TRAPZ UTILIZA ESE TIPO DE DATOS
#print(np_array)
df3_normalizado = df3.copy()
#print(df3)
# Cálculo del área bajo la curva para cada columna
#print("\nÁreas bajo la curva para cada columna:")
for col in df3.columns:
    #print(df3[col])
    #print(df3_normalizado[col])
    area = (np.trapz(df3[col], np_array)) *-1  #MULTIPLIQUE POR -1 PARA QUE EL GRAFICO SALGA TODO HACIA ARRIBA ESTO SE DEBE A QUE EL RAMAN_SHIFT ESTA EN FORMA DECRECIENTE
    if area != 0:
        df3_normalizado[col] = df3[col] / area
    else:
        print(f"Advertencia: El área de la columna {col} es cero y no se puede normalizar.") #seguro contra errores de división por cero 
#print(df3_normalizado)
df_concatenado_area = pd.concat([cabecera,df3_normalizado], axis=0, ignore_index=True)
#print(df_concatenado_area)
# Paso 1: Convertir la primera fila en cabecera
df_concatenado_area.columns = df_concatenado_area.iloc[0]  # Asigna la primera fila como nombres de columna
# Paso 2: Eliminar la primera fila (ahora es la cabecera) y resetear el índice
df_concatenado_cabecera_nueva_area = df_concatenado_area[1:].reset_index(drop=True)
# AHORA ELIMINAMOS LA COLUMNA CON VALORES NAN
df_concatenado_cabecera_nueva_area = df_concatenado_cabecera_nueva_area.dropna(axis=1, how='all')
#print("ESPECTRO NORMALIZADO POR EL AREA")
#print(df_concatenado_cabecera_nueva_area) #ESTA VARIABLE SE USA PARA EL PCA TAMBIEN
#print('entro 10')
df_concatenado_cabecera_nueva_area.to_csv("nor_area_df.csv", index=False)



# MOSTRAMOS LA LEYENDA PARA CADA TIPO
plt.figure(figsize=(2,2))    
for index, row in diccionario.iterrows():
    #print('entro 15')
    tipo = row[0]   # Nombre del tipo (por ejemplo, 'collagen')
    color = row[1]  # Color asociado (por ejemplo, '#ff0000')
    plt.plot([], [], color=color, label=tipo) 
# Mostrar la leyenda y el gráfico
#print('entro 20')
plt.legend(loc='center')
plt.grid(False)
plt.title(f'Cantidad de tipos encontrados {cant_tipos}')
plt.axis('off')
plt.show()




def mostrar_espectros(datos,metodo,opcion,m_suavi):
    
    
    
    print("ENTRO EN MOSTRAR ESPECTROS")
    print(datos)
    # Graficar los espectros
    plt.figure(figsize=(10, 6))
    

    #DESCOMENTAR EL CODIGO DE ABAJO ESE ES MIO, EL DE ARRIBA ES CHATGPT  
  
    leyendas_tipos = set()  # almacenamos los tipos que enocntramos y la funcion set() nos ayuda a quer no se repitan
    pos_y=0
    for col in datos.columns :
        #print('entro normal')
        for tipo in asignacion_colores:
            #print("wwwwwww")
            if tipo == col :
                color_actual= asignacion_colores[tipo] #ACA YA ENCONTRAMOS EL COLOR CORRESPONDIENTE A ESE TIPO   
                #print(tipo,'==',col,'color=',color_actual) 
                #print("ccccccc")
                if isinstance(col, str):  #Verifica que el nombre de la columna sea un string
                    #print('RAMAN SHIFT')
                    #print(raman_shift)
                    #print('INTENSIDADES')
                    #print(df2.iloc[:,pos_y]) 
                    print(col)
                    #print("xdxdxdxdxdxd")
                    if tipo in leyendas_tipos:
                        plt.plot(raman_shift , datos[col], color=color_actual, alpha=0.3, linewidth = 0.1,label=col) 
                        '''raman_shift:LE PASAMOS TODAS LAS INTENSIDADES , df2[col]= LE PASAMOS TODAS LAS COLUMNAS CON EL MISMO TIPO'''
                        #print('entro4')
                        #print(pos_y)   
                        break
                    else:
                        plt.plot(raman_shift , df2[col], color=color_actual, alpha=0.3, linewidth = 0.1) 
                        leyendas_tipos.add(tipo) 
                pos_y+=1 
    
    # TODO ESTE CONDICIONAL ES SOLO PARA QUE EL TITULO DEL GRAFICO MUESTRE POR CUAL METODO SE NORMALIZO
    if metodo == 1:
        #print(leyendas_tipos) 
        #print('entro 13')
        # Etiquetas y título
        plt.xlabel('Longitud de onda / Frecuencia')
        plt.ylabel('Intensidad')
        plt.title(f'Espectros del archivo {bd_name}')
        plt.show()
    elif metodo == 2:
        #print(leyendas_tipos) 
        #print('entro 13')
        # Etiquetas y título
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
        #print(leyendas_tipos) 
        #print('entro 13')
        # Etiquetas y título
        if opcion == '1':
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Espectros del archivo {bd_name} Suavizado por Saviztky_golay y Normalizado por la media')
            plt.show()   
        elif opcion == '2':
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
        #print(leyendas_tipos) 
        #print('entro 13')
        # Etiquetas y título
        if opcion == '1':
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Espectros del archivo {bd_name} Suavizado por Filtro Gaussiano y Normalizado por la media')
            plt.show()   
        elif opcion == '2':
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Espectros del archivo {bd_name} Suavizado por Filtro Gaussiano y Normalizado Area')
            plt.show() 
        else:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Espectros del archivo {bd_name} Suavizado por Filtro Gaussiano y sin Normalizar ')
            plt.show() 
    elif metodo == 6:
        #print(leyendas_tipos) 
        #print('entro 13')
        # Etiquetas y título
        if opcion == '1':
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Espectros del archivo {bd_name} Suavizado por Media Movil y Normalizado por la media')
            plt.show()   
        elif opcion == '2':
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Espectros del archivo {bd_name} Suavizado por Media Movil y Normalizado Area')
            plt.show() 
        else:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Espectros del archivo {bd_name} Suavizado por Media Movil y sin Normalizar ')
            plt.show() 
    elif metodo == 7:
            print("hola PCA")
    elif metodo == 8:
            print("entro en opcion 8")
  
    else:
        print("NO HAY GRAFICA DISPONIBLE PARA ESTA OPCION")



# SUAVIZADO POR SAVIZTKY-GOLAY

def suavizado_saviztky_golay(normalizado_pca, pca_op):  #acordarse que se puede suavizar por la media, area y directo
    if pca_op == 0:
        print("NORMALIZAR POR:")
        print("1-MEDIA")
        print("2-AREA")
        print("3-SIN NORMALIZAR")
        opcion = input("Selecciona una opción: ")
        ventana = int(input("INGRESE EL VALOR DE LA VENTANA: "))
        orden = int(input("INGRESE EL VALOR DEL ORDEN: "))
        print("Procesando los datos")
        print("Por favor espere un momento...")
     
        if opcion == '1'  :
            normalizado_pca = df_media_pca
            #print(normalizado_pca)
        elif opcion == '2' :
            normalizado_pca = df_concatenado_cabecera_nueva_area
            #print(normalizado_pca)
        elif opcion == '3' :
            normalizado_pca = df2
            #print(normalizado_pca)
        else:
            print("OPCION NO VALIDA")
            print("SALIR...")
            #mostrar_menu()
    else:
        ventana = int(input("INGRESE EL VALOR DE LA VENTANA: "))
        orden = int(input("INGRESE EL VALOR DEL ORDEN: "))
        print("Procesando los datos")
        print("Por favor espere un momento...")
     
      
    
    dato = normalizado_pca.to_numpy() #PASAMOS LOS DATOS A NUMPY POR QUE SAVGOL_FILTER USA SOLO NUMPY COMO PARAMETRO (PIERDE LA CABECERA DE TIPOS AL HACER ESTO)

    suavizado = savgol_filter(dato, window_length=ventana, polyorder=orden)
    suavizado_pd = pd.DataFrame(suavizado) # PASAMOS SUAVIZADO A PANDAS Y GUARDAMOS EN SUAVIZADO_PD
    suavizado_pd.columns = normalizado_pca.columns # AGREGAMOS LA CABECERA DE TIPOS
    
    if pca_op == 0:
        print("ESPECTRO SUAVIZADO POR SAVITZKY GOLAY")
        print(suavizado_pd) 
        mostrar_espectros(suavizado_pd,4,opcion,0)
    else:
        print("ESPECTRO SUAVIZADO POR SAVITZKY GOLAY")
        print("suavizado savitkz golay:",suavizado_pd.shape) 
        print(suavizado_pd)
        return suavizado_pd
    



 
# SUAVIZADO POR FILTRO GAUSIANO

def suavizado_filtroGausiano(normalizado_pca, pca_op):  #acordarse que se puede suavizar por la media, area y directo
    if pca_op == 0:
        print("NORMALIZAR POR:")
        print("1-MEDIA")
        print("2-Area")
        print("3-Sin normalizar")
        opcion = input("Selecciona una opción: ")
        sigma = int(input("INGRESE EL VALOR DE SIGMA: ")) #Un valor mayor de sigma produce un suavizado más fuerte
        print("Procesando los datos")
        print("Por favor espere un momento...")
        
     
        if opcion == '1'  :
            normalizado = df_media_pca
        elif opcion == '2' :
            normalizado = df_concatenado_cabecera_nueva_area
        elif opcion == '3' :
            normalizado = df2
        else:
            print("OPCION NO VALIDA")
            print("Salir...")
    else:
        sigma = int(input("INGRESE EL VALOR DE SIGMA: ")) #Un valor mayor de sigma produce un suavizado más fuerte
        normalizado = normalizado_pca
        print("Procesando los datos")
        print("Por favor espere un momento...")
         
    #print(type(normalizado))  
    #print(normalizado)
    dato = normalizado.to_numpy() #PASAMOS LOS DATOS A NUMPY (PIERDE LA CABECERA DE TIPOS AL HACER ESTO)
    #print(dato)
    #print(type(dato))
    #print(dato.dtype)  # me tira que es  Object, eso quiere decir que el array numpy contiene datos que no son de un tipo numerico uniforme
    # por lo que tendremos que forza su conversion con astype(float)
    dato = np.array(dato, dtype=float)
    #print(dato)
    #print(dato.dtype)
    suavizado_gaussiano = gaussian_filter(dato,sigma=sigma)
    #print(suavizado_gaussiano)
    suavizado_gaussiano_pd = pd.DataFrame(suavizado_gaussiano)
    #print("xxxxxxxxxxxxxxxxxxxxxxxxxx")
    #print(suavizado_gaussiano_pd)
     
    suavizado_gaussiano_pd.columns = cabecera.iloc[0,1:].values #agregamos la cabecera 
    #print(suavizado_gaussiano_pd)
    #print("RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRrr")
    if pca_op == 0:
        #print("ESPECTRO SUAVIZADO POR FILTRO GAUSSIANO")
        #print(suavizado_gaussiano_pd)
        mostrar_espectros(suavizado_gaussiano_pd,5,opcion,0)
    else:
        #print("ESPECTRO SUAVIZADO POR FILTRO GAUSSIANO")
        #print(suavizado_gaussiano_pd)
        #print("hizo bien su suavizado xDDDDDDDD")
        return suavizado_gaussiano_pd





def suavizado_mediamovil(normalizado_pca, pca_op):
    if pca_op == 0:
        print("NORMALIZAR POR:")
        print("1-Media")
        print("2-Area")
        print("3-Sin normalizar")
        opcion = input("Selecciona una opción: ")
        ventana = int(input("INGRESE EL VALOR DE LA VENTANA: "))
        print("Procesando los datos")
        print("Por favor espere un momento...")
        
     
        if opcion == '1'  :
            normalizado = df_media_pca
        elif opcion == '2' :
            normalizado = df_concatenado_cabecera_nueva_area
        elif opcion == '3' :
            normalizado = df2
        else:
            print("OPCION NO VALIDA")
            print("Salir...")
    else:
        ventana = int(input("INGRESE EL VALOR DE LA VENTANA: "))
        normalizado = normalizado_pca
        print("Procesando los datos")
        print("Por favor espere un momento...")
      
    suavizado_media_movil = pd.DataFrame()
    
    
    suavizado_media_movil = normalizado.rolling(window=ventana, center=True).mean() # mean() es para hallar el promedio
    
    if pca_op == 0:
        #print("ESPECTRO SUAVIZADO POR MEDIA MOVIL")
        #print(suavizado_media_movil)
        mostrar_espectros(suavizado_media_movil,6,opcion,0)
    else:
        #print("ESPECTRO SUAVIZADO POR MEDIA MOVIL")
        #print(suavizado_media_movil)
        return suavizado_media_movil
    
    
   # print(suavizado_media_movil)




def primera_derivada(normalizado, pca_op):
            
    print("entro en la funcion")
            
    if pca_op == 0:
        print("NORMALIZAR POR:")
        print("1-Media")
        print("2-Area")
        print("3-Sin normalizar")
        opcion = input("Selecciona una opción: ")
           
        if opcion == '1'  :
            normalizado = df_media_pca
        elif opcion == '2' :
            normalizado = df_concatenado_cabecera_nueva_area
        elif opcion == '3' :
            normalizado = df2
        else:
            print("OPCION NO VALIDA")
            print("Salir...")
       
        print("EL normalizado.SHAPE antes ES:", normalizado.shape)
        
        print("DESEA SUAVIZAR")
        print("1. SI")
        print("2. NO")
        opcion_s =  int(input("OPCION: "))
        if opcion_s == 1:
            print("\n--- POR CUAL METODO DESEAS SUAVIZAR ---")
            print("1. SUAVIZADO POR SAVIZTKY-GOLAY")
            print("2. SUAVIZADO POR FILTRO GAUSIANO")
            print("3. SUAVIZADO POR MEDIA MOVIL")
            metodo_suavizado = input("OPCION: ")
            if metodo_suavizado == '1':
                aux = suavizado_saviztky_golay(normalizado,1)
                print("EL aux.SHAPE antes ES:", aux.shape)
                #normalizado = suavizado_saviztky_golay(normalizado,1)
            elif metodo_suavizado == '2':
                print("normalizado shape por FG = ", normalizado.shape)
                #normalizado = suavizado_filtroGausiano(normalizado,1)
                aux = suavizado_filtroGausiano(normalizado,1)
                print("normalizado shape despues por FG = ", normalizado.shape)
            elif metodo_suavizado == '3':
                #normalizado = suavizado_mediamovil(normalizado,1) 
                aux = suavizado_mediamovil(normalizado,1) 
        else:
                metodo_suavizado = '4'         
                print("No se va a normalizar, directamente la derivada")
                 
                  # si viene 0 que haga todo eso pero si viene 1 desde la funcion del PCA que haller directo la primera derivada con esos parametros
           
    else:
       #daba error por que no retornaba nada
       normalizado_pca = normalizado

    if pca_op == 0 and opcion_s == 1:
        #print("entro aca")
        normalizado_f = aux
        print("EL DF NORMALIZADO_F ES:")
        print(normalizado_f)
    elif pca_op == 0 and opcion_s == 2:
        normalizado_f = normalizado
    else:
        normalizado_f = normalizado_pca #este es para cuando sea la funcion derivada sea llamado por la funcion del PCA
        print("aca va lo del PCA")
    
    
    df_derivada = normalizado_f.apply(pd.to_numeric, errors='coerce') # PASAMOS A NUMERICO SI ES NECESARIO
    print("xXXXXXXXxxXXXX")
    print(df_derivada)
    
    # Calcular la primera derivada
    df_derivada_diff = df_derivada.diff()
    
    # Imprimir la primera derivada
    print("Primera Derivada:")
    print(df_derivada_diff)
    
    
    plt.figure(figsize=(10, 6))
    

    if pca_op == 0 :
        print("LA PRIMERA DERIVADA ES:")
        print(df_derivada_diff.shape)
        #mostrar_espectros(df_derivada_diff, 8,opcion,metodo_suavizado)
        leyendas_tipos = set()  # almacenamos los tipos que enocntramos y la funcion set() nos ayuda a quer no se repitan
        pos_y=0
        for col in df_derivada_diff.columns :
            #print('entro normal')
            for tipo in asignacion_colores:
                #print("wwwwwww")
                if tipo == col :
                    color_actual= asignacion_colores[tipo] #ACA YA ENCONTRAMOS EL COLOR CORRESPONDIENTE A ESE TIPO   
                    #print(tipo,'==',col,'color=',color_actual) 
                    #print("ccccccc")
                    if isinstance(col, str):  #Verifica que el nombre de la columna sea un string
                        #print('RAMAN SHIFT')
                        #print(raman_shift)
                        #print('INTENSIDADES')
                        #print(df2.iloc[:,pos_y]) 
                        #print(col)
                        #print("xdxdxdxdxdxd")
                        if tipo in leyendas_tipos:
                            plt.plot(raman_shift , df_derivada_diff[col], color=color_actual, alpha=0.3, linewidth = 0.1,label=col) 
                            '''raman_shift:LE PASAMOS TODAS LAS INTENSIDADES , df2[col]= LE PASAMOS TODAS LAS COLUMNAS CON EL MISMO TIPO'''
                            print('entro4')
                            #print(pos_y)   
                            break
                        else:
                            plt.plot(raman_shift , df2[col], color=color_actual, alpha=0.3, linewidth = 0.1) 
                            leyendas_tipos.add(tipo)
                            print(leyendas_tipos)
                            print("entro 5")
                    pos_y+=1 
                        
        if opcion == '1':
            if metodo_suavizado == '1':
                plt.xlabel('Longitud de onda / Frecuencia')
                plt.ylabel('Intensidad')
                plt.title(f'Primera Derivada del archivo {bd_name} Suavizado por Saviztky_golay y Normalizado por la media ')
                plt.show()
            elif metodo_suavizado == '2':
                plt.xlabel('Longitud de onda / Frecuencia')
                plt.ylabel('Intensidad')
                plt.title(f'Primera Derivada del archivo {bd_name} Suavizado por FILTRO GAUSIANO y Normalizado por la media ')
                plt.show()
            elif metodo_suavizado == '3':
                plt.xlabel('Longitud de onda / Frecuencia')
                plt.ylabel('Intensidad')
                plt.title(f'Primera Derivada  del archivo {bd_name} Suavizado por MEDIA MOVIL y Normalizado por la media ')
                plt.show()
            else:
                plt.xlabel('Longitud de onda / Frecuencia')
                plt.ylabel('Intensidad')
                plt.title(f'Primera Derivada  del archivo {bd_name} Sin Suavizar y Normalizado por la media ')
                plt.show()
        elif opcion == '2':
            if metodo_suavizado == '1':
                plt.xlabel('Longitud de onda / Frecuencia')
                plt.ylabel('Intensidad')
                plt.title(f'Primera Derivada  del archivo {bd_name} Suavizado por Saviztky_golay y Normalizado por Area ')
                plt.show()
            elif metodo_suavizado == '2':
                plt.xlabel('Longitud de onda / Frecuencia')
                plt.ylabel('Intensidad')
                plt.title(f'Primera Derivada del archivo {bd_name} Suavizado por FILTRO GAUSIANO y Normalizado por Area ')
                plt.show()
            elif metodo_suavizado == '3':
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
            if metodo_suavizado == '1':
                plt.xlabel('Longitud de onda / Frecuencia')
                plt.ylabel('Intensidad')
                plt.title(f'Primera Derivada del archivo {bd_name} Suavizado por Saviztky_golay y sin Normalizar')
                plt.show()
            elif metodo_suavizado == '2':
                plt.xlabel('Longitud de onda / Frecuencia')
                plt.ylabel('Intensidad')
                plt.title(f'Primera Derivada del archivo {bd_name} Suavizado por FILTRO GAUSIANO y sin Normalizar ')
                plt.show()
            elif metodo_suavizado == '3':
                plt.xlabel('Longitud de onda / Frecuencia')
                plt.ylabel('Intensidad')
                plt.title(f'Primera Derivada del archivo {bd_name} Suavizado por MEDIA MOVIL y sin Normalizar')
                plt.show()
            else:
                plt.xlabel('Longitud de onda / Frecuencia')
                plt.ylabel('Intensidad')
                plt.title(f'Primera Derivada del archivo {bd_name} Sin Suavizar y sin Normalizar')
                plt.show()
                
    else:
        print("LA PRIMERA DERIVADA ES:")
        print(df_derivada_diff)
        return df_derivada_diff
        #para la llamada del PCA



    


 
def segunda_derivada(normalizado, pca_op):
            
    print("entro en la funcion")
            
    if pca_op == 0:
        print("NORMALIZAR POR:")
        print("1-Media")
        print("2-Area")
        print("3-Sin normalizar")
        opcion = input("Selecciona una opción: ")
           
        if opcion == '1'  :
            normalizado = df_media_pca
        elif opcion == '2' :
            normalizado = df_concatenado_cabecera_nueva_area
        elif opcion == '3' :
            normalizado = df2
        else:
            print("OPCION NO VALIDA")
            print("Salir...")
       
        print("EL normalizado.SHAPE antes ES:", normalizado.shape)
        
        print("DESEA SUAVIZAR")
        print("1. SI")
        print("2. NO")
        opcion_s =  int(input("OPCION: "))
        if opcion_s == 1:
            print("\n--- POR CUAL METODO DESEAS SUAVIZAR ---")
            print("1. SUAVIZADO POR SAVIZTKY-GOLAY")
            print("2. SUAVIZADO POR FILTRO GAUSIANO")
            print("3. SUAVIZADO POR MEDIA MOVIL")
            metodo_suavizado = input("OPCION: ")
            if metodo_suavizado == '1':
                aux = suavizado_saviztky_golay(normalizado,1)
                print("EL aux.SHAPE antes ES:", aux.shape)
                #normalizado = suavizado_saviztky_golay(normalizado,1)
            elif metodo_suavizado == '2':
                print("normalizado shape por FG = ", normalizado.shape)
                #normalizado = suavizado_filtroGausiano(normalizado,1)
                aux = suavizado_filtroGausiano(normalizado,1)
                print("normalizado shape despues por FG = ", normalizado.shape)
            elif metodo_suavizado == '3':
                #normalizado = suavizado_mediamovil(normalizado,1) 
                aux = suavizado_mediamovil(normalizado,1) 
        else:
                metodo_suavizado = '4'         
                print("No se va a normalizar, directamente la derivada")
                 
                  # si viene 0 que haga todo eso pero si viene 1 desde la funcion del PCA que haller directo la primera derivada con esos parametros
           
    else:
       #daba error por que no retornaba nada
       normalizado_pca = normalizado

    if pca_op == 0 and opcion_s == 1:
        #print("entro aca")
        normalizado_f = aux
        print("EL DF NORMALIZADO_F ES:")
        print(normalizado_f)
    elif pca_op == 0 and opcion_s == 2:
        normalizado_f = normalizado
    else:
        normalizado_f = normalizado_pca #este es para cuando sea la funcion derivada sea llamado por la funcion del PCA
        print("aca va lo del PCA")
    
    
    df_derivada = normalizado_f.apply(pd.to_numeric, errors='coerce') # PASAMOS A NUMERICO SI ES NECESARIO
    print("xXXXXXXXxxXXXX")
    print(df_derivada)
    
    # Calcular la primera derivada
    df_derivada_diff = df_derivada.diff()
    print("primera derivada")
    print(df_derivada_diff)
    # Calculamos la segunda derivada
    df_derivada_diff = df_derivada_diff.diff()
    print("segunda derivada")
    print(df_derivada_diff)
    # Imprimir la primera derivada
    print("Primera Derivada:")
    print(df_derivada_diff)
    
    
    plt.figure(figsize=(10, 6))
    

    if pca_op == 0 :
        print("LA PRIMERA DERIVADA ES:")
        print(df_derivada_diff.shape)
        #mostrar_espectros(df_derivada_diff, 8,opcion,metodo_suavizado)
        leyendas_tipos = set()  # almacenamos los tipos que enocntramos y la funcion set() nos ayuda a quer no se repitan
        pos_y=0
        for col in df_derivada_diff.columns :
            #print('entro normal')
            for tipo in asignacion_colores:
                #print("wwwwwww")
                if tipo == col :
                    color_actual= asignacion_colores[tipo] #ACA YA ENCONTRAMOS EL COLOR CORRESPONDIENTE A ESE TIPO   
                    #print(tipo,'==',col,'color=',color_actual) 
                    #print("ccccccc")
                    if isinstance(col, str):  #Verifica que el nombre de la columna sea un string
                        #print('RAMAN SHIFT')
                        #print(raman_shift)
                        #print('INTENSIDADES')
                        #print(df2.iloc[:,pos_y]) 
                        #print(col)
                        #print("xdxdxdxdxdxd")
                        if tipo in leyendas_tipos:
                            plt.plot(raman_shift , df_derivada_diff[col], color=color_actual, alpha=0.3, linewidth = 0.1,label=col) 
                            '''raman_shift:LE PASAMOS TODAS LAS INTENSIDADES , df2[col]= LE PASAMOS TODAS LAS COLUMNAS CON EL MISMO TIPO'''
                            print('entro4')
                            #print(pos_y)   
                            break
                        else:
                            plt.plot(raman_shift , df2[col], color=color_actual, alpha=0.3, linewidth = 0.1) 
                            leyendas_tipos.add(tipo)
                            print(leyendas_tipos)
                            print("entro 5")
                    pos_y+=1 
                        
        if opcion == '1':
            if metodo_suavizado == '1':
                plt.xlabel('Longitud de onda / Frecuencia')
                plt.ylabel('Intensidad')
                plt.title(f'Segunda Derivada del archivo {bd_name} Suavizado por Saviztky_golay y Normalizado por la media ')
                plt.show()
            elif metodo_suavizado == '2':
                plt.xlabel('Longitud de onda / Frecuencia')
                plt.ylabel('Intensidad')
                plt.title(f'Segunda Derivada del archivo {bd_name} Suavizado por FILTRO GAUSIANO y Normalizado por la media ')
                plt.show()
            elif metodo_suavizado == '3':
                plt.xlabel('Longitud de onda / Frecuencia')
                plt.ylabel('Intensidad')
                plt.title(f'Segunda Derivada  del archivo {bd_name} Suavizado por MEDIA MOVIL y Normalizado por la media ')
                plt.show()
            else:
                plt.xlabel('Longitud de onda / Frecuencia')
                plt.ylabel('Intensidad')
                plt.title(f'Segunda Derivada  del archivo {bd_name} Sin Suavizar y Normalizado por la media ')
                plt.show()
        elif opcion == '2':
            if metodo_suavizado == '1':
                plt.xlabel('Longitud de onda / Frecuencia')
                plt.ylabel('Intensidad')
                plt.title(f'Segunda Derivada  del archivo {bd_name} Suavizado por Saviztky_golay y Normalizado por Area ')
                plt.show()
            elif metodo_suavizado == '2':
                plt.xlabel('Longitud de onda / Frecuencia')
                plt.ylabel('Intensidad')
                plt.title(f'Segunda Derivada del archivo {bd_name} Suavizado por FILTRO GAUSIANO y Normalizado por Area ')
                plt.show()
            elif metodo_suavizado == '3':
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
            if metodo_suavizado == '1':
                plt.xlabel('Longitud de onda / Frecuencia')
                plt.ylabel('Intensidad')
                plt.title(f'Segunda Derivada del archivo {bd_name} Suavizado por Saviztky_golay y sin Normalizar')
                plt.show()
            elif metodo_suavizado == '2':
                plt.xlabel('Longitud de onda / Frecuencia')
                plt.ylabel('Intensidad')
                plt.title(f'Segunda Derivada del archivo {bd_name} Suavizado por FILTRO GAUSIANO y sin Normalizar ')
                plt.show()
            elif metodo_suavizado == '3':
                plt.xlabel('Longitud de onda / Frecuencia')
                plt.ylabel('Intensidad')
                plt.title(f'Segunda Derivada del archivo {bd_name} Suavizado por MEDIA MOVIL y sin Normalizar')
                plt.show()
            else:
                plt.xlabel('Longitud de onda / Frecuencia')
                plt.ylabel('Intensidad')
                plt.title(f'Segunda Derivada del archivo {bd_name} Sin Suavizar y sin Normalizar')
                plt.show()
                
    else:
        print("LA PRIMERA DERIVADA ES:")
        print(df_derivada_diff)
        return df_derivada_diff
        #para la llamada del PCA
   




#COMENZAR CON EL PCA

def  mostrar_pca():
    print("NORMALIZAR POR:")
    print("1-MEDIA")
    print("2-AREA")
    print("3-SIN NORMALIZAR")
    
    opcion = input("Selecciona una opción: ")
    
 
    if opcion == '1'  :
        normalizado_pca = df_media_pca
        print("Deseas Suavizar?")
        print("1- SI")
        print("2- NO")
        suavizar = int(input("OPCION: "))
        if suavizar == 1:
            print("\n--- POR CUAL METODO DESEAS SUAVIZAR ---")
            print("1. SUAVIZADO POR SAVIZTKY-GOLAY")
            print("2. SUAVIZADO POR FILTRO GAUSIANO")
            print("3. SUAVIZADO POR MEDIA MOVIL")
            metodo_suavizado = int(input("OPCION: "))
            if metodo_suavizado == 1:
                normalizado_pca = suavizado_saviztky_golay(normalizado_pca,1)
                # print("volvio")
                #print(metodo_suavizado)
            elif metodo_suavizado == 2:
                normalizado_pca = suavizado_filtroGausiano(normalizado_pca,1)
            else:
                normalizado_pca = suavizado_mediamovil(normalizado_pca,1) 
            
        print("\n--- DESEAS REALIZAR ALGUNA DERIVADA ---")  
        print("1- SI")
        print("2- NO")
        opcion = int(input("OPCION: "))
        if opcion == 1:
            print("1- PRIMERA DERIVADA")
            print("2- SEGUNDA DERIVADA")
            op_der= int(input("OPCION: "))
            if op_der == 1:
                 normalizado_pca = primera_derivada(normalizado_pca,1)
            else:
                 normalizado_pca = segunda_derivada(normalizado_pca,1)
        
    elif opcion == '2' :
        print("entor op 2")
        normalizado_pca = df_concatenado_cabecera_nueva_area
        print("Deseas Suavizar?")
        print("1- SI")
        print("2- NO")
        suavizar = int(input("OPCION: "))
        if suavizar == 1:
            print("\n--- POR CUAL METODO DESEAS SUAVIZAR ---")
            print("1. SUAVIZADO POR SAVIZTKY-GOLAY")
            print("2. SUAVIZADO POR FILTRO GAUSIANO")
            print("3. SUAVIZADO POR MEDIA MOVIL")
            metodo_suavizado = int(input("OPCION: "))
            if metodo_suavizado == 1:
                normalizado_pca = suavizado_saviztky_golay(normalizado_pca,1)
            elif metodo_suavizado == 2:
                normalizado_pca = suavizado_filtroGausiano(normalizado_pca,1)
            else:
                normalizado_pca = suavizado_mediamovil(normalizado_pca,1) 
        print("no suavizar xdd")
        print("\n--- DESEAS REALIZAR ALGUNA DERIVADA ---")  
        print("1- SI")
        print("2- NO")
        opcion = int(input("OPCION: "))
        if opcion == 1:
           print("1- PRIMERA DERIVADA")
           print("2- SEGUNDA DERIVADA")
           op_der= int(input("OPCION: "))
           if op_der == 1:
                print("si quiero derivar")
                #print(normalizado_pca)
                normalizado_pca = primera_derivada(normalizado_pca,1)
           else:
                normalizado_pca = segunda_derivada(normalizado_pca,1)      


    elif opcion == '3' :
        normalizado_pca = df2
        print("Deseas Suavizar?")
        print("1- SI")
        print("2- NO")
        suavizar = int(input("OPCION: "))
        if suavizar == 1:
            print("\n--- POR CUAL METODO DESEAS SUAVIZAR ---")
            print("1. SUAVIZADO POR SAVIZTKY-GOLAY")
            print("2. SUAVIZADO POR FILTRO GAUSIANO")
            print("3. SUAVIZADO POR MEDIA MOVIL")
            metodo_suavizado = int(input("OPCION: "))
            if metodo_suavizado == 1:
                normalizado_pca = suavizado_saviztky_golay(normalizado_pca,1)
            elif metodo_suavizado == 2:
                normalizado_pca = suavizado_filtroGausiano(normalizado_pca,1)
            else:
                normalizado_pca = suavizado_mediamovil(normalizado_pca,1) 
        print("\n--- DESEAS REALIZAR ALGUNA DERIVADA ---")  
        print("1- SI")
        print("2- NO")
        opcion = int(input("OPCION: "))
        if opcion == 1:
            print("1- PRIMERA DERIVADA")
            print("2- SEGUNDA DERIVADA")
            op_der= int(input("OPCION: "))
            if op_der == 1:
                normalizado_pca = primera_derivada(normalizado_pca,1)
            else:
                normalizado_pca = segunda_derivada(normalizado_pca,1)      

    else:
        print("OPCION NO VALIDA")
        print("SAlir...")
        #mostrar_menu()

    datos = pd.DataFrame(normalizado_pca)
    #print(datos)
    
    datos_df = datos.transpose() #PASAMOS LA CABECERA DE TIPOS A LA COLUMNA
    #print('prueba')
    #print(datos_df)
    
    datos_np = datos_df.to_numpy() # PASAMOS DE UN DATAFRAME PANDAS A UN ARRAY NUMPY
    #print(datos_np)
    
    pca = PCA(n_components=2)
    # Ajustar y transformar los datos
    dato_pca = pca.fit_transform(datos_np)
    #print(dato_pca)
    
    
    #print("Varianza explicada por cada componente:", pca.explained_variance_ratio_)
    colores_pca_original = [asignacion_colores.get(type_, 'black') for type_ in types]
    plt.figure(figsize=(10, 6))
    plt.scatter(dato_pca[:, 0], dato_pca[:, 1], c=colores_pca_original, alpha=0.7)
    
    if opcion == '1':   
        plt.xlabel('Raman_shift')
        plt.ylabel('Intensidad')
        plt.title('PCA NORMALIZADA POR LA MEDIA')
        plt.grid()
        plt.show()
    elif opcion == '2':
        plt.xlabel('Raman_shift')
        plt.ylabel('Intensidad')
        plt.title('PCA NORMALIZADA POR AREA')
        plt.grid()
        plt.show()
    else:
        plt.xlabel('Raman_shift')
        plt.ylabel('Intensidad')
        plt.title('PCA SIN NORMALIZAR')
        plt.grid()
        plt.show()        











if __name__ == "__main__":
     main()




    
    
    
    
    
    
    
