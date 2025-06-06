#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 11:42:51 2024

@author: diego
"""

# main.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler # PARA LA NORMALIZACION POR LA MEDIA Y PCA
from sklearn.decomposition import PCA


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


#GRAFICAMOS LOS ESPECTROS SIN NORMALIZAR#

raman_shift = df.iloc[1:, 0].reset_index(drop=True)  # EXTRAEMOS TODA LA PRIMERA COLUMNA, reset_index(drop=True) SIRVE PARA QUE EL INDICE COMIENCE EN 0 Y NO EN 1
#print(raman_shift)
#print(raman_shift.head(50))

intensity = df.iloc[1:, 1:] # EXTRAEMOS TODAS DEMAS COLUMNAS EXCEPTO LA PRIMERA FILA Y PRIMERA COLUMNA
#print(intensity)   

tipos = df.iloc[0, 1:] # EXTRAEMOS LA PRIMERA FILA MENOS DE LA PRIMERA COLUMNA
print(tipos)
types=tipos.tolist()
print(types)

cabecera = df.iloc[[0]].copy() # EXTRAEMOS LA PRIMERA FILA 
cabecera.drop( 0 ,axis=1, inplace=True) #eliminamos la primera columna no me sirve el indice cero
#print(cabecera)


cant_tipos = tipos.nunique() # PARA EL EJEMPLO DE LIMPIO.CSV CANT_TIPOS TENDRA VALOR 4 YA QUE HAY 4 TIPOS (collagen,lipids,glycogen,DNA)
#print(cant_tipos)

tipos_nombres = df.iloc[0, 1:].unique() # OBTENEMOS LOS NOMBRES DE LOS TIPOS
#print(tipos_nombres)




colores = plt.cm.get_cmap('hsv', cant_tipos)

# Crear el diccionario de asignación de colores
asignacion_colores = {tipo: mcolors.to_hex(colores(i)) for i, tipo in enumerate(tipos_nombres)}

# Mostrar el diccionario de colores
#print("Diccionario de asignación de colores:")
#print(asignacion_colores)

diccionario=pd.DataFrame(asignacion_colores.items())
#print(diccionario)
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
#print(df2) #ESTA VARIABLE SE USA PARA EL PCA TAMBIEN
#print(df2.shape)

"""
VARIABLES DE NORMALIZAR POR LA MEDIA
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
df_media_pca= pd.DataFrame(df_concatenado_cabecera_nueva)
#print(df_media_pca) #ESTA VARIABLE SE USA PARA EL PCA TAMBIEN
#print('normalizacion media')



"""
VARIABLES DE NORMALIZAR POR AREA
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
    area = (np.trapz(df3[col], np_array))*-1  #MULTIPLIQUE POR -1 PARA QUE EL GRAFICO SALGA TODO HACIA ARRIBA ESTO SE DEBE A QUE EL RAMAN_SHIFT ESTA EN FORMA DECRECIENTE
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
#print(df_concatenado_cabecera_nueva_area) #ESTA VARIABLE SE USA PARA EL PCA TAMBIEN
#print('entro 10')




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






'''
    PREPARAMOS EL SIGUIENTE MENU
'''

def mostrar_menu():
    print("\n--- Menú Principal ---")
    print("1. MOSTRAR ESPECTROS")
    print("2. NORMALIZAR POR MEDIA")
    print("3. NORMALIZAR POR AREA")
    print("4. PCA")
    print("5. SUAVIZADO POR SAVIZTKY-GOLAY")
    print("6. SUAVIZADO POR FILTRO GAUSIANO")
    print("7. SUAVIZADO POR MEDIA MOVIL")
    print("8. PRIMERA DERIVADA")
    print("9. SEGUNDA DERIVADA")
    print("10. Salir")




      # 
      # elif opcion == '5':
      #     suavizado_saviztky-golay()
      # elif opcion == '6':
      #      suavizado_filtroGausiano()
      # elif opcion == '7':
      #      suavizado_mediamovil()
      # elif opcion == '8':
      #      primera_derivada()
      # elif opcion == '9':
      #      segunda_derivada()
 

def main():
    while True:
        mostrar_menu()
        opcion = input("Selecciona una opción: ")
        
        if opcion == '1':
            metodo = 1
            print("Procesando los datos")
            print("Por favor espere un momento")
            mostrar_espectros(df2,metodo)
        elif opcion == '2':
            metodo = 2
            print("Procesando los datos")
            print("Por favor espere un momento")
            mostrar_espectros(df_media_pca,metodo)
        elif opcion == '3':
            metodo = 3
            print("Procesando los datos")
            print("Por favor espere un momento")
            mostrar_espectros(df_concatenado_cabecera_nueva_area,metodo)
        elif opcion == '4':
            mostrar_pca()
        elif opcion == '10':
            print("Saliendo del programa...")
            break
        else:
            print("Opción no válida. Inténtalo de nuevo.")





def mostrar_espectros(datos,metodo):
    
    
    # Graficar los espectros
    plt.figure(figsize=(10, 6))
  
    
    leyendas_tipos = set()  # almacenamos los tipos que enocntramos y la funcion set() nos ayuda a quer no se repitan
    pos_y=0
    for col in datos.columns :
        #print('entro normal')
        for tipo in asignacion_colores:
            if tipo == col :
                color_actual= asignacion_colores[tipo] #ACA YA ENCONTRAMOS EL COLOR CORRESPONDIENTE A ESE TIPO   
                #print(tipo,'==',col,'color=',color_actual) 
                if isinstance(col, str):  #Verifica que el nombre de la columna sea un string
                    #print('RAMAN SHIFT')
                    #print(raman_shift)
                    #print('INTENSIDADES')
                    #print(df2.iloc[:,pos_y]) 
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
    
    #TODO ESTE CONDICIONAL ES SOLO PARA QUE EL TITULO DEL GRAFICO MUESTRE POR CUAL METODO SE NORMALIZO
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
        plt.title(f'Espectros del archivo Normalizado por la Media {bd_name}')
        plt.show()
    else:
        #print(leyendas_tipos) 
        #print('entro 13')
        # Etiquetas y título
        plt.xlabel('Longitud de onda / Frecuencia')
        plt.ylabel('Intensidad')
        plt.title(f'Espectros del archivo Normalizado por Area {bd_name}')
        plt.show()




#COMENZAR CON EL PCA

def  mostrar_pca():
    print("NORMALIZAR POR:")
    print("1-MEDIA")
    print("2-Area")
    print("3-Sin normalizar")
    opcion = input("Selecciona una opción: ")
    
 
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
        print("SAlir...")
        #mostrar_menu()


    pca = PCA(n_components=2)

    # Ajustar y transformar los datos
    dato_pca = pca.fit_transform(normalizado_pca)
    #print(dato_pca)
    print("Varianza explicada por cada componente:", pca.explained_variance_ratio_)
    colores_pca_original = [asignacion_colores.get(type_, 'black') for type_ in types]
    plt.figure(figsize=(8, 6))
    plt.scatter(dato_pca[:, 0], dato_pca[:, 1], c='black', alpha=0.7)
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.title('Proyección en las Primeras 2 Componentes Principales')
    plt.grid()
    plt.show()






if __name__ == "__main__":
     main()






























