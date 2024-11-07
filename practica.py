#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 11:42:51 2024

@author: diego
"""

# main.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler # PARA LA NORMALIZACION POR LA MEDIA


#nombre = input("Por favor, ingresa tu nombre: ")
#print(f"Hola, {nombre}!")


bd_name = 'limpio.csv' #Este archivo contiene los datos espectroscópicos que serán leídos
df = pd.read_csv(bd_name, delimiter = ',' , header=None)
#print(df)


#GRAFICAMOS LOS ESPECTROS SIN NORMALIZAR#

raman_shift = df.iloc[1:, 0].reset_index(drop=True)  # EXTRAEMOS TODA LA PRIMERA COLUMNA, reset_index(drop=True) SIRVE PARA QUE EL INDICE COMIENCE EN 0 Y NO EN 1
print(raman_shift)
print(raman_shift.head(50))

intensity = df.iloc[1:, 1:] # EXTRAEMOS TODAS DEMAS COLUMNAS EXCEPTO LA PRIMERA FILA Y PRIMERA COLUMNA
print(intensity)

tipos = df.iloc[0, 1:] # EXTRAEMOS LA PRIMERA FILA MENOS DE LA PRIMERA COLUMNA
print(tipos)

cabecera = df.iloc[[0]].copy() # EXTRAEMOS LA PRIMERA FILA 
cabecera.drop( 0 ,axis=1, inplace=True) #eliminamos la primera columna no me sirve el indice cero
print(cabecera)

cant_tipos = tipos.nunique() # PARA EL EJEMPLO DE LIMPIO.CSV CANT_TIPOS TENDRA VALOR 4 YA QUE HAY 4 TIPOS (collagen,lipids,glycogen,DNA)
print(cant_tipos)

tipos_nombres = df.iloc[0, 1:].unique() # OBTENEMOS LOS NOMBRES DE LOS TIPOS
print(tipos_nombres)




colores = plt.cm.get_cmap('hsv', cant_tipos)

# Crear el diccionario de asignación de colores
asignacion_colores = {tipo: mcolors.to_hex(colores(i)) for i, tipo in enumerate(tipos_nombres)}

# Mostrar el diccionario de colores
#print("Diccionario de asignación de colores:")
#print(asignacion_colores)

diccionario=pd.DataFrame(asignacion_colores.items())
#print(diccionario)
#AHORA QUE YA TENGO ASIGNADO UN COLOR POR CADA TIPO TENGO QUE GRAFICAR LOS ESPECTROS#



#print('entro 10')

plt.figure(figsize=(1,1))    
for index, row in diccionario.iterrows():
    print('entro 15')
    tipo = row[0]   # Nombre del tipo (por ejemplo, 'collagen')
    color = row[1]  # Color asociado (por ejemplo, '#ff0000')
    plt.plot([], [], color=color, label=tipo) 
# Mostrar la leyenda y el gráfico
#print('entro 20')
plt.legend(loc='center')
plt.grid(False)
plt.axis('off')
plt.show()



# Graficar los espectros
plt.figure(figsize=(10, 6))

'''
########## DESCOMENTAR ESTE PARA VER COMO MUESTRA LA LEYENDA Y VER LA FORMA DE PONER EN EL MISMO PLOT QUE EN LOS DEMAS GRAFICO
for index, row in diccionario.iterrows():
    tipo = row[0]   # Nombre del tipo (por ejemplo, 'collagen')
    color = row[1]  # Color asociado (por ejemplo, '#ff0000')
    plt.plot([], [], color=color, label=tipo) 
# Mostrar la leyenda y el gráfico
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
'''      
   

cant_col = intensity.shape[1] #PARA SABER LA CANTIDAD DE COLUMNAS QUE TIENE EL ARCHIVO ORIGINAL
#print(cant_col)


#df2 = df.iloc[1:,:] # eliminamos los nombres de los tipos que estan en la cabecera
#print(df)

df2 = df.copy()
df2.columns = df2.iloc[0]
#print(df2)
df2 = df2.drop(0).reset_index(drop=True) #eliminamos la primera fila
df2 = df2.drop(df2.columns[0], axis=1) #eliminamos la primera columna el del rama_shift
#print(df2) # aca ya tenemos la tabla de la manera que necesitamos, fila cero es la cabecera con los nombres de los tipos anteriormente eran indice numericos consecutivos
df2 = df2.apply(pd.to_numeric, errors='coerce') #CONVERTIMOS A NUMERICO
#print(df2)
#print(df2.shape)


#print('entro 12')
#### GRAFICAMOS EL ESPECTRO SIN NORMALIZAR ##########

leyendas_tipos = set()  # almacenamos los tipos que enocntramos y la funcion set() nos ayuda a quer no se repitan
pos_y=0
for col in df2.columns :
    print('entro normal')
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
                    plt.plot(raman_shift , df2[col], color=color_actual, alpha=0.3, linewidth = 0.1,label=col) 
                    '''raman_shift:LE PASAMOS TODAS LAS INTENSIDADES , df2[col]= LE PASAMOS TODAS LAS COLUMNAS CON EL MISMO TIPO'''
                    #print('entro4')
                    #print(pos_y)   
                    break
                else:
                    plt.plot(raman_shift , df2[col], color=color_actual, alpha=0.3, linewidth = 0.1) 
                    leyendas_tipos.add(tipo) 
            pos_y+=1 



#print(leyendas_tipos) 
#print('entro 13')
# Etiquetas y título
plt.xlabel('Longitud de onda / Frecuencia')
plt.ylabel('Intensidad')
plt.title(f'Espectros del archivo {bd_name}')
plt.show()




##### VEMOS PARA  IMPLEMENTAR EL NORMALIZADO POR LA MEDIA

'''
La normalización por la media es una técnica que ajusta los valores de una variable o conjunto de datos
 para que tengan una media de 0 y una desviación estándar de 1.
'''

'''
NORMALMENTE NO SE NORMALIZA EL RAMAN_SHIFT , TAMBIEN ATENDER QUE TIENE QUE SER TODO NUMERO PARA EL fit_transform()
OSEA BORRAR CABECERA
'''

scaler = StandardScaler() 
cal_nor = scaler.fit_transform(intensity) #calcula la media y desviación estándar
dato_normalizado = pd.DataFrame(cal_nor, columns=intensity.columns) # lo convertimos de vuelta en un DataFrame

print(dato_normalizado)

df_concatenado = pd.concat([cabecera,dato_normalizado], axis=0, ignore_index=True)
print(df_concatenado)


# Paso 1: Convertir la primera fila en cabecera
df_concatenado.columns = df_concatenado.iloc[0]  # Asigna la primera fila como nombres de columna

# Paso 2: Eliminar la primera fila (ahora es la cabecera) y resetear el índice
df_concatenado_cabecera_nueva = df_concatenado[1:].reset_index(drop=True)
print(df_concatenado_cabecera_nueva.head(50))

print('normalizacion media')

leyendas_tipos = set()  # almacenamos los tipos que enocntramos y la funcion set() nos ayuda a quer no se repitan
pos_y=0
for col in df_concatenado_cabecera_nueva.columns :
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
                    plt.plot(raman_shift , df_concatenado_cabecera_nueva[col], color=color_actual, alpha=0.3, linewidth = 0.1,label=col) 
                  # raman_shift:LE PASAMOS TODAS LAS INTENSIDADES , df2[col]= LE PASAMOS TODAS LAS COLUMNAS CON EL MISMO TIPO'''
                    #print('entro4')
                    #print(pos_y)   
                    break
                else:
                    plt.plot(raman_shift , df_concatenado_cabecera_nueva[col], color=color_actual, alpha=0.3, linewidth = 0.1) 
                    leyendas_tipos.add(tipo) 
            pos_y+=1 



#print(leyendas_tipos) 
#print('entro 13')
# Etiquetas y título
plt.xlabel('Longitud de onda / Frecuencia')
plt.ylabel('Intensidad')
plt.title(f'Espectros del archivo Normalizado por la media {bd_name}')
plt.show()





##### VEMOS PARA  IMPLEMENTAR EL NORMALIZADO POR AREA



df3 = pd.DataFrame(intensity)
print("DataFrame de Intensidades:")
print(df3)
df3 = df3.apply(pd.to_numeric, errors='coerce')  # Convierte a numérico, colocando NaN donde haya problemas
print(df3)

np_array = raman_shift.to_numpy() #CONVERTIMOS INTENSITY AL TIPO NUMPY POR QUE POR QUE NP.TRAPZ UTILIZA ESE TIPO DE DATOS
print(np_array)



df3_normalizado = df3.copy()
print(df3)
# Cálculo del área bajo la curva para cada columna
print("\nÁreas bajo la curva para cada columna:")
for col in df3.columns:
    print(df3[col])
    print(df3_normalizado[col])
    area = (np.trapz(df3[col], np_array))*-1  #MULTIPLIQUE POR -1 PARA QUE EL GRAFICO SALGA TODO HACIA ARRIBA
    if area != 0:
        df3_normalizado[col] = df3[col] / area
    else:
        print(f"Advertencia: El área de la columna {col} es cero y no se puede normalizar.") #seguro contra errores de división por cero 
print(df3_normalizado)


df_concatenado_area = pd.concat([cabecera,df3_normalizado], axis=0, ignore_index=True)
print(df_concatenado_area)


# Paso 1: Convertir la primera fila en cabecera
df_concatenado_area.columns = df_concatenado_area.iloc[0]  # Asigna la primera fila como nombres de columna
# Paso 2: Eliminar la primera fila (ahora es la cabecera) y resetear el índice
df_concatenado_cabecera_nueva_area = df_concatenado_area[1:].reset_index(drop=True)
print(df_concatenado_cabecera_nueva_area)


leyendas_tipos = set()  # almacenamos los tipos que enocntramos y la funcion set() nos ayuda a quer no se repitan
pos_y=0
for col in df_concatenado_cabecera_nueva_area.columns :
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
                    plt.plot(raman_shift , df_concatenado_cabecera_nueva_area[col], color=color_actual, alpha=0.3, linewidth = 0.1,label=col) 
                  # raman_shift:LE PASAMOS TODAS LAS INTENSIDADES , df2[col]= LE PASAMOS TODAS LAS COLUMNAS CON EL MISMO TIPO'''
                    #print('entro4')
                    #print(pos_y)   
                    break
                else:
                    plt.plot(raman_shift , df_concatenado_cabecera_nueva_area[col], color=color_actual, alpha=0.3, linewidth = 0.1) 
                    leyendas_tipos.add(tipo) 
            pos_y+=1 



#print(leyendas_tipos) 
#print('entro 13')
# Etiquetas y título
plt.xlabel('Longitud de onda / Frecuencia')
plt.ylabel('Intensidad')
plt.title(f'Espectros del archivo Normalizado por la Area {bd_name}')
plt.show()






#COMENZAR CON LOS METODOS DE SUAVIZADO


























