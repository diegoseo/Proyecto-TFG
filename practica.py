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

bd_name = 'limpio.csv' #Este archivo contiene los datos espectroscópicos que serán leídos
df = pd.read_csv(bd_name, delimiter = ',' , header=None)
print(df)


#GRAFICAMOS LOS ESPECTROS SIN NORMALIZAR#

raman_shift = df.iloc[1:, 0].reset_index(drop=True)  # EXTRAEMOS TODA LA PRIMERA COLUMNA, reset_index(drop=True) SIRVE PARA QUE EL INDICE COMIENCE EN 0 Y NO EN 1
print(raman_shift)

intensity = df.iloc[1:, 1:] # EXTRAEMOS TODAS DEMAS COLUMNAS EXCEPTO LA PRIMERA FILA Y PRIMERA COLUMNA
print(intensity)

tipos = df.iloc[0, 1:] # EXTRAEMOS LA PRIMERA FILA MENOS DE LA PRIMERA COLUMNA
print(tipos)

cant_tipos = tipos.nunique() # PARA EL EJEMPLO DE LIMPIO.CSV CANT_TIPOS TENDRA VALOR 4 YA QUE HAY 4 TIPOS (collagen,lipids,glycogen,DNA)
print(cant_tipos)

tipos_nombres = df.iloc[0, 1:].unique() # OBTENEMOS LOS NOMBRES DE LOS TIPOS
print(tipos_nombres)




colores = plt.cm.get_cmap('hsv', cant_tipos)

# Crear el diccionario de asignación de colores
asignacion_colores = {tipo: mcolors.to_hex(colores(i)) for i, tipo in enumerate(tipos_nombres)}

# Mostrar el diccionario de colores
print("Diccionario de asignación de colores:")
print(asignacion_colores)

diccionario=pd.DataFrame(asignacion_colores.items())
print(diccionario)
#AHORA QUE YA TENGO ASIGNADO UN COLOR POR CADA TIPO TENGO QUE GRAFICAR LOS ESPECTROS#





plt.figure(figsize=(1,1))    
for index, row in diccionario.iterrows():
    tipo = row[0]   # Nombre del tipo (por ejemplo, 'collagen')
    color = row[1]  # Color asociado (por ejemplo, '#ff0000')
    plt.plot([], [], color=color, label=tipo) 
# Mostrar la leyenda y el gráfico
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
print(cant_col)


#df2 = df.iloc[1:,:] # eliminamos los nombres de los tipos que estan en la cabecera
print(df)

df2 = df.copy()
df2.columns = df2.iloc[0]
print(df2)
df2 = df2.drop(0).reset_index(drop=True) #eliminamos la primera fila
df2 = df2.drop(df2.columns[0], axis=1) #eliminamos la primera columna el del rama_shift
print(df2) # aca ya tenemos la tabla de la manera que necesitamos, fila cero es la cabecera con los nombres de los tipos anteriormente eran indice numericos consecutivos
df2 = df2.apply(pd.to_numeric, errors='coerce') #CONVERTIMOS A NUMERICO
print(df2)
print(df2.shape)



#### GRAFICAMOS EL ESPECTRO SIN NORMALIZAR ##########

leyendas_tipos = set()  # almacenamos los tipos que enocntramos y la funcion set() nos ayuda a quer no se repitan
pos_y=0
for col in df2.columns :
    for tipo in asignacion_colores:
        if tipo == col :
            color_actual= asignacion_colores[tipo] #ACA YA ENCONTRAMOS EL COLOR CORRESPONDIENTE A ESE TIPO   
            print(tipo,'==',col,'color=',color_actual) 
            if isinstance(col, str):  #Verifica que el nombre de la columna sea un string
                print('RAMAN SHIFT')
                print(raman_shift)
                print('INTENSIDADES')
                #print(df2.iloc[:,pos_y]) 
                if tipo in leyendas_tipos:
                    plt.plot(raman_shift , df2[col], color=color_actual, alpha=0.3, linewidth = 0.1,label=col) 
                    '''raman_shift:LE PASAMOS TODAS LAS INTENSIDADES , df2[col]= LE PASAMOS TODAS LAS COLUMNAS CON EL MISMO TIPO'''
                    print('entro4')
                    print(pos_y)   
                    break
                else:
                    plt.plot(raman_shift , df2[col], color=color_actual, alpha=0.3, linewidth = 0.1) 
                    leyendas_tipos.add(tipo) 
            pos_y+=1 



#print(leyendas_tipos) 

# Etiquetas y título
plt.xlabel('Longitud de onda / Frecuencia')
plt.ylabel('Intensidad')
plt.title(f'Espectros del archivo {bd_name}')

